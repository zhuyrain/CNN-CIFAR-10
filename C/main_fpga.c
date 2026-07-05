#include <stdint.h>
#include "my_printf.h"
#include "npu.h"
#include "../python/test_vector_fpga.h"
#include "conv2d_int8_naive.h"

// ==========================================
// 中断与系统控制逻辑
// ==========================================
extern void enable_npu_irq(uint32_t mask);
extern void wait_for_irq(void);

volatile int npu_task_finished = 0;

void npu_isr(void) {
    npu_task_finished = 1;
    NPU_REG_CTRL_STATUS = 0x00000004; 
}

void wait_npu_done() {
    while (!npu_task_finished) { wait_for_irq(); }
    npu_task_finished = 0;
    my_printf("-"); // 打印进度条
}

// ==========================================
// 内存分配 (软硬分离的安全复用池)
// ==========================================

// 1. 硬件运算内存池 (64KB): 专供 NPU 卷积和 FC 层使用
int8_t buf_conv_base[64 * 32 * 32] __attribute__((aligned(4))) = {0}; 

// 2. 软件运算内存池 (16KB): 专供 CPU 池化层使用 (防 Read-After-Write 践踏)
int8_t buf_pool_base[64 * 16 * 16] __attribute__((aligned(4))) = {0};

// ==========================================
// 零代码修改：宏定义自动路由
// ==========================================

// Layer 1 [Conv1]: 写入 64KB 池
#define buf_conv1 (buf_conv_base)

// Layer 2 [Pool1]: 读 64KB 池，写入 16KB 池 (软硬物理分离)
#define buf_pool1 (buf_pool_base)

// Layer 3 [Conv2]: 读 16KB 池，写回 64KB 池 (硬件安全复用)
#define buf_conv2 (buf_conv_base)

// Layer 4 [Pool2]: 读 64KB 池，写回 16KB 池的首部安全区域
#define buf_pool2 (buf_pool_base)

// Layer 5 [FC]: 读 16KB 池，写回 64KB 池的安全空闲区域
#define buf_fc    (buf_conv_base + 32768)

int main() {
    // 开启中断
    enable_npu_irq(~(1 << 4));

    my_printf("\n========================================\n");
    my_printf(" [FPGA TEST] Simple 2-Layer CNN Inference \n");
    my_printf("             Heterogeneous Pipeline (NPU + CPU)\n");
    my_printf("========================================\n");
    
    // 设置全局的流水线安全控制参数
    uint32_t drain_timeout  = 132;   
    uint32_t fifo_af_thresh = 40; 

    // -----------------------------------------------------------
    // [1/5] Layer 1: Hardware Conv1 (32x32x3 -> 32x32x16)
    // -----------------------------------------------------------
    my_printf(" -> [1/5] Executing NPU Conv1 (4 Passes for 16 OC)\n  ");
    NPU_REG_CFG_IMG_DIM  = (32 << 16) | 32;      
    
    // Row_words = 32*(0+1) = 32. Weight_words = 9(w/OC) * 4(OC) = 36.
    NPU_REG_CFG_MEM_CNT  = (32 << 16) | 36;
    
    // out_stride=16, lb_ic_groups_r=0, sa_weight_num=9, lb_line_width=34, lb_ic_groups=0
    NPU_REG_CFG_DATAPATH = (16 << 23) | (0 << 19) | (9 << 11) | (34 << 4) | 0;
    
    // oc_num=4 (每次算4个OC), col_group_en=0x0001 (激活1个Group=4列)
    NPU_REG_CFG_SPATIAL  = (4 << 24) | (0x0001 << 8) | (0 << 5) | (1 << 4) | (2 << 2) | 2;
    
    NPU_REG_CFG_QUANT    = (0 << 16) | L1_MULT;
    NPU_REG_CFG_POST     = (drain_timeout << 22) | (fifo_af_thresh << 12) | (L1_SHIFT << 1) | 1;
    
    NPU_REG_ACT_BASE     = (uint32_t)test_input_img;

    for (int pass = 0; pass < 4; pass++) {
        NPU_REG_BIAS_BASE    = (uint32_t)test_conv1_bias + (pass * 16);           // 4 OCs * 4 Bytes
        NPU_REG_WEIGHT_BASE  = (uint32_t)test_conv1_weight + (pass * 144);        // 36 Words * 4 Bytes
        NPU_REG_OUT_BASE     = (uint32_t)buf_conv1 + (pass * 4);                  // 4 OCs * 1 Byte offset
        NPU_REG_CTRL_STATUS  = 0x00000001;
        wait_npu_done();
    }
    my_printf("\n");

    // -----------------------------------------------------------
    // [2/5] Layer 2: Software Pool1 (32x32x16 -> 16x16x16)
    // -----------------------------------------------------------
    my_printf(" -> [2/5] CPU Executing MaxPool 1...\n");
    maxpool2d_int8_hwc(buf_conv1, buf_pool1, 16, 32, 2, 2);

    // -----------------------------------------------------------
    // [3/5] Layer 3: Hardware Conv2 (16x16x16 -> 16x16x32)
    // -----------------------------------------------------------
    my_printf(" -> [3/5] Executing NPU Conv2 (8 Passes for 32 OC)\n  ");
    NPU_REG_CFG_IMG_DIM  = (16 << 16) | 16;      
    
    // Row_words = 16*(3+1) = 64. Weight_words = 36(w/OC) * 4(OC) = 144.
    NPU_REG_CFG_MEM_CNT  = (64 << 16) | 144;
    
    // 黄金时间复用折叠：
    // out_stride=32, lb_ic_groups_r=3, sa_weight_num=36(压入144物理极限内!), lb_line_width=18, lb_ic_groups=3
    NPU_REG_CFG_DATAPATH = (32 << 23) | (3 << 19) | (36 << 11) | (18 << 4) | 3;
    
    NPU_REG_CFG_SPATIAL  = (4 << 24) | (0x0001 << 8) | (0 << 5) | (1 << 4) | (2 << 2) | 2;
    NPU_REG_CFG_QUANT    = (0 << 16) | L2_MULT;
    NPU_REG_CFG_POST     = (drain_timeout << 22) | (fifo_af_thresh << 12) | (L2_SHIFT << 1) | 1;
    
    NPU_REG_ACT_BASE     = (uint32_t)buf_pool1;

    for (int pass = 0; pass < 8; pass++) {
        NPU_REG_BIAS_BASE    = (uint32_t)test_conv2_bias + (pass * 16);           // 4 OCs * 4 Bytes
        NPU_REG_WEIGHT_BASE  = (uint32_t)test_conv2_weight + (pass * 576);        // 144 Words * 4 Bytes
        NPU_REG_OUT_BASE     = (uint32_t)buf_conv2 + (pass * 4);
        NPU_REG_CTRL_STATUS  = 0x00000001;
        wait_npu_done();
    }
    my_printf("\n");

    // -----------------------------------------------------------
    // [4/5] Layer 4: Software Pool2 (16x16x32 -> 8x8x32)
    // -----------------------------------------------------------
    my_printf(" -> [4/5] CPU Executing MaxPool 2...\n");
    maxpool2d_int8_hwc(buf_conv2, buf_pool2, 32, 16, 2, 2);

    // -----------------------------------------------------------
    // [5/5] Layer 5: Software FC Layer (Heterogeneous Fallback)
    // -----------------------------------------------------------
    // FC 输入节点数为 32 * 8 * 8 = 2048。
    my_printf(" -> [5/5] CPU Executing FC Layer (NPU PE Capacity Fallback)...\n");
    
    linear_int8_hwc(
        buf_pool2, test_fc_weight, test_fc_bias, buf_fc,
        32, 8, 8, 10, FC_MULT, FC_SHIFT
    );

    // ===========================================================
    // 终极审判：端到端输出比对
    // ===========================================================
    my_printf("\n>>> Final Verification: Network Classification Output <<<\n");
    
    int8_t* ptr_hw_out  = (int8_t*)buf_fc;
    int8_t* ptr_exp_out = (int8_t*)expected_fc_output;
    int error_cnt = 0;
    
    for (int ch = 0; ch < 10; ch++) {
        int8_t val = ptr_hw_out[ch];
        int8_t exp = ptr_exp_out[ch];
        
        if (val == exp) {
            my_printf("    CH[%02d]: MATCH (Logit = %d)\n", ch, val);
        } else {
            my_printf("    [ERROR] CH[%02d] -> HW: %d | EXP: %d\n", ch, val, exp);
            error_cnt++;
        }
    }
    
    if (error_cnt == 0) {
        my_printf("\n  [PASS] Simple CNN Inference is 100%% Accurate!\n");
        my_printf("========================================\n");
        my_printf(" FPGA HARDWARE DEPLOYMENT READY!\n");
        my_printf("========================================\n");
    } else {
        my_printf("\n  [FAIL] Network inference failed with %d errors.\n", error_cnt);
        while(1);
    }

    while(1);
    return 0;
}