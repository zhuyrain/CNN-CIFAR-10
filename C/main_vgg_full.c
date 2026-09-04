#include <stdint.h>
#include "my_printf.h"
#include "npu.h"
#include "../python/test_vector_vgg_full.h"
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
    my_printf("-");
}
 
// 考虑到 VGG-Lite 中 Conv2 需要读 64KB 并同时写 64KB，
// 我们必须准备两个 64KB 的物理内存池进行 Ping-Pong 交替，
// 这样可以绝对防止脉动阵列与 DMA 之间的 Read-After-Write (RAW) 数据践踏。
// 总内存占用从静态的 ~180KB 完美降至 128KB！

// 池 A (64KB): 默认给奇数层作输出缓冲
int8_t mem_pool_A[64 * 32 * 32] __attribute__((aligned(4))) = {0}; 
// 池 B (64KB): 默认给偶数层作输出缓冲
int8_t mem_pool_B[64 * 32 * 32] __attribute__((aligned(4))) = {0}; 

// ==========================================
// 零代码修改：宏定义自动 Ping-Pong 路由
// ==========================================

// [1/7] Conv1: 3x32x32 -> 64x32x32. 读输入图片，写入 池A
#define buf_conv1 (mem_pool_A)

// [2/7] Conv2: 64x32x32 -> 64x32x32. 读 池A，写入 池B (双池火力全开)
#define buf_conv2 (mem_pool_B)

// [3/7] Pool1: 64x32x32 -> 64x16x16. 读 池B，写入 池A 的前 16KB
#define buf_pool1 (mem_pool_A)

// [4/7] Conv3: 64x16x16 -> 64x16x16. 读 池A，写入 池B 的前 16KB
#define buf_conv3 (mem_pool_B)

// [5/7] Conv4: 64x16x16 -> 64x16x16. 读 池B，写入 池A 的前 16KB
#define buf_conv4 (mem_pool_A)

// [6/7] Pool2: 64x16x16 -> 64x8x8. 读 池A，写入 池B 的前 4KB
#define buf_pool2 (mem_pool_B)

// [7/7] FC: 64x8x8 -> 10. 读 池B，写入 池A 的空闲区域 (安全起见复用在池A起始)
#define buf_fc    (mem_pool_A)

int main() {
    // 开启中断
    enable_npu_irq(~(1 << 4));

    my_printf("\n========================================\n");
    my_printf(" [TEST 11] End-to-End VGG Inference \n");
    my_printf("           Heterogeneous Pipeline (NPU + CPU)\n");
    my_printf("========================================\n");
    
    // 设置全局的流水线安全控制参数
    uint32_t drain_timeout  = 132;   
    uint32_t fifo_af_thresh = 40; 

    // -----------------------------------------------------------
    // [1/7] Layer 1: Hardware Conv1 (32x32x3 -> 32x32x64)
    // -----------------------------------------------------------
    my_printf(" -> [1/7] Executing NPU Conv1 (64 OC)...\n");
    NPU_REG_CFG_IMG_DIM  = (32 << 16) | 32;      
    NPU_REG_CFG_MEM_CNT  = (32 << 16) | 576;
    NPU_REG_CFG_DATAPATH = (64 << 23) | (0 << 19) | (9 << 11) | (34 << 4) | 0;
    NPU_REG_CFG_SPATIAL  = (64 << 24) | (0xFFFF << 8) | (0 << 5) | (1 << 4) | (2 << 2) | 2;
    NPU_REG_CFG_QUANT    = (0 << 16) | L1_MULT;
    NPU_REG_CFG_POST     = (drain_timeout << 22) | (fifo_af_thresh << 12) | (L1_SHIFT << 1) | 1;
    
    NPU_REG_ACT_BASE     = (uint32_t)test_input_img;
    NPU_REG_BIAS_BASE    = (uint32_t)test_conv1_bias;
    NPU_REG_WEIGHT_BASE  = (uint32_t)test_conv1_weight;
    NPU_REG_OUT_BASE     = (uint32_t)buf_conv1;
    NPU_REG_CTRL_STATUS  = 0x00000001;
    wait_npu_done();

    // -----------------------------------------------------------
    // [2/7] Layer 2: Hardware Conv2 (32x32x64 -> 32x32x64)
    // -----------------------------------------------------------
    my_printf(" -> [2/7] Executing NPU Conv2 (64 OC)...\n");
    NPU_REG_CFG_IMG_DIM  = (32 << 16) | 32;      
    NPU_REG_CFG_MEM_CNT  = (512 << 16) | 9216;
    NPU_REG_CFG_DATAPATH = (64 << 23) | (0 << 19) | (9 << 11) | (34 << 4) | 15;
    NPU_REG_CFG_SPATIAL  = (64 << 24) | (0xFFFF << 8) | (0 << 5) | (1 << 4) | (2 << 2) | 2;
    NPU_REG_CFG_QUANT    = (0 << 16) | L2_MULT;
    NPU_REG_CFG_POST     = (drain_timeout << 22) | (fifo_af_thresh << 12) | (L2_SHIFT << 1) | 1;
    
    NPU_REG_ACT_BASE     = (uint32_t)buf_conv1;
    NPU_REG_BIAS_BASE    = (uint32_t)test_conv2_bias;
    NPU_REG_WEIGHT_BASE  = (uint32_t)test_conv2_weight;
    NPU_REG_OUT_BASE     = (uint32_t)buf_conv2;
    NPU_REG_CTRL_STATUS  = 0x00000001;
    wait_npu_done();

    // -----------------------------------------------------------
    // [3/7] Layer 3: Software Pool1 (32x32x64 -> 16x16x64)
    // -----------------------------------------------------------
    my_printf(" -> [3/7] CPU Executing MaxPool 1...\n");
    maxpool2d_int8_hwc(buf_conv2, buf_pool1, 64, 32, 2, 2);

    // -----------------------------------------------------------
    // [4/7] Layer 4: Hardware Conv3 (16x16x64 -> 16x16x64)
    // -----------------------------------------------------------
    my_printf(" -> [4/7] Executing NPU Conv3 (64 OC)...\n");
    NPU_REG_CFG_IMG_DIM  = (16 << 16) | 16;      
    NPU_REG_CFG_MEM_CNT  = (256 << 16) | 9216;
    NPU_REG_CFG_DATAPATH = (64 << 23) | (0 << 19) | (9 << 11) | (18 << 4) | 15; 
    NPU_REG_CFG_SPATIAL  = (64 << 24) | (0xFFFF << 8) | (0 << 5) | (1 << 4) | (2 << 2) | 2;
    NPU_REG_CFG_QUANT    = (0 << 16) | L3_MULT;
    NPU_REG_CFG_POST     = (drain_timeout << 22) | (fifo_af_thresh << 12) | (L3_SHIFT << 1) | 1;
    
    NPU_REG_ACT_BASE     = (uint32_t)buf_pool1;
    NPU_REG_BIAS_BASE    = (uint32_t)test_conv3_bias;
    NPU_REG_WEIGHT_BASE  = (uint32_t)test_conv3_weight;
    NPU_REG_OUT_BASE     = (uint32_t)buf_conv3;
    NPU_REG_CTRL_STATUS  = 0x00000001;
    wait_npu_done();

    // -----------------------------------------------------------
    // [5/7] Layer 5: Hardware Conv4 (16x16x64 -> 16x16x64)
    // -----------------------------------------------------------
    my_printf(" -> [5/7] Executing NPU Conv4 (64 OC)...\n");
    NPU_REG_CFG_IMG_DIM  = (16 << 16) | 16;      
    NPU_REG_CFG_MEM_CNT  = (256 << 16) | 9216;
    NPU_REG_CFG_DATAPATH = (64 << 23) | (0 << 19) | (9 << 11) | (18 << 4) | 15; 
    NPU_REG_CFG_SPATIAL  = (64 << 24) | (0xFFFF << 8) | (0 << 5) | (1 << 4) | (2 << 2) | 2;
    NPU_REG_CFG_QUANT    = (0 << 16) | L4_MULT;
    NPU_REG_CFG_POST     = (drain_timeout << 22) | (fifo_af_thresh << 12) | (L4_SHIFT << 1) | 1;
    
    NPU_REG_ACT_BASE     = (uint32_t)buf_conv3;
    NPU_REG_BIAS_BASE    = (uint32_t)test_conv4_bias;
    NPU_REG_WEIGHT_BASE  = (uint32_t)test_conv4_weight;
    NPU_REG_OUT_BASE     = (uint32_t)buf_conv4;
    NPU_REG_CTRL_STATUS  = 0x00000001;
    wait_npu_done();

    // -----------------------------------------------------------
    // [6/7] Layer 6: Software Pool2 (16x16x64 -> 8x8x64)
    // -----------------------------------------------------------
    my_printf(" -> [6/7] CPU Executing MaxPool 2...\n");
    maxpool2d_int8_hwc(buf_conv4, buf_pool2, 64, 16, 2, 2);

    // -----------------------------------------------------------
    // [7/7] Layer 7: Hardware FC Layer (Folded Reshape Map)
    // -----------------------------------------------------------
    my_printf(" -> [7/7] Executing NPU FC Layer (2x32 Reshape)...\n");
    NPU_REG_CFG_IMG_DIM  = (2 << 16) | 32;      
    NPU_REG_CFG_MEM_CNT  = (512 << 16) | 12288;
    NPU_REG_CFG_DATAPATH = (12 << 23) | (0 << 19) | (64 << 11) | (32 << 4) | 15;
    NPU_REG_CFG_SPATIAL  = (12 << 24) | (0x0007 << 8) | (1 << 5) | (0 << 4) | (0 << 2) | 0;
    NPU_REG_CFG_QUANT    = (0 << 16) | FC_MULT;
    NPU_REG_CFG_POST     = (drain_timeout << 22) | (fifo_af_thresh << 12) | (FC_SHIFT << 1) | 0;
    
    NPU_REG_ACT_BASE     = (uint32_t)buf_pool2;
    NPU_REG_BIAS_BASE    = (uint32_t)test_fc_bias;
    NPU_REG_WEIGHT_BASE  = (uint32_t)test_fc_weight;
    NPU_REG_OUT_BASE     = (uint32_t)buf_fc;
    NPU_REG_CTRL_STATUS  = 0x00000001;
    wait_npu_done();

    // ===========================================================
    // 端到端输出比对
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
        my_printf("\n  [PASS] End-to-End Inference is 100%% Accurate!\n");
        my_printf("========================================\n");
        my_printf(" FULL SYSTEM INTEGRATION PASSED!\n");
        my_printf("========================================\n");
        SIM_EXIT_PORT = 1;
    } else {
        my_printf("\n  [FAIL] Network inference failed with %d errors.\n", error_cnt);
        SIM_EXIT_PORT = 0;
    }

    while(1);
    return 0;
}