#include <stdint.h>
#include "my_printf.h"

// ==========================================
// 1. NPU 寄存器映射
// ==========================================
#include "npu.h"
// ==========================================
// 2. 中断与系统控制逻辑
// ==========================================
extern void enable_npu_irq(uint32_t mask);
extern void wait_for_irq(void);

volatile int npu_task_finished = 0;

void npu_isr(void) {
    npu_task_finished = 1;
    NPU_REG_CTRL_STATUS = 0x00000004; // 清除 NPU Done 中断
}

void wait_npu_done() {
    while (!npu_task_finished) {
        wait_for_irq(); 
    }
    npu_task_finished = 0;
}

// ==========================================
// 3. 数据阵列分配与初始化 (强制 4 字节对齐)
// ==========================================
// 任务设定：1x1 特征图，64 输入通道，64 输出通道
#define IC 64
#define OC 64

// Act: 1像素 * 64通道 = 64 Bytes
int8_t  act_buf[IC] __attribute__((aligned(4)));

// Weight: 1x1卷积核 * 64输入通道 * 64输出通道 = 4096 Bytes
int8_t  weight_buf[OC * IC] __attribute__((aligned(4)));

// Bias: 64输出通道 = 64 个 int32_t = 256 Bytes
int32_t bias_buf[OC] __attribute__((aligned(4)));

// Out: 1像素 * 64输出通道 = 64 Bytes
int8_t  out_buf[OC] __attribute__((aligned(4)));


// 数据装载函数 (初始化为纯 1 矩阵)
void init_ones_matrix() {
    for (int i = 0; i < IC; i++) {
        act_buf[i] = 1;
    }
    for (int i = 0; i < OC * IC; i++) {
        weight_buf[i] = 1;
    }
    for (int i = 0; i < OC; i++) {
        bias_buf[i] = 1; // 偏置也是 1
    }
    for (int i = 0; i < OC; i++) {
        out_buf[i] = 0;  // 输出清零
    }
}

int main() {
    // 开启 NPU 中断
    enable_npu_irq(~(1 << 4));

    my_printf("\n========================================\n");
    my_printf(" [TEST 4] Pure Math Core Verification \n");
    my_printf("          All-Ones Matrix (MAC x 4096)\n");
    my_printf("========================================\n");

    my_printf(" -> Initializing SRAM with all 1s...\n");
    init_ones_matrix();

    // -----------------------------------------------------------
    // NPU 寄存器配置：极简 1x1 Pointwise 卷积模式
    // 利用“时空不对称”的高级策略！
    // -----------------------------------------------------------
    my_printf(" -> Configuring NPU (1x1 Conv, No Bypass MUX)...\n");

    // [31:16] H=1, [15:0] W=1 
    NPU_REG_CFG_IMG_DIM  = (1 << 16) | 1;      

    // row_word = IC / 4 = 16. weight_word = (IC/4) * OC = 16 * 64 = 1024
    NPU_REG_CFG_MEM_CNT  = (16 << 16) | 1024;

    // [31:23] out_stride=64 (凑满整个输出总线)
    // [22:19] lb_ic_groups_r=0
    // [18:11] sa_weight_num=16 (每组 OC 需要 16 个 Word 的权重)
    // [10:4] lb_line_width=1 (既然 W=1, Line Buffer 宽度为 1 即可)
    // [3:0] lb_ic_groups=15 (IC/4 - 1 = 15)
    NPU_REG_CFG_DATAPATH = (64 << 23) | (0 << 19) | (1 << 11) | (1 << 4) | 15;

    // 核心降维魔法：
    // [31:24] oc_num=64
    // [23:8] col_group_en=0xFFFF (64列全开)
    // [5] ky_skew=1 (死锁在中心物理行)
    // [4] pad_size=0 (去除两端填充气泡)
    // [3:2] kx_max=0, [1:0] ky_max=0 (坍缩为 1x1 循环)
    NPU_REG_CFG_SPATIAL  = (64 << 24) | (0xFFFF << 8) | (1 << 5) | (0 << 4) | (0 << 2) | 0;

    // 量化关闭： multiplier=1, zp=0
    NPU_REG_CFG_QUANT    = (0 << 16) | 1;
    
    // shift=0, relu=0
    uint32_t drain_timeout  = 132;   
    uint32_t fifo_af_thresh = 40; 
    NPU_REG_CFG_POST = (drain_timeout << 22) | (fifo_af_thresh << 12) | (0 << 1) | 0;

    // 基址配置
    NPU_REG_ACT_BASE    = (uint32_t)act_buf;
    NPU_REG_WEIGHT_BASE = (uint32_t)weight_buf;
    NPU_REG_BIAS_BASE   = (uint32_t)bias_buf;
    NPU_REG_OUT_BASE    = (uint32_t)out_buf;

    // 启动计算
    my_printf(" -> Firing NPU!\n");
    NPU_REG_CTRL_STATUS = 0x00000001;
    wait_npu_done();
    my_printf(" -> NPU Execution Completed.\n");

    // -----------------------------------------------------------
    // 算力铁证校验
    // -----------------------------------------------------------
    my_printf("\n>>> Checking 4096 MACs Output <<<\n");
    
    int error_cnt = 0;
    
    // 预期值计算：(IC=64 个 1 相加) + (Bias=1) = 65
    int8_t expected_val = 65; 
    
    for (int ch = 0; ch < OC; ch++) {
        if (out_buf[ch] != expected_val) {
            my_printf("  [ERROR] CH[%d] HW: %d | EXP: %d\n", ch, out_buf[ch], expected_val);
            error_cnt++;
        }
    }
    
    if (error_cnt == 0) {
        my_printf("  [PASS] All 64 Channels exactly match 65!\n");
        my_printf("  [PASS] Spatial 1x1 Zero-Padding mode verified!\n");
        my_printf("========================================\n");
        my_printf(" MATH CORE VERIFICATION PASSED!\n");
        my_printf("========================================\n");
        SIM_EXIT_PORT = 1;
    } else {
        my_printf("  [FAIL] Math Core Verification FAILED with %d errors.\n", error_cnt);
        SIM_EXIT_PORT = 0;
    }

    while(1);
    return 0;
}