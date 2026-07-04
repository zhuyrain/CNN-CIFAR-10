#include <stdint.h>
#include "my_printf.h"
#include "npu.h"

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
}

// ==========================================
// 极限压榨张量配置 (H=32, W=32, IC=4, OC=64)
// ==========================================
#define H 32
#define W 32
#define IC 4
#define OC 64

// 输入特征图: 32 * 32 * 4 = 4096 Bytes
int8_t act_buf[H * W * IC] __attribute__((aligned(4)));

// 1x1 权重: 1 * (IC/4) * OC = 64 Words = 256 Bytes
// 因为使用了你的 INT8 4合1 (DP4A) 架构，这里直接用 8-bit 定义，其实占 256 字节
int8_t weight_1x1[IC * OC] __attribute__((aligned(4)));

// 偏置: 64 个通道 = 256 Bytes
int32_t bias_buf[OC] __attribute__((aligned(4)));

// 输出: 8 * 8 * 64 = 4096 Bytes
int8_t out_buf[H * W * OC] __attribute__((aligned(4)));

// 初始化测试数据
void init_stress_data() {
    // 激活全 1
    for (int i = 0; i < H * W * IC; i++) { act_buf[i] = 1; }
    
    // 权重交织填充 (逻辑等效全 1)
    for (int i = 0; i < IC * OC; i++) { weight_1x1[i] = 1; }
    
    // 偏置等于通道索引
    for (int i = 0; i < OC; i++) { bias_buf[i] = i; }
    
    // 输出清零
    for (int i = 0; i < H * W * OC; i++) { out_buf[i] = 0; }
}

int main() {
    enable_npu_irq(~(1 << 4));

    my_printf("\n========================================\n");
    my_printf(" [TEST 8] Internal Backpressure & Bus Stress\n");
    my_printf("          Prod-Cons Ratio = 16:1 (Massive Stalls)\n");
    my_printf("========================================\n");

    init_stress_data();

    my_printf(" -> Configuring 1x1 Conv for Max Throughput Mismatch...\n");

    // 1. 维度与搬运
    NPU_REG_CFG_IMG_DIM  = (H << 16) | W;      
    // row_word = W * IC / 4 = 32 * 4 / 4 = 32
    // weight_word = 1 * 64 = 64
    NPU_REG_CFG_MEM_CNT  = (32 << 16) | 64;
    
    // 2. 数据通路
    // out_stride=64, sa_weight_num=1, lb_line_width=32, lb_ic_groups=0
    NPU_REG_CFG_DATAPATH = (64 << 23) | (0 << 19) | (1 << 11) | (32 << 4) | 0;

    // 3. 空间映射 (全速开火！)
    // oc_num=64, col_group_en=0xFFFF(64列全开), ky_skew=1, pad=0, kx=0, ky=0
    NPU_REG_CFG_SPATIAL  = (64 << 24) | (0xFFFF << 8) | (1 << 5) | (0 << 4) | (0 << 2) | 0;

    // 4. 量化参数 (关闭)
    NPU_REG_CFG_QUANT    = (0 << 16) | 1;
    
    // 5. 【极限施压核心】将 FIFO 将满阈值设为极小值 (如 16)！
    // 假设 FIFO 深度很大，我们强制它在只装了 16 个 Word 时就拉高 fifo_af 信号，
    // 疯狂且高频地向前端施加反压！
    uint32_t drain_timeout  = 132;   
    uint32_t fifo_af_thresh = 40;  // <--- 核心反压诱发点！
    NPU_REG_CFG_POST = (drain_timeout << 22) | (fifo_af_thresh << 12) | (0 << 1) | 0;

    // 6. 基址配置
    NPU_REG_ACT_BASE    = (uint32_t)act_buf;
    NPU_REG_WEIGHT_BASE = (uint32_t)weight_1x1;
    NPU_REG_BIAS_BASE   = (uint32_t)bias_buf;
    NPU_REG_OUT_BASE    = (uint32_t)out_buf;

    my_printf(" -> Firing NPU! Array will blast data, Bus will struggle...\n");
    NPU_REG_CTRL_STATUS = 0x00000001;
    wait_npu_done();
    my_printf(" -> NPU Survived the Backpressure Storm!\n");

    // =======================================================
    // 结果校验
    // =======================================================
    my_printf("\n>>> Checking Data Integrity Under Severe Stalls <<<\n");
    
    int error_cnt = 0;
    
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < OC; c++) {
                int idx = (h * W + w) * OC + c;
                // 预期：4 个 1 相加 + 通道索引 = 4 + c
                int expected = 4 + c; 
                
                if (out_buf[idx] != expected) {
                    if (error_cnt < 10) { // 避免终端被海量报错淹没
                        my_printf("    [ERROR] Pixel(%d,%d) CH[%d] HW: %d | EXP: %d\n", h, w, c, out_buf[idx], expected);
                    }
                    error_cnt++;
                }
            }
        }
    }

    if (error_cnt == 0) {
        my_printf("  [PASS] All 4096 Bytes exactly match! Zero data dropped!\n");
        my_printf("========================================\n");
        my_printf(" BACKPRESSURE & STALL RECOVERY PASSED!\n");
        my_printf("========================================\n");
    } else {
        my_printf("  [FAIL] %d Errors found! Pipeline stalled and dropped data!\n", error_cnt);
        while(1);
    }

    while(1);
    return 0;
}