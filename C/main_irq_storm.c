#include <stdint.h>
#include "my_printf.h"
#include "npu.h"

// ==========================================
// 中断与系统控制逻辑
// ==========================================
extern void enable_npu_irq(uint32_t mask);
extern void wait_for_irq(void);

volatile int npu_task_finished = 0;
volatile int total_irq_count = 0; // 全局中断计数器

void npu_isr(void) {
    npu_task_finished = 1;
    total_irq_count++;
    
    // 【生死攸关的硬件清除】
    // 必须在 ISR 中极其迅速地清掉硬件的中断 Flag，
    // 否则一退出 ISR 就会被挂在电平上的 IRQ 信号再次拉回来，导致死机。
    NPU_REG_CTRL_STATUS = 0x00000004; 
}

void wait_npu_done() {
    while (!npu_task_finished) { 
        wait_for_irq(); // CPU 执行 WFI 挂起
    }
    npu_task_finished = 0;
}

// ==========================================
// 极小张量 (H=4, W=4, IC=4, OC=4) - 为了让计算极速结束
// ==========================================
#define H 4
#define W 4
#define IC 4
#define OC 4

int8_t  act_buf[H * W * IC] __attribute__((aligned(64)));
int8_t  weight_buf[OC * IC] __attribute__((aligned(64)));
int32_t bias_buf[OC]        __attribute__((aligned(64)));
int8_t  out_buf[H * W * OC] __attribute__((aligned(64)));

void init_fast_task_data() {
    for (int i = 0; i < H * W * IC; i++) act_buf[i] = 1;
    for (int i = 0; i < OC * IC; i++) weight_buf[i] = 1;
    for (int i = 0; i < OC; i++) bias_buf[i] = 0;
    for (int i = 0; i < H * W * OC; i++) out_buf[i] = 0;
}

int main() {
    // 开启中断
    enable_npu_irq(~(1 << 4));

    my_printf("\n========================================\n");
    my_printf(" [TEST 10] High-Frequency IRQ Storm Test\n");
    my_printf("           1000 Continuous Micro-Tasks\n");
    my_printf("========================================\n");

    init_fast_task_data();

    // 提前配置好所有静态寄存器，循环中只控制启动
    NPU_REG_CFG_IMG_DIM  = (H << 16) | W;      
    NPU_REG_CFG_MEM_CNT  = (4 << 16) | 4;
    NPU_REG_CFG_DATAPATH = (4 << 23) | (0 << 19) | (1 << 11) | (4 << 4) | 0;
    NPU_REG_CFG_SPATIAL  = (4 << 24) | (0x0001 << 8) | (1 << 5) | (0 << 4) | (0 << 2) | 0;
    NPU_REG_CFG_QUANT    = (0 << 16) | 1;
    NPU_REG_CFG_POST     = (132 << 22) | (40 << 12) | (0 << 1) | 0;

    NPU_REG_ACT_BASE    = (uint32_t)act_buf;
    NPU_REG_WEIGHT_BASE = (uint32_t)weight_buf;
    NPU_REG_BIAS_BASE   = (uint32_t)bias_buf;
    NPU_REG_OUT_BASE    = (uint32_t)out_buf;

    my_printf(" -> Commencing IRQ Storm (Target: 1000)...\n");

    int target_runs = 1000;
    
    // 核心风暴循环
    for (int i = 1; i <= target_runs; i++) {
        
        // 1. 瞬间点火
        NPU_REG_CTRL_STATUS = 0x00000001; 
        
        // 2. CPU 立即休眠，交出总线和功耗，等待被 NPU 一脚踹醒
        wait_npu_done();
        
        // 3. 打印进度 (每 100 次打印一次，防止 UART 打印太慢拖累了风暴的节奏)
        if (i % 100 == 0) {
            my_printf("    [INFO] Handled %d / %d IRQs flawlessly...\n", i, target_runs);
        }
    }

    // =======================================================
    // 验证成果
    // =======================================================
    my_printf(" -> IRQ Storm Finished!\n");
    
    int error_cnt = 0;
    // 抽查最后一次运算的结果是否依然正确 (1*4 = 4)
    for (int i = 0; i < H * W * OC; i++) {
        if (out_buf[i] != 4) {
            error_cnt++;
        }
    }

    if (total_irq_count == target_runs && error_cnt == 0) {
        my_printf("  [PASS] Exactly 1000 Interrupts Handled.\n");
        my_printf("  [PASS] Final Data Integrity is 100%% Correct.\n");
        my_printf("========================================\n");
        my_printf(" SYSTEM IRQ ROBUSTNESS PASSED!\n");
        my_printf("========================================\n");
        SIM_EXIT_PORT = 1;
    } else {
        my_printf("  [FAIL] IRQ Count: %d | Data Errors: %d\n", total_irq_count, error_cnt);
        SIM_EXIT_PORT = 0;
    }

    while(1);
    return 0;
}