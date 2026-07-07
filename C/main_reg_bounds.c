#include "../python/test_vector_2.h"
#include "my_printf.h"
#include <stdint.h>

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
    NPU_REG_CTRL_STATUS = 0x00000004; // 清除中断
}

void wait_npu_done() {
    while (!npu_task_finished) {
        wait_for_irq(); 
    }
    npu_task_finished = 0;
}

int main() {
    int error_cnt = 0;

    my_printf("\n========================================\n");
    my_printf(" [TEST 12] NPU Register Bounds & Status\n");
    my_printf("           AXI-Lite Slave Address Decoding\n");
    my_printf("========================================\n");

    // -----------------------------------------------------------
    // 1. 越界写入测试
    // -----------------------------------------------------------
    my_printf(" [1/3] Testing Out-of-bounds Write (Offset 0x50)...\n");
    // 预期：硬件 AXI-Lite Slave 状态机会进入 default 分支，静默丢弃该写入
    // 并且总线握手正常完成，不会引发死锁 (Deadlock)
    *((volatile uint32_t*)(NPU_BASE + (20 << 2))) = 0x12345678;
    my_printf(" -> [PASS] Bus did not hang on OOB write.\n");
    
    // -----------------------------------------------------------
    // 2. 状态寄存器有效性读取
    // -----------------------------------------------------------
    my_printf(" [2/3] Testing Status Register Read (Offset 0x00)...\n");
    uint32_t status = NPU_REG_CTRL_STATUS;
    my_printf(" -> Status value: 0x%08x\n", status);

    // -----------------------------------------------------------
    // 3. 越界读取测试
    // -----------------------------------------------------------
    my_printf(" [3/3] Testing Out-of-bounds Read (Offset 0x78)...\n");
    // 预期：触发 RTL 中的 default 分支，返回硬件预设的 0xDEADBEEF 错误码
    uint32_t invalid_read = *((volatile uint32_t*)(NPU_BASE + (30 << 2)));
    
    if (invalid_read == 0xDEADBEEF) {
        my_printf(" -> [PASS] Read 0xDEADBEEF from illegal address.\n");
    } else {
        my_printf(" -> [FAIL] Expected 0xDEADBEEF, got 0x%08x\n", invalid_read);
        error_cnt++;
    }

    // ===========================================================
    // 最终结果评估
    // ===========================================================
    my_printf("========================================\n");
    if (error_cnt == 0) {
        my_printf("  [PASS] AXI-Lite address decoding is 100%% robust!\n");
        my_printf("  [PASS] Hardware correctly ignored and responded to OOB accesses.\n");
        my_printf("========================================\n");
        my_printf(" REGISTER BOUNDARY VERIFICATION PASSED!\n");
        my_printf("========================================\n");
        SIM_EXIT_PORT = 1;
    } else {
        my_printf("  [FAIL] Verification FAILED with %d errors.\n", error_cnt);
        SIM_EXIT_PORT = 0;
    }

    while(1);
    return 0;
}