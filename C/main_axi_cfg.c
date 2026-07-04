#include <stdint.h>
#include "my_printf.h"
#include "npu.h"

// 声明汇编中的自定义函数
extern void enable_npu_irq(uint32_t mask);
extern void wait_for_irq(void);

// 必须使用 volatile，防止编译器过度优化导致死循环
volatile int npu_task_finished = 0;

// ==========================================
// [新增] 中断服务程序 (由 start.S 中的 trap_entry 汇编调用)
// ==========================================
void npu_isr(void) {
    // 1. 软件标志位置 1，通知主程序的 WFI 循环可以退出了
    npu_task_finished = 1;

    // 2. 硬件标志位清零 (向 Bit 2 写 1 清除 DONE 标志)
    // 极其关键：必须在这里清硬件标志，否则 irq 引脚一直拉高会导致无限触发中断！
    NPU_REG_CTRL_STATUS = 0x00000004; 
}

// ==========================================
// 测试逻辑表
// ==========================================
// 避开 CTRL_STATUS，我们将 10 个安全寄存器的地址装入数组，方便循环遍历测试
volatile uint32_t* const cfg_regs[] = {
    &NPU_REG_ACT_BASE,
    &NPU_REG_WEIGHT_BASE,
    &NPU_REG_BIAS_BASE,
    &NPU_REG_OUT_BASE,
    &NPU_REG_CFG_IMG_DIM,
    &NPU_REG_CFG_MEM_CNT,
    &NPU_REG_CFG_DATAPATH,
    &NPU_REG_CFG_SPATIAL,
    &NPU_REG_CFG_QUANT,
    &NPU_REG_CFG_POST
};

#define REG_COUNT (sizeof(cfg_regs) / sizeof(cfg_regs[0]))

int test_axi_cfg_rw() {
    // ---------------------------------------------------------
    // 测试模式 1：全 0 (边界值清零测试)
    // ---------------------------------------------------------
    for (int i = 0; i < REG_COUNT; i++) {
        *(cfg_regs[i]) = 0x00000000;
    }
    for (int i = 0; i < REG_COUNT; i++) {
        if (*(cfg_regs[i]) != 0x00000000) return 0;
    }

    // ---------------------------------------------------------
    // 测试模式 2：全 1 (测试数据总线 32 根线是否全部连通且无短路到地)
    // ---------------------------------------------------------
    for (int i = 0; i < REG_COUNT; i++) {
        *(cfg_regs[i]) = 0xFFFFFFFF;
    }
    for (int i = 0; i < REG_COUNT; i++) {
        if (*(cfg_regs[i]) != 0xFFFFFFFF) return 0;
    }

    // ---------------------------------------------------------
    // 测试模式 3：交替 01 和 10 (测试相邻数据线之间是否有串扰)
    // ---------------------------------------------------------
    for (int i = 0; i < REG_COUNT; i++) {
        *(cfg_regs[i]) = (i % 2 == 0) ? 0x55555555 : 0xAAAAAAAA;
    }
    for (int i = 0; i < REG_COUNT; i++) {
        uint32_t expected = (i % 2 == 0) ? 0x55555555 : 0xAAAAAAAA;
        if (*(cfg_regs[i]) != expected) return 0;
    }

    // ---------------------------------------------------------
    // 测试模式 4：独立流水号特征值 (测试 AXI 地址路由逻辑是否发生错位/覆盖)
    // ---------------------------------------------------------
    for (int i = 0; i < REG_COUNT; i++) {
        *(cfg_regs[i]) = 0x12340000 + i;
    }
    for (int i = 0; i < REG_COUNT; i++) {
        if (*(cfg_regs[i]) != (0x12340000 + i)) return 0;
    }

    return 1; // 4 轮严苛测试全部通过
}

int main() {
    my_printf("\n========================================\n");
    my_printf(" [TEST 2] AXI Config Bus & Echo Logic\n");
    my_printf("========================================\n");

    my_printf("-> Writing boundary patterns to %d NPU Config Regs...\n", REG_COUNT);
    my_printf("-> Reading back via AXI-Full Slave interface...\n");

    if (test_axi_cfg_rw()) {
        my_printf("   [PASS] AXI Config Read/Write Match!\n");
    } else {
        my_printf("   [FAIL] AXI Config Data Mismatch!\n");
        while(1); // 触发总线错误，死循环挂起，方便抓取波形定位
    }

    my_printf("========================================\n");
    my_printf(" ALL AXI CONFIG TESTS PASSED!\n");
    my_printf("========================================\n");

    // 测试结束挂起
    while(1);
    return 0;
}