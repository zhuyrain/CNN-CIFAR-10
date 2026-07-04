#include "my_printf.h"
#include <stdint.h>

// ==========================================
// 1. NPU AXI-Lite 寄存器映射宏定义 (已对齐全新重构版本)
// ==========================================
#define NPU_BASE            0x40000000
#define NPU_REG_CTRL_STATUS (*(volatile uint32_t*)(NPU_BASE + 0x00))
#define NPU_REG_ACT_BASE    (*(volatile uint32_t*)(NPU_BASE + 0x04))
#define NPU_REG_WEIGHT_BASE (*(volatile uint32_t*)(NPU_BASE + 0x08))
#define NPU_REG_BIAS_BASE   (*(volatile uint32_t*)(NPU_BASE + 0x0C))
#define NPU_REG_OUT_BASE    (*(volatile uint32_t*)(NPU_BASE + 0x10))

// 全新重构的 6 个配置寄存器
#define NPU_REG_CFG_IMG_DIM  (*(volatile uint32_t*)(NPU_BASE + 0x14))
#define NPU_REG_CFG_MEM_CNT  (*(volatile uint32_t*)(NPU_BASE + 0x18))
#define NPU_REG_CFG_DATAPATH (*(volatile uint32_t*)(NPU_BASE + 0x1C))
#define NPU_REG_CFG_SPATIAL  (*(volatile uint32_t*)(NPU_BASE + 0x20))
#define NPU_REG_CFG_QUANT    (*(volatile uint32_t*)(NPU_BASE + 0x24))
#define NPU_REG_CFG_POST     (*(volatile uint32_t*)(NPU_BASE + 0x28))

// =========================================================================
// 1. CPU ALU & 分支预测原子测试
// 必须使用 volatile，强迫 CPU 在运行时真实调用硬件加减乘除逻辑，防止编译器优化
// =========================================================================
int test_alu_and_branch() {
    volatile int32_t a = 125;
    volatile int32_t b = 5;
    volatile int32_t result;

    // 测试基础运算
    result = a + b; if (result != 130) return 0;
    result = a - b; if (result != 120) return 0;
    
    // 测试硬件乘法器 (ENABLE_FAST_MUL=1)
    result = a * b; if (result != 625) return 0;
    
    // 测试硬件除法器 (ENABLE_DIV=1)
    result = a / b; if (result != 25) return 0;

    // 测试位运算 (逻辑与控制流)
    volatile uint32_t mask = 0xDEADBEEF;
    result = (mask >> 16) & 0xFFFF;
    if (result != 0xDEAD) return 0;

    result = mask ^ 0xFFFFFFFF;
    if (result != 0x21524110) return 0;

    // 测试多重条件分支 (Branching)
    volatile int loop_cnt = 0;
    for (volatile int i = 0; i < 100; i++) {
        if (i % 2 == 0) {
            loop_cnt++;
        }
    }
    if (loop_cnt != 50) return 0;

    return 1; // 全部通过
}

// =========================================================================
// 2. SRAM 连续读写与数据完整性测试
// 验证 AXI 互联矩阵 M0 端口的数据通路和访存正确性
// =========================================================================
#define MEM_TEST_WORDS 2048 // 测试 8KB 的 SRAM 空间

// 将测试数组强制分配到常规的 SRAM 数据段
uint32_t sram_test_buffer[MEM_TEST_WORDS]; 

int test_sram_integrity() {
    // 第一阶段：写入复杂的校验伪随机模式
    for (int i = 0; i < MEM_TEST_WORDS; i++) {
        // 使用地址偏移和固定魔数生成独特的数据块
        sram_test_buffer[i] = 0x5AA50000 | i; 
    }

    // 第二阶段：全量读回并校验
    for (int i = 0; i < MEM_TEST_WORDS; i++) {
        volatile uint32_t read_back = sram_test_buffer[i];
        if (read_back != (0x5AA50000 | i)) {
            // 发生数据踩踏或读写失败
            return 0; 
        }
    }

    return 1; // 内存完整性完美
}

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

// =========================================================================
// 3. 主函数入口
// =========================================================================
int main() {
    my_printf("\n========================================\n");
    my_printf(" [TEST 1] CPU & SRAM Infrastructure\n");
    my_printf("========================================\n");

    my_printf("-> Running PicoRV32 ALU & ISA Tests...\n");
    if (test_alu_and_branch()) {
        my_printf("   [PASS] ALU & Branching OK.\n");
    } else {
        my_printf("   [FAIL] ALU Hardware Error!\n");
        while(1); // 测试失败，死循环挂起
    }

    my_printf("-> Running AXI-SRAM Memory Integrity Tests...\n");
    if (test_sram_integrity()) {
        my_printf("   [PASS] SRAM Read/Write OK.\n");
    } else {
        my_printf("   [FAIL] SRAM Data Corruption!\n");
        while(1); 
    }

    my_printf("========================================\n");
    my_printf(" ALL INFRASTRUCTURE TESTS PASSED!\n");
    my_printf("========================================\n");

    // 成功结束，进入休眠或死循环等待仿真工具截获结束信号
    while(1);
    return 0;
}