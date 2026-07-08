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
// 边界条件张量配置
// ==========================================
#define IC 64
#define LOGICAL_OC 4
#define HW_OC 16 

// 强制 64 字节对齐
int8_t  act_buf[256]     __attribute__((aligned(64)));
int8_t  weight_buf[1024] __attribute__((aligned(64)));
int32_t bias_buf[64]     __attribute__((aligned(64)));
int8_t  out_buf[64]      __attribute__((aligned(64)));

// 测试函数
int run_quant_test(const char* test_name, 
                   int8_t act_val, int8_t weight_val, int32_t bias_val, 
                   int32_t mult, int shift, int8_t expected) 
{
    // 使用 volatile 指针，防范 CPU/编译器做任何 Cache 或循环优化
    volatile int8_t* p_act = (volatile int8_t*)act_buf;
    volatile int8_t* p_wt  = (volatile int8_t*)weight_buf;
    volatile int32_t* p_bias = (volatile int32_t*)bias_buf;
    volatile int8_t* p_out = (volatile int8_t*)out_buf;

    // 1. 初始化测试内存 (CPU 执行 Store 指令)
    for (int i = 0; i < IC; i++) p_act[i] = act_val;
    for (int i = 0; i < HW_OC * IC; i++) p_wt[i] = weight_val;
    for (int i = 0; i < HW_OC; i++) p_bias[i] = bias_val;
    for (int i = 0; i < HW_OC; i++) p_out[i] = 0;

    // =======================================================
    // 【内存屏障 (Memory Barrier)】
    // 通过 CPU 强制读取刚刚写入的内存，强迫 AXI Interconnect
    // 清空内部的 Write Buffer，确保所有数据真正落入 SRAM。
    // =======================================================
    my_printf("    [DEBUG] CPU Read-back Bias: ");
    int bias_write_error = 0;
    for (int i = 0; i < LOGICAL_OC; i++) {
        int32_t read_val = p_bias[i]; // CPU 发起 AXI Read
        my_printf("%d ", read_val);
        if (read_val != bias_val) {
            bias_write_error = 1;
        }
    }
    my_printf("\n");
    if (bias_write_error) {
        my_printf("    [FATAL] CPU failed to write/read Bias correctly!\n");
        return 0; // 提前退出，保护现场
    }

    // 2. 寄存器配置 (锁定 1x1 单点复用)
    NPU_REG_CFG_IMG_DIM  = (1 << 16) | 1;      
    NPU_REG_CFG_MEM_CNT  = (16 << 16) | (16 * HW_OC); 
    NPU_REG_CFG_DATAPATH = (HW_OC << 23) | (0 << 19) | (1 << 11) | (1 << 4) | 15;
    NPU_REG_CFG_SPATIAL  = (HW_OC << 24) | (0x000F << 8) | (1 << 5) | (0 << 4) | (0 << 2) | 0;
    
    // 3. 量化核心配置
    NPU_REG_CFG_QUANT    = (0 << 16) | mult;
    NPU_REG_CFG_POST     = (132 << 22) | (40 << 12) | (shift << 1) | 0;

    // 4. 基址挂载与启动 
    NPU_REG_ACT_BASE    = (uint32_t)act_buf;
    NPU_REG_WEIGHT_BASE = (uint32_t)weight_buf;
    NPU_REG_BIAS_BASE   = (uint32_t)bias_buf;
    NPU_REG_OUT_BASE    = (uint32_t)out_buf;

    NPU_REG_CTRL_STATUS = 0x00000001; // CPU 启动 NPU
    wait_npu_done(); // 等待 NPU 计算完成

    // 5. 自检
    int error_cnt = 0;
    for (int c = 0; c < LOGICAL_OC; c++) {
        if (p_out[c] != expected) {
            my_printf("    [ERROR] %s -> CH[%d] HW: %d | EXP: %d\n", test_name, c, p_out[c], expected);
            error_cnt++;
        }
    }
    
    if (error_cnt == 0) {
        my_printf("  [PASS] %s\n", test_name);
        return 1;
    }
    return 0;
}

int main() {
    enable_npu_irq(~(1 << 4));

    my_printf("\n========================================\n");
    my_printf(" [TEST 9] Quantization Boundaries & Saturation\n");
    my_printf("          Symmetric Range [-127, 127] Enforcement\n");
    my_printf("========================================\n");

    int pass = 1;

    pass &= run_quant_test("Max Positive Saturation (+127)", -128, -128, 0, 1, 0, 127);
    pass &= run_quant_test("Max Negative Saturation (-127)", -128, 127, 0, 1, 0, -127);
    pass &= run_quant_test("Arithmetic Right Shift (Neg)", -128, 127, 0, 1, 13, -127);
    
    pass &= run_quant_test("Huge Bias Cancellation", 127, 127, -1032256, 1, 0, 0);
    pass &= run_quant_test("Multiplier & Shift Logic", 1, 1, 0, 10, 4, 40);

    my_printf("========================================\n");
    if (pass) {
        my_printf(" QUANTIZATION & DATAPATH BOUNDARIES PASSED!\n");
        my_printf("========================================\n");
        SIM_EXIT_PORT = 1;
    } else {
        my_printf(" QUANTIZATION VERIFICATION FAILED!\n");
        my_printf("========================================\n");
        SIM_EXIT_PORT = 0;
    }
    while(1);
    return 0;
}