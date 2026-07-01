// 这里包含你用 Python 重新打包好的 HWC 格式的头文件！
#include "../python/test_vector_2.h"
#include "conv2d_int8_naive.h"
#include "my_printf.h"

#include <stdint.h>
// ==========================================
// 1. NPU AXI-Lite 寄存器映射宏定义
// ==========================================
#define NPU_BASE            0x40000000
#define NPU_REG_CTRL_STATUS (*(volatile uint32_t*)(NPU_BASE + 0x00))
#define NPU_REG_ACT_BASE    (*(volatile uint32_t*)(NPU_BASE + 0x04))
#define NPU_REG_WEIGHT_BASE (*(volatile uint32_t*)(NPU_BASE + 0x08))
#define NPU_REG_BIAS_BASE   (*(volatile uint32_t*)(NPU_BASE + 0x0C))
#define NPU_REG_OUT_BASE    (*(volatile uint32_t*)(NPU_BASE + 0x10))
#define NPU_REG_CFG_IMG_DIM (*(volatile uint32_t*)(NPU_BASE + 0x14))
#define NPU_REG_CFG_CH      (*(volatile uint32_t*)(NPU_BASE + 0x18))
#define NPU_REG_CFG_QUANT   (*(volatile uint32_t*)(NPU_BASE + 0x1C))
#define NPU_REG_CFG_DPATH   (*(volatile uint32_t*)(NPU_BASE + 0x20))
#define NPU_REG_CFG_DPATH2  (*(volatile uint32_t*)(NPU_BASE + 0x24))
#define NPU_REG_CFG_KERNEL  (*(volatile uint32_t*)(NPU_BASE + 0x28))

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
// 2. 内存分配 (适配 64x64 阵列，强行 64 字节对齐保平安)
// ==========================================
int8_t buf_conv1[64 * 32 * 32] __attribute__((aligned(4))) = {0}; 
int8_t buf_conv2[64 * 32 * 32] __attribute__((aligned(4))) = {0}; 
int8_t buf_pool1[64 * 16 * 16] __attribute__((aligned(4))) = {0}; 
int8_t buf_conv3[64 * 16 * 16] __attribute__((aligned(4))) = {0}; 
int8_t buf_conv4[64 * 16 * 16] __attribute__((aligned(4))) = {0}; 
int8_t buf_pool2[64 *  8 *  8] __attribute__((aligned(4))) = {0}; 
int8_t buf_fc[12]              __attribute__((aligned(4))) = {0};
// ==========================================
// 3. 辅助函数：休眠等待 NPU 完成 (替换死等轮询)
// ==========================================
void wait_npu_done() {
    // 当 NPU 还没算完时，执行休眠
    while (!npu_task_finished) {
        wait_for_irq(); // 完美的自定义 WFI
    }
    
    // CPU 被唤醒并跳出循环后，重置标志位，为下一层的复用做准备
    npu_task_finished = 0;
    
    // 表示npu进行完一轮运算
    my_printf("-");
}

int main() {
    // 【系统级初始化：开启硬件中断】
    // PicoRV32 没有全局中断开关，Mask 寄存器就是一切！
    // 我们传入 ~(1 << 4)：即除了 bit 4 是 0 以外，其余 31 个 bit 都是 1。
    // 这意味着：放行 NPU 的 IRQ 4，屏蔽其他所有中断。
    enable_npu_irq(~(1 << 4));
    
    my_printf("Input Base: %d\n", test_input_img);
    my_printf("Weight Base: %d\n", test_conv1_weight);
    my_printf("Conv1 Out Base: %d\n", buf_conv1);
    my_printf("===== RISC-V & NPU 64x64 FULL POWER Inference =====\n");

    // -----------------------------------------------------------
    // Layer 1: Hardware Conv1 (32x32x3 -> 32x32x64)
    // -----------------------------------------------------------
    my_printf("[1/7] Executing FC Layer 1 (Conv1: 64 OC)...\n");

    NPU_REG_CFG_IMG_DIM = (1 << 16) | 64;       
    NPU_REG_CFG_QUANT   = (12 << 24) | (0 << 21) | (FC_SHIFT << 16) | FC_MULT; 
    // Out_Stride=64, IC_G_R=0(空间全并行!), W_Num=9, LB_W=34, IC_G=15
    NPU_REG_CFG_DPATH   = (12 << 22) | (0 << 18) | (64 << 10) | (0 << 4) | 15;
    NPU_REG_CFG_DPATH2  = (0 << 16) | (0xFFFF);

    NPU_REG_CFG_KERNEL  = (1 << 5) | (0 << 4) |(0 << 2) | 0;
    // Row_words = 32*(0+1) = 32; Weight_words = 144 * 4(OC) = 576
    NPU_REG_CFG_CH      = (1024 << 16) | 12288;
    
    NPU_REG_ACT_BASE    = (uint32_t)expected_pool2_output;

    // ===========================================================
    // [Debug] CPU 探内存：按 8-bit 小端序逐字节打印前 16 个字节
    // ===========================================================
    my_printf("\n>>> CPU Memory Dump at expected_pool2_output <<<\n");
    
    // 强转为 8-bit 无符号指针，完美规避小端序在 32-bit 下的视觉干扰
    uint8_t* dump_ptr = (uint8_t*)expected_pool2_output;
    
    // 打印前 16 个字节 (即前 4 个 32-bit Word)
    // 根据你的头文件，这 4 个 Word 是：0x00000000, 0x00000000, 0x00000103, 0x03000000
    for (int i = 0; i < 16; i++) {
        // 由于没有 16 进制打印，直接打印十进制无符号数 (0~255)
        my_printf("Byte Offset [%d] : %d\n", i, dump_ptr[i]);
    }
    my_printf("------------------------------------------------\n\n");

    // 启动 NPU 计算 (当前 FC 规模较小，单次 Pass 即可完成)
    for (int pass = 0; pass < 1; pass++) {
        // 修正为 FC 层对应的偏移量：Bias(12字节), Weight(49152字节), Out(12字节)
        NPU_REG_BIAS_BASE   = (uint32_t)test_fc_bias + (pass * 12);
        NPU_REG_WEIGHT_BASE = (uint32_t)test_fc_weight + (pass * 49152);
        NPU_REG_OUT_BASE    = (uint32_t)buf_fc + (pass * 12);

        NPU_REG_CTRL_STATUS = 0x00000001;
        // CPU 进入休眠，交出控制权，等待硬件中断唤醒！
        wait_npu_done();
    }
    
    // ===========================================================
    // 校验 FC 层 (10 个有效输出通道)
    // ===========================================================
    my_printf("\n>>> Checking NPU FC Output (10 Valid Channels) <<<\n");
    
    // 【关键修复】直接使用 int8_t 指针进行单字节比对，彻底杜绝内存越界
    int8_t* ptr_hw_out  = (int8_t*)buf_fc;
    int8_t* ptr_exp_out = (int8_t*)expected_fc_output;
    
    int error_cnt = 0;
    
    // 只遍历预期的 10 个通道，忽略 buf_fc[10] 和 buf_fc[11] 的 Padding 数据
    for (int ch = 0; ch < 10; ch++) {
        int8_t val = ptr_hw_out[ch];
        int8_t exp = ptr_exp_out[ch];
        
        if (val == exp) {
            my_printf("  CH[%d]: MATCH (%d)\n", ch, val);
        } else {
            my_printf("  [ERROR] CH[%d] Mismatch -> HW: %d | EXP: %d\n", ch, val, exp);
            error_cnt++;
        }
    }
    
    if (error_cnt == 0) {
        my_printf("  FC Output Verification PASSED!\n");
    } else {
        my_printf("  FC Output Verification FAILED with %d errors.\n", error_cnt);
    }

    return 0;
}