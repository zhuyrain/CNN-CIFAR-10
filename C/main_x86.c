#include <stdio.h>
#include <stdint.h>
// 引入刚刚 Python 生成的真实测试向量！
#include "../python/test_vector.h" 

extern void conv2d_int8_naive_x86(
    const int8_t* input, const int8_t* weight, const int32_t* bias, int8_t* output,
    int in_channels, int out_channels, int in_size, int kernel_size,
    int out_zp, int32_t multiplier, int shift);

// 定义一个指向外部仿真器假串口地址的挥发性指针
// 将地址改到物理 SRAM 范围内，借用 SRAM 的硬件握手逻辑
// volatile unsigned int * const UART_TX = (unsigned int *)0x000E0000;

// // 自定义的最简打印函数
// void my_putc(char c) {
//     *UART_TX = c; // 往这个地址写数据，Testbench 就会在终端打印出来
// }

// // 如果你需要打印字符串
// void printf(const char *str) {
//     while (*str) {
//         my_putc(*str++);
//     }
// }

int main() {
    printf("===== RISC-V Bit-True Golden Model Test =====\n");

    // 为 C 语言输出分配一块内存
    int8_t c_output[16 * 32 * 32] = {0};

    // 从 test_vector.h 中填入你刚刚生成的实际参数 (请根据 Python 打印的内容检查)
    int multiplier = 151;  // 假设 Python 打印的是 58，如果不是，请改成你实际生成的！
    int shift = 16;
    int out_zp = 0;

    printf("Executing Conv2D with real CIFAR-10 Image...\n");
    
    // 执行运算
    conv2d_int8_naive_x86(
        test_input_img, test_conv1_weight, test_conv1_bias, c_output,
        3, 16, 32, 3,
        out_zp, multiplier, shift
    );

    printf("Conv2D finished! Validating Results...\n");

    // 【核心】比特级对齐检查
    int error_count = 0;
    int total_pixels = 16 * 32 * 32;

    for (int i = 0; i < total_pixels; i++) {
        if (c_output[i] != expected_conv1_output[i]) {
            if (error_count < 5) { // 打印前几个错误
                printf("Mismatch at index %d: C_out=%d, Expected=%d\n", i, c_output[i], expected_conv1_output[i]);
            }
            error_count++;
        }
    }

    if (error_count == 0) {
        printf("\n[SUCCESS] PERFECT MATCH! 0 errors...\n");
    } else {
        // printf("\n[FAILED] Found %d errors out of %d elements.\n", error_count, total_pixels);
        printf("\n[FAILED] Found errors...\n");
    }
    return 0;
}