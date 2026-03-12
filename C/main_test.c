#include <stdio.h>
#include <stdint.h>

extern void conv2d_int8_naive(
    const int8_t* input, const int8_t* weight, const int32_t* bias, int8_t* output,
    int in_channels, int out_channels, int in_size, int kernel_size,
    int out_zp, int32_t multiplier, int shift);

int main() {
    printf("===== RISC-V C Baseline Test Started =====\n");

    int8_t dummy_input[3 * 32 * 32] = {1};
    int8_t dummy_weight[16 * 3 * 3 * 3] = {1};
    int32_t dummy_bias[16] = {0};
    int8_t dummy_output[16 * 32 * 32] = {0};

    printf("Executing Conv2D...\n");
    
    conv2d_int8_naive(
        dummy_input, dummy_weight, dummy_bias, dummy_output,
        3, 16, 32, 3,
        0, 1024, 10
    );

    printf("Conv2D finished! First output value: %d\n", dummy_output[0]);
    printf("GCC Environment is OK!\n");

    return 0;
}