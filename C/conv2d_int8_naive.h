#ifndef CONV2D_INT8_NAIVE_H
#define CONV2D_INT8_NAIVE_H

#include <stdint.h>

void conv2d_int8_naive(
    const int8_t* input, 
    const int8_t* weight, 
    const int32_t* bias, 
    int8_t* output,
    int in_channels, 
    int out_channels, 
    int in_size, 
    int kernel_size,
    int out_zp,
    int32_t multiplier,
    int shift);

#endif