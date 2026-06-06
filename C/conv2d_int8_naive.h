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

void relu_int8(int8_t* data, int size);
void maxpool2d_int8_hwc(
    const int8_t* input, int8_t* output, 
    int channels, int in_size, int pool_size, int stride);
void linear_int8(
    const int8_t* input, const int8_t* weight, const int32_t* bias, int8_t* output,
    int in_features, int out_features, 
    int32_t multiplier, int shift);
    
#endif