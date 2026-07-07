#ifndef OTHER_INT8_NAIVE_X86_H
#define OTHER_INT8_NAIVE_X86_H

#include <stdint.h>
#include <stdio.h>

void relu_int8(int8_t* data, int size);
void maxpool2d_int8(
    const int8_t* input, int8_t* output, 
    int channels, int in_size, int pool_size, int stride);
void linear_int8(
    const int8_t* input, const int8_t* weight, const int32_t* bias, int8_t* output,
    int in_features, int out_features, 
    int32_t multiplier, int shift);
#endif