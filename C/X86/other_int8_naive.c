#include "other_int8_naive.h"
#include <stdint.h>
#include <stdio.h>

// 辅助宏：用于将 32 位累加结果钳位（Clamp）到 8 位有效范围，防止溢出变号
#define CLAMP_INT8(x)  ((x) > 127 ? 127 : ((x) < -127 ? -127 : (x)))

/**
 * @brief 原地激活的 ReLU 算子 (INT8)
 * 物理意义：将所有负数特征值置零，硬件上等价于一个符号位比较器和一个多路选择器。
 */
void relu_int8(int8_t* data, int size) {
    // 因为是原地操作（In-place），直接遍历整个平铺的内存即可，完全不需要关心通道和维度的形状
    for (int i = 0; i < size; i++) {
        if (data[i] < 0) {
            data[i] = 0;
        }
    }
}

/**
 * @brief 2D 最大池化算子 (INT8)
 * 默认前提：输入数据的内存排布为标准的 CHW (Channel, Height, Width) 格式
 */
void maxpool2d_int8(
    const int8_t* input, int8_t* output, 
    int channels, int in_size, int pool_size, int stride) 
{
    // 计算输出特征图的边长尺寸
    int out_size = (in_size - pool_size) / stride + 1;
    
    // 遍历每一个通道
    for (int c = 0; c < channels; c++) {
        int channel_in_offset = c * in_size * in_size;
        int channel_out_offset = c * out_size * out_size;
        
        // 遍历输出特征图的高 (Y) 和 宽 (X)
        for (int out_y = 0; out_y < out_size; out_y++) {
            for (int out_x = 0; out_x < out_size; out_x++) {
                
                // 初始化最大值为 INT8 的理论最小值 (-128)
                int8_t max_val = -128;
                
                // 在 pool_size x pool_size 的滑动窗口内寻找最大值
                for (int py = 0; py < pool_size; py++) {
                    for (int px = 0; px < pool_size; px++) {
                        // 映射回输入特征图的绝对坐标
                        int in_y = out_y * stride + py;
                        int in_x = out_x * stride + px;
                        
                        // 计算 1D 数组中的绝对索引
                        int in_idx = channel_in_offset + (in_y * in_size) + in_x;
                        
                        if (input[in_idx] > max_val) {
                            max_val = input[in_idx];
                        }
                    }
                }
                
                // 将找到的最大值写入输出特征图
                int out_idx = channel_out_offset + (out_y * out_size) + out_x;
                output[out_idx] = max_val;
            }
        }
    }
}

/**
 * @brief 全连接层 (Linear) 算子 (INT8) - 带 64 次累加切片 Debug 打印
 * 物理意义：一维向量与二维权重矩阵的乘法，并附带偏置相加和重量化
 */
void linear_int8(
    const int8_t* input, const int8_t* weight, const int32_t* bias, int8_t* output,
    int in_features, int out_features, 
    int32_t multiplier, int shift) 
{
    // 遍历每一个输出神经元 (OC)
    for (int out_idx = 0; out_idx < out_features; out_idx++) {
        
        // 1. 初始化累加器为当前神经元的 Bias（利用 32 位寄存器防溢出）
        int32_t acc = bias[out_idx];
        
        // 【Debug 逻辑】：仅针对第一个输出通道 (OC = 0) 打印信息
        if (out_idx == 0) {
            printf("\n--- Debugging OC 0 (Chunk Size = 64) ---\n");
            printf("Initial Bias: %d\n", acc);
        }
        
        int weight_row_offset = out_idx * in_features;
        
        // 2. 分块执行向量内积 (每 64 个 MAC 算作一个 Chunk)
        int chunk_size = 64;
        int num_chunks = in_features / chunk_size;
        
        for (int chunk = 0; chunk < num_chunks-60; chunk++) {
            int32_t psum = 0; // 局部部分和 (64个元素的累加)
            
            if (out_idx == 0) {
                printf("\n>>> Starting Chunk[%2d] (in_idx %4d~%4d) <<<\n", 
                       chunk, chunk * chunk_size, (chunk + 1) * chunk_size - 1);
            }

            for (int i = 0; i < chunk_size; i++) {
                int in_idx = chunk * chunk_size + i;
                
                // 提取输入和权重并转换为 32-bit 以防溢出
                int32_t in_val = (int32_t)input[in_idx];
                int32_t w_val  = (int32_t)weight[weight_row_offset + in_idx];
                int32_t mac_res = in_val * w_val;
                
                // 累加到局部 psum
                psum += mac_res;
                
                // 【Debug 逻辑】：打印每一步的详细 MAC 过程
                if (out_idx == 0) {
                    printf("  MAC[%2d]: In(%4d) * W(%4d) = %6d | Current psum: %6d\n", 
                           i, in_val, w_val, mac_res, psum);
                }
            }
            
            // 累加到全局 acc 中
            acc += psum;
            
            // 【Debug 逻辑】：打印每个 64 元素的最终 psum 结果
            if (out_idx == 0) {
                printf("<<< Chunk[%2d] Finished. Final psum: %6d >>>\n", chunk, psum);
            }
        }
        
        // 处理不能被 64 整除的剩余部分（防御性编程，尽管 4096 能整除）
        int remainder_start = num_chunks * chunk_size;
        for (int in_idx = remainder_start; in_idx < in_features; in_idx++) {
            int32_t in_val = (int32_t)input[in_idx];
            int32_t w_val  = (int32_t)weight[weight_row_offset + in_idx];
            acc += in_val * w_val;
        }

        // 【Debug 逻辑】：打印所有部分和加偏置后的最终 32-bit 累加和
        if (out_idx == 0) {
            printf("\n----------------------------------------\n");
            printf("Final 32-bit Accumulated Sum (All psums + Bias): %d\n", acc);
            printf("----------------------------------------\n\n");
        }
        
        // 3. 硬件友好的重量化 (Re-quantization)
        // 完美复刻 Python 中的: q_fc_out = clamp((mac_fc * mult_fc) >> shift_fc, -127, 127)
        int32_t scaled_acc = (acc * multiplier) >> shift;
        
        // 4. 钳位并写入输出缓存
        output[out_idx] = (int8_t)CLAMP_INT8(scaled_acc);
    }
}