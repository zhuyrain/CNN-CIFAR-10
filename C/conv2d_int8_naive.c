#include <stdint.h>
#include "npu_weights_int8.h"
// 假设这些来自你 Python 导出的头文件
// 包含了量化参数 (Scale 和 Zero Point)
extern const int8_t conv1_weight_int8[];
extern const int32_t conv1_bias_int32[];
extern void my_print(const char *str);

/**
 * @brief 朴素版 INT8 2D 卷积 (带有重量化逻辑)
 * * @param input       输入特征图 (INT8)
 * @param weight      卷积核权重 (INT8)
 * @param bias        偏置 (INT32)
 * @param output      输出特征图 (INT8)
 * @param in_channels 输入通道数
 * @param out_channels 输出通道数
 * @param in_size     输入特征图尺寸 (假设宽高一致)
 * @param kernel_size 卷积核尺寸 (例如 3x3)
 * @param out_zp      下一层的 Zero Point (用于重量化)
 * @param multiplier  量化缩放乘数 (由 Scale 换算而来)
 * @param shift       量化缩放移位数 (由 Scale 换算而来)
 */
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
    int shift) 
{
    // Padding 为 1 时的输出尺寸 (为了简化 Demo，这里写死 stride=1, padding=1 的计算逻辑)
    int out_size = in_size; 
    int pad = 1;

    // 1. 遍历输出通道
    for (int oc = 0; oc < out_channels; oc++) {
        my_print("-");
        // 2. 遍历输出特征图的 Y 坐标
        for (int oy = 0; oy < out_size; oy++) {
            // 3. 遍历输出特征图的 X 坐标
            for (int ox = 0; ox < out_size; ox++) {
                
                // 【关键点 1：累加器必须是 32 位整数！】
                // 即使输入和权重都是 8 位，多次乘加后极易溢出 16 位
                int32_t acc = bias[oc]; 

                // 4. 遍历输入通道
                for (int ic = 0; ic < in_channels; ic++) {
                    // 5. 遍历卷积核 Y 坐标
                    for (int ky = 0; ky < kernel_size; ky++) {
                        // 6. 遍历卷积核 X 坐标
                        for (int kx = 0; kx < kernel_size; kx++) {
                            
                            // 计算在原图上的绝对坐标 (考虑 Padding)
                            int iy = oy + ky - pad;
                            int ix = ox + kx - pad;

                            // 边界检查：如果超出边界，相当于 Padding 了 0，跳过乘加
                            if (iy >= 0 && iy < in_size && ix >= 0 && ix < in_size) {
                                // 一维数组寻址公式：(c * H * W) + (y * W) + x
                                int in_idx = (ic * in_size * in_size) + (iy * in_size) + ix;
                                int w_idx = (oc * in_channels * kernel_size * kernel_size) + 
                                            (ic * kernel_size * kernel_size) + 
                                            (ky * kernel_size) + kx;

                                // 核心 MAC (乘加) 操作
                                acc += (int32_t)input[in_idx] * (int32_t)weight[w_idx];
                            }
                        }
                    }
                }

                // 【关键点 2：重量化 (Re-quantization)】
                // 将 INT32 的累加结果，缩放并平移回 INT8，供下一层使用
                // 在没有 FPU 的微处理器上，(acc * scale) 通常转化为 (acc * multiplier) >> shift
                int32_t scaled_acc = (acc * multiplier) >> shift;
                int32_t requantized = scaled_acc + out_zp;

                // 饱和截断 (Clip to INT8)
                if (requantized > 127) requantized = 127;
                if (requantized < -128) requantized = -128;

                // 写入输出数组
                int out_idx = (oc * out_size * out_size) + (oy * out_size) + ox;
                output[out_idx] = (int8_t)requantized;
            }
        }
    }
}