#include "conv2d_int8_naive.h"
#include "my_printf.h"

#include <stdint.h>

// 辅助宏：用于将 32 位累加结果钳位（Clamp）到 8 位有效范围，防止溢出变号
#define CLAMP_INT8(x)  ((x) > 127 ? 127 : ((x) < -127 ? -127 : (x)))

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
        my_printf("-");
        // 2. 遍历输出特征图的 Y 坐标
        for (int oy = 0; oy < out_size; oy++) {
            // 3. 遍历输出特征图的 X 坐标
            for (int ox = 0; ox < out_size; ox++) {
                
                // 关键点 1：累加器必须使用 32 位整数
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

                // 关键点 2：重量化 (Re-quantization)
                // 将 INT32 的累加结果缩放并平移回 INT8，供下一层使用
                // 在没有 FPU 的微处理器上，(acc * scale) 通常转化为 (acc * multiplier) >> shift
                int32_t scaled_acc = (acc * multiplier) >> shift;
                int32_t requantized = scaled_acc + out_zp;

                // 饱和截断 (Clip to INT8)
                if (requantized > 127) requantized = 127;
                if (requantized < -127) requantized = -127;

                // 写入输出数组
                int out_idx = (oc * out_size * out_size) + (oy * out_size) + ox;
                output[out_idx] = (int8_t)requantized;
            }
        }
    }
}

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
 * @brief 2D 最大池化算子 (INT8) - HWC (通道交织) 专用版
 * 匹配 NPU 写回的 AXI 连续数据包格式
 */
void maxpool2d_int8_hwc(
    const int8_t* input, int8_t* output, 
    int channels, int in_size, int pool_size, int stride) 
{
    int out_size = (in_size - pool_size) / stride + 1;
    
    // HWC 格式的遍历最佳顺序：先遍历空间 H 和 W，最后遍历通道 C (保持缓存局部性)
    for (int out_y = 0; out_y < out_size; out_y++) {
        for (int out_x = 0; out_x < out_size; out_x++) {
            for (int c = 0; c < channels; c++) {
                
                int8_t max_val = -128;
                
                for (int py = 0; py < pool_size; py++) {
                    for (int px = 0; px < pool_size; px++) {
                        int in_y = out_y * stride + py;
                        int in_x = out_x * stride + px;
                        
                        // 核心修改：HWC 一维数组绝对寻址公式
                        // Index = (Y * W * C) + (X * C) + C
                        int in_idx = (in_y * in_size * channels) + (in_x * channels) + c;
                        
                        if (input[in_idx] > max_val) {
                            max_val = input[in_idx];
                        }
                    }
                }
                
                // 将找到的最大值写入输出 (同样保持 HWC 格式！)
                int out_idx = (out_y * out_size * channels) + (out_x * channels) + c;
                output[out_idx] = max_val;
            }
        }
    }
}

/**
 * @brief 全连接层 (Linear) 算子 (INT8) - 适配 HWC 输入！
 * 物理意义：权重期望 CHW 展开，但实际输入为 HWC 格式。我们在运算时进行动态地址重映射。
 */
void linear_int8_hwc(
    const int8_t* input_hwc, const int8_t* weight_chw, const int32_t* bias, int8_t* output,
    int in_c, int in_h, int in_w, int out_features, 
    int32_t multiplier, int shift) 
{
    // 遍历每一个输出分类神经元 (0~9)
    for (int out_idx = 0; out_idx < out_features; out_idx++) {
        
        // 1. 初始化累加器为当前神经元的 Bias
        int32_t acc = bias[out_idx];
        
        int weight_row_offset = out_idx * (in_c * in_h * in_w);
        
        // 2. 遍历 CHW 逻辑坐标，但分别映射到 Weight 和 Input 的物理一维数组中
        for (int c = 0; c < in_c; c++) {
            for (int h = 0; h < in_h; h++) {
                for (int w = 0; w < in_w; w++) {
                    
                    // Python PyTorch 导出的权重期望的展开索引 (CHW)
                    // Index = c * (H * W) + h * W + w
                    int chw_idx = c * (in_h * in_w) + h * in_w + w;
                    
                    // 硬件 NPU 实际吐出的物理特征图索引 (HWC)
                    // Index = h * (W * C) + w * C + c
                    int hwc_idx = h * (in_w * in_c) + w * in_c + c;
                    
                    // 动态坐标对齐乘加
                    acc += (int32_t)input_hwc[hwc_idx] * (int32_t)weight_chw[weight_row_offset + chw_idx];
                }
            }
        }
        
        // 3. 重量化 (Re-quantization)
        int32_t scaled_acc = (acc * multiplier) >> shift;
        
        // 4. 饱和截断 (Clip to INT8) -> 直接当做 ReLU
        if (scaled_acc > 127)       scaled_acc = 127;
        else if (scaled_acc < -128) scaled_acc = -128; // 如果不带ReLU，底线是 -128
        
        output[out_idx] = (int8_t)scaled_acc;
    }
}