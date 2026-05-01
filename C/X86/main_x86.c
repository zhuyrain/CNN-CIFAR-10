// 引入 Python 导出的真实常量、权重和黄金对照组
#include "../../python/test_vector.h" 
#include "conv2d_int8_naive_x86.h"
#include "other_int8_naive.h"

#include <stdio.h>
#include <stdint.h>
// ==========================================
// 1. 核心算子声明 (部分为待实现的占位函数)
// ==========================================

// // 已有的纯 INT8 卷积计算算子
// extern void conv2d_int8_naive_x86(
//     const int8_t* input, const int8_t* weight, const int32_t* bias, int8_t* output,
//     int in_channels, int out_channels, int in_size, int kernel_size,
//     int out_zp, int32_t multiplier, int shift);

// // [待实现] 原地激活的 ReLU 算子 (因为是 max(0, x)，直接覆盖原数组最省内存)
// extern void relu_int8(int8_t* data, int size);

// // [待实现] 2x2 最大池化算子
// extern void maxpool2d_int8(
//     const int8_t* input, int8_t* output, 
//     int channels, int in_size, int pool_size, int stride);

// // [待实现] 全连接层算子
// extern void linear_int8(
//     const int8_t* input, const int8_t* weight, const int32_t* bias, int8_t* output,
//     int in_features, int out_features, 
//     int32_t multiplier, int shift);


// ==========================================
// 2. 主流程：端到端 CIFAR-10 推理
// ==========================================
int main() {
    printf("===== RISC-V End-to-End W8A8 Golden Model Test =====\n");

    // -----------------------------------------------------------
    // [内存分配] 模拟 NPU 的内部 SRAM (SRAM Memory Map)
    // -----------------------------------------------------------
    // Conv1 输出: 16通道 * 32 * 32 = 16384 bytes
    int8_t buf_conv1[16 * 32 * 32] = {0}; 
    // Pool1 输出: 16通道 * 16 * 16 = 4096 bytes
    int8_t buf_pool1[16 * 16 * 16] = {0}; 
    // Conv2 输出: 32通道 * 16 * 16 = 8192 bytes
    int8_t buf_conv2[32 * 16 * 16] = {0}; 
    // Pool2 输出: 32通道 * 8 * 8 = 2048 bytes
    int8_t buf_pool2[32 *  8 *  8] = {0}; 
    // FC 层最终分类输出: 10 bytes
    int8_t buf_fc[10] = {0};

    // 严格对称量化，Zero-Point 锁定为 0
    int out_zp = 0; 

    // -----------------------------------------------------------
    // Layer 1: Conv1 -> ReLU -> Pool1
    // -----------------------------------------------------------
    printf("[1/3] Executing Layer 1 (Conv -> ReLU -> Pool)...\n");
    
    conv2d_int8_naive_x86(
        test_input_img, test_conv1_weight, test_conv1_bias, buf_conv1,
        3, 16, 32, 3, out_zp, L1_MULT, L1_SHIFT
    );
    
    // 原地 ReLU 激活
    relu_int8(buf_conv1, 16 * 32 * 32);
    
    // 2x2 池化，步长为 2，输入尺寸为 32
    maxpool2d_int8(buf_conv1, buf_pool1, 16, 32, 2, 2);

    // -----------------------------------------------------------
    // Layer 2: Conv2 -> ReLU -> Pool2
    // -----------------------------------------------------------
    printf("\n[2/3] Executing Layer 2 (Conv -> ReLU -> Pool)...\n");
    
    conv2d_int8_naive_x86(
        buf_pool1, test_conv2_weight, test_conv2_bias, buf_conv2,
        16, 32, 16, 3, out_zp, L2_MULT, L2_SHIFT
    );
    
    // 原地 ReLU 激活
    relu_int8(buf_conv2, 32 * 16 * 16);
    
    // 2x2 池化，步长为 2，输入尺寸为 16
    maxpool2d_int8(buf_conv2, buf_pool2, 32, 16, 2, 2);

    // -----------------------------------------------------------
    // Layer 3: Fully Connected (Linear)
    // -----------------------------------------------------------
    printf("\n[3/3] Executing Layer 3 (Fully Connected)...\n");
    
    // FC 层的输入就是 Pool2 展平后的结果，长度为 2048 (32*8*8)
    linear_int8(
        buf_pool2, test_fc_weight, test_fc_bias, buf_fc,
        2048, 10, FC_MULT, FC_SHIFT
    );

    // -----------------------------------------------------------
    // 终极对账：检验最终 10 分类向量是否完美对齐
    // -----------------------------------------------------------
    printf("\n>>> Pipeline Finished! Validating Final FC Output...\n");
    
    int error_count = 0;
    int max_idx_c = 0;
    int max_idx_py = 0;
    int8_t max_val_c = -128;
    int8_t max_val_py = -128;

    printf("Index | C_Model | Python_Expected\n");
    printf("----------------------------------\n");
    for (int i = 0; i < 10; i++) {
        // 打印详细对比
        printf("  %d   |   %4d  |   %4d\n", i, buf_fc[i], expected_fc_output[i]);
        
        // 找 C 模型的预测结果 (Argmax)
        if (buf_fc[i] > max_val_c) {
            max_val_c = buf_fc[i];
            max_idx_c = i;
        }
        // 找 Python 模型的预测结果
        if (expected_fc_output[i] > max_val_py) {
            max_val_py = expected_fc_output[i];
            max_idx_py = i;
        }
        
        // 记录绝对错误
        if (buf_fc[i] != expected_fc_output[i]) {
            error_count++;
        }
    }

    printf("\n================ SUMMARY ================\n");
    printf("Prediction C_Model : Class %d\n", max_idx_c);
    printf("Prediction Python  : Class %d\n", max_idx_py);

    if (error_count == 0) {
        printf("[SUCCESS] END-TO-END PERFECT MATCH! 0 errors.\n");
        printf("你的软硬件协同设计已彻底跑通！可以放心流片/上板。\n");
    } else {
        printf("[WARNING] Found %d bit-level mismatches in the final 10 elements.\n", error_count);
        if (max_idx_c == max_idx_py) {
            printf("[NOTE] Top-1 Prediction is STILL CORRECT despite value mismatches.\n");
        } else {
            printf("[FAILED] Top-1 Prediction MISMATCH!\n");
        }
    }

    return 0;
}