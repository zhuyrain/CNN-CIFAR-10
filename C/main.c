// 引入刚刚 Python 生成的真实测试向量！
#include "../python/test_vector.h" 
#include "conv2d_int8_naive.h"
#include "my_printf.h"

#include <stdint.h>

// ==========================================
// [内存分配] 模拟 NPU 的内部 SRAM (SRAM Memory Map)
// ==========================================
// 【关键修改】：将特征图数组定义为全局变量，分配在物理内存的 .bss 段
// 防止 RV32 极小的栈空间被瞬间撑爆导致硬件跑飞
int8_t buf_conv1[16 * 32 * 32] = {0}; 
int8_t buf_pool1[16 * 16 * 16] = {0}; 
int8_t buf_conv2[32 * 16 * 16] = {0}; 
int8_t buf_pool2[32 *  8 *  8] = {0}; 
int8_t buf_fc[10] = {0};

int main() {
    my_printf("===== RISC-V Bare-metal End-to-End Test =====\n");

    // 严格对称量化，Zero-Point 锁定为 0
    int out_zp = 0; 

    // -----------------------------------------------------------
    // Layer 1: Conv1 -> ReLU -> Pool1
    // -----------------------------------------------------------
    my_printf("[1/3] Executing Layer 1...\n");
    
    // 注意：函数名已改回你在 RV32 环境下的 conv2d_int8_naive
    conv2d_int8_naive(
        test_input_img, test_conv1_weight, test_conv1_bias, buf_conv1,
        3, 16, 32, 3, out_zp, L1_MULT, L1_SHIFT
    );
    
    relu_int8(buf_conv1, 16 * 32 * 32);
    maxpool2d_int8(buf_conv1, buf_pool1, 16, 32, 2, 2);

    // -----------------------------------------------------------
    // Layer 2: Conv2 -> ReLU -> Pool2
    // -----------------------------------------------------------
    my_printf("\n[2/3] Executing Layer 2...\n");
    
    conv2d_int8_naive(
        buf_pool1, test_conv2_weight, test_conv2_bias, buf_conv2,
        16, 32, 16, 3, out_zp, L2_MULT, L2_SHIFT
    );
    
    relu_int8(buf_conv2, 32 * 16 * 16);
    maxpool2d_int8(buf_conv2, buf_pool2, 32, 16, 2, 2);

    // -----------------------------------------------------------
    // Layer 3: Fully Connected (Linear)
    // -----------------------------------------------------------
    my_printf("\n[3/3] Executing Layer 3...\n");
    
    linear_int8(
        buf_pool2, test_fc_weight, test_fc_bias, buf_fc,
        2048, 10, FC_MULT, FC_SHIFT
    );

    // -----------------------------------------------------------
    // 终极对账
    // -----------------------------------------------------------
    my_printf("\n>>> Validating Final FC Output...\n");
    
    int error_count = 0;
    int max_idx_c = 0;
    int max_idx_py = 0;
    int8_t max_val_c = -128;
    int8_t max_val_py = -128;

    my_printf("Index | C_Model | Python\n");
    my_printf("------------------------\n");
    for (int i = 0; i < 10; i++) {
        // 裸机环境下的自定义 my_printf 通常不支持 %4d，这里退化为最基础安全的 %d
        my_printf("  %d   |   %d   |   %d\n", i, buf_fc[i], expected_fc_output[i]);
        
        if (buf_fc[i] > max_val_c) {
            max_val_c = buf_fc[i];
            max_idx_c = i;
        }
        if (expected_fc_output[i] > max_val_py) {
            max_val_py = expected_fc_output[i];
            max_idx_py = i;
        }
        
        if (buf_fc[i] != expected_fc_output[i]) {
            error_count++;
        }
    }

    my_printf("\n====== SUMMARY ======\n");
    my_printf("Pred C_Model : Class %d\n", max_idx_c);
    my_printf("Pred Python  : Class %d\n", max_idx_py);

    if (error_count == 0) {
        my_printf("[SUCCESS] PERFECT MATCH! 0 errors.\n");
    } else {
        my_printf("[WARNING] Found bit-level mismatches.\n");
        if (max_idx_c == max_idx_py) {
            my_printf("[NOTE] Top-1 Pred STILL CORRECT.\n");
        } else {
            my_printf("[FAILED] Top-1 Pred MISMATCH!\n");
        }
    }

    return 0;
}