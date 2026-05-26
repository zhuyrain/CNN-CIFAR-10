// 引入 Python 导出的真实常量、权重和黄金对照组
// 作为 Verilog仿真的对照组，对比每层的精确结果，目前仅对比第一conv层及其后的relu层结果
#include "../../../python/test_vector.h" 
#include "../conv2d_int8_naive_x86.h"
#include "../other_int8_naive.h"

#include <stdio.h>
#include <stdint.h>

// ==========================================
// [内存分配] 模拟 NPU 的内部 SRAM (SRAM Memory Map)
// ==========================================
// 仅保留 Layer 1 相关的 Buffer，节约内存
int8_t buf_conv1[16 * 32 * 32] = {0}; 

int main() {
    printf("\n===== RISC-V & NPU Bit-Accurate Verification =====\n");

    // 严格对称量化，Zero-Point 锁定为 0
    int out_zp = 0; 

    // -----------------------------------------------------------
    // 1. 软件手工核对 32-bit 原始累加和 (Raw Psum)
    // -----------------------------------------------------------
    // 专门针对滑动窗口第一个位置 (ox=0, oy=0)，前 4 个输出通道 (oc=0~3)
    printf("\n[Step 1] Verifying 32-bit Raw Psum (ox=0, oy=0, oc=0~3)...\n");
    
    int pad = 1;
    for (int oc = 0; oc < 4; oc++) {
        int32_t acc = test_conv1_bias[oc]; // 初始装填 Bias
        
        for (int ic = 0; ic < 3; ic++) {
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int iy = 0 + ky - pad;
                    int ix = 0 + kx - pad;

                    if (iy >= 0 && iy < 32 && ix >= 0 && ix < 32) {
                        int in_idx = (ic * 32 * 32) + (iy * 32) + ix;
                        int w_idx = (oc * 3 * 3 * 3) + (ic * 3 * 3) + (ky * 3) + kx;
                        acc += (int32_t)test_input_img[in_idx] * (int32_t)test_conv1_weight[w_idx];
                    }
                }
            }
        }
        printf("  Hardware Psum for Col %d (OC %d) should be : %d\n", oc, oc, acc);
    }

    // -----------------------------------------------------------
    // 2. 执行完整的 Layer 1 C 模型卷积运算
    // -----------------------------------------------------------
    printf("\n[Step 2] Executing Full Layer 1 C-Model Convolution...\n");
    
    // 执行包含量化截断的 naive 卷积
    conv2d_int8_naive_x86(
        test_input_img, test_conv1_weight, test_conv1_bias, buf_conv1,
        3, 16, 32, 3, out_zp, L1_MULT, L1_SHIFT
    );

    // -----------------------------------------------------------
    // 3. 打印最终写入 SRAM 的 8-bit 特征图结果
    // -----------------------------------------------------------
    printf("\n[Step 3] Final 8-bit Quantized Output (Before ReLU/Pool)...\n");
    
    for (int oc = 0; oc < 4; oc++) {
        // 计算输出数组的一维索引 (oc * H * W) + (y * W) + x
        int out_idx = (oc * 32 * 32) + (0 * 32) + 0; 
        printf("  SRAM Output for OC %d (ox=0, oy=0) : %d\n", oc, buf_conv1[out_idx]);
    }

    printf("\n====== VERIFICATION COMPLETE ======\n");

    return 0;
}