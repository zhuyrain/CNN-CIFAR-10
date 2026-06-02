// ============================================================================
// RISC-V & NPU Bit-Accurate Verification with Hardwired Packing (Full 16 ICs & 32 ICs Second Conv)
// ============================================================================
#include "../../../python/test_vector.h" 
#include "../conv2d_int8_naive_x86.h"
#include "../other_int8_naive.h"

#include <stdio.h> 
#include <stdint.h> 
#include <stdlib.h> // 为 FILE 操作引入

// ========================================== 
// 1. 核心算子声明 
// ==========================================
// 已于conv2d_int8_naive_x86.h和other_int8_naive.h中声明

// ========================================== 
// 2. 主流程：端到端 CIFAR-10 推理 
// ========================================== 
int main() { 
    printf("===== RISC-V End-to-End W8A8 Golden Model Test =====\n");

    // -----------------------------------------------------------
    // [内存分配] 模拟 NPU 的内部 SRAM (SRAM Memory Map)
    // -----------------------------------------------------------
    int8_t buf_conv1[16 * 32 * 32] = {0}; 
    int8_t buf_pool1[16 * 16 * 16] = {0}; 
    int8_t buf_conv2[32 * 16 * 16] = {0}; 
    int8_t buf_pool2[32 * 8 * 8] = {0}; 
    int8_t buf_fc[10] = {0};

    int out_zp = 0; 

    // -----------------------------------------------------------
    // Layer 1: Conv1 -> ReLU -> Pool1
    // -----------------------------------------------------------
    printf("[1/3] Executing Layer 1 (Conv -> ReLU -> Pool)...\n");

    conv2d_int8_naive_x86(
        test_input_img, test_conv1_weight, test_conv1_bias, buf_conv1,
        3, 16, 32, 3, out_zp, L1_MULT, L1_SHIFT
    );

    relu_int8(buf_conv1, 16 * 32 * 32);
    maxpool2d_int8(buf_conv1, buf_pool1, 16, 32, 2, 2);

// ===========================================================
    // [修改] 导出 Conv2 的输入 (Pool1 的输出)
    // 形状: 16通道 * 16 * 16，按 4 通道分组。
    // 排列顺序: HWC 交织 (先排满一个像素的 16 个通道，再切下一个像素)
    // ===========================================================
    printf("  >>> Dumping Conv2 Input (buf_pool1) to 'conv2_input_packed.txt'...\n");
    FILE *fp_in = fopen("conv2_input_packed.txt", "w");
    if (fp_in) {
        // 空间维度 (Y, X) 在最外层
        for (int y = 0; y < 16; y++) {
            for (int x = 0; x < 16; x++) {
                
                // 通道分组 (Group) 在最内层：连续吐出一个像素的所有深度数据
                for (int ic_group = 0; ic_group < 4; ic_group++) {
                    int base_ic = ic_group * 4;
                    
                    // 从 C 模型的 CHW 一维数组中精准拾取对应坐标的值
                    uint32_t val0 = (uint32_t)(uint8_t)buf_pool1[(base_ic + 0) * 16 * 16 + y * 16 + x];
                    uint32_t val1 = (uint32_t)(uint8_t)buf_pool1[(base_ic + 1) * 16 * 16 + y * 16 + x];
                    uint32_t val2 = (uint32_t)(uint8_t)buf_pool1[(base_ic + 2) * 16 * 16 + y * 16 + x];
                    uint32_t val3 = (uint32_t)(uint8_t)buf_pool1[(base_ic + 3) * 16 * 16 + y * 16 + x];

                    // 硬件打包拼接: {IC3, IC2, IC1, IC0}
                    uint32_t packed = (val3 << 24) | (val2 << 16) | (val1 << 8) | val0;
                    
                    // 输出小写十六进制
                    fprintf(fp_in, "%08x\n", packed);
                }
            }
        }
        fclose(fp_in);
    }

    // -----------------------------------------------------------
    // Layer 2: Conv2 -> ReLU -> Pool2
    // -----------------------------------------------------------
    printf("\n[2/3] Executing Layer 2 (Conv -> ReLU -> Pool)...\n");

    conv2d_int8_naive_x86(
        buf_pool1, test_conv2_weight, test_conv2_bias, buf_conv2,
        16, 32, 16, 3, out_zp, L2_MULT, L2_SHIFT
    );

    // ===========================================================
    // [修改] 导出 Conv2 的输出 (量化后，ReLU前)
    // 形状: 32通道 * 16 * 16，按 4 通道分组。
    // 排列顺序: HWC 交织 (先排满一个像素的 32 个通道，再切下一个像素)
    // ===========================================================
    printf("  >>> Dumping Conv2 Output (buf_conv2) to 'conv2_output_packed.txt'...\n");
    FILE *fp_out = fopen("conv2_output_packed.txt", "w");
    if (fp_out) {
        // 空间维度 (Y, X) 在最外层
        for (int y = 0; y < 16; y++) {
            for (int x = 0; x < 16; x++) {
                
                // 通道分组 (Group) 在最内层 (32通道共计8组)
                for (int oc_group = 0; oc_group < 8; oc_group++) { 
                    int base_oc = oc_group * 4;
                    
                    // 拾取坐标
                    uint32_t val0 = (uint32_t)(uint8_t)buf_conv2[(base_oc + 0) * 16 * 16 + y * 16 + x];
                    uint32_t val1 = (uint32_t)(uint8_t)buf_conv2[(base_oc + 1) * 16 * 16 + y * 16 + x];
                    uint32_t val2 = (uint32_t)(uint8_t)buf_conv2[(base_oc + 2) * 16 * 16 + y * 16 + x];
                    uint32_t val3 = (uint32_t)(uint8_t)buf_conv2[(base_oc + 3) * 16 * 16 + y * 16 + x];

                    // 硬件打包拼接: {OC3, OC2, OC1, OC0}
                    uint32_t packed = (val3 << 24) | (val2 << 16) | (val1 << 8) | val0;
                    
                    // 输出小写十六进制
                    fprintf(fp_out, "%08x\n", packed);
                }
            }
        }
        fclose(fp_out);
    }

    // 原地 ReLU 激活 (发生在 Dump 之后，确保验证的是纯 Conv 的 MAC 结果)
    relu_int8(buf_conv2, 32 * 16 * 16);
    maxpool2d_int8(buf_conv2, buf_pool2, 32, 16, 2, 2);

    // -----------------------------------------------------------
    // Layer 3: Fully Connected (Linear)
    // -----------------------------------------------------------
    printf("\n[3/3] Executing Layer 3 (Fully Connected)...\n");

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
        printf("  %d   |   %4d  |   %4d\n", i, buf_fc[i], expected_fc_output[i]);
        
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