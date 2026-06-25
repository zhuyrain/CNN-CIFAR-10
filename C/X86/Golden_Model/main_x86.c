// ============================================================================
// RISC-V & NPU Bit-Accurate Verification (X86 Host Golden Model)
// 适配 64 通道 VGG-Style 网络，CHW 计算，HWC 打印比对！
// ============================================================================
#include "../../../python/test_vector.h" 
#include "../conv2d_int8_naive_x86.h"
#include "../other_int8_naive.h"

#include <stdio.h> 
#include <stdint.h> 
#include <stdlib.h>

// ===========================================================
// [导出专用] 导出 Conv2 的权重 (Weight) 为 C 语言头文件格式
// 原始形状: [64 OC][64 IC][3 H][3 W]
// 硬件打包: 每行 1 个 32-bit 数据，包含 4 个 IC 的权重
// 排列顺序: OC分组(外) -> IC分组 -> H -> W -> OC(内)
// ===========================================================
void dump_conv2_weight_to_h() {
    printf("  >>> Dumping Conv2 Weight to 'conv2_weight_packed.h'...\n");
    FILE *fp_w2 = fopen("conv2_weight_packed.h", "w");
    if (fp_w2) {
        int w_match_count = 0;
        int words_per_line = 0;

        fprintf(fp_w2, "// ===================================================================\n");
        fprintf(fp_w2, "// NPU Packed Weight Data for Conv2 (64 OC, 64 IC, 3x3)\n");
        fprintf(fp_w2, "// 格式: {IC3, IC2, IC1, IC0}\n");
        fprintf(fp_w2, "// ===================================================================\n");
        fprintf(fp_w2, "const uint32_t test_conv2_weight_packed[9216] = {\n");

        // 1. 最外层：输出通道分组 (64 OC / 4列 = 16 组)
        for (int oc_group = 0; oc_group < 16; oc_group++) {
            int base_oc = oc_group * 4; // 当前组的起始 OC: 0, 4, 8 ... 60

            // 2. 输入通道分组 (64 IC / 4 = 16 组)
            for (int ic_group = 0; ic_group < 16; ic_group++) {
                int base_ic = ic_group * 4; // 当前组的起始 IC: 0, 4, 8 ... 60

                // 3. 空间维度遍历 (按要求：w 先变化，所以 w 在最内侧)
                for (int h = 0; h < 3; h++) {
                    for (int w = 0; w < 3; w++) {
                        
                        // 4. 在当前的空间位置上，连续输出 4 个 OC 的打包数据
                        for (int oc_offset = 0; oc_offset < 4; oc_offset++) {
                            int oc = base_oc + oc_offset;
                            
                            // 提取 4 个 IC 的坐标: (oc, base_ic+x, h, w)
                            // 一维索引计算: (oc * 64 * 3 * 3) + (ic * 3 * 3) + (h * 3) + w
                            // 即: (oc * 576) + (ic * 9) + (h * 3) + w
                            int idx0 = (oc * 576) + ((base_ic + 0) * 9) + (h * 3) + w;
                            int idx1 = (oc * 576) + ((base_ic + 1) * 9) + (h * 3) + w;
                            int idx2 = (oc * 576) + ((base_ic + 2) * 9) + (h * 3) + w;
                            int idx3 = (oc * 576) + ((base_ic + 3) * 9) + (h * 3) + w;

                            // 防符号位扩展截断处理
                            uint32_t w0 = (uint32_t)(uint8_t)test_conv2_weight[idx0];
                            uint32_t w1 = (uint32_t)(uint8_t)test_conv2_weight[idx1];
                            uint32_t w2 = (uint32_t)(uint8_t)test_conv2_weight[idx2];
                            uint32_t w3 = (uint32_t)(uint8_t)test_conv2_weight[idx3];

                            // 硬件拼接: {IC3, IC2, IC1, IC0} 
                            uint32_t packed_w = (w3 << 24) | (w2 << 16) | (w1 << 8) | w0;
                            
                            // 每行打印 8 个数据，保持代码美观
                            if (words_per_line == 0) {
                                fprintf(fp_w2, "    ");
                            }
                            fprintf(fp_w2, "0x%08x, ", packed_w);
                            
                            words_per_line++;
                            if (words_per_line == 8) {
                                fprintf(fp_w2, "\n");
                                words_per_line = 0;
                            }
                            w_match_count++;
                        }
                    }
                }
            }
        }
        
        if (words_per_line != 0) {
            fprintf(fp_w2, "\n"); // 补齐最后一行的换行
        }
        fprintf(fp_w2, "};\n");
        fclose(fp_w2);
        printf("      Weight dump complete (%d words expected). \n\n", w_match_count);
    } else {
        printf("      [Error] Failed to open conv2_weight_packed.h for writing.\n");
    }
}

// ==========================================
// 辅助对比函数：将 X86 的 CHW 结果与 Python 直接导出的 CHW 黄金模型对比
// 打印时依然按 HWC (Word) 格式输出，以保持与 NPU 嵌入式日志绝对对齐
// ==========================================
void check_8_words_chw(const int8_t* chw_buf, const int8_t* exp_chw_buf, int H, int W, const char* layer_name) {
    printf("\n>>> Checking x86 %s Output (First 8 Words / 32 Channels) <<<\n", layer_name);
    
    for (int i = 0; i < 8; i++) {
        int ch_base = i * 4;
        
        // 【核心转换】：从 X86 算子输出的 CHW 数组中提取 Pixel(0,0) 的 4 个通道数据
        // 在 CHW 格式中，通道 c 的 (0,0) 像素地址跨度刚好是 c * (H * W)
        int8_t c0 = chw_buf[(ch_base + 0) * H * W];
        int8_t c1 = chw_buf[(ch_base + 1) * H * W];
        int8_t c2 = chw_buf[(ch_base + 2) * H * W];
        int8_t c3 = chw_buf[(ch_base + 3) * H * W];

        // 【黄金数据】：以完全相同的逻辑，从 Python 导出的 CHW 数组中提取对应通道
        int8_t e0 = exp_chw_buf[(ch_base + 0) * H * W];
        int8_t e1 = exp_chw_buf[(ch_base + 1) * H * W];
        int8_t e2 = exp_chw_buf[(ch_base + 2) * H * W];
        int8_t e3 = exp_chw_buf[(ch_base + 3) * H * W];

        // 验证对比
        if (c0 == e0 && c1 == e1 && c2 == e2 && c3 == e3) {
            printf("  Word[%d] CH[%d~%d]: MATCH\n", i, ch_base+3, ch_base);
        } else {
            printf("  [ERROR] Word[%d] CH[%d~%d] Mismatch:\n", i, ch_base+3, ch_base);
            
            // 逐个指出具体的错误通道
            if (c3 != e3) printf("    -> CH[%d] x86: %d | EXP: %d\n", ch_base+3, c3, e3);
            if (c2 != e2) printf("    -> CH[%d] x86: %d | EXP: %d\n", ch_base+2, c2, e2);
            if (c1 != e1) printf("    -> CH[%d] x86: %d | EXP: %d\n", ch_base+1, c1, e1);
            if (c0 != e0) printf("    -> CH[%d] x86: %d | EXP: %d\n", ch_base,   c0, e0);
        }
    }
}
// ========================================== 
// 主流程：端到端 CIFAR-10 推理 (X86 纯软件平台)
// ========================================== 
int main() { 
    printf("===== X86 Host End-to-End CHW Golden Model Test =====\n");

    // -----------------------------------------------------------
    // [内存分配] 适配 64 通道的 VGG-Style 网络
    // -----------------------------------------------------------
    int8_t* buf_conv1 = (int8_t*)calloc(64 * 32 * 32, sizeof(int8_t));
    int8_t* buf_conv2 = (int8_t*)calloc(64 * 32 * 32, sizeof(int8_t));
    int8_t* buf_pool1 = (int8_t*)calloc(64 * 16 * 16, sizeof(int8_t));
    int8_t* buf_conv3 = (int8_t*)calloc(64 * 16 * 16, sizeof(int8_t));
    int8_t* buf_conv4 = (int8_t*)calloc(64 * 16 * 16, sizeof(int8_t));
    int8_t* buf_pool2 = (int8_t*)calloc(64 *  8 *  8, sizeof(int8_t));
    int8_t  buf_fc[10] = {0};

    // -----------------------------------------------------------
    // Layer 1: Conv1 -> ReLU (3 -> 64)
    // -----------------------------------------------------------
    printf("\n[1/7] Executing x86 Layer 1 (Conv1: 64 OC)...\n");
    conv2d_int8_naive_x86(test_input_img, test_conv1_weight, test_conv1_bias, buf_conv1,
                          3, 64, 32, 3, 0, L1_MULT, L1_SHIFT);
    relu_int8(buf_conv1, 64 * 32 * 32);
    check_8_words_chw(buf_conv1, expected_act1_output, 32, 32, "Conv1+ReLU");

    // -----------------------------------------------------------
    // Layer 2: Conv2 -> ReLU (64 -> 64)
    // -----------------------------------------------------------
    printf("\n[2/7] Executing x86 Layer 2 (Conv2: 64 OC, 64 IC)...\n");
    conv2d_int8_naive_x86(buf_conv1, test_conv2_weight, test_conv2_bias, buf_conv2,
                          64, 64, 32, 3, 0, L2_MULT, L2_SHIFT);
    relu_int8(buf_conv2, 64 * 32 * 32);
    // 请确保使用你生成 Python 头文件时的实际名字 (expected_conv2_output 或 expected_act2_output)
    check_8_words_chw(buf_conv2, expected_act2_output, 32, 32, "Conv2+ReLU");
    dump_conv2_weight_to_h();
    // -----------------------------------------------------------
    // Layer 3: Pool1 (32x32 -> 16x16)
    // -----------------------------------------------------------
    printf("\n[3/7] CPU Executing MaxPool 1...\n");
    maxpool2d_int8(buf_conv2, buf_pool1, 64, 32, 2, 2);
    check_8_words_chw(buf_pool1, expected_pool1_output, 16, 16, "Pool1");

    // -----------------------------------------------------------
    // Layer 4: Conv3 -> ReLU (64 -> 64)
    // -----------------------------------------------------------
    printf("\n[4/7] Executing x86 Layer 3 (Conv3: 64 OC, 64 IC)...\n");
    conv2d_int8_naive_x86(buf_pool1, test_conv3_weight, test_conv3_bias, buf_conv3,
                          64, 64, 16, 3, 0, L3_MULT, L3_SHIFT);
    relu_int8(buf_conv3, 64 * 16 * 16);
    check_8_words_chw(buf_conv3, expected_act3_output, 16, 16, "Conv3+ReLU");

    // -----------------------------------------------------------
    // Layer 5: Conv4 -> ReLU (64 -> 64)
    // -----------------------------------------------------------
    printf("\n[5/7] Executing x86 Layer 4 (Conv4: 64 OC, 64 IC)...\n");
    conv2d_int8_naive_x86(buf_conv3, test_conv4_weight, test_conv4_bias, buf_conv4,
                          64, 64, 16, 3, 0, L4_MULT, L4_SHIFT);
    relu_int8(buf_conv4, 64 * 16 * 16);
    check_8_words_chw(buf_conv4, expected_act4_output, 16, 16, "Conv4+ReLU");

    // -----------------------------------------------------------
    // Layer 6: Pool2 (16x16 -> 8x8)
    // -----------------------------------------------------------
    printf("\n[6/7] CPU Executing MaxPool 2...\n");
    maxpool2d_int8(buf_conv4, buf_pool2, 64, 16, 2, 2);
    check_8_words_chw(buf_pool2, expected_pool2_output, 8, 8, "Pool2");

    // -----------------------------------------------------------
    // Layer 7: Fully Connected (64*8*8 -> 10)
    // -----------------------------------------------------------
    printf("\n[7/7] CPU Executing Fully Connected Layer...\n");
    linear_int8(buf_pool2, test_fc_weight, test_fc_bias, buf_fc, 64 * 8 * 8, 10, FC_MULT, FC_SHIFT);

    // -----------------------------------------------------------
    // 终极对账 (10 分类结果)
    // -----------------------------------------------------------
    printf("\n>>> Validating Final FC Output <<<\n");
    
    int error_count = 0;
    int max_idx_x86 = 0;
    int max_idx_py = 0;
    int8_t max_val_x86 = -128;
    int8_t max_val_py = -128;

    printf("Index | X86_C_Model | Python_Golden\n");
    printf("------------------------------------\n");
    for (int i = 0; i < 10; i++) {
        printf("  %d   |    %5d    |    %5d\n", i, buf_fc[i], expected_fc_output[i]);
        
        if (buf_fc[i] > max_val_x86) {
            max_val_x86 = buf_fc[i];
            max_idx_x86 = i;
        }
        if (expected_fc_output[i] > max_val_py) {
            max_val_py = expected_fc_output[i];
            max_idx_py = i;
        }
        if (buf_fc[i] != expected_fc_output[i]) {
            error_count++;
        }
    }

    printf("\n====== SUMMARY ======\n");
    printf("Pred X86_C_Model : Class %d\n", max_idx_x86);
    printf("Pred Python      : Class %d\n", max_idx_py);

    if (error_count == 0) {
        printf("[SUCCESS] PERFECT MATCH! 0 errors.\n");
    } else {
        printf("[WARNING] Found bit-level mismatches.\n");
    }

    free(buf_conv1); free(buf_conv2); free(buf_pool1); 
    free(buf_conv3); free(buf_conv4); free(buf_pool2);
    return 0;
}