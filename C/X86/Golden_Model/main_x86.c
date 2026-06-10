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

    // ===========================================================
    // [新增] 导出 Conv2 的偏置 (Bias)
    // 形状: 32 个 int32_t 数据
    // 排列顺序: OC0 到 OC31，每行 1 个 32-bit 小写十六进制数据
    // ===========================================================
    printf("\n  >>> Dumping Conv2 Bias to 'conv2_bias_packed.txt'...\n");
    FILE *fp_bias2 = fopen("conv2_bias_packed.txt", "w");
    if (fp_bias2) {
        for (int oc = 0; oc < 32; oc++) {
            // 直接转为无符号 32-bit，打印十六进制补码
            uint32_t b_val = (uint32_t)test_conv2_bias[oc];
            fprintf(fp_bias2, "%08x\n", b_val);
        }
        fclose(fp_bias2);
        printf("      Bias dump complete (32 lines expected).\n");
    }

    // ===========================================================
    // [新增] 导出 Conv2 的权重 (Weight)
    // 原始形状: [32 OC][16 IC][3 H][3 W]
    // 硬件打包: 每行 1 个 32-bit 数据，包含 4 个 IC 的权重
    // 排列顺序: OC分组(外) -> IC分组 -> H -> W -> OC(内)
    // ===========================================================
    printf("  >>> Dumping Conv2 Weight to 'conv2_weight_packed.txt'...\n");
    FILE *fp_w2 = fopen("conv2_weight_packed.txt", "w");
    if (fp_w2) {
        int w_match_count = 0;

        // 1. 最外层：输出通道分组 (32 OC / 4 = 8 组)
        for (int oc_group = 0; oc_group < 8; oc_group++) {
            int base_oc = oc_group * 4; // 当前组的起始 OC: 0, 4, 8...

            // 2. 输入通道分组 (16 IC / 4 = 4 组)
            for (int ic_group = 0; ic_group < 4; ic_group++) {
                int base_ic = ic_group * 4; // 当前组的起始 IC: 0, 4, 8, 12

                // 3. 空间维度遍历 (按要求：w 先变化，所以 w 在最内侧)
                for (int h = 0; h < 3; h++) {
                    for (int w = 0; w < 3; w++) {
                        
                        // 4. 在当前的空间位置上，连续输出 4 个 OC 的打包数据
                        // 这样 DMA 只要连读 4 次，就能喂饱 4 列 PE
                        for (int oc_offset = 0; oc_offset < 4; oc_offset++) {
                            int oc = base_oc + oc_offset;
                            
                            // 提取 4 个 IC 的坐标: (oc, base_ic+x, h, w)
                            // C 语言一维索引计算: (oc * 16 * 3 * 3) + (ic * 3 * 3) + (h * 3) + w
                            int idx0 = (oc * 144) + ((base_ic + 0) * 9) + (h * 3) + w;
                            int idx1 = (oc * 144) + ((base_ic + 1) * 9) + (h * 3) + w;
                            int idx2 = (oc * 144) + ((base_ic + 2) * 9) + (h * 3) + w;
                            int idx3 = (oc * 144) + ((base_ic + 3) * 9) + (h * 3) + w;

                            // 防截断处理
                            uint32_t w0 = (uint32_t)(uint8_t)test_conv2_weight[idx0];
                            uint32_t w1 = (uint32_t)(uint8_t)test_conv2_weight[idx1];
                            uint32_t w2 = (uint32_t)(uint8_t)test_conv2_weight[idx2];
                            uint32_t w3 = (uint32_t)(uint8_t)test_conv2_weight[idx3];

                            // 硬件拼接: {IC3, IC2, IC1, IC0} 
                            uint32_t packed_w = (w3 << 24) | (w2 << 16) | (w1 << 8) | w0;
                            
                            // 写入文本，小写十六进制
                            fprintf(fp_w2, "%08x\n", packed_w);
                            
                            // 调试输出：打印极度关键的第一组数据，方便肉眼校对 FSM
                            if (oc_group == 0 && ic_group == 0 && h == 0 && w == 0) {
                                printf("      [Debug] OC=%d, H=0, W=0 -> Pack{ic3,ic2,ic1,ic0} = %08x\n", oc, packed_w);
                            }
                            w_match_count++;
                        }
                    }
                }
            }
        }
        fclose(fp_w2);
        printf("      Weight dump complete (%d lines expected). \n\n", w_match_count);
    }

// ===========================================================
    // [专属定制 升级版] 提取 Conv2 第二层，OC=0, H=0, W=0 的计算过程
    // 包含 144 次乘加的 32-bit 原始累加和 (Raw Accumulation Sum) 验证
    // ===========================================================
    printf("\n>>> Extracting specific calculation data & Raw MAC Sum for Conv2 (OC=0, H=0, W=0)...\n");

    FILE *fp_single_in   = fopen("conv2_first_calc_in.txt", "w");
    FILE *fp_single_w    = fopen("conv2_first_calc_w.txt", "w");
    FILE *fp_single_bias = fopen("conv2_first_calc_bias.txt", "w");
    FILE *fp_single_out  = fopen("conv2_first_calc_out.txt", "w");
    FILE *fp_single_mac  = fopen("conv2_first_calc_raw_sum.txt", "w"); // 新增：保存累加过程的文件

    if (fp_single_in && fp_single_w && fp_single_bias && fp_single_out && fp_single_mac) {
        
        int oc = 0;
        int out_y = 0;
        int out_x = 0;
        int pad = 1; 

        // 记录 32-bit 累加和的变量
        int32_t raw_mac_sum = 0;
        int32_t bias_val = test_conv2_bias[0];

        // 1. 导出 Bias
        fprintf(fp_single_bias, "OC=0 Bias: %d (0x%08x)\n", bias_val, (uint32_t)bias_val);
        
        // 2. 导出最终 8-bit Output
        int8_t first_out = buf_conv2[0 * 16 * 16 + 0 * 16 + 0];
        fprintf(fp_single_out, "OC=0,H=0,W=0 Output: %d (0x%02x)\n", first_out, (uint8_t)first_out);

        // 3. 遍历 144 次计算过程
        fprintf(fp_single_in, "// Format: [ky][kx][ic] -> Value (Hex)\n");
        fprintf(fp_single_w,  "// Format: [ky][kx][ic] -> Value (Hex)\n");
        fprintf(fp_single_mac, "===== 144 MAC Operations for (OC=0, H=0, W=0) =====\n");

        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                
                int in_y = out_y + ky - pad;
                int in_x = out_x + kx - pad;
                int is_pad = (in_y < 0 || in_y >= 16 || in_x < 0 || in_x >= 16);

                for (int ic = 0; ic < 16; ic++) {
                    // --- 获取输入激活值 ---
                    int8_t in_val = 0;
                    if (!is_pad) {
                        in_val = buf_pool1[ic * 16 * 16 + in_y * 16 + in_x];
                    }
                    fprintf(fp_single_in, "[ky=%d][kx=%d][ic=%2d] in_y=%2d in_x=%2d PAD=%d -> %4d (0x%02x)\n", 
                            ky, kx, ic, in_y, in_x, is_pad, in_val, (uint8_t)in_val);

                    // --- 获取权重值 ---
                    int weight_idx = (oc * 16 * 3 * 3) + (ic * 3 * 3) + (ky * 3) + kx;
                    int8_t w_val = test_conv2_weight[weight_idx];
                    fprintf(fp_single_w, "[ky=%d][kx=%d][ic=%2d] -> %4d (0x%02x)\n", 
                            ky, kx, ic, w_val, (uint8_t)w_val);

                    // --- 【核心逻辑】执行单次乘加，并累加到 32-bit 寄存器中 ---
                    int32_t current_mac = (int32_t)in_val * (int32_t)w_val;
                    raw_mac_sum += current_mac;

                    // 写入到 MAC 记录文件，方便你一步一步核对累加器内部的值
                    fprintf(fp_single_mac, "[Step %3d] in(%4d) * w(%4d) = %6d | Current MAC Sum = %8d (0x%08x)\n", 
                            (ky*48 + kx*16 + ic + 1), in_val, w_val, current_mac, raw_mac_sum, (uint32_t)raw_mac_sum);
                }
            }
        }

        // 4. 计算加上 Bias 之后的最终量化前总和 (Total Raw Sum)
        int32_t total_raw_sum = raw_mac_sum + bias_val;

        fprintf(fp_single_mac, "\n================ SUMMARY ================\n");
        fprintf(fp_single_mac, "1. Pure MAC Sum (144 mults) = %d (0x%08x)\n", raw_mac_sum, (uint32_t)raw_mac_sum);
        fprintf(fp_single_mac, "2. Bias Value               = %d (0x%08x)\n", bias_val, (uint32_t)bias_val);
        fprintf(fp_single_mac, "3. Total Pre-quant Sum      = %d (0x%08x)\n", total_raw_sum, (uint32_t)total_raw_sum);
        fprintf(fp_single_mac, "=========================================\n");
        fprintf(fp_single_mac, "Next step in HW: (Total Pre-quant Sum * L2_MULT) >> L2_SHIFT -> cast to 8-bit -> Out = %d\n", first_out);

        fclose(fp_single_in);
        fclose(fp_single_w);
        fclose(fp_single_bias);
        fclose(fp_single_out);
        fclose(fp_single_mac);
        printf("      Specific extraction complete! Check 'conv2_first_calc_raw_sum.txt' for exact MAC accumulation.\n");
    } else {
        printf("      [Error] Failed to open specific dump files.\n");
    }
    // ================== [定制结束] =============================================
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