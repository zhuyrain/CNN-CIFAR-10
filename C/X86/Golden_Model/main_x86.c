// ============================================================================
// RISC-V & NPU Bit-Accurate Verification with Hardwired Packing
// ============================================================================
#include "../../../python/test_vector.h" 
#include "../conv2d_int8_naive_x86.h"
#include "../other_int8_naive.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// [内存分配] 模拟 NPU 的内部 SRAM
int8_t buf_conv1[16 * 32 * 32] = {0}; 

int main() {
    printf("\n===== RISC-V & NPU Bit-Accurate Verification =====\n");

    int out_zp = 0; 

    // -----------------------------------------------------------
    // 1. 执行完整的 Layer 1 C 模型卷积运算
    // -----------------------------------------------------------
    printf("[Step 1] Executing Full Layer 1 C-Model Convolution...\n");
    
    conv2d_int8_naive_x86(
        test_input_img, test_conv1_weight, test_conv1_bias, buf_conv1,
        3, 16, 32, 3, out_zp, L1_MULT, L1_SHIFT
    );

    // -----------------------------------------------------------
    // 2. 导出与 Verilog 仿真严格对应的交织拼接结果
    // -----------------------------------------------------------
    printf("[Step 2] Packing OC3, OC2, OC1, OC0 and dumping to file...\n");
    
    // 打开输出文件
    FILE *fp = fopen("c_golden_packed.txt", "w");
    if (fp == NULL) {
        printf("❌ Error: Cannot create output file 'c_golden_packed.txt'\n");
        return -1;
    }

    int match_count = 0;

    // 按照空间扫描顺序（Raster Scan: H -> W）遍历 32x32 的每个像素点
    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
            
            // 提取前 4 个输出通道在一维存储中的真实量化值
            // CHW 排列下的索引计算: (oc * H * W) + (y * W) + x
            uint32_t val0 = (uint32_t)(uint8_t)buf_conv1[(0 * 32 * 32) + (y * 32) + x];
            uint32_t val1 = (uint32_t)(uint8_t)buf_conv1[(1 * 32 * 32) + (y * 32) + x];
            uint32_t val2 = (uint32_t)(uint8_t)buf_conv1[(2 * 32 * 32) + (y * 32) + x];
            uint32_t val3 = (uint32_t)(uint8_t)buf_conv1[(3 * 32 * 32) + (y * 32) + x];

            // 严格执行硬件拼接: {OC3, OC2, OC1, OC0}
            // [31:24] = OC3, [23:16] = OC2, [15:8] = OC1, [7:0] = OC0
            uint32_t packed_word = (val3 << 24) | (val2 << 16) | (val1 << 8) | val0;

            // 写入文本文件，保持 8 位小写十六进制形态，和Verilog对齐
            fprintf(fp, "%08x\n", packed_word);

            // 在控制台打印前 4 个特殊点，方便快速肉眼核对
            if (y == 0 && x == 0) {
                printf("\n  [Debug Check] At (ox=0, oy=0):\n");
                printf("    OC0 = %4d (0x%02X)\n", buf_conv1[(0*32*32)], val0);
                printf("    OC1 = %4d (0x%02X)\n", buf_conv1[(1*32*32)], val1);
                printf("    OC2 = %4d (0x%02X)\n", buf_conv1[(2*32*32)], val2);
                printf("    OC3 = %4d (0x%02X)\n", buf_conv1[(3*32*32)], val3);
                printf("    Packed Word => %08X\n\n", packed_word);
            }
            match_count++;
        }
    }

    fclose(fp);

    printf("====== VERIFICATION COMPLETE ======\n");
    printf("Successfully generated 'c_golden_packed.txt' with %d words.\n", match_count);
    printf("You can now directly diff it with your Verilog SRAM dump file.\n\n");

    return 0;
}