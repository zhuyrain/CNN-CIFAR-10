// 这里包含你用 Python 重新打包好的 HWC 格式的头文件！
#include "../python/test_vector_2.h"
#include "conv2d_int8_naive.h"
#include "my_printf.h"

#include <stdint.h>
// ==========================================
// 1. NPU AXI-Lite 寄存器映射宏定义
// ==========================================
#define NPU_BASE            0x40000000
#define NPU_REG_CTRL_STATUS (*(volatile uint32_t*)(NPU_BASE + 0x00))
#define NPU_REG_ACT_BASE    (*(volatile uint32_t*)(NPU_BASE + 0x04))
#define NPU_REG_WEIGHT_BASE (*(volatile uint32_t*)(NPU_BASE + 0x08))
#define NPU_REG_BIAS_BASE   (*(volatile uint32_t*)(NPU_BASE + 0x0C))
#define NPU_REG_OUT_BASE    (*(volatile uint32_t*)(NPU_BASE + 0x10))
#define NPU_REG_CFG_IMG_DIM (*(volatile uint32_t*)(NPU_BASE + 0x14))
#define NPU_REG_CFG_CH      (*(volatile uint32_t*)(NPU_BASE + 0x18))
#define NPU_REG_CFG_QUANT   (*(volatile uint32_t*)(NPU_BASE + 0x1C))
#define NPU_REG_CFG_DPATH   (*(volatile uint32_t*)(NPU_BASE + 0x20))

// ==========================================
// 2. 内存分配 (适配 64x64 阵列，强行 64 字节对齐保平安)
// ==========================================
int8_t buf_conv1[64 * 32 * 32] __attribute__((aligned(4))) = {0}; 
int8_t buf_conv2[64 * 32 * 32] __attribute__((aligned(4))) = {0}; 
int8_t buf_pool1[64 * 16 * 16] __attribute__((aligned(4))) = {0}; 
int8_t buf_conv3[64 * 16 * 16] __attribute__((aligned(4))) = {0}; 
int8_t buf_conv4[64 * 16 * 16] __attribute__((aligned(4))) = {0}; 
int8_t buf_pool2[64 *  8 *  8] __attribute__((aligned(4))) = {0}; 
int8_t buf_fc[10]              __attribute__((aligned(4))) = {0};
// ==========================================
// 3. 辅助函数：启动 NPU 并轮询等待
// ==========================================
void wait_npu_done() {
    // 检查 Bit 2 (DONE 标志)
    while ((NPU_REG_CTRL_STATUS & 0x00000004) == 0) {
        // 死等 (Polling)
    }
    // 写 1 清除 DONE 标志
    NPU_REG_CTRL_STATUS = 0x00000004; 
    // 表示npu进行完一轮运算
    my_printf("-");
}

int main() {
    my_printf("Input Base: %d\n", test_input_img);
    my_printf("Weight Base: %d\n", test_conv1_weight);
    my_printf("Conv1 Out Base: %d\n", buf_conv1);
    my_printf("===== RISC-V & NPU 64x64 FULL POWER Inference =====\n");

    // -----------------------------------------------------------
    // Layer 1: Hardware Conv1 (32x32x3 -> 32x32x64)
    // -----------------------------------------------------------
    my_printf("[1/7] Executing NPU Layer 1 (Conv1: 64 OC)...\n");

    NPU_REG_CFG_IMG_DIM = (32 << 16) | 32;       
    NPU_REG_CFG_QUANT   = (L1_SHIFT << 16) | L1_MULT; 
    // Out_Stride=64, IC_G_R=0(空间全并行!), W_Num=9, LB_W=34, IC_G=0(输入仅4通道)
    NPU_REG_CFG_DPATH   = (64 << 22) | (0 << 18) | (9 << 10) | (34 << 4) | 0;
    // Row_words = 32*(0+1) = 32; Weight_words = 9 * 64(OC) = 576
    NPU_REG_CFG_CH      = (32 << 16) | 576;
    
    NPU_REG_ACT_BASE    = (uint32_t)test_input_img;
    NPU_REG_BIAS_BASE   = (uint32_t)test_conv1_bias;
    NPU_REG_WEIGHT_BASE = (uint32_t)test_conv1_weight;
    NPU_REG_OUT_BASE    = (uint32_t)buf_conv1;

    // 【震撼瘦身】：因为有 64 列，一层只需要启动一次！没有 pass 循环了！
    NPU_REG_CTRL_STATUS = 0x00000001; 
    wait_npu_done();
    
    // ===========================================================
    // 校验 Conv1
    // ===========================================================
    my_printf("\n>>> Checking NPU Conv1 Output (First 8 Words / 32 Channels) <<<\n");
    uint32_t* ptr_conv1_32b = (uint32_t*)buf_conv1;
    uint32_t* ptr_exp1_32b  = (uint32_t*)expected_act1_output;
    
    for (int i = 0; i < 8; i++) {
        uint32_t val = ptr_conv1_32b[i];
        uint32_t exp = ptr_exp1_32b[i];
        
        int ch_base = i * 4;
        
        // 提取硬件计算输出的 4 个通道 (8-bit)
        int8_t c0 = (int8_t)(val & 0xFF);
        int8_t c1 = (int8_t)((val >> 8) & 0xFF);
        int8_t c2 = (int8_t)((val >> 16) & 0xFF);
        int8_t c3 = (int8_t)((val >> 24) & 0xFF);
        
        // 提取预期输出的 4 个通道 (8-bit)
        int8_t e0 = (int8_t)(exp & 0xFF);
        int8_t e1 = (int8_t)((exp >> 8) & 0xFF);
        int8_t e2 = (int8_t)((exp >> 16) & 0xFF);
        int8_t e3 = (int8_t)((exp >> 24) & 0xFF);
        
        if (val == exp) {
            my_printf("  Word[%d] CH[%d~%d]: MATCH\n", i, ch_base+3, ch_base);
        } else {
            my_printf("  [ERROR] Word[%d] CH[%d~%d] Mismatch:\n", i, ch_base+3, ch_base);
            if (c3 != e3) my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+3, c3, e3);
            if (c2 != e2) my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+2, c2, e2);
            if (c1 != e1) my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+1, c1, e1);
            if (c0 != e0) my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base,   c0, e0);
        }
    }

    // -----------------------------------------------------------
    // Layer 2: Hardware Conv2 (32x32x64 -> 32x32x64)
    // -----------------------------------------------------------
    my_printf("\n[2/7] Executing NPU Layer 2 (Conv2: 64 OC, 64 IC)...\n");

    NPU_REG_CFG_IMG_DIM = (32 << 16) | 32;       
    NPU_REG_CFG_QUANT   = (L2_SHIFT << 16) | L2_MULT; 
    // Out_Stride=64, IC_G_R=0(魔法核心!), W_Num=144(16组*9), LB_W=34, IC_G=15(写通道需16组)
    NPU_REG_CFG_DPATH   = (64 << 22) | (0 << 18) | (9 << 10) | (34 << 4) | 15; 
    // Row_words = 32*(15+1) = 512; Weight_words = 144 * 64(OC) = 9216
    NPU_REG_CFG_CH      = (512 << 16) | 9216;
    
    NPU_REG_ACT_BASE    = (uint32_t)buf_conv1;
    NPU_REG_BIAS_BASE   = (uint32_t)test_conv2_bias;
    NPU_REG_WEIGHT_BASE = (uint32_t)test_conv2_weight;
    NPU_REG_OUT_BASE    = (uint32_t)buf_conv2;

    NPU_REG_CTRL_STATUS = 0x00000001;
    wait_npu_done();

    // ===========================================================
    // 校验 Conv2 + Relu
    // ===========================================================
    my_printf("\n>>> Checking NPU Conv2 + Relu Output (First 8 Words / 32 Channels) <<<\n");
    ptr_conv1_32b = (uint32_t*)buf_conv2;
    ptr_exp1_32b  = (uint32_t*)expected_act2_output;
    
    for (int i = 0; i < 8; i++) {
        uint32_t val = ptr_conv1_32b[i];
        uint32_t exp = ptr_exp1_32b[i];
        
        int ch_base = i * 4;
        
        // 提取硬件计算输出的 4 个通道 (8-bit)
        int8_t c0 = (int8_t)(val & 0xFF);
        int8_t c1 = (int8_t)((val >> 8) & 0xFF);
        int8_t c2 = (int8_t)((val >> 16) & 0xFF);
        int8_t c3 = (int8_t)((val >> 24) & 0xFF);
        
        // 提取预期输出的 4 个通道 (8-bit)
        int8_t e0 = (int8_t)(exp & 0xFF);
        int8_t e1 = (int8_t)((exp >> 8) & 0xFF);
        int8_t e2 = (int8_t)((exp >> 16) & 0xFF);
        int8_t e3 = (int8_t)((exp >> 24) & 0xFF);
        
        if (val == exp) {
            my_printf("  Word[%d] CH[%d~%d]: MATCH\n", i, ch_base+3, ch_base);
        } else {
            my_printf("  [ERROR] Word[%d] CH[%d~%d] Mismatch:\n", i, ch_base+3, ch_base);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+3, c3, e3);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+2, c2, e2);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+1, c1, e1);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base,   c0, e0);
        }
    }

    // -----------------------------------------------------------
    // Layer 3: Software Pool1 (32x32x64 -> 16x16x64)
    // -----------------------------------------------------------
    my_printf("\n[3/7] CPU Executing MaxPool 1...\n");
    maxpool2d_int8_hwc(buf_conv2, buf_pool1, 64, 32, 2, 2);

    // ===========================================================
    // 校验 Pool1
    // ===========================================================
    my_printf("\n>>> Checking CPU Pool1 Output (First 8 Words / 32 Channels) <<<\n");
    uint32_t* ptr_pool1_32b = (uint32_t*)buf_pool1;
    uint32_t* ptr_expp1_32b = (uint32_t*)expected_pool1_output;
    
    for (int i = 0; i < 8; i++) {
        uint32_t val = ptr_pool1_32b[i];
        uint32_t exp = ptr_expp1_32b[i];
        int ch_base = i * 4;
        
        int8_t c0 = (int8_t)(val & 0xFF), c1 = (int8_t)((val >> 8) & 0xFF), c2 = (int8_t)((val >> 16) & 0xFF), c3 = (int8_t)((val >> 24) & 0xFF);
        int8_t e0 = (int8_t)(exp & 0xFF), e1 = (int8_t)((exp >> 8) & 0xFF), e2 = (int8_t)((exp >> 16) & 0xFF), e3 = (int8_t)((exp >> 24) & 0xFF);
        
        if (val == exp) {
            my_printf("  Word[%d] CH[%d~%d]: MATCH\n", i, ch_base+3, ch_base);
        } else {
            my_printf("  [ERROR] Word[%d] CH[%d~%d] Mismatch:\n", i, ch_base+3, ch_base);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+3, c3, e3);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+2, c2, e2);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+1, c1, e1);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base,   c0, e0);
        }
    }

    // -----------------------------------------------------------
    // Layer 4: Hardware Conv3 (16x16x64 -> 16x16x64)
    // -----------------------------------------------------------
    my_printf("\n[4/7] Executing NPU Layer 3 (Conv3: 64 OC, 64 IC)...\n");

    NPU_REG_CFG_IMG_DIM = (16 << 16) | 16;       
    NPU_REG_CFG_QUANT   = (L3_SHIFT << 16) | L3_MULT; 
    // Out_Stride=64, IC_G_R=0, W_Num=144, LB_W=18, IC_G=15
    NPU_REG_CFG_DPATH   = (64 << 22) | (0 << 18) | (9 << 10) | (18 << 4) | 15; 
    // Row_words = 16*(15+1) = 256; Weight_words = 144 * 64 = 9216
    NPU_REG_CFG_CH      = (256 << 16) | 9216;
    
    NPU_REG_ACT_BASE    = (uint32_t)buf_pool1;
    NPU_REG_BIAS_BASE   = (uint32_t)test_conv3_bias;
    NPU_REG_WEIGHT_BASE = (uint32_t)test_conv3_weight;
    NPU_REG_OUT_BASE    = (uint32_t)buf_conv3;

    NPU_REG_CTRL_STATUS = 0x00000001;
    wait_npu_done();

    // ===========================================================
    // 校验 Conv3
    // ===========================================================
    my_printf("\n>>> Checking NPU Conv3 Output (First 8 Words / 32 Channels) <<<\n");
    uint32_t* ptr_conv3_32b = (uint32_t*)buf_conv3;
    uint32_t* ptr_exp3_32b  = (uint32_t*)expected_act3_output;
    
    for (int i = 0; i < 8; i++) {
        uint32_t val = ptr_conv3_32b[i];
        uint32_t exp = ptr_exp3_32b[i];
        int ch_base = i * 4;
        
        int8_t c0 = (int8_t)(val & 0xFF), c1 = (int8_t)((val >> 8) & 0xFF), c2 = (int8_t)((val >> 16) & 0xFF), c3 = (int8_t)((val >> 24) & 0xFF);
        int8_t e0 = (int8_t)(exp & 0xFF), e1 = (int8_t)((exp >> 8) & 0xFF), e2 = (int8_t)((exp >> 16) & 0xFF), e3 = (int8_t)((exp >> 24) & 0xFF);
        
        if (val == exp) {
            my_printf("  Word[%d] CH[%d~%d]: MATCH\n", i, ch_base+3, ch_base);
        } else {
            my_printf("  [ERROR] Word[%d] CH[%d~%d] Mismatch:\n", i, ch_base+3, ch_base);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+3, c3, e3);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+2, c2, e2);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+1, c1, e1);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base,   c0, e0);
        }
    }

    // -----------------------------------------------------------
    // Layer 5: Hardware Conv4 (16x16x64 -> 16x16x64)
    // -----------------------------------------------------------
    my_printf("\n[5/7] Executing NPU Layer 4 (Conv4: 64 OC, 64 IC)...\n");

    NPU_REG_CFG_IMG_DIM = (16 << 16) | 16;       
    NPU_REG_CFG_QUANT   = (L4_SHIFT << 16) | L4_MULT; 
    // Out_Stride=64, IC_G_R=0, W_Num=144, LB_W=18, IC_G=15
    NPU_REG_CFG_DPATH   = (64 << 22) | (0 << 18) | (9 << 10) | (18 << 4) | 15; 
    // Row_words = 16*(15+1) = 256; Weight_words = 144 * 64 = 9216
    NPU_REG_CFG_CH      = (256 << 16) | 9216;
    
    NPU_REG_ACT_BASE    = (uint32_t)buf_conv3;
    NPU_REG_BIAS_BASE   = (uint32_t)test_conv4_bias;
    NPU_REG_WEIGHT_BASE = (uint32_t)test_conv4_weight;
    NPU_REG_OUT_BASE    = (uint32_t)buf_conv4;

    NPU_REG_CTRL_STATUS = 0x00000001;
    wait_npu_done();

    // ===========================================================
    // 校验 Conv4
    // ===========================================================
    my_printf("\n>>> Checking NPU Conv4 Output (First 8 Words / 32 Channels) <<<\n");
    uint32_t* ptr_conv4_32b = (uint32_t*)buf_conv4;
    uint32_t* ptr_exp4_32b  = (uint32_t*)expected_act4_output;
    
    for (int i = 0; i < 8; i++) {
        uint32_t val = ptr_conv4_32b[i];
        uint32_t exp = ptr_exp4_32b[i];
        int ch_base = i * 4;
        
        int8_t c0 = (int8_t)(val & 0xFF), c1 = (int8_t)((val >> 8) & 0xFF), c2 = (int8_t)((val >> 16) & 0xFF), c3 = (int8_t)((val >> 24) & 0xFF);
        int8_t e0 = (int8_t)(exp & 0xFF), e1 = (int8_t)((exp >> 8) & 0xFF), e2 = (int8_t)((exp >> 16) & 0xFF), e3 = (int8_t)((exp >> 24) & 0xFF);
        
        if (val == exp) {
            my_printf("  Word[%d] CH[%d~%d]: MATCH\n", i, ch_base+3, ch_base);
        } else {
            my_printf("  [ERROR] Word[%d] CH[%d~%d] Mismatch:\n", i, ch_base+3, ch_base);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+3, c3, e3);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+2, c2, e2);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base+1, c1, e1);
            my_printf("    -> CH[%d] HW: %d | EXP: %d\n", ch_base,   c0, e0);
        }
    }

    // -----------------------------------------------------------
    // Layer 6: Software Pool2 (16x16x64 -> 8x8x64)
    // -----------------------------------------------------------
    my_printf("\n[6/7] CPU Executing MaxPool 2...\n");
    maxpool2d_int8_hwc(buf_conv4, buf_pool2, 64, 16, 2, 2);

    // -----------------------------------------------------------
    // Layer 7: Software Fully Connected (8x8x64 -> 10)
    // -----------------------------------------------------------
    my_printf("\n[7/7] CPU Executing Fully Connected Layer...\n");
    
    linear_int8_hwc(
        buf_pool2, test_fc_weight, test_fc_bias, buf_fc,
        64, 8, 8, 10, FC_MULT, FC_SHIFT
    );

    // -----------------------------------------------------------
    // 终极对账 (10 分类结果)
    // -----------------------------------------------------------
    my_printf("\n>>> Validating Final FC Output <<<\n");
    
    int error_count = 0;
    int max_idx_c = 0, max_idx_py = 0;
    int8_t max_val_c = -128, max_val_py = -128;

    my_printf("Index | Hardware SoC | Python Golden\n");
    my_printf("--------------------------------------\n");
    for (int i = 0; i < 10; i++) {
        my_printf("  %d   |    %d      |    %d\n", i, buf_fc[i], expected_fc_output[i]);
        
        if (buf_fc[i] > max_val_c) { max_val_c = buf_fc[i]; max_idx_c = i; }
        if (expected_fc_output[i] > max_val_py) { max_val_py = expected_fc_output[i]; max_idx_py = i; }
        if (buf_fc[i] != expected_fc_output[i]) error_count++;
    }

    my_printf("\n====== SUMMARY ======\n");
    my_printf("Pred Hardware : Class %d\n", max_idx_c);
    my_printf("Pred Python   : Class %d\n", max_idx_py);

    if (error_count == 0) {
        my_printf("[SUCCESS] PERFECT MATCH! 0 errors.\n");
        my_printf("THE 64x64 NPU BEAST IS OFFICIALLY ALIVE!\n");
    } else {
        my_printf("[WARNING] Found bit-level mismatches.\n");
    }

    return 0;
}