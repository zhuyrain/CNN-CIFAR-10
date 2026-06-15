// 这里包含你用 Python 重新打包好的 HWC 格式的头文件！
#include "../python/test_vector_2.h"
#include "conv2d_int8_naive.h"
#include "my_printf.h"

#include <stdint.h>
// ==========================================
// 1. NPU AXI-Lite 寄存器映射宏定义
// NPU 挂在互联矩阵的 M1 端口，基地址为 0x4000_0000
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
// 2. 内存分配 (由 GCC 链接器自动分配物理地址)
// ==========================================
// 保证 4 字节对齐，这对于 AXI 总线极其重要
int8_t buf_conv1[16 * 32 * 32] __attribute__((aligned(4))) = {0}; 
int8_t buf_pool1[16 * 16 * 16] __attribute__((aligned(4))) = {0}; 
int8_t buf_conv2[32 * 16 * 16] __attribute__((aligned(4))) = {0}; 
int8_t buf_pool2[32 *  8 *  8] __attribute__((aligned(4))) = {0}; 
int8_t buf_fc[10]              __attribute__((aligned(4))) = {0};

// ==========================================
// 3. 辅助函数：启动 NPU 并轮询等待
// ==========================================
void wait_npu_done() {
    // 检查 Bit 2 (DONE 标志)
    while ((NPU_REG_CTRL_STATUS & 0x00000004) == 0) {
        // 死等 (Polling)，实际工程里可以插入 wfi 指令休眠
    }
    // 写 1 清除 DONE 标志
    NPU_REG_CTRL_STATUS = 0x00000004; 
}

// ==========================================
// 4. 主函数 (全网端到端推理)
// ==========================================
int main() {
    my_printf("Input Base: %d\n", test_input_img_packed32);
    my_printf("Weight Base: %d\n", test_conv1_weight_packed32);
    my_printf("Conv1 Out Base: %d\n", buf_conv1);
    my_printf("===== RISC-V & NPU End-to-End CNN Inference =====\n");

    // -----------------------------------------------------------
    // Layer 1: Hardware Conv1 (32x32x3 -> 32x32x16)
    // -----------------------------------------------------------
    my_printf("[1/5] Executing NPU Layer 1 (Conv1: 16 OC)...\n");

    NPU_REG_CFG_IMG_DIM = (32 << 16) | 32;       // H=32, W=32
    NPU_REG_CFG_QUANT   = (L1_SHIFT << 16) | L1_MULT; 
    // Out_Stride=16, W_Num=9, LB_W=34, IC_G=1
    NPU_REG_CFG_DPATH   = (16 << 16) | (9 << 10) | (34 << 4) | 1;
    
    // 输入图片地址是固定的
    NPU_REG_ACT_BASE    = (uint32_t)test_input_img_packed32;

    // Conv1: 16 输出通道，分 4 次 Pass (4 * 4 = 16)
    for (int pass = 0; pass < 4; pass++) {
        NPU_REG_BIAS_BASE   = (uint32_t)test_conv1_bias + (pass * 16);
        NPU_REG_WEIGHT_BASE = (uint32_t)test_conv1_weight_packed32 + (pass * 144);
        NPU_REG_OUT_BASE    = (uint32_t)buf_conv1 + (pass * 4);

        NPU_REG_CTRL_STATUS = 0x00000001; // START!
        wait_npu_done();
    }
    my_printf("      [OK] NPU Conv1 Complete!\n");
    // -----------------------------------------------------------
    // 打印池化前 (Conv1 输出) 的前 8 个 32-bit Word
    // -----------------------------------------------------------
    my_printf(">>> Checking NPU Conv1 Output (First 8 Words / 2 Pixels) <<<\n");
    uint32_t* ptr_conv1_32b = (uint32_t*)buf_conv1;
    
    // 前 8 个 Word 刚好包含：Pixel(0,0) 的 16通道 和 Pixel(0,1) 的 16通道！
    for (int i = 0; i < 8; i++) {
        uint32_t val = ptr_conv1_32b[i];
        
        // 拆解出 4 个 8-bit 通道 (验证小端序 OC3, OC2, OC1, OC0)
        int8_t c0 = (int8_t)(val & 0xFF);
        int8_t c1 = (int8_t)((val >> 8) & 0xFF);
        int8_t c2 = (int8_t)((val >> 16) & 0xFF);
        int8_t c3 = (int8_t)((val >> 24) & 0xFF);
        
        int pixel_idx = i / 4;  // 每 4 个 Word 是一个像素
        int ch_base = (i % 4) * 4;
        
        // 由于你的 my_printf 可能不支持 %08x，用 %d 打印分解的值最稳妥
        my_printf("  Pixel(0,%d) CH[%d~%d]: OC%d=%d, OC%d=%d, OC%d=%d, OC%d=%d\n", 
                  pixel_idx, ch_base+3, ch_base,
                  ch_base+3, c3, ch_base+2, c2, ch_base+1, c1, ch_base+0, c0);
    }

    // -----------------------------------------------------------
    // Layer 1: Software CPU Pool1 (32x32x16 -> 16x16x16)
    // -----------------------------------------------------------
    my_printf("[2/5] CPU Executing MaxPool 1...\n");
    maxpool2d_int8_hwc(buf_conv1, buf_pool1, 16, 32, 2, 2);
    my_printf("      [OK] CPU Pool1 Complete!\n");
    // -----------------------------------------------------------
    // 打印池化后 (Pool1 输出) 的前 8 个 32-bit Word
    // -----------------------------------------------------------
    my_printf(">>> Checking CPU Pool1 Output (First 8 Words / 2 Pils) <<<\n");
    uint32_t* ptr_pool1_32b = (uint32_t*)buf_pool1;
    
    // Pool 完之后，长宽变成 16x16，但通道交织格式和深度 (16) 完全没变！
    for (int i = 0; i < 8; i++) {
        uint32_t val = ptr_pool1_32b[i];
        
        // 拆解出 4 个 8-bit 通道 (验证小端序 OC3, OC2, OC1, OC0)
        int8_t c0 = (int8_t)(val & 0xFF);
        int8_t c1 = (int8_t)((val >> 8) & 0xFF);
        int8_t c2 = (int8_t)((val >> 16) & 0xFF);
        int8_t c3 = (int8_t)((val >> 24) & 0xFF);
        
        int pixel_idx = i / 4;  // 每 4 个 Word 是一个像素
        int ch_base = (i % 4) * 4;
        
        // 由于你的 my_printf 可能不支持 %08x，用 %d 打印分解的值最稳妥
        my_printf("  Pixel(0,%d) CH[%d~%d]: OC%d=%d, OC%d=%d, OC%d=%d, OC%d=%d\n", 
                  pixel_idx, ch_base+3, ch_base,
                  ch_base+3, c3, ch_base+2, c2, ch_base+1, c1, ch_base+0, c0);
    }
    // -----------------------------------------------------------
    // Layer 2: Hardware Conv2 (16x16x16 -> 16x16x32)
    // -----------------------------------------------------------
    my_printf("[3/5] Executing NPU Layer 2 (Conv2: 32 OC, 16 IC)...\n");

    NPU_REG_CFG_IMG_DIM = (16 << 16) | 16;       // H=16, W=16
    NPU_REG_CFG_QUANT   = (L2_SHIFT << 16) | L2_MULT; 
    // Out_Stride=32, W_Num=36, LB_W=18, IC_G=4
    NPU_REG_CFG_DPATH   = (32 << 16) | (36 << 10) | (18 << 4) | 4; 
    NPU_REG_ACT_BASE    = (uint32_t)buf_pool1;   // ⬅️ 数据源变成了 Pool1 的输出！

    // Conv2: 32 输出通道，分 8 次 Pass (8 * 4 = 32)
    for (int pass = 0; pass < 8; pass++) {
        NPU_REG_BIAS_BASE   = (uint32_t)test_conv2_bias + (pass * 16);
        // Conv2 每个 pass 需要装载 36 个权重字 (36 * 4 * 4 = 576 bytes)
        NPU_REG_WEIGHT_BASE = (uint32_t)test_conv2_weight_packed32 + (pass * 576);
        NPU_REG_OUT_BASE    = (uint32_t)buf_conv2 + (pass * 4);

        NPU_REG_CTRL_STATUS = 0x00000001; // START!
        wait_npu_done();
    }
    my_printf("      [OK] NPU Conv2 Complete!\n");
    // -----------------------------------------------------------
    // 打印池化前 (Conv2 输出) 的前 8 个 32-bit Word
    // -----------------------------------------------------------
    my_printf(">>> Checking NPU Conv2 Output (First 8 Words / 1 Pixels) <<<\n");
    uint32_t* ptr_conv2_32b = (uint32_t*)buf_conv2;
    
    // 前 8 个 Word 刚好包含：Pixel(0,0) 的 16通道 和 Pixel(0,1) 的 16通道！
    for (int i = 0; i < 8; i++) {
        uint32_t val = ptr_conv2_32b[i];
        
        // 拆解出 4 个 8-bit 通道 (验证小端序 OC3, OC2, OC1, OC0)
        int8_t c0 = (int8_t)(val & 0xFF);
        int8_t c1 = (int8_t)((val >> 8) & 0xFF);
        int8_t c2 = (int8_t)((val >> 16) & 0xFF);
        int8_t c3 = (int8_t)((val >> 24) & 0xFF);
        
        int pixel_idx = i / 4;  // 每 4 个 Word 是一个像素
        int ch_base = (i % 4) * 4;
        
        // 由于你的 my_printf 可能不支持 %08x，用 %d 打印分解的值最稳妥
        my_printf("  Pixel(0,%d) CH[%d~%d]: OC%d=%d, OC%d=%d, OC%d=%d, OC%d=%d\n", 
                  pixel_idx, ch_base+3, ch_base,
                  ch_base+3, c3, ch_base+2, c2, ch_base+1, c1, ch_base+0, c0);
    }

    // -----------------------------------------------------------
    // Layer 2: Software CPU Pool2 (16x16x32 -> 8x8x32)
    // -----------------------------------------------------------
    my_printf("[4/5] CPU Executing MaxPool 2...\n");
    maxpool2d_int8_hwc(buf_conv2, buf_pool2, 32, 16, 2, 2);

    // -----------------------------------------------------------
    // Layer 3: Software Fully Connected (Linear: HWC 2048 -> 10)
    // -----------------------------------------------------------
    my_printf("\n[5/5] CPU Executing Fully Connected Layer...\n");
    
    // 输入维度: Channels=32, Height=8, Width=8 (Pool2 之后的尺寸)
    linear_int8_hwc(
        buf_pool2, test_fc_weight, test_fc_bias, buf_fc,
        32, 8, 8, 10, FC_MULT, FC_SHIFT
    );
    // -----------------------------------------------------------
    // 终极对账 (10 分类结果)
    // -----------------------------------------------------------
    my_printf("\n>>> Validating Final FC Output...\n");
    
    int error_count = 0;
    int max_idx_c = 0;
    int max_idx_py = 0;
    int8_t max_val_c = -128;
    int8_t max_val_py = -128;

    my_printf("Index | Hardware SoC | Python Golden\n");
    my_printf("--------------------------------------\n");
    for (int i = 0; i < 10; i++) {
        my_printf("  %d   |    %d      |    %d\n", i, buf_fc[i], expected_fc_output[i]);
        
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
    my_printf("Pred Hardware : Class %d\n", max_idx_c);
    my_printf("Pred Python   : Class %d\n", max_idx_py);

    if (error_count == 0) {
        my_printf("[SUCCESS] PERFECT MATCH! 0 errors.\n");
        my_printf("THE NPU IS OFFICIALLY ALIVE!\n");
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