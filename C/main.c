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
// 4. 主函数
// ==========================================
int main() {
    my_printf("===== RISC-V SoC & NPU Hardware Execution =====\n");

    // -----------------------------------------------------------
    // Layer 1: Hardware Conv1 (分 8 次 Pass 算出 32 个输出通道)
    // -----------------------------------------------------------
    my_printf("[1/3] Executing NPU Layer 1 (Conv1)...\n");

    // 全局静态配置
    NPU_REG_CFG_IMG_DIM = (16 << 16) | 16;       // H=32, W=32
    NPU_REG_CFG_QUANT   = (L2_SHIFT << 16) | L2_MULT;      // L1_SHIFT=16, L1_MULT=104
    NPU_REG_CFG_DPATH   = (32 << 16) | (36 << 10) | (18 << 4) | 4; // Out_Stride=16, W_Num=9, LB_W=34, IC_G=1
    
    // 输入图片地址是固定的
    NPU_REG_ACT_BASE    = (uint32_t)test_input_img_packed32;

    // 分 8 次循环，每次算 4 个输出通道
    for (int pass = 0; pass < 8; pass++) {
        my_printf("      -> Running Pass %d (OC %d ~ %d)...\n", pass, pass*4, pass*4+3);
        
        // --- 指针魔法：计算本次 Pass 的地址偏移 ---
        // 偏置偏移：每次 4 个 Bias，即 16 字节
        NPU_REG_BIAS_BASE   = (uint32_t)test_conv1_bias + (pass * 16);
        
        // 权重偏移：每次 36个Cycle * 4个Word(16字节) = 576 字节
        NPU_REG_WEIGHT_BASE = (uint32_t)test_conv1_weight_packed32 + (pass * 576);
        
        // 输出偏移：由于我们设置了 Out_Stride=16，
        // 第一轮写在 buf_conv1[0,16,32...]，第二轮必须错开4个字节写在 buf_conv1[4,20,36...]
        NPU_REG_OUT_BASE    = (uint32_t)buf_conv1 + (pass * 4);

        // --- 扣动扳机 ---
        NPU_REG_CTRL_STATUS = 0x00000001; // START!
        
        // --- 阻塞等待 ---
        wait_npu_done();
    }
    my_printf("      [OK] NPU Conv1 Complete!\n");
    // -----------------------------------------------------------
    // 打印池化前 (Conv1 输出) 的前 8 个 32-bit Word
    // -----------------------------------------------------------
    my_printf("\n>>> Checking NPU Conv1 Output (First 8 Words / 2 Pixels) <<<\n");
    uint32_t* ptr_conv_32b = (uint32_t*)buf_conv1;
    
    // 前 8 个 Word 刚好包含：Pixel(0,0) 的 16通道 和 Pixel(0,1) 的 16通道！
    for (int i = 0; i < 8; i++) {
        uint32_t val = ptr_conv_32b[i];
        
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
    // Layer 1: Software CPU Pool1 (调用全新的 HWC 版)
    // -----------------------------------------------------------
    my_printf("\n[2/3] CPU Executing MaxPool 1...\n");
    // 16 通道，输入 32x32，Pool核 2x2，步长 2
    maxpool2d_int8_hwc(buf_conv1, buf_pool1, 16, 32, 2, 2);
    my_printf("      [OK] CPU Pool1 Complete!\n");

    // -----------------------------------------------------------
    // 打印池化后 (Pool1 输出) 的前 8 个 32-bit Word
    // -----------------------------------------------------------
    my_printf("\n>>> Checking CPU Pool1 Output (First 8 Words / 2 Pixels) <<<\n");
    uint32_t* ptr_pool_32b = (uint32_t*)buf_pool1;
    
    // Pool 完之后，长宽变成 16x16，但通道交织格式和深度 (16) 完全没变！
    for (int i = 0; i < 8; i++) {
        uint32_t val = ptr_pool_32b[i];
        
        int8_t c0 = (int8_t)(val & 0xFF);
        int8_t c1 = (int8_t)((val >> 8) & 0xFF);
        int8_t c2 = (int8_t)((val >> 16) & 0xFF);
        int8_t c3 = (int8_t)((val >> 24) & 0xFF);
        
        int pixel_idx = i / 4;
        int ch_base = (i % 4) * 4;
        
        my_printf("  Pixel(0,%d) CH[%d~%d]: OC%d=%d, OC%d=%d, OC%d=%d, OC%d=%d\n", 
                  pixel_idx, ch_base+3, ch_base,
                  ch_base+3, c3, ch_base+2, c2, ch_base+1, c1, ch_base+0, c0);
    }

    my_printf("===== ALL DONE =====\n");
    return 0;
}