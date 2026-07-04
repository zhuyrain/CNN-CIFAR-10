// 这里包含你用 Python 重新打包好的 HWC 格式的头文件！
#include "../python/test_vector_2.h"
#include "conv2d_int8_naive.h"
#include "my_printf.h"

#include <stdint.h>
// ==========================================
// 1. NPU AXI-Lite 寄存器映射宏定义 (已对齐全新重构版本)
// ==========================================
#define NPU_BASE            0x40000000
#define NPU_REG_CTRL_STATUS (*(volatile uint32_t*)(NPU_BASE + 0x00))
#define NPU_REG_ACT_BASE    (*(volatile uint32_t*)(NPU_BASE + 0x04))
#define NPU_REG_WEIGHT_BASE (*(volatile uint32_t*)(NPU_BASE + 0x08))
#define NPU_REG_BIAS_BASE   (*(volatile uint32_t*)(NPU_BASE + 0x0C))
#define NPU_REG_OUT_BASE    (*(volatile uint32_t*)(NPU_BASE + 0x10))

// 全新重构的 6 个配置寄存器
#define NPU_REG_CFG_IMG_DIM  (*(volatile uint32_t*)(NPU_BASE + 0x14))
#define NPU_REG_CFG_MEM_CNT  (*(volatile uint32_t*)(NPU_BASE + 0x18))
#define NPU_REG_CFG_DATAPATH (*(volatile uint32_t*)(NPU_BASE + 0x1C))
#define NPU_REG_CFG_SPATIAL  (*(volatile uint32_t*)(NPU_BASE + 0x20))
#define NPU_REG_CFG_QUANT    (*(volatile uint32_t*)(NPU_BASE + 0x24))
#define NPU_REG_CFG_POST     (*(volatile uint32_t*)(NPU_BASE + 0x28))
// 声明汇编中的自定义函数
extern void enable_npu_irq(uint32_t mask);
extern void wait_for_irq(void);

// 必须使用 volatile，防止编译器过度优化导致死循环
volatile int npu_task_finished = 0;

// ==========================================
// [新增] 中断服务程序 (由 start.S 中的 trap_entry 汇编调用)
// ==========================================
void npu_isr(void) {
    // 1. 软件标志位置 1，通知主程序的 WFI 循环可以退出了
    npu_task_finished = 1;

    // 2. 硬件标志位清零 (向 Bit 2 写 1 清除 DONE 标志)
    // 极其关键：必须在这里清硬件标志，否则 irq 引脚一直拉高会导致无限触发中断！
    NPU_REG_CTRL_STATUS = 0x00000004; 
}


// ==========================================
// 2. 内存分配 (适配 64x64 阵列，强行 64 字节对齐保平安)
// ==========================================
int8_t buf_conv1[64 * 32 * 32] __attribute__((aligned(4))) = {0}; 
int8_t buf_conv2[64 * 32 * 32] __attribute__((aligned(4))) = {0}; 
int8_t buf_pool1[64 * 16 * 16] __attribute__((aligned(4))) = {0}; 
int8_t buf_conv3[64 * 16 * 16] __attribute__((aligned(4))) = {0}; 
int8_t buf_conv4[64 * 16 * 16] __attribute__((aligned(4))) = {0}; 
int8_t buf_pool2[64 *  8 *  8] __attribute__((aligned(4))) = {0}; 
int8_t buf_fc[12]              __attribute__((aligned(4))) = {0};
// ==========================================
// 3. 辅助函数：休眠等待 NPU 完成 (替换死等轮询)
// ==========================================
void wait_npu_done() {
    // 当 NPU 还没算完时，执行休眠
    while (!npu_task_finished) {
        wait_for_irq(); // 完美的自定义 WFI
    }
    
    // CPU 被唤醒并跳出循环后，重置标志位，为下一层的复用做准备
    npu_task_finished = 0;
    
    // 表示npu进行完一轮运算
    my_printf("-");
}

int main() {
    // 【系统级初始化：开启硬件中断】
    // PicoRV32 没有全局中断开关，Mask 寄存器就是一切！
    // 我们传入 ~(1 << 4)：即除了 bit 4 是 0 以外，其余 31 个 bit 都是 1。
    // 这意味着：放行 NPU 的 IRQ 4，屏蔽其他所有中断。
    enable_npu_irq(~(1 << 4));
    
    my_printf("Input Base: %d\n", test_input_img);
    my_printf("Weight Base: %d\n", test_conv1_weight);
    my_printf("Conv1 Out Base: %d\n", buf_conv1);
    my_printf("===== RISC-V & NPU 64x64 FULL POWER Inference =====\n");

    // 设置全局的流水线安全控制参数
    uint32_t drain_timeout  = 132;   // 强制排空等待拍数 (64行+12列+4保险 = 80)
    uint32_t fifo_af_thresh = 40; // FIFO 将满阈值 (留出 24 拍飞行裕量)

    // -----------------------------------------------------------
    // Layer 1: Hardware Conv1 (32x32x3 -> 32x32x64)
    // -----------------------------------------------------------
    my_printf("[1/7] Executing NPU Layer 1 (Conv1: 64 OC)...\n");

    // [31:16] H=32, [15:0] W=32
    NPU_REG_CFG_IMG_DIM  = (32 << 16) | 32;      
    // [31:16] row_word_count=32, [15:0] weight_word_count=576
    NPU_REG_CFG_MEM_CNT  = (32 << 16) | 576;
    // [31:23] out_stride=64, [22:19] lb_ic_groups_r=0, [18:11] sa_weight_num=9, [10:4] lb_line_width=34, [3:0] lb_ic_groups=0
    NPU_REG_CFG_DATAPATH = (64 << 23) | (0 << 19) | (9 << 11) | (34 << 4) | 0;
    // [31:24] oc_num=64, [23:8] col_group_en=0xFFFF, [5] ky_skew=0, [4] pad_size=1, [3:2] kx_max=2, [1:0] ky_max=2 (标准3x3参数)
    NPU_REG_CFG_SPATIAL  = (64 << 24) | (0xFFFF << 8) | (0 << 5) | (1 << 4) | (2 << 2) | 2;
    // [31:16] out_zp=0, [15:0] multiplier=L1_MULT
    NPU_REG_CFG_QUANT    = (0 << 16) | L1_MULT;
    // [31:22] drain_timeout, [21:12] fifo_af_thresh, [5:1] shift=L1_SHIFT, [0] relu_en=1
    NPU_REG_CFG_POST     = (drain_timeout << 22) | (fifo_af_thresh << 12) | (L1_SHIFT << 1) | 1;
    
    NPU_REG_ACT_BASE     = (uint32_t)test_input_img;

    // 启动 NPU 计算
    for (int pass = 0; pass < 1; pass++) {
        NPU_REG_BIAS_BASE   = (uint32_t)test_conv1_bias + (pass * 4 * 32);
        NPU_REG_WEIGHT_BASE = (uint32_t)test_conv1_weight + (pass * 1152);
        NPU_REG_OUT_BASE    = (uint32_t)buf_conv1 + (pass * 32);

        NPU_REG_CTRL_STATUS = 0x00000001;
        // CPU 进入休眠，交出控制权，等待硬件中断唤醒！
        wait_npu_done();
    }
    
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

    // ===========================================================
    // Layer 2: Hardware Conv2 (32x32x64 -> 32x32x64)
    // ===========================================================
    my_printf("\n[2/7] Executing NPU Layer 2 (Conv2: 64 OC, 64 IC)...\n");

    NPU_REG_CFG_IMG_DIM  = (32 << 16) | 32;      
    // Row_words = 32*(15+1) = 512; Weight_words = 144 * 32(OC) = 4608 (假设分次计算的参数)
    NPU_REG_CFG_MEM_CNT  = (512 << 16) | 9216;
    // out_stride=64, lb_ic_groups_r=0, sa_weight_num=9, lb_line_width=34, lb_ic_groups=15
    NPU_REG_CFG_DATAPATH = (64 << 23) | (0 << 19) | (9 << 11) | (34 << 4) | 15;
    NPU_REG_CFG_SPATIAL  = (64 << 24) | (0xFFFF << 8) | (0 << 5) | (1 << 4) | (2 << 2) | 2;
    NPU_REG_CFG_QUANT    = (0 << 16) | L2_MULT;
    NPU_REG_CFG_POST     = (drain_timeout << 22) | (fifo_af_thresh << 12) | (L2_SHIFT << 1) | 1;
    
    NPU_REG_ACT_BASE     = (uint32_t)buf_conv1;
    // 启动 NPU 计算 (仅需 1 个 Pass)
    for (int pass = 0; pass < 1; pass++) {
        // 每次偏移 32 个 Bias (32 * 4 bytes = 128 bytes)
        NPU_REG_BIAS_BASE   = (uint32_t)test_conv2_bias + (pass * 128);
        // 每次偏移 4608 个 Word 权重 (4608 * 4 bytes = 18432 bytes)
        NPU_REG_WEIGHT_BASE = (uint32_t)test_conv2_weight + (pass * 18432);
        // 每次偏移 32 个 Output Channel (32 * 1 byte = 32 bytes)
        NPU_REG_OUT_BASE    = (uint32_t)buf_conv2 + (pass * 32);

        NPU_REG_CTRL_STATUS = 0x00000001;
        wait_npu_done();
    }

    // ===========================================================
    // 校验 Conv2 + Relu
    // ===========================================================
    my_printf("\n>>> Checking NPU Conv2 + Relu Output (First 8 Words / 32 Channels) <<<\n");
    uint32_t* ptr_conv2_32b = (uint32_t*)buf_conv2;
    uint32_t* ptr_exp2_32b  = (uint32_t*)expected_act2_output;
    
    for (int i = 0; i < 8; i++) {
        uint32_t val = ptr_conv2_32b[i];
        uint32_t exp = ptr_exp2_32b[i];
        
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

    // ===========================================================
    // Layer 4: Hardware Conv3 (16x16x64 -> 16x16x64)
    // ===========================================================
    my_printf("\n[4/7] Executing NPU Layer 3 (Conv3: 64 OC, 64 IC)...\n");

    NPU_REG_CFG_IMG_DIM  = (16 << 16) | 16;      
    // Row_words = 16*(15+1) = 256; Weight_words = 144 * 32 = 4608
    NPU_REG_CFG_MEM_CNT  = (256 << 16) | 9216;
    // out_stride=64, lb_ic_groups_r=0, sa_weight_num=9, lb_line_width=18, lb_ic_groups=15
    NPU_REG_CFG_DATAPATH = (64 << 23) | (0 << 19) | (9 << 11) | (18 << 4) | 15; 
    NPU_REG_CFG_SPATIAL  = (64 << 24) | (0xFFFF << 8) | (0 << 5) | (1 << 4) | (2 << 2) | 2;
    NPU_REG_CFG_QUANT    = (0 << 16) | L3_MULT;
    NPU_REG_CFG_POST     = (drain_timeout << 22) | (fifo_af_thresh << 12) | (L3_SHIFT << 1) | 1;
    
    NPU_REG_ACT_BASE     = (uint32_t)buf_pool1;

    for (int pass = 0; pass < 1; pass++) {
        NPU_REG_BIAS_BASE   = (uint32_t)test_conv3_bias + (pass * 128);
        NPU_REG_WEIGHT_BASE = (uint32_t)test_conv3_weight + (pass * 18432);
        NPU_REG_OUT_BASE    = (uint32_t)buf_conv3 + (pass * 32);

        NPU_REG_CTRL_STATUS = 0x00000001;
        wait_npu_done();
    }

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

    // ===========================================================
    // Layer 5: Hardware Conv4 (16x16x64 -> 16x16x64)
    // ===========================================================
    my_printf("\n[5/7] Executing NPU Layer 4 (Conv4: 64 OC, 64 IC)...\n");

    NPU_REG_CFG_IMG_DIM  = (16 << 16) | 16;      
    NPU_REG_CFG_MEM_CNT  = (256 << 16) | 9216;
    NPU_REG_CFG_DATAPATH = (64 << 23) | (0 << 19) | (9 << 11) | (18 << 4) | 15; 
    NPU_REG_CFG_SPATIAL  = (64 << 24) | (0xFFFF << 8) | (0 << 5) | (1 << 4) | (2 << 2) | 2;
    NPU_REG_CFG_QUANT    = (0 << 16) | L4_MULT;
    NPU_REG_CFG_POST     = (drain_timeout << 22) | (fifo_af_thresh << 12) | (L4_SHIFT << 1) | 1;
    
    NPU_REG_ACT_BASE     = (uint32_t)buf_conv3;

    for (int pass = 0; pass < 1; pass++) {
        NPU_REG_BIAS_BASE   = (uint32_t)test_conv4_bias + (pass * 128);
        NPU_REG_WEIGHT_BASE = (uint32_t)test_conv4_weight + (pass * 18432);
        NPU_REG_OUT_BASE    = (uint32_t)buf_conv4 + (pass * 32);

        NPU_REG_CTRL_STATUS = 0x00000001;
        wait_npu_done();
    }

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
    // Layer 1: Hardware Conv1 (32x32x3 -> 32x32x64) 转换为 FC 配置
    // -----------------------------------------------------------
    my_printf("[1/7] Executing FC Layer 1 (Conv1: 64 OC)...\n");

    // 1. 外部维度与搬运计数
    // [31:16] H=1, [15:0] W=64
    NPU_REG_CFG_IMG_DIM  = (1 << 16) | 64;      
    // [31:16] row_word_count=1024, [15:0] weight_word_count=12288
    NPU_REG_CFG_MEM_CNT  = (1024 << 16) | 12288;

    // 2. 数据通路与片上缓存参数
    // [31:23] out_stride=12, [22:19] lb_ic_groups_r=0, [18:11] sa_weight_num=64, [10:4] lb_line_width=64, [3:0] lb_ic_groups=15
    NPU_REG_CFG_DATAPATH = (12 << 23) | (0 << 19) | (64 << 11) | (64 << 4) | 15;

    // 3. 阵列空间映射与滑窗边界
    // [31:24] oc_num=12, [23:8] col_group_en=0xFFFF, [5] ky_skew=1, [4] pad_size=0, [3:2] kx_max=0, [1:0] ky_max=0
    NPU_REG_CFG_SPATIAL  = (12 << 24) | (0x0007 << 8) | (1 << 5) | (0 << 4) | (0 << 2) | 0;

    // 4. 量化数学参数
    // [31:16] out_zp=0, [15:0] multiplier=FC_MULT
    NPU_REG_CFG_QUANT    = (0 << 16) | FC_MULT;

    // // 5. 后处理与系统级流水线安全控制 (新增亮点!)
    // // [31:22] drain_timeout: 强制排空等待拍数 (64行+64列+4保险 = 132)
    // // [21:12] fifo_af_thresh: FIFO 将满阈值 (假设你的 FIFO 深度是 1024，设为 1000 留出 24 拍飞行裕量)
    // // [5:1] shift=FC_SHIFT, [0] relu_en=0
    // uint32_t drain_timeout  = 132;   
    // uint32_t fifo_af_thresh = 40; 
    NPU_REG_CFG_POST = (drain_timeout << 22) | (fifo_af_thresh << 12) | (FC_SHIFT << 1) | 0;
    
    // -----------------------------------------------------------
    // 启动运算与基地址配置
    // -----------------------------------------------------------
    NPU_REG_ACT_BASE    = (uint32_t)buf_pool2;

    // ===========================================================
    // [Debug] CPU 探内存：按 8-bit 小端序逐字节打印前 16 个字节
    // ===========================================================
    my_printf("\n>>> CPU Memory Dump at expected_pool2_output <<<\n");
    
    // 强转为 8-bit 无符号指针，完美规避小端序在 32-bit 下的视觉干扰
    uint8_t* dump_ptr = (uint8_t*)expected_pool2_output;
    
    // 打印前 16 个字节 (即前 4 个 32-bit Word)
    // 根据你的头文件，这 4 个 Word 是：0x00000000, 0x00000000, 0x00000103, 0x03000000
    for (int i = 0; i < 16; i++) {
        // 由于没有 16 进制打印，直接打印十进制无符号数 (0~255)
        my_printf("Byte Offset [%d] : %d\n", i, dump_ptr[i]);
    }
    my_printf("------------------------------------------------\n\n");

    // 启动 NPU 计算 (当前 FC 规模较小，单次 Pass 即可完成)
    for (int pass = 0; pass < 1; pass++) {
        // 修正为 FC 层对应的偏移量：Bias(12字节), Weight(49152字节), Out(12字节)
        NPU_REG_BIAS_BASE   = (uint32_t)test_fc_bias + (pass * 12);
        NPU_REG_WEIGHT_BASE = (uint32_t)test_fc_weight + (pass * 49152);
        NPU_REG_OUT_BASE    = (uint32_t)buf_fc + (pass * 12);

        NPU_REG_CTRL_STATUS = 0x00000001;
        // CPU 进入休眠，交出控制权，等待硬件中断唤醒！
        wait_npu_done();
    }
    
    // ===========================================================
    // 校验 FC 层 (10 个有效输出通道)
    // ===========================================================
    my_printf("\n>>> Checking NPU FC Output (10 Valid Channels) <<<\n");
    
    // 【关键修复】直接使用 int8_t 指针进行单字节比对，彻底杜绝内存越界
    int8_t* ptr_hw_out  = (int8_t*)buf_fc;
    int8_t* ptr_exp_out = (int8_t*)expected_fc_output;
    
    int error_cnt = 0;
    
    // 只遍历预期的 10 个通道，忽略 buf_fc[10] 和 buf_fc[11] 的 Padding 数据
    for (int ch = 0; ch < 10; ch++) {
        int8_t val = ptr_hw_out[ch];
        int8_t exp = ptr_exp_out[ch];
        
        if (val == exp) {
            my_printf("  CH[%d]: MATCH (%d)\n", ch, val);
        } else {
            my_printf("  [ERROR] CH[%d] Mismatch -> HW: %d | EXP: %d\n", ch, val, exp);
            error_cnt++;
        }
    }
    
    if (error_cnt == 0) {
        my_printf("  FC Output Verification PASSED!\n");
    } else {
        my_printf("  FC Output Verification FAILED with %d errors.\n", error_cnt);
    }

    return 0;
}