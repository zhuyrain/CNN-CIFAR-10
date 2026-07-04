#include <stdint.h>
#include "my_printf.h"
#include "npu.h"
// 引入你的 Python 测试向量
#include "../python/test_vector_2.h" 

// ==========================================
// 中断与系统控制逻辑
// ==========================================
extern void enable_npu_irq(uint32_t mask);
extern void wait_for_irq(void);

volatile int npu_task_finished = 0;

void npu_isr(void) {
    npu_task_finished = 1;
    NPU_REG_CTRL_STATUS = 0x00000004; // 清除中断
}

void wait_npu_done() {
    while (!npu_task_finished) {
        wait_for_irq(); 
    }
    npu_task_finished = 0;
}

// ==========================================
// 内存分配 (复用 12 字节输出即可)
// ==========================================
int8_t buf_fc[12] __attribute__((aligned(4))) = {0};

// 核心测试函数：传入 H 和 W 进行 FC 形态重塑验证
int run_fc_reshape_test(int H, int W) {
    // 1. 每次运算前清空输出缓冲区，防止结果残留
    for(int i = 0; i < 12; i++) {
        buf_fc[i] = 0;
    }

    // 2. 动态计算当前折叠形态下的配置参数
    // 在 FC 层映射中，总 IC 元素为 4096 (64个通道 * H * W)。
    // 这里我们相当于把 4096 个数据折叠成了 H * W 个虚拟像素，每个像素深度为 64。
    // row_word_count = (虚拟像素宽度 W * 64深度) / 4(AXI位宽) = W * 16
    uint32_t row_word = W * 16;
    
    // 权重数量是固定的 12288 words (49152 Bytes)
    uint32_t weight_word = 12288;

    // 3. 下发寄存器配置
    NPU_REG_CFG_IMG_DIM  = (H << 16) | W;      
    NPU_REG_CFG_MEM_CNT  = (row_word << 16) | weight_word;

    // out_stride=12, sa_weight_num=64, lb_line_width=W, lb_ic_groups=15 (64/4-1)
    NPU_REG_CFG_DATAPATH = (12 << 23) | (0 << 19) | (64 << 11) | (W << 4) | 15;

    // oc_num=12, col_group_en=0x0007 (激活 3 个 Group=12列), ky_skew=1
    // pad_size=0, kx_max=0, ky_max=0 (FC 层无空间滑动)
    NPU_REG_CFG_SPATIAL  = (12 << 24) | (0x0007 << 8) | (1 << 5) | (0 << 4) | (0 << 2) | 0;

    // 量化参数
    NPU_REG_CFG_QUANT    = (0 << 16) | FC_MULT;
    
    // 排空与后处理
    uint32_t drain_timeout  = 132;   
    uint32_t fifo_af_thresh = 40; 
    NPU_REG_CFG_POST = (drain_timeout << 22) | (fifo_af_thresh << 12) | (FC_SHIFT << 1) | 0;

    // 基址配置
    NPU_REG_ACT_BASE    = (uint32_t)expected_pool2_output; // 用池化层2的结果作为 FC 输入
    NPU_REG_WEIGHT_BASE = (uint32_t)test_fc_weight;
    NPU_REG_BIAS_BASE   = (uint32_t)test_fc_bias;
    NPU_REG_OUT_BASE    = (uint32_t)buf_fc;

    // 4. 发射任务
    NPU_REG_CTRL_STATUS = 0x00000001;
    wait_npu_done();

    // 5. 校验 FC 层的 10 个有效输出通道
    int error_cnt = 0;
    int8_t* ptr_hw_out  = (int8_t*)buf_fc;
    int8_t* ptr_exp_out = (int8_t*)expected_fc_output;

    for (int ch = 0; ch < 10; ch++) {
        int8_t val = ptr_hw_out[ch];
        int8_t exp = ptr_exp_out[ch];
        
        if (val != exp) {
            my_printf("    [ERROR] CH[%d] Mismatch -> HW: %d | EXP: %d\n", ch, val, exp);
            error_cnt++;
        }
    }

    if (error_cnt == 0) {
        my_printf("  [PASS] Reshape %dx%d (H x W) Verified!\n", H, W);
        return 1;
    } else {
        my_printf("  [FAIL] Reshape %dx%d Failed with %d errors.\n", H, W, error_cnt);
        return 0;
    }
}

int main() {
    // 放行 NPU IRQ
    enable_npu_irq(~(1 << 4));

    my_printf("\n========================================\n");
    my_printf(" [TEST 6] FC Layer Tensor Reshape \n");
    my_printf("          Folding: 1x64, 2x32, 4x16\n");
    my_printf("========================================\n");

    int pass = 1;

    // 测试形态 1：高度极度压缩 (1行 64列)
    my_printf("-> Testing Reshape Mode 1: H=1, W=64 ...\n");
    pass &= run_fc_reshape_test(1, 64);

    // 测试形态 2：折半展开 (2行 32列)
    my_printf("-> Testing Reshape Mode 2: H=2, W=32 ...\n");
    pass &= run_fc_reshape_test(2, 32);

    // 测试形态 3：黄金参考形态 (4行 16列)
    my_printf("-> Testing Reshape Mode 3: H=4, W=16 ...\n");
    pass &= run_fc_reshape_test(4, 16);

    my_printf("========================================\n");
    if (pass) {
        my_printf(" FC TENSOR RESHAPE VERIFICATION PASSED!\n");
    } else {
        my_printf(" FC TENSOR RESHAPE VERIFICATION FAILED!\n");
        while(1);
    }
    my_printf("========================================\n");

    while(1);
    return 0;
}