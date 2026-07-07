#include "../python/test_vector_2.h"
#include "my_printf.h"
#include <stdint.h>

// ==========================================
// 1. NPU 寄存器映射
// ==========================================
#include "npu.h"

// ==========================================
// 2. 中断与系统控制逻辑
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
// 3. 跨页缓冲区分配 (多分配 4KB 以确保内部包含边界)
// ==========================================
// 分配 5KB 空间，确保里面绝对能找到一个以 0x1000 对齐的地址
uint8_t raw_bias_buf[5120] = {0};
uint8_t raw_out_buf[5120] = {0};

int main() {
    // 开启 NPU 中断
    enable_npu_irq(~(1 << 4));

    my_printf("\n========================================\n");
    my_printf(" [TEST 3] AXI Burst Limits & 4KB Boundary\n");
    my_printf("========================================\n");

    // -----------------------------------------------------------
    // 核心魔法：动态计算 4KB 恶意边界地址
    // -----------------------------------------------------------
    // 1. 偏置 (Bias) 恶意地址
    uint32_t raw_bias_addr = (uint32_t)raw_bias_buf;
    uint32_t boundary_4k_bias = (raw_bias_addr + 0x0FFF) & ~(0x0FFF);
    // 退后 8 个字节 (2个Word)。由于 FC Bias 长 12 字节，这必将导致跨页！
    uint32_t evil_bias_addr = boundary_4k_bias - 8; 

    // 2. 输出 (Output) 恶意地址
    uint32_t raw_out_addr = (uint32_t)raw_out_buf;
    uint32_t boundary_4k_out = (raw_out_addr + 0x0FFF) & ~(0x0FFF);
    uint32_t evil_out_addr = boundary_4k_out - 8;

    my_printf(" -> Calculated 4KB Boundary for Bias: 0x%08x\n", boundary_4k_bias);
    my_printf(" -> Evil Bias Base (Crosses 4KB)  : 0x%08x\n", evil_bias_addr);
    my_printf(" -> Evil Out Base  (Crosses 4KB)  : 0x%08x\n", evil_out_addr);

    // -----------------------------------------------------------
    // CPU 提前搬运数据到跨页内存
    // -----------------------------------------------------------
    uint32_t* evil_bias_ptr = (uint32_t*)evil_bias_addr;
    uint32_t* orig_bias_ptr = (uint32_t*)test_fc_bias;
    
    // FC Bias 只有 12 字节 (包含 10 个有效数据和 2 个 Padding)
    for (int i = 0; i < 12; i++) {
        evil_bias_ptr[i] = orig_bias_ptr[i];
    }
    my_printf(" -> CPU Bias Data Pre-loading Done.\n");

    // -----------------------------------------------------------
    // NPU 寄存器配置 (利用 FC 层进行测试)
    // -----------------------------------------------------------
    my_printf("\n -> Executing NPU FC Layer with malicious addresses...\n");

    NPU_REG_CFG_IMG_DIM  = (4 << 16) | 16;      
    NPU_REG_CFG_MEM_CNT  = (256 << 16) | 12288;
    NPU_REG_CFG_DATAPATH = (12 << 23) | (0 << 19) | (64 << 11) | (16 << 4) | 15;
    NPU_REG_CFG_SPATIAL  = (12 << 24) | (0x0007 << 8) | (1 << 5) | (0 << 4) | (0 << 2) | 0;
    NPU_REG_CFG_QUANT    = (0 << 16) | FC_MULT;
    
    uint32_t drain_timeout  = 132;   
    uint32_t fifo_af_thresh = 40; 
    NPU_REG_CFG_POST = (drain_timeout << 22) | (fifo_af_thresh << 12) | (FC_SHIFT << 1) | 0;

    // 配置长数据基址 (天生会触发 256 beat 限制)
    NPU_REG_ACT_BASE    = (uint32_t)expected_pool2_output;
    NPU_REG_WEIGHT_BASE = (uint32_t)test_fc_weight;
    
    // 配置我们精心准备的跨页地址！
    NPU_REG_BIAS_BASE   = evil_bias_addr;
    NPU_REG_OUT_BASE    = evil_out_addr;

    // 启动计算
    NPU_REG_CTRL_STATUS = 0x00000001;
    wait_npu_done();
    my_printf(" -> NPU Execution Completed.\n");

    // -----------------------------------------------------------
    // 校验输出结果 (直接从跨页的 evil_out_addr 读取)
    // -----------------------------------------------------------
    my_printf("\n>>> Checking NPU Output Integrity <<<\n");
    
    int8_t* ptr_hw_out  = (int8_t*)evil_out_addr;
    int8_t* ptr_exp_out = (int8_t*)expected_fc_output;
    int error_cnt = 0;
    
    // 遍历 10 个有效通道
    for (int ch = 0; ch < 10; ch++) {
        int8_t val = ptr_hw_out[ch];
        int8_t exp = ptr_exp_out[ch];
        
        if (val != exp) {
            my_printf("  [ERROR] CH[%d] Mismatch -> HW: %d | EXP: %d\n", ch, val, exp);
            error_cnt++;
        }
    }
    
    if (error_cnt == 0) {
        my_printf("  [PASS] Output Data is 100%% Correct!\n");
        my_printf("  [PASS] DMA correctly handled the 4KB boundary wrap-around.\n");
        my_printf("========================================\n");
        my_printf(" ALL AXI BURST BOUNDARY TESTS PASSED!\n");
        my_printf("========================================\n");
        SIM_EXIT_PORT = 1;
    } else {
        my_printf("  [FAIL] Output Verification FAILED with %d errors.\n", error_cnt);
        SIM_EXIT_PORT = 0;
    }

    while(1);
    return 0;
}