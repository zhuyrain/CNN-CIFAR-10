#include <stdint.h>
#include "my_printf.h"
#include "npu.h" // 完美引入你封装好的底层寄存器宏

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
// 数据阵列分配
// ==========================================
#define IC 64
#define MAX_OC 64

int8_t  act_buf[IC] __attribute__((aligned(4)));
int8_t  weight_buf[MAX_OC * IC] __attribute__((aligned(4)));
int32_t bias_buf[MAX_OC] __attribute__((aligned(4)));
int8_t  out_buf[MAX_OC] __attribute__((aligned(4)));

// 初始化公共数据 (特征图在整个测试中不变)
void init_test_data() {
    for (int i = 0; i < IC; i++) { act_buf[i] = 1; }
}

// 核心测试函数：动态维度伸缩 (支持任意逻辑 OC，软件自动对齐到底层 4 通道)
int run_dim_test(int logical_oc) {
    // ========================================================
    // 【架构师软件补丁：Software Padding】
    // 将逻辑输出通道数 (logical_oc) 向上对齐到硬件的 32-bit 粒度 (hw_oc)
    // ========================================================
    int hw_oc = (logical_oc + 3) & ~3; 

    // 1. 初始化偏置和输出
    for (int i = 0; i < MAX_OC; i++) {
        out_buf[i] = 0; // 彻底清空输出，防止残留
        
        if (i < logical_oc) {
            // 真实有效通道：注入真实的偏置(i)
            bias_buf[i] = i; 
        } else if (i < hw_oc) {
            // Padding 通道：强制抹零
            bias_buf[i] = 0;
        } else {
            // 物理门控通道：故意填入脏数据 (i)
            bias_buf[i] = i;
        }
    }

    // ========================================================
    // 【架构师核心修正：权重交织 (基于 Python 黄金模型的 INT8 4合1 映射)】
    // 根据 Python 代码，1个 32-bit 的 packed_word 包含的是：
    // 同一个 OC 的连续 4 个 IC (ig*4+0 ~ ig*4+3)。
    // 并且打包的外层循环是 ic_group (ig)，内层循环是 oc。
    // ========================================================
    
    // 先把整个内存铺满脏数据 (作为门控泄漏的陷阱)
    for (int i = 0; i < MAX_OC * IC; i++) {
        weight_buf[i] = 1; 
    }

    int ic_groups = IC / 4;
    for (int ig = 0; ig < ic_groups; ig++) {
        for (int oc = 0; oc < hw_oc; oc++) {
            // 算出这 4 个 IC (即 1 个 Word) 在内存中的字节基址
            int word_base_idx = (ig * hw_oc + oc) * 4; 
            
            if (oc < logical_oc) {
                // 真实通道：4 个 IC 全填 1
                weight_buf[word_base_idx + 0] = 1; // c0
                weight_buf[word_base_idx + 1] = 1; // c1
                weight_buf[word_base_idx + 2] = 1; // c2
                weight_buf[word_base_idx + 3] = 1; // c3
            } else {
                // Padding 通道：为了让硬件算出 0，精准地把这 4 个 IC 抹零
                weight_buf[word_base_idx + 0] = 0;
                weight_buf[word_base_idx + 1] = 0;
                weight_buf[word_base_idx + 2] = 0;
                weight_buf[word_base_idx + 3] = 0;
            }
        }
    }

    // 2. 动态计算硬件参数 (全部喂给硬件对齐后的 hw_oc)
    uint32_t weight_words = 16 * hw_oc; // IC/4 * hw_oc
    int group_count = hw_oc / 4; 
    uint32_t col_group_en = (1 << group_count) - 1;

    // 3. 配置 NPU 寄存器
    NPU_REG_CFG_IMG_DIM  = (1 << 16) | 1;      
    NPU_REG_CFG_MEM_CNT  = (16 << 16) | weight_words;
    
    // out_stride 使用硬件写回的 hw_oc
    NPU_REG_CFG_DATAPATH = (hw_oc << 23) | (0 << 19) | (1 << 11) | (1 << 4) | 15;

    NPU_REG_CFG_SPATIAL  = (hw_oc << 24) | (col_group_en << 8) | (1 << 5) | (0 << 4) | (0 << 2) | 0;

    NPU_REG_CFG_QUANT    = (0 << 16) | 1; // 缩放比例 1
    
    uint32_t drain_timeout  = 132;   
    uint32_t fifo_af_thresh = 40; 
    NPU_REG_CFG_POST = (drain_timeout << 22) | (fifo_af_thresh << 12) | (0 << 1) | 0;

    // 基址配置
    NPU_REG_ACT_BASE    = (uint32_t)act_buf;
    NPU_REG_WEIGHT_BASE = (uint32_t)weight_buf;
    NPU_REG_BIAS_BASE   = (uint32_t)bias_buf;
    NPU_REG_OUT_BASE    = (uint32_t)out_buf;

    // 4. 发射任务
    NPU_REG_CTRL_STATUS = 0x00000001;
    wait_npu_done();

    // 5. 结果严格自检
    int error_cnt = 0;
    
    // 校验 A：检查被激活的逻辑通道 (预期: 64 个 1 相加 + 偏置 = 64 + ch)
    for (int ch = 0; ch < logical_oc; ch++) {
        int8_t exp = 64 + ch;
        if (out_buf[ch] != exp) {
            my_printf("    [ERROR-LOGICAL] CH[%d] HW: %d | EXP: %d\n", ch, out_buf[ch], exp);
            error_cnt++;
        }
    }

    // 校验 B：检查软件 Padding 通道 (预期: 算出的绝对是 0)
    for (int ch = logical_oc; ch < hw_oc; ch++) {
        if (out_buf[ch] != 0) {
            my_printf("    [ERROR-PADDING] Padded CH[%d] was NOT 0 (Got: %d)!\n", ch, out_buf[ch]);
            error_cnt++;
        }
    }
    
    // 校验 C：检查彻底休眠的门控通道 (预期: 根本没往 SRAM 写数，维持清空时的 0)
    for (int ch = hw_oc; ch < MAX_OC; ch++) {
        if (out_buf[ch] != 0) {
            my_printf("    [ERROR-LEAK] Gated CH[%d] was modified to %d!\n", ch, out_buf[ch]);
            error_cnt++;
        }
    }

    if (error_cnt == 0) {
        my_printf("  [PASS] Logical OC = %d (HW Padded to: %d) Verified.\n", logical_oc, hw_oc);
        return 1;
    } else {
        my_printf("  [FAIL] Logical OC = %d Failed with %d errors.\n", logical_oc, error_cnt);
        return 0;
    }
}

int main() {
    enable_npu_irq(~(1 << 4));

    my_printf("\n========================================\n");
    my_printf(" [TEST 5] Dynamic Output Channels & ICG \n");
    my_printf("          Stretching: 64 -> 32 -> 16 -> 4\n");
    my_printf("========================================\n");

    init_test_data();

    int pass = 1;
    
    my_printf("-> Testing OC = 64 (Full Load 16 Groups)...\n");
    pass &= run_dim_test(64);

    my_printf("-> Testing OC = 32 (Half Array 8 Groups)...\n");
    pass &= run_dim_test(32);

    my_printf("-> Testing OC = 16 (Quarter Array 4 Groups)...\n");
    pass &= run_dim_test(16);

    // 极限边界测试：软件传入逻辑 OC=1，依靠软件抹零填充 (Software Padding) 凑够 32-bit
    my_printf("-> Testing OC = 1 (Extreme Corner Case w/ Software Padding)...\n");
    pass &= run_dim_test(1);

    my_printf("========================================\n");
    if (pass) {
        my_printf(" DYNAMIC DIMENSION VERIFICATION PASSED!\n");
        my_printf("========================================\n");
        SIM_EXIT_PORT = 1;
    } else {
        my_printf(" DYNAMIC DIMENSION VERIFICATION FAILED!\n");
        my_printf("========================================\n");
        SIM_EXIT_PORT = 0;
    }
    while(1);
    return 0;
}