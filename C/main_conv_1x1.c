#include <stdint.h>
#include "my_printf.h"
#include "npu.h"

// ==========================================
// 中断与系统控制逻辑
// ==========================================
extern void enable_npu_irq(uint32_t mask);
extern void wait_for_irq(void);

volatile int npu_task_finished = 0;

void npu_isr(void) {
    npu_task_finished = 1;
    NPU_REG_CTRL_STATUS = 0x00000004; 
}

void wait_npu_done() {
    while (!npu_task_finished) { wait_for_irq(); }
    npu_task_finished = 0;
}

// ==========================================
// 迷你测试张量 (H=4, W=4, IC=4, OC=4)
// ==========================================
#define H 4
#define W 4
#define IC 4
#define OC 4

int8_t act_buf[H * W * IC] __attribute__((aligned(4)));

// 3x3 权重: 9 个位置 * (IC/4) * OC = 9 * 1 * 4 = 36 words = 144 Bytes
int8_t weight_3x3[9 * IC * OC] __attribute__((aligned(4)));

// 1x1 权重: 1 个位置 * (IC/4) * OC = 1 * 1 * 4 = 16 Bytes
int8_t weight_1x1[1 * IC * OC] __attribute__((aligned(4)));

int32_t bias_buf[OC] __attribute__((aligned(4)));
int8_t  out_buf[H * W * OC] __attribute__((aligned(4)));

// 初始化为全 1 矩阵，方便通过数学规律进行边界自检
void init_tensors() {
    for (int i = 0; i < H * W * IC; i++) act_buf[i] = 1;
    for (int i = 0; i < 9 * IC * OC; i++) weight_3x3[i] = 1;
    for (int i = 0; i < 1 * IC * OC; i++) weight_1x1[i] = 1;
    for (int i = 0; i < OC; i++) bias_buf[i] = 0; // 偏置为 0，纯测乘加
}

// -----------------------------------------------------------
// 验证 A：标准 3x3 卷积
// -----------------------------------------------------------
int run_conv_3x3() {
    for (int i = 0; i < H * W * OC; i++) out_buf[i] = 0;

    my_printf(" -> Configuring 3x3 Convolution (pad=1, kx=2, ky=2)...\n");

    NPU_REG_CFG_IMG_DIM  = (H << 16) | W;      
    // row_word = W * IC / 4 = 4. weight_word = 9 * (IC/4) * OC = 36
    NPU_REG_CFG_MEM_CNT  = (4 << 16) | 36;
    
    // out_stride=4, sa_weight_num=9 (3x3), lb_line_width=4, lb_ic_groups=0
    NPU_REG_CFG_DATAPATH = (4 << 23) | (0 << 19) | (9 << 11) | (6 << 4) | 0;

    // oc_num=4, col_group_en=1(4列), ky_skew=0, pad_size=1, kx_max=2, ky_max=2
    NPU_REG_CFG_SPATIAL  = (4 << 24) | (0x0001 << 8) | (0 << 5) | (1 << 4) | (2 << 2) | 2;

    NPU_REG_CFG_QUANT    = (0 << 16) | 1;
    NPU_REG_CFG_POST     = (132 << 22) | (40 << 12) | (0 << 1) | 0;

    NPU_REG_ACT_BASE    = (uint32_t)act_buf;
    NPU_REG_WEIGHT_BASE = (uint32_t)weight_3x3;
    NPU_REG_BIAS_BASE   = (uint32_t)bias_buf;
    NPU_REG_OUT_BASE    = (uint32_t)out_buf;

    NPU_REG_CTRL_STATUS = 0x00000001;
    wait_npu_done();

    int error_cnt = 0;
    // 3x3 Padding 数学校验
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int expected = 0;
            if ((h == 0 || h == 3) && (w == 0 || w == 3)) expected = 16; // 4个角: 2x2=4个有效像素 * 4通道 = 16
            else if (h == 0 || h == 3 || w == 0 || w == 3) expected = 24; // 边缘: 2x3=6个有效像素 * 4通道 = 24
            else expected = 36; // 中心: 3x3=9个有效像素 * 4通道 = 36

            for (int c = 0; c < OC; c++) {
                int idx = (h * W + w) * OC + c;
                if (out_buf[idx] != expected) {
                    my_printf("    [ERROR 3x3] (h=%d, w=%d, c=%d) HW: %d | EXP: %d\n", h, w, c, out_buf[idx], expected);
                    error_cnt++;
                }
            }
        }
    }

    if (error_cnt == 0) return 1;
    return 0;
}

// -----------------------------------------------------------
// 验证 B：极致 1x1 卷积 (时空不对称复用)
// -----------------------------------------------------------
int run_conv_1x1() {
    for (int i = 0; i < H * W * OC; i++) out_buf[i] = 0;

    my_printf(" -> Configuring 1x1 Pointwise Conv (pad=0, kx=0, ky=0, skew=1)...\n");

    NPU_REG_CFG_IMG_DIM  = (H << 16) | W;      
    // weight_word = 1 * (IC/4) * OC = 4
    NPU_REG_CFG_MEM_CNT  = (4 << 16) | 4;
    
    // sa_weight_num 变为 1
    NPU_REG_CFG_DATAPATH = (4 << 23) | (0 << 19) | (1 << 11) | (4 << 4) | 0;

    // 核心时空魔法：
    // ky_skew=1 (强行锁定物理第二行), pad_size=0 (消灭X轴气泡), kx_max=0, ky_max=0 (坍缩为单点)
    NPU_REG_CFG_SPATIAL  = (4 << 24) | (0x0001 << 8) | (1 << 5) | (0 << 4) | (0 << 2) | 0;

    NPU_REG_CFG_QUANT    = (0 << 16) | 1;
    NPU_REG_CFG_POST     = (132 << 22) | (40 << 12) | (0 << 1) | 0;

    NPU_REG_ACT_BASE    = (uint32_t)act_buf;
    NPU_REG_WEIGHT_BASE = (uint32_t)weight_1x1;
    NPU_REG_BIAS_BASE   = (uint32_t)bias_buf;
    NPU_REG_OUT_BASE    = (uint32_t)out_buf;

    NPU_REG_CTRL_STATUS = 0x00000001;
    wait_npu_done();

    int error_cnt = 0;
    // 1x1 数学校验
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int expected = 4; // 1个像素 * 4通道 = 4
            for (int c = 0; c < OC; c++) {
                int idx = (h * W + w) * OC + c;
                if (out_buf[idx] != expected) {
                    my_printf("    [ERROR 1x1] (h=%d, w=%d, c=%d) HW: %d | EXP: %d\n", h, w, c, out_buf[idx], expected);
                    error_cnt++;
                }
            }
        }
    }

    if (error_cnt == 0) return 1;
    return 0;
}

int main() {
    enable_npu_irq(~(1 << 4));

    my_printf("\n========================================\n");
    my_printf(" [TEST 7] Multi-Scale Conv Core Re-use \n");
    my_printf("          Spatiotemporal Asymmetry (3x3 vs 1x1)\n");
    my_printf("========================================\n");

    init_tensors();

    int pass = 1;

    // 运行 3x3 卷积测试
    if (run_conv_3x3()) {
        my_printf("  [PASS] Standard 3x3 Conv (Padding = 1) Verified!\n");
    } else { pass = 0; }

    // 运行 1x1 卷积测试
    if (run_conv_1x1()) {
        my_printf("  [PASS] Pointwise 1x1 Conv (Padding = 0) Verified!\n");
    } else { pass = 0; }

    my_printf("========================================\n");
    if (pass) {
        my_printf(" MULTI-SCALE CONV KERNEL VERIFICATION PASSED!\n");
    } else {
        my_printf(" MULTI-SCALE CONV KERNEL VERIFICATION FAILED!\n");
        while(1);
    }
    my_printf("========================================\n");

    while(1);
    return 0;
}