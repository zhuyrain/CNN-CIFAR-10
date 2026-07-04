#ifndef NPU_H
#define NPU_H

// ==========================================
// NPU 寄存器映射 (建议后续移入 npu.h)
// ==========================================
#define NPU_BASE            0x40000000
#define NPU_REG_CTRL_STATUS (*(volatile uint32_t*)(NPU_BASE + 0x00))

// 4 个基址寄存器
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

#endif