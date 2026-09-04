# CNN-CIFAR-10：面向 RISC-V + 自定义 NPU 的全栈软硬件协同设计

一套完整的端到端系统，用于在搭载专用**神经网络处理器（NPU）**加速器的**自定义 RISC-V SoC** 上部署 VGG 风格的 CNN。覆盖完整流程：PyTorch 模型训练 → INT8 量化 → C 头文件导出 → 裸机 RISC-V 固件 → NPU 寄存器级编程 → 硬件验证。

**目标任务：** CIFAR-10 图像分类（10 分类）
**精度：** INT8 量化精度 84.96%（在全量 1 万张测试集上验证）
**目标平台：** PicoRV32 软核 CPU + 自定义 NPU（基址 0x40000000），1 MB SRAM

---

## 项目架构

```
┌──────────────────────────────────────────────────────────┐
│  python/          │  C/              │  RISCV/           │
│                   │                  │                   │
│  训练 (.ipynb)    │  固件 (.c)       │  启动 (start.S)    │
│  量化             │  NPU 驱动        │  链接脚本 (lds)    │
│  黄金参考模型     │  测试套件        │  PicoRV32 中断服务 │
│  导出 .h          │                  │                   │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │  自定义 NPU (Verilog) │
              │  基址: 0x40000000     │
              │  AXI-Lite 配置接口    │
              │  AXI-Full DMA 引擎    │
              └──────────────────────┘
```

### 数据流

```
Python 训练 → 融合 BN → INT8 量化 → 导出 .h
                                        │
               ┌────────────────────────┤
               ▼                        ▼
        python黄金参考模型           RISC-V 固件
        (逐比特验证)              (NPU 寄存器配置)
               │                        │
               └────────┬───────────────┘
                        ▼
              比对最终分类 logits
              SIM_EXIT_PORT = 1 (PASS)
```

---

## 仓库结构

```
├── python/                     # 模型训练与量化
│   ├── cifar10_fp32_baseline.ipynb       # 训练 FP32 模型，融合 BatchNorm
│   ├── export_test_vector.ipynb          # INT8 量化，导出 C 头文件
│   ├── test_scaled_model_accuracy.ipynb  # 验证 INT8 精度 (84.96%)
│   ├── convert_hex.py                    # 后处理 objcopy hex → 32 位字对齐
│   ├── test_vector.h                     # 16/32 通道模型数据（供 Python 黄金参考模型使用）
│   ├── test_vector_2.h                   # 64 通道模型数据（供 RISC-V 固件使用）
│   ├── test_vector_fpga.h                # FPGA 优化模型数据
│   ├── *.pth                             # 已保存的 FP32 融合权重
│   └── data/                             # CIFAR-10 数据集（本地缓存）
│
├── C/                           # C 固件源码
│   ├── npu.h                     # NPU 寄存器映射（基址 0x40000000）
│   ├── conv2d_int8_naive.c/h     # CPU 后备算子：Conv2D、ReLU、MaxPool、FC（HWC 格式）
│   ├── my_printf.c/h             # 裸机 UART printf（无 libc 依赖）
│   ├── main_vgg_full.c           # 端到端 VGG 推理（7 层流水线）
│   ├── main_fpga.c               # FPGA 可部署 2 层 CNN
│   ├── main_test.c               # 3 层测试流水线
│   ├── main_cpu_alu.c            # 测试 1：CPU 算术逻辑单元 + SRAM 完整性
│   ├── main_axi_cfg.c            # 测试 2：AXI 配置寄存器读写
│   ├── main_axi_burst_edge.c     # 测试 3：4KB AXI 突发传输边界
│   ├── main_math_ones.c          # 测试 4：NPU 乘加阵列验证（4096 次 MAC）
│   ├── main_dim_stretch.c        # 测试 5：可变输出通道维度
│   ├── main_fc_reshape.c         # 测试 6：FC 层 H×W 重映射折叠
│   ├── main_conv_1x1.c           # 测试 7：3×3 与 1×1 卷积多尺度测试
│   ├── main_bus_stress.c         # 测试 8：AXI 总线吞吐压力
│   ├── main_quant_bound.c        # 测试 9：INT8 边界条件
│   ├── main_irq_storm.c          # 测试 10：中断风暴韧性测试
│   └── main_reg_bounds.c         # 测试 12：寄存器地址越界检测
│
├── RISCV/                       # RISC-V 平台支持
│   ├── start.S                   # 启动向量、中断服务例程、PicoRV32 自定义指令
│   └── custom.lds                # 1 MB SRAM 链接脚本（基址 0x00000000）
│
├── firmware/                    # 构建产物（各测试用例的 ELF、HEX 文件）
├── makefile                     # 根构建编排文件（RISC-V 交叉编译）
├── .gitignore
└── readme.md
```

---

## CNN 网络架构

项目包含两套网络配置，分别用于**仿真验证**和 **FPGA 上板部署**：

- **仿真（复杂模型）：** 使用 `my_hardware_cnn_fp32_64ch_fused.pth`，导出 `test_vector_2.h`
- **FPGA 上板（简单模型）：** 使用 `my_hardware_cnn_fp32_69p.pth`，导出 `test_vector_fpga.h`
- **Python 黄金参考：** 使用 `test_vector.h`，在 `test_scaled_model_accuracy.ipynb` 中直接加载以获取标准分类结果

### 仿真模型：完整 VGG 流水线（64 通道，7 层）

> 权重文件：`my_hardware_cnn_fp32_64ch_fused.pth` | 头文件：`test_vector_2.h` | 固件：`main_vgg_full.c`

| 层    | 类型             | 输入尺寸     | 输出尺寸    | 卷积核 | 执行单元 |
| ----- | ---------------- | ------------ | ----------- | ------ | -------- |
| Conv1 | Conv2D + ReLU    | 32×32×3      | 32×32×64    | 3×3    | **NPU**  |
| Conv2 | Conv2D + ReLU    | 32×32×64     | 32×32×64    | 3×3    | **NPU**  |
| Pool1 | MaxPool 2×2      | 32×32×64     | 16×16×64    | 步长 2 | **CPU**  |
| Conv3 | Conv2D + ReLU    | 16×16×64     | 16×16×64    | 3×3    | **NPU**  |
| Conv4 | Conv2D + ReLU    | 16×16×64     | 16×16×64    | 3×3    | **NPU**  |
| Pool2 | MaxPool 2×2      | 16×16×64     | 8×8×64      | 步长 2 | **CPU**  |
| FC    | 全连接（重映射） | 8×8×64=4096  | 10          | —      | **NPU**  |

### FPGA 上板模型：轻量 3 层流水线（16→32 通道）

> 权重文件：`my_hardware_cnn_fp32_69p.pth` | 头文件：`test_vector_fpga.h` | 固件：`main_fpga.c`

| 层    | 类型                 | 输入尺寸     | 输出尺寸    | 卷积核 |执行单元 |
| ----- | -------------------- | ------------ | ----------- | ------ |-------- |
| Conv1 | Conv2D + ReLU + Pool | 32×32×3      | 16×16×16    | 3×3    |**NPU**  |
| Conv2 | Conv2D + ReLU + Pool | 16×16×16     | 8×8×32      | 3×3    |**NPU**  |
| FC    | 全连接               | 8×8×32=2048  | 10          | —      |**CPU**  |

---

## NPU 硬件接口

自定义 NPU 加速器通过存储映射方式挂载在基址 **0x40000000**，寄存器映射如下：

| 偏移 | 寄存器       | 功能描述                                  |
| ---- | ------------ | ----------------------------------------- |
| 0x00 | CTRL_STATUS  | Bit 0：启动，Bit 1：完成，Bit 2：清除中断 |
| 0x04 | ACT_BASE     | 输入激活值缓冲区指针                      |
| 0x08 | WEIGHT_BASE  | 权重缓冲区指针                            |
| 0x0C | BIAS_BASE    | 偏置缓冲区指针                            |
| 0x10 | OUT_BASE     | 输出缓冲区指针                            |
| 0x14 | CFG_IMG_DIM  | 图像尺寸（高度 << 16\| 宽度）             |
| 0x18 | CFG_MEM_CNT  | 行字数与权重字数                          |
| 0x1C | CFG_DATAPATH | 输出步长、脉动阵列配置、行缓冲宽度        |
| 0x20 | CFG_SPATIAL  | 输出通道数、列分组、卷积核/填充配置       |
| 0x24 | CFG_QUANT    | 重量化乘法因子                            |
| 0x28 | CFG_POST     | 排空超时、FIFO 阈值、移位量、ReLU 使能    |

**其他存储映射外设：**

- **UART TX** @ 0x80000000（AXI UART Lite）—— 调试用 printf 输出
- **SIM_EXIT** @ 0x80000010 —— 写 1 = PASS，写 0 = FAIL（仿真退出）

---

## 关键技术特性

### 1. INT8 对称量化

- **方案：** 逐张量对称 INT8，零点 = 0
- **重量化公式：** `output = (累加器 × 乘法因子) >> 移位`
- 推理阶段**零浮点运算** —— 纯整数算术
- BatchNorm 层在量化前通过数学方法**融合**到卷积权重/偏置中，硬件无需 BatchNorm 逻辑

### 2. 异构计算（NPU + CPU）

- **NPU 负责：** 所有卷积层和全连接层（计算密集型）
- **CPU 负责：** MaxPool 下采样（轻量级，不值得占用脉动阵列资源）
- 此设计在最大化硬件利用率的同时，保持 CPU 可用于控制流调度

### 3. HWC 与 CHW 数据布局

- **NPU 采用 HWC**（高-宽-通道交织）格式，以获得最优 AXI 突发传输效率
- **Python 黄金参考模型采用 CHW**（通道-高-宽）格式，便于标准互操作
- `linear_int8_hwc()` 函数在 FC 层动态完成两种格式之间的映射转换

### 4. PicoRV32 自定义指令

固件使用了 PicoRV32 特有的非标准 RISC-V 指令：

| 指令        | 机器码         | 用途                     |
| ----------- | -------------- | ------------------------ |
| `maskirq` | `0x0605600B` | 屏蔽/使能中断线          |
| `waitirq` | `0x0800400B` | 休眠等待中断（WFI 等价） |
| `retirq`  | `0x0400000B` | 从中断处理程序返回       |

### 5. 中断驱动的 NPU 控制

RISC-V 固件通过寄存器写入发起 NPU 计算，随后进入 WFI 状态（`waitirq`）。NPU 通过 IRQ 线发出完成信号，中断服务例程记录 `npu_task_finished` 标志，主循环继续执行——最大限度降低功耗和轮询开销。

### 6. 内存高效的缓冲区复用

- FPGA 测试用例（`main_fpga.c`）采用 **64 KB 硬件内存池** + **16 KB 软件内存池**
- 通过宏别名在不同层之间复用缓冲区 —— 无需数据拷贝，仅需指针重映射
- 在 1 MB SRAM 约束下即可运行完整流水线

---

## 快速开始

请参考 [快速开始指南](QuickStart.md)

---



## 硬件验证测试套件

项目包含 12 项回归测试，覆盖硬件验证的全阶段：

| 编号 | 测试用例           | 验证内容                                     |
| ---- | ------------------ | -------------------------------------------- |
| 1    | `cpu_alu`        | PicoRV32 算术逻辑单元运算 + 8 KB SRAM 完整性 |
| 2    | `axi_cfg`        | 全部 10 个 NPU 配置寄存器的 AXI-Lite 读写    |
| 3    | `axi_burst_edge` | 4 KB 页边界 AXI DMA 突发传输跨页             |
| 4    | `math_ones`      | NPU 乘加阵列正确性（4096 次运算）            |
| 5    | `dim_stretch`    | 输出通道维度可伸缩性（4→64 通道）           |
| 6    | `fc_reshape`     | FC 层在 3 种不同 H×W 几何排布下的折叠映射   |
| 7    | `conv_1x1`       | 多尺度卷积（3×3 与 1×1）                   |
| 8    | `bus_stress`     | 极限负载下 AXI 总线吞吐量                    |
| 9    | `quant_bound`    | INT8 上溢/下溢/极限量化参数                  |
| 10   | `irq_storm`      | 2500 次连续快速中断下的中断服务例程韧性      |
| 11   | `vgg_full`       | 端到端 7 层 VGG 推理                         |
| 12   | `reg_bounds`     | NPU 寄存器地址越界检测                       |

---

## 性能指标

| 指标                   | 数值                       |
| ---------------------- | -------------------------- |
| FP32 基线精度          | ~86.52%                    |
| INT8 量化精度          | 84.96%                     |
| 精度损失（FP32→INT8） | ~1.56%                     |
| 验证范围               | 全量 CIFAR-10 1 万张测试集 |
| SRAM 占用              | 1 MB（总计）               |
| NPU 地址空间           | 0x40000000 – 0x4000003F   |

---
