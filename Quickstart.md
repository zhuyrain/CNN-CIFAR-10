## 快速开始

### 环境依赖

| 组件          | 要求                                                        |
| ------------- | ----------------------------------------------------------- |
| RISC-V 工具链 | `riscv-none-elf-gcc`（目标：`rv32imc`，ABI：`ilp32`） |
| Python        | 3.8+ 及 PyTorch、torchvision、numpy                         |

### 运行 Python Notebook

按顺序执行 Jupyter notebook：

1. `cifar10_fp32_baseline.ipynb` —— 训练 FP32 模型，融合 BN，保存权重
2. `export_test_vector.ipynb` —— INT8 量化，导出 C 头文件test_vector.h。执行代码块2以将test_vector.h转化为硬件友好的头文件test_vector_2.h
3. `test_scaled_model_accuracy.ipynb` —— 在测试集上验证量化后模型正确率
### 构建 RISC-V 固件

```bash
# 构建单个测试用例（对应C文件夹下main_<case>.c）
# 默认编译main_test.c
mingw32-make CASE=vgg_full

# 构建全部 12 项回归测试，收集 firmware_byte_<case>.hex产物至 regress目录
mingw32-make regress

# 清理所有构建产物
mingw32-make clean
```

各测试用例输出：

- `firmware/<case>/firmware_<case>.elf` —— RISC-V ELF 二进制文件
- `firmware/<case>/firmware_<case>.hex` —— 原始 hex
- `firmware/<case>/firmware_byte_<case>.hex` —— 32 位字对齐 hex（Verilog 兼容）

### 与picorv32_npu代码仓交互
1. 将firmware/regress文件夹下的固件整体复制到picorv32_npu代码仓的iVerdi/regress/firmwares下，即可一键进行覆盖率回归测试
2. 将任意一个firmware_byte_`<case>`.hex文件内容复制到iVerdi/firmware.hex文件中，即可进行单次仿真验证
