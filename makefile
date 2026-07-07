# ==========================================
# RISC-V 裸机多用例编译 Makefile (目录隔离版)
# ==========================================

# 兼容大小写传参 (允许 mingw32-make CASE=xxx 或 make case=xxx)
ifdef case
    CASE = $(case)
endif
CASE ?= test

# ==========================================
# 跨平台环境检测与命令包装 (彻底解决 Windows/Linux 兼容性)
# ==========================================
ifeq ($(OS),Windows_NT)
    # Windows 环境 (回滚至最稳定的原生 cmd /c 调用)
    RM      = cmd /c del /Q /F
    RMDIR   = cmd /c rmdir /Q /S
    CP      = cmd /c copy /Y
    FixPath = $(subst /,\,$1)
    MKDIR   = cmd /c if not exist $(subst /,\,$1) mkdir $(subst /,\,$1)
else
    # Linux / macOS 环境 (bash)
    RM      = rm -f
    RMDIR   = rm -rf
    CP      = cp -f
    FixPath = $1
    MKDIR   = mkdir -p $1
endif

# 1. 工具链定义
PREFIX  = riscv-none-elf-
CC      = $(PREFIX)gcc
OBJCOPY = $(PREFIX)objcopy
PY      = python

# 2. 文件夹路径规划 (完美文件隔离)
C_DIR       = C
RV_DIR      = RISCV
PYTHON_DIR  = python
FW_DIR      = firmware
# 每个 CASE 生成一个独立的子文件夹
CASE_DIR    = $(FW_DIR)/$(CASE)
# 批量回归测试的汇总文件夹
REGRESS_DIR = $(FW_DIR)/regress

# 3. 源文件定义
MAIN_SRC    = $(C_DIR)/main_$(CASE).c
COMMON_SRCS = $(C_DIR)/conv2d_int8_naive.c $(C_DIR)/my_printf.c
ASM_SRC     = $(RV_DIR)/start.S
LDSCRIPT    = $(RV_DIR)/custom.lds
CONV_HEX    = $(PYTHON_DIR)/convert_hex.py

# 最终输出目标 (全部分配到对应 CASE_DIR 之下)
TARGET_ELF      = $(CASE_DIR)/firmware_$(CASE).elf
TARGET_HEX      = $(CASE_DIR)/firmware_$(CASE).hex
TARGET_BYTE_HEX = $(CASE_DIR)/firmware_byte_$(CASE).hex

# 4. 编译选项
CFLAGS  = -march=rv32imc -mabi=ilp32 -I$(C_DIR) -Wall -O2 \
          -ffunction-sections -fdata-sections
LDFLAGS = -T $(LDSCRIPT) -nostartfiles -Wl,--gc-sections

# ==========================================
# 5. 基础编译目标 (Targets)
# ==========================================
all: conv

conv: $(TARGET_BYTE_HEX)

# 确保输出子目录存在，然后生成 ELF
$(TARGET_ELF): $(ASM_SRC) $(MAIN_SRC) $(COMMON_SRCS)
	@$(call MKDIR, $(CASE_DIR))
	@echo "========================================"
	@echo " Building Test Case : $(CASE)"
	@echo " Target Directory   : $(CASE_DIR)"
	@echo "========================================"
	$(CC) $(CFLAGS) $(LDFLAGS) $(ASM_SRC) $(MAIN_SRC) $(COMMON_SRCS) -o $@

# 生成标准 Verilog Hex
$(TARGET_HEX): $(TARGET_ELF)
	$(OBJCOPY) -O verilog $< $@

# 生成 8-bit 单字节对齐 Hex
$(TARGET_BYTE_HEX): $(TARGET_HEX)
	$(PY) $(CONV_HEX) $< $@
	@echo " -> Build Complete: $@"

# ==========================================
# 6. 回归测试自动化 (Regression Test Batch)
# ==========================================
# 修正为你现有的 11 个核心测试用例
REGRESS_CASES = cpu_alu axi_cfg axi_burst_edge math_ones dim_stretch fc_reshape conv_1x1 bus_stress quant_bound irq_storm vgg_full reg_bounds

# 使用 Make 的依赖推导机制，避免在不同系统的 Shell 下写 for 循环
REGRESS_TARGETS = $(addprefix regress_, $(REGRESS_CASES))

regress: $(REGRESS_TARGETS)
	@echo ""
	@echo "========================================"
	@echo " Regression Firmware Build Completed! "
	@echo " All byte-aligned HEX files are gathered in:"
	@echo " -> $(REGRESS_DIR)"
	@echo "========================================"

# 模式匹配规则：针对每个回归用例执行编译，并将终极 _byte.hex 复制到 regress 目录
regress_%:
	@$(MAKE) CASE=$*
	@$(call MKDIR, $(REGRESS_DIR))
	-@$(CP) $(call FixPath, $(FW_DIR)/$*/firmware_byte_$*.hex) $(call FixPath, $(REGRESS_DIR)/)

# ==========================================
# 7. 清理规则
# ==========================================
# 【还原原始删除逻辑】：去除了导致 PowerShell 卡死的 >nul，保留最稳定的 cmd /c rmdir
clean:
	@echo "Cleaning firmware directory..."
	-@$(RMDIR) $(call FixPath, $(FW_DIR))
	@echo "Clean completed."