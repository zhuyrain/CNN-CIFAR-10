# ==========================================
# RISC-V 裸机编译 Makefile
# ==========================================

# 1. 工具链定义
PREFIX  = riscv-none-elf-
CC      = $(PREFIX)gcc
OBJCOPY = $(PREFIX)objcopy

# 2. 文件夹路径
C_DIR   = C
RV_DIR  = RISCV

# 3. 源文件定义
C_SRCS  = $(C_DIR)/main.c $(C_DIR)/conv2d_int8_naive.c
ASM_SRC = $(RV_DIR)/start.S
LDSCRIPT= $(RV_DIR)/custom.lds

# 4. 编译选项
# -I$(C_DIR) 的作用是告诉编译器去 C/ 文件夹下寻找引用的 .h 头文件
CFLAGS  = -march=rv32imc -mabi=ilp32 -I$(C_DIR) -Wall -O2
LDFLAGS = -T $(LDSCRIPT) -nostartfiles

# 5. 编译目标 (Targets)
all: firmware_byte.hex

firmware.elf: $(ASM_SRC) $(C_SRCS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(ASM_SRC) $(C_SRCS) -o $@

firmware_byte.hex: firmware.elf
	$(OBJCOPY) -O verilog $< $@

clean:
	rm firmware.elf,firmware.hex,firmware_byte.hex