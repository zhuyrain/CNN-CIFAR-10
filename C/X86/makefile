# ==========================================
# x86 宿主机端到端模型验证 Makefile
# ==========================================

# ==========================================
# 跨平台环境检测 (Cross-Platform OS Detection)
# ==========================================
ifeq ($(OS),Windows_NT)
	# Windows 环境 (cmd.exe)
	RM = cmd /c del /Q /F
	FixPath = $(subst /,\,$1)
	# Windows 下可执行文件带有 .exe 后缀
	TARGET = main_x86.exe
else
	# Linux / macOS 环境 (bash/sh)
	RM = rm -f
	FixPath = $1
	# Linux/Mac 下可执行文件通常没有后缀
	TARGET = main_x86
endif

# 1. 工具链定义
# 宿主机直接使用系统原生 GCC 即可，不需要 riscv-none-elf- 前缀
CC = gcc

# 2. 文件夹路径
SRC_DIR = .

# 3. 源文件定义
# 将截图中的 3 个 .c 文件全部加入编译列表
C_SRCS = $(SRC_DIR)/main_x86.c \
         $(SRC_DIR)/conv2d_int8_naive_x86.c \
         $(SRC_DIR)/other_int8_naive.c

# 4. 编译选项
# -I$(SRC_DIR) 的作用是告诉编译器去 C/X86 文件夹下寻找引用的 .h 头文件
# -Wall 开启所有警告，-O2 开启通用优化
CFLAGS = -I$(SRC_DIR) -Wall -O2

# 5. 编译目标 (Targets)
# 默认目标：编译生成可执行文件
all: $(TARGET)

# 链接并生成最终的 x86 可执行文件
$(TARGET): $(C_SRCS)
	$(CC) $(CFLAGS) $(C_SRCS) -o $@

# 运行测试的便捷目标 (输入 make run 可以直接编译并执行)
run: all
ifeq ($(OS),Windows_NT)
	.\$(TARGET)
else
	./$(TARGET)
endif

# 清理生成的文件
clean:
	$(RM) $(call FixPath, $(TARGET))