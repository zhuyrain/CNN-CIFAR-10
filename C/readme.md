# 文件目录

### npu_weights_int8.h

1. 权重头文件

## C文件主程序

### main_x86.c

1. x86平台的测试文件，详细的计算过程log输出

### main.c

1. picorv32平台的测试文件，详细的计算过程log输出

### main_test.c（暂时忽略）

1. 初始的测试文件，跨平台

## C文件调用程序

### conv2d_int8_naive.c

1. picorv32平台的主要cnn计算函数
2. 通过extern在main文件中声明
