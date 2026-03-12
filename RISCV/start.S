.section .text.init, "ax"
    .global _start

_start:
    /* 1. 初始化栈指针 (sp) */
    /* 符号 _sp 由我们的链接脚本提供 */
    la sp, _sp

    /* 2. 跳转到你的 C 语言 main 函数 */
    jal ra, main

    /* 3. 如果 main() 意外返回，则进入死循环 (安全网) */
trap_loop:
    j trap_loop