.section .text.init, "ax"
    .global _start

_start:
    la sp, _sp
    jal ra, main
    /* main 正常结束后，进入死循环 */
end_loop:
    j end_loop

/* 【神级修复】强制对齐到 0x10 物理地址 */
.org 0x10
trap_loop:
    /* 如果 CPU 发生任何异常，都会跳到这里无限循环，而不是坠毁 */
    j trap_loop