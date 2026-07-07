.section .text.init, "ax"
    .global _start

/* ================================================== */
/* 1. 中断向量表 (Vector Table)*/
/* 严格掌控 0x00 到 0x10 的绝对物理空间*/
/* ================================================== */

/* 绝对地址 0x00000000：复位入口 */
_start:
    j real_start        /* 直接跳到后面去，把空间让出来！占 4 字节 */

/* 绝对地址 0x00000010：异常入口 */
.org 0x10
trap_entry:
    /* 1. 保护现场：分配 64 字节栈空间 (16个寄存器 * 4字节) */
    addi sp, sp, -64
    sw ra,  0(sp)
    sw t0,  4(sp)
    sw t1,  8(sp)
    sw t2, 12(sp)
    sw a0, 16(sp)
    sw a1, 20(sp)
    sw a2, 24(sp)
    sw a3, 28(sp)
    sw a4, 32(sp)
    sw a5, 36(sp)
    sw a6, 40(sp)
    sw a7, 44(sp)
    sw t3, 48(sp)
    sw t4, 52(sp)
    sw t5, 56(sp)
    sw t6, 60(sp)

    /* 2. 调用 C 中断服务函数 */
    call npu_isr

    /* 3. 恢复现场 */
    lw ra,  0(sp)
    lw t0,  4(sp)
    lw t1,  8(sp)
    lw t2, 12(sp)
    lw a0, 16(sp)
    lw a1, 20(sp)
    lw a2, 24(sp)
    lw a3, 28(sp)
    lw a4, 32(sp)
    lw a5, 36(sp)
    lw a6, 40(sp)
    lw a7, 44(sp)
    lw t3, 48(sp)
    lw t4, 52(sp)
    lw t5, 56(sp)
    lw t6, 60(sp)
    addi sp, sp, 64

    /* 4. 硬件级完美返回 */
    /* 删除原来的 mret，替换为 PicoRV32 专用的 retirq 机器码 */
    .word 0x0400000B

/* ================================================== */
/* 2. 真正的启动代码 (可以无限长) */
/* 紧跟在 0x10 的代码之后（通常从 0x14 开始）*/
/* ================================================== */
real_start:
    /* 初始化栈指针 */
    la sp, _sp

    /* 清零 BSS 段 */
    la t0, _bss_start
    la t1, _bss_end
    bgeu t0, t1, 2f
1:
    sw zero, 0(t0)
    addi t0, t0, 4
    bltu t0, t1, 1b
2:

    /* 进入 C 语言世界 */
    jal ra, main
    
end_loop:
    j end_loop

/* ================================================== */
/* 供 C 语言调用的 PicoRV32 自定义中断指令 API         */
/* ================================================== */

.global enable_npu_irq
enable_npu_irq:
    /* PicoRV32 专属指令：maskirq zero, a0 */
    /* a0 (x10) 存放 C 语言传进来的 mask 参数 */
    .word 0x0605600B
    ret

.global wait_for_irq
wait_for_irq:
    /* PicoRV32 专属指令：waitirq zero */
    /* CPU 将立刻挂起断电，直到任何未被 Mask 的中断发生 */
    .word 0x0800400B
    ret