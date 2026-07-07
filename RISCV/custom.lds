/* 将入口点设置为我们汇编里的 _start 标签 */
ENTRY(_start)

/* 定义物理内存布局 (起始于 0x0 的 1MB SRAM) */
/* SRAM 为 1MB */
MEMORY {
    SRAM (rwx) : ORIGIN = 0x00000000, LENGTH = 1024K
}

SECTIONS
{
    /* .text 段：代码和指令 */
    .text :
    {
        /* 确保启动汇编被放在最开头 (即 0x0 地址) */
        *(.text.init)
        *(.text)
        *(.text.*)
    } > SRAM

    /* .rodata 段：只读数据 (你的 test_vector.h 里的权重全存在这里！) */
    .rodata :
    {
        . = ALIGN(4);
        *(.rodata)
        *(.rodata.*)
    } > SRAM

    /* .data 段：已初始化的变量 */
    .data :
    {
        . = ALIGN(4);
        *(.data)
        *(.data.*)
        *(.sdata)
    } > SRAM

    /* .bss 段：未初始化的变量 (清零) */
    .bss :
    {
        . = ALIGN(4);
        _bss_start = .;
        *(.bss)
        *(.bss.*)
        *(.sbss)
        _bss_end = .;
    } > SRAM

    /* 将栈顶定义在 1MB SRAM 的最末端 */
    /* 这个符号 (_sp) 会被 start.S 使用 */
    _sp = ORIGIN(SRAM) + LENGTH(SRAM);
}