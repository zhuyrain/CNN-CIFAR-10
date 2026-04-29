#include "my_printf.h"
#include <stdarg.h>
// #include <stdio.h>
#include <stdint.h>


// 定义一个指向外部仿真器假串口地址的挥发性指针
// 将地址改到物理 SRAM 范围内，借用 SRAM 的硬件握手逻辑
// 强制使用确定位宽的指针，避免任何对齐或总线截断问题
volatile uint32_t * const UART_TX = (uint32_t *)0x000E0000;

// 自定义的最简打印函数
void my_putc(char c) {
    *UART_TX = c; // 往这个地址写数据，Testbench 就会在终端打印出来
}

// 保持你原有的 my_putc 不变
// volatile unsigned int * const UART_TX = ...
// void my_putc(char c) { ... }

// 封装一个裸机版的 my_printf
void my_printf(const char *format, ...) {
    va_list args;
    va_start(args, format);

    while (*format) {
        if (*format == '%') {
            format++;
            if (*format == 'd') {
                int n = va_arg(args, int);
                // 这里的逻辑就是你之前写的 my_print_int，直接内联进来
                if (n == 0) { my_putc('0'); }
                else {
                    if (n < 0) { my_putc('-'); n = -n; }
                    char buf[16]; int i = 0;
                    while (n > 0) { buf[i++] = (n % 10) + '0'; n /= 10; }
                    while (i > 0) { my_putc(buf[--i]); }
                }
            } else if (*format == 's') {
                char *s = va_arg(args, char *);
                while (*s) { my_putc(*s++); }
            } else if (*format == 'c') {
                char c = (char)va_arg(args, int);
                my_putc(c);
            } else if (*format == '%') {
                my_putc('%');
            }
        } else {
            my_putc(*format);
        }
        format++;
    }
    va_end(args);
}
