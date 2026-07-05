#include "my_printf.h"
#include <stdarg.h>
#include <stdint.h>

// 定义硬件寄存器绝对物理地址
#define AXI_UARTLITE_BASE 0x80000000
#define UART_TX_FIFO      (*(volatile uint32_t*)(AXI_UARTLITE_BASE + 0x04))
#define UART_STATUS       (*(volatile uint32_t*)(AXI_UARTLITE_BASE + 0x08))

void my_putc(char c) {
    // 【核心防丢包逻辑】：轮询状态寄存器的 Bit 3
    // 如果 TX FIFO 是满的 (UART_STATUS & 0x08 为真)，CPU 就死循环死等！
    while (UART_STATUS & (1 << 3)) {
        // do nothing, just wait...
    }
    
    // 等到 FIFO 有空位了，再把字符塞进去
    UART_TX_FIFO = (uint32_t)c;
}

// 封装一个裸机版的 my_printf (支持 %d, %s, %c, %x, %X 以及 %08x 等前导零补齐)
void my_printf(const char *format, ...) {
    va_list args;
    va_start(args, format);

    while (*format) {
        if (*format == '%') {
            format++;
            
            // ====================================================
            // [新增] 解析前导零和宽度，例如 "%08x"
            // ====================================================
            int pad_zero = 0;
            int pad_width = 0;
            
            if (*format == '0') {
                pad_zero = 1;
                format++;
            }
            while (*format >= '0' && *format <= '9') {
                pad_width = pad_width * 10 + (*format - '0');
                format++;
            }

            // ====================================================
            // 原有格式化逻辑及扩展
            // ====================================================
            if (*format == 'd') {
                int n = va_arg(args, int);
                if (n == 0) { 
                    int pad = pad_width > 1 ? pad_width - 1 : 0;
                    while(pad_zero && pad-- > 0) my_putc('0');
                    my_putc('0'); 
                } else {
                    if (n < 0) { my_putc('-'); n = -n; }
                    char buf[16]; int i = 0;
                    while (n > 0) { buf[i++] = (n % 10) + '0'; n /= 10; }
                    int pad = pad_width > i ? pad_width - i : 0;
                    while(pad_zero && pad-- > 0) my_putc('0');
                    while (i > 0) { my_putc(buf[--i]); }
                }
            } 
            // [新增] 16进制打印支持 (支持小写 x 和大写 X)
            else if (*format == 'x' || *format == 'X') {
                uint32_t n = va_arg(args, uint32_t); // 地址/寄存器通常视为无符号数
                if (n == 0) {
                    int pad = pad_width > 1 ? pad_width - 1 : 0;
                    while(pad_zero && pad-- > 0) my_putc('0');
                    my_putc('0');
                } else {
                    char buf[16]; int i = 0;
                    int is_upper = (*format == 'X');
                    while (n > 0) {
                        int rem = n % 16;
                        buf[i++] = (rem < 10) ? (rem + '0') : (rem - 10 + (is_upper ? 'A' : 'a'));
                        n /= 16;
                    }
                    int pad = pad_width > i ? pad_width - i : 0;
                    while(pad_zero && pad-- > 0) my_putc('0');
                    while (i > 0) { my_putc(buf[--i]); }
                }
            } 
            else if (*format == 's') {
                char *s = va_arg(args, char *);
                while (*s) { my_putc(*s++); }
            } 
            else if (*format == 'c') {
                char c = (char)va_arg(args, int);
                my_putc(c);
            } 
            else if (*format == '%') {
                my_putc('%');
            }
        } else {
            my_putc(*format);
        }
        format++;
    }
    va_end(args);
}