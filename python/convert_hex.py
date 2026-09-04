import sys

def convert_hex(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 使用字典模拟真正的物理 SRAM，键为物理字节地址，默认值为 '00'
    memory = {}
    current_byte_addr = 0
    max_addr = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('@'):
            # 捕获 objcopy 吐出的物理基地址
            current_byte_addr = int(line[1:], 16)
        else:
            tokens = line.split()
            for token in tokens:
                # 将字节精确地存入对应的物理地址
                memory[current_byte_addr] = token
                if current_byte_addr > max_addr:
                    max_addr = current_byte_addr
                current_byte_addr += 1

    # 计算一共需要输出多少个 32-bit Word (向上取整)
    max_word_addr = max_addr // 4

    with open(output_file, 'w') as f:
        # 严格按照物理地址递增，每 4 个字节打包成一个 Word
        for i in range(max_word_addr + 1):
            base = i * 4
            
            # 如果链接器在这个地址留了空白，.get() 会自动用 '00' 填充 (保留 Padding)
            b0 = memory.get(base + 0, '00')
            b1 = memory.get(base + 1, '00')
            b2 = memory.get(base + 2, '00')
            b3 = memory.get(base + 3, '00')
            
            # 小端序：高地址放在高位
            word_32bit = b3 + b2 + b1 + b0
            f.write(word_32bit + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_hex.py <input.hex> <output.hex>")
        sys.exit(1)
    
    convert_hex(sys.argv[1], sys.argv[2])
    print(f"Successfully packed {sys.argv[1]} into 32-bit words in {sys.argv[2]}")