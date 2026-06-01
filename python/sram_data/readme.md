## 文件说明

### bias_readmemh.txt，image_readmemh.txt和weights_interleaved.txt是直接复制到sram中的数据

1. 用于和第一层conv层的前4输出通道计算结果进行比对
2. 已经足够用于生成第一层conv层的所有输出通道（共16通道）结果
3. 来源于../目录，生成后拖拽过来单独存放
