## python子文件夹中各文件的作用

### cifar10_fp32_baseline.ipynb

1. 训练FP32下的baseline模型

### export_test_vector.ipynb

1. 导出量化参数文件test_vector.h
2. 完成对称量化过程（非泄露量化）

### test_scaled_model_accuracy.ipynb

1. 测试量化后模型在训练集上的正确率
2. 量化参数完全来源于test_vector.h

### convert_hex.py

1. 转换机器码的格式为一行32bits/4bytes数据
2. 用于在makefile文件中完成firmware.hex -> firmware_byte.hex

## 常用术语

1. 量化配置/量化参数/量化权重
2. 量化激活值/输入输出特征图/IFM/OFM
