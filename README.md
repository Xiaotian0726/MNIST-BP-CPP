# MNIST-BP-CPP
C++实现BP神经网络识别手写数据集MNIST

## 数据集
60000训练集，10000测试集。

每幅图像为28*28的灰度图像，单个像素值范围0~255，作为输入前需要进行/=255的归一化处理。每个真实值标签是一个0~9的整数值，读入后类型转换为double型。


## BP神经网络架构
784个输入层单元，30个隐藏层单元，10个输出层单元。两两之间全连接。

w1[i][j]：第i个输入单元到第j个隐藏单元的权重

bias1[j]：第j个隐藏单元的偏置

w2[j][k]：第j个隐藏单元到第k个隐藏单元的权重

bias2[k]：第k个隐藏单元的偏置

激活函数：sigmoid

损失函数：1/2均方误差

训练方法：单图梯度下降


## 备注
梯度计算部分较为复杂，可自行打草稿验证

运行程序时若无故出现长时间未响应的情况，可按Enter继续。目前出现这种情况的原因未知。
