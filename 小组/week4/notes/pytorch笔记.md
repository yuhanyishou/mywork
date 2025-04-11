# pytorch的相关基本知识
## （1）张量（Tensors）
相当于多维数组，可以用到GPU上加速计算
### 1.导入相关的库
```py
from __future__ import print_function
import torch
```
### 2.声明和定义
#### 声明一个张量
```py
#创建出来的矩阵形式为五列三行
x=torch.empty(5,3)  #未初始化
x_rand=torch.rand(5,3)   #随机初始化 
x_zeros=torch.zeros(5,3,dtype=torch.long)    #零矩阵
#下面直接传递数值创建
tensor1=torch.tensor([5,3])  #输出为（[5,3]）
```
#### 获取tensors尺寸大小
```py
print(tensor1.size())
```
#### 修改尺寸的大小   torch.view() 
```py
# x为四行四列
x=troch.randn(4,4)
# y为一行十六列
y=x.view(16)
z=x.view(-1,8)  #8为给定的维度，另一个维度通过计算得到（16/8=2）
```
### 3.基本的运算
#### 加法：
1. 直接用“+”符号相加
2. 用"add"
```py   
torch.add(tensor1,tensor2,out=tensor3)
```
##### 取部分的数据
使用索引进行访问
```py
# 访问第一列的数据
print(tensor1[:,0])
```
注：若Tensor中只有一个元素，可用x.item()来获取
### 4.和numpy的转化
#### tensor转为numpy:  tensor.numpy()
```py
a=torch.ones(5)
b=a.numpy   #b转为了numpy形式
```
注：在这种情况下，a若发生变化，b也会跟着变
#### numpy转为tensor:     torch.from_numpy(numpy_array)
```py
a=np.ones(5)
b=torch.from_numpy(a)
```
### 5.cuda张量  
用.to 的方法把tensor转到不同的设备上
```py
device = torch.device("cuda")  #定义一个cuda设备对象
x=x.to(device)      #转到cuda上
```
## （2）autograd库
提供了计算梯度的功能（对 Tensors 上所有运算操作的自动微分功能）
### 1.追踪计算
```py
x=torch.ones(2,2,requires_grad=True) #requires_grad=True能够对变量的相关的计算进行追踪
```
接下来若对该张量进行计算，则结果后会带上一个“grad_fn=”的标记。例：
```py
y = x + 2
print(y)
#输出的结果：
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
```
### 2.计算梯度
对于一个张量x，当对x进行一些运算时，计算后的得到的张量会带有一个标记（与用到的运算符号有关），最终得到的结果设为y。运算结束后，调用y.backward(x)，则能够求出y相对于x的梯度，即dy/dx。例：
```py
x = torch.randn(3, requires_grad=True) #当设置requires_grad=True时能够对每个运算过程进行追踪，从而在反向传播时能够得到梯度
y = x * 2
y.backward(x)
print(x.grad)     #梯度保存在了grad中
```
