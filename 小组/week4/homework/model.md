```python
'''
下面的代码用来构成一个resnet网络
'''
#导入用的到的库
import torch
import torch.nn as nn
import PIL
import os
import torch.optim as optim
from torchvision import transforms, datasets
from PIL import Image 
```

```python
#定义残差结构
class BasicBlock(nn.Module):
    #判断残差结构中主分支的卷积核个数是否变化，默认为1
    expansion=1
    
    #初始化，声明各层定义
    def __init__(self,in_channel,outchannel,stride=1,downsample=None,**kwargs):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        #批量归一化
        self.bn1=nn.BatchNorm2d(out_channel)
        #设置激活函数
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.downsample=downsample
```


```python
#定义向前传播过程，描述了各层之间的联系
    def forward(self,x):
        #残差快保留原始输入
        identity=x
        #如果是虚线残差结构则采样
        if self.downsample is not None:
            identity=self.downsample(x)
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        #主分支和侧枝数据相加
        out=out+identity
        out=self.relu(out)
        return out
```


```python
#定义resnet类
class ResNet(nn.Module):
    def __init__(self,block,blocks_num,num_classes=1000,include_top=True,groups=1,width_per_group=64):
        super(ResNet,self).__init__()
        self.include_top=include_top
        #输出通道为64，残差结构输入通道为64
        self.in_channel=64
        self.groups=groups
        self.width_per_group=width_per_group
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #block:定义的两种残差模块，block_num：模块中的残差块的个数
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            #平均池化
            self.avgpool=nn.AdaptiveAvgPool2d((1, 1))
            #全连接层
            self.fc=nn.Linear(512*block.expansion,num_classes)
            #遍历网络中每一层
            for i in self.modules():
                '''
                isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
                '''
                if isinstance(m,nn.Conv2d):   #如果是卷积层
                    nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
```


```python
#定义残差模块
    def _make_layer(self,block,channelblock_num,stride=1):
        downsample=None
        #若满足条件，则为虚线残差结构
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample=nn.Sequential(nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                       nn.BatchNorm2d(channel * block.expansion))
        layers=[]
        layers.append(block(self.in_channer,channel,downsample=downsample,stride=stride,groups=self.groups,width_per_group=self.width_per_group))
        self.in_channel=channel*block.expansion
        for _ in range(1,block_num):
            layers.append(block(self.in_channer,channel,downsample=downsample,stride=stride,groups=self.groups,width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
```


```python
#定义前向传播的输入处理
#静态层
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #动态层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

def resnet(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
     
model = resnet(num_classes=1000) 
model = model.to(device)
```


```python
'''
下面的代码用来训练模型
'''
# 转到GPU训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

```


```python
data_transform={
    #训练
    "train":transform.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    #验证 
    "val":transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

#定义一个函数用来读取二进制的数据集
class CIFAR10BinDataset(Dataset):
    def __init__(self, bin_file, transform=None):
        self.data = self._load_bin(bin_file)
        self.transform = transform
    def _load_bin(self, path):
        with open(path, 'rb') as f:
            raw_data = np.frombuffer(f.read(), dtype=np.uint8)
        raw_data = raw_data.reshape(-1, 3073)
        labels = raw_data[:, 0]
        pixels = raw_data[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return {'images': pixels, 'labels': labels}
    def __getitem__(self, idx):
        image = self.data["images"][idx]  # numpy数组格式的图像 (32, 32, 3)
        label = int(self.data["labels"][idx])
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
    def __len__(self):  
        return len(self.data["images"])
        return image, label

#读取数据集
data_root = os.path.abspath(os.getcwd())
bin_file = r"C:\Users\asus\Desktop\工作室学习\数据资料\cifar10\cifar-10-batches-bin\data_batch_1.bin"
assert os.path.exists(bin_file),"文件路径错误"
train_dataset = CIFAR10BinDataset(bin_file, transform=data_transform["train"])
#训练集长度
train_num = len(train_dataset)

```


```python
#确定一次训练的样本数量
batch_size = 200
train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=False,
                        num_workers=0)
#定义损失函数（交叉熵损失）
loss_function = nn.CrossEntropyLoss()
#定义优化器
optimizer = optim.Adam(net.parameters(), lr=0.0001)
#定义迭代次数
epochs = 4
#开始训练：
for epoch in range(epochs):
    net.train()
    for images,labels in train_loader:
        #梯度清零：
        optimizer.zero_grad()
        #前向传播
        images = images.to(device)
        labels = labels.to(device)
        #计算训练值
        logits = net(images.to(device))
        #计算损失
        loss = loss_function(logits, labels.to(device))
        #计算梯度
        loss.backward()
        optimizer.step()

```



