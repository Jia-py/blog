---
layout:     post
title:      "安装Pytorch以及配置远端Jupyter"
subtitle:   "Pytorch and Jupyter"
date:       2021-03-23 14:47:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
# 生活，工作，笔记（个人理解消化心得），文档（方便后续查阅的资料整理），其他
tags:
    - 文档
---
# 本地PyTorch

若只需要使用CPU版本 参考：[windows下安装pytorch教程 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/130524345)

## 安装anaconda

因为我之前有安装，所以这里直接用即可

```
conda create -n pytorch python=3.6
conda activate pytorch
```

## 安装CUDA Toolkit（若要用GPU版本需要安装）

网址：[CUDA Toolkit 11.2 Update 2 Downloads NVIDIA Developer](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)

建议下载 `exe`，复制下载链接后用 `迅雷`下载会很快。

## 安装cuDNN（若要使用GPU版本需要安装）

网址：[cuDNN Download NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-download)

下载后是一个压缩文件夹，解压文件夹，将里面的 3 个文件夹：bin，include，lib 里面的内容分别放入 CUDA 安装位置的对应文件夹。

CUDA 安装的位置在：

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0

## 安装PyTorch

注意首先在 `C:\Users\user\.condarc`文件中添加镜像

```
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
show_channel_urls: true
```

网址：[Start Locally PyTorch](https://pytorch.org/get-started/locally/)

```
conda install pytorch torchvision torchaudio cudatoolkit=11.1
```

输入代码请删去-c及其后的字符串，其含义为使用pytorch官方下载源，但我们此处需要使用国内镜像。

根据网站选择CUDA版本，安装信息等等。CUDA版本可以在 `NVIDIA设置-系统信息-组件-NVCUDA 后的产品信息`中查看。

搞了半天，到凌晨了，anaconda删了重装，终于好了！

![](https://cdn.jsdelivr.net/gh/Jia-py/blog_picture/21_3/Snipaste_2021-03-23_00-32-42.jpg)

## 安装pytorch geometric

[rusty1s/pytorch_geometric: Geometric Deep Learning Extension Library for PyTorch (github.com)](https://github.com/rusty1s/pytorch_geometric)

```
conda install pytorch-geometric -c rusty1s -c conda-forge
```

## jupyter使用虚拟环境

```
conda install -n pytorch ipykernel
python -m ipykernel install --user --name pytorch --display-name "pytorch"
```

## 安装tqdm出现IProgress错误

```python
conda install -n base -c conda-forge widgetsnbextension
conda install -n py36 -c conda-forge ipywidgets
```

最后启动IDE

# Linux服务器安装pytorch

安装conda，这一步实验室一般都已经安装好了，此处略过，网络上资源也很多。

因为我是在师兄的conda中新建了环境直接做的，cudatookit以及cuDNN的安装并未接触，此处也需要各位自行搜索了。

## 安装pytorch

基本步骤和本地安装pytorch是相似的，但要注意的是服务器的Cuda版本，安装的pytorch要与cuda版本对应。

## 安装pyg

```
conda install pytorch-geometric -c rusty1s -c conda-forge
```

## 替换源

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

## 删除源改为默认源
conda config --remove-key channels
```

如果存在服务器上下载超时的包，可以根据所显示的链接先保存到本地再上传至服务器文件夹中，通过`conda install package`的方式安装

## 查看端口占用

```bash
lsof -i:端口号
kill -9 PID
```

## 使用本地浏览器连接远端服务器的jupyter

在linux服务器开启jupyter后，查看端口等情况，一般为8888

可以直接在本地电脑浏览器输入服务器的`公网ip:8888`访问，也可以通过xshell等软件监听端口转发访问到。

## 保持服务器进程在退出shell后不关闭

使用命令`nohup 命令 &`

nohup是在退出后不关闭，&命令作用为后台运行

需要关闭服务的话则输入`ps`，通过`kill -9 PID`的方式关闭服务

**需要特别注意的是退出时使用`exit`，不要直接将xshell软件关闭，否则服务仍将关闭**

查看后台程序，如jupyter

```bash
ps -aux|grep jupyter
```

## 查看jupyter token

有时候忘记了打开的jupyter的token，可以通过以下语句查看

```
jupyter notebook list
```

# Pytorch基础教程

主要参考：[Learn the Basics — PyTorch Tutorials 1.8.0 documentation](https://pytorch.org/tutorials/beginner/basics/intro.html)  这里官方的教程都挺好的，另外有API不知道的可以在Doc里面查询

中文教程：[Pytorch中文教程](https://pytorch.apachecn.org/docs/1.7/)

## Tensor

张量，与Numpy的ndarrays很相似，但他可以自动计算梯度以及使用GPU加速。

## Load Data

Pytorch有两种载入数据方式：`torch.utils.data.DataLoader`与 `torch.utils.data.Dataset`

## Transforms

`torchvision.transforms`可以将不符合训练要求的数据转为符合要求的数据

pytorch需要标准化张量的特征，以及one-hot编码的张量作为标签。

## Build the Neural Network

这些模块都在 `torch.nn`中

```python
class NeuralNetwork(nn.Module):
	def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() #flatten相当于把单张图像铺平为一个一维数组
        self.linear_relu_stack = nn.Sequential( # 模型的容器，数据按照顺序输入
            # 填具体的结构
            nn.Linear(28*28, 512), #输入28*28的是数据,输出512位的feature
            nn.ReLU(),# 非线性激活，在输入和输出之间创建复杂映射
        )
    def forward(self,x):
        # 前向方法
```

```python
model = NeuralNetwork().to(device)
```

最后经过 `nn.Softmax()`层，输入logits，logits是一个[0,1]范围的数组，表示模型预测的为每一类的可能性。

## Autograd

`torch.autograd`

计算前需要设定所需计算梯度的tensor属性，`requires_grad=True`或 `x.requires_grad_(True)`

声明的张量在 `loss.backward()`之后调用 `张量.grad()`即可获得对应梯度。

## Optimize

Hyperparameters：Number of Epochs, Batch Size, Learning Rate

Optimization Loop:The Train Loop, The Validation/Test Loop

```python
loss_fn = nn.CrossEntropyLoss() #定义损失函数
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## save and load the model

```python
import torchvision.models as models
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(),path) # 保存模型权重
torch.save(model,path)

model = torch.load(path)
model = models.vgg16()
model.load_state_dict(torch.load('path'))
model.eval() # 重要的步骤
```
