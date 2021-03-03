---

layout:     post
title:      "Earth Data Diamond 云计算"
subtitle:   "Diamond cloud computing"
date:       2021-02-07 17:33:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - Cloud
---

## 创建镜像文件

#### 创建requirements.txt文件

使用`pipreqs`，`pip install pipreqs`

cmd输入`pipreqs (your code path) --encoding=utf8 --force`生成requirements.txt文件。**只能用于.py文件，notebook把所有import提取出来创建成一个py文件也可用**

查看python包版本`cmd>> pip list`

> 可以删去requirements.txt后的版本数字默认下载最新版本

#### Ubuntu安装Docker

https://www.runoob.com/docker/ubuntu-docker-install.html

#### 创建Dockerfile

```
From python:3

WORKDIR /app

COPY . /app

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

#### 镜像相关操作

浏览镜像`docker images`

删除镜像`docker rmi -f (IMAGE ID)`

创建镜像`docker build -t (name):latest .`

登录华为云 `个人中心复制代码`

Docker改名`docker tag (your file name:latest) (Docker push 中的文件名)/(your file name:latest)  `

推送 `个人中心复制代码`

docker清理`docker container prune` `docker image prune` `docker volume prune` `docker builder prune` 或全清理`docker system prune`

## 华为云notebook环境配置

目前华为云还未支持在`开发环境`中加载自建镜像，因此只能在默认镜像中添加python库

### 查看python版本及支持安装包版本

```python
import sys
sys.version
#示例输出：'3.7.6 | packaged by conda-forge | (default, Jan  7 2020, 22:33:48) \n[GCC 7.3.0]'
```

```python
! pip debug --verbose
```

此处输出的内容中主要关注`Compatible tags`，只有符合这些命名方式的安装包才能被正常安装，否则会报错` is not a supported wheel on this platform.`

### 从[[PyPI ](https://pypi.org/)]上下载缺失的包

* 输入代码查看已安装包

```python
#查看是否已安装scikit-learn
! pip show scikit-learn
#查看所有已安装包
! pip list
#结合网页的ctrl+F效率较高
```

* 打开[PyPI ](https://pypi.org/)搜索包
* 选择包的具体版本
* 查看`Dependencies`(这些包都必须被安装)
* 点击Download files
* 选择与之前的**Compatible tags**所匹配的安装包，下载

### 华为云平台中安装包

* 将安装包上传至构建开发环境所选的文件夹中
* 打开开发环境，打开一个notebook
* 输入代码

```python
! pip install file_name.whl
```

* 等待出现`Successfully installed`字样，则完成安装

> 若出现http错误等字样。是因为当前所装包的Dependencies未被安装，需要先安装该依赖。如：`...(connect timeout=15)'/':/simple/threadpoolctl`，缺失了`threadpoolctl`包。

jpy 21/02/08

