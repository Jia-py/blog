---

layout:     post
title:      "Earth_Data_Diamond云计算"
subtitle:   "Diamond cloud computing"
date:       2021-02-07 17:33:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - Study
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

