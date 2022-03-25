---
layout:     post
title:      "HKU CS GPU Farm Notes"
subtitle:   "HKU CS GPU Farm 使用指南"
date:       2022-03-24 23:00:00
updatedate:
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: false
tags:
    - CS
    - HKU CS
---

以下使用体验基于Xshell

通过Xshell新建会话，设置账号密码，即可登录至服务器。

```bash
# 开启GPU模式
gpu-interactive
# 创建文件夹
mkdir filename
# remove 
rm -rf filename
# 清空cuda显存
torch.cuda.empty_cache()
# 查看cuda使用情况
watch -n 1 nvidia-smi
```

# 如何远程查看jupyter lab

```bash
# 设置jupyter密码
jupyter notebook password
conda activate pytorch
hostname -I
>>> 10.xx.xx.xxx
jupyter-lab --no-browser --FileContentsManager.delete_to_trash=False
>>> port:8888
```

再在xshell中新开一个会话，地址填`10.xx.xx.xxx`，账号密码同gpu farm账号，在隧道中添加一个转移

![image-20220324231202836](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220324231202836.png)

在本地浏览器打开`http://localhost:8888`即可

# 同时打开tensorboard与jupyter

```bash
tensorboard --logdir=log_path &
tensorboard --logdir='./train_output/runs/map_translation' &
# 在本地打开localhost:6006

jupyter-lab --no-browser --FileContentsManager.delete_to_trash=False
# 在本地打开localhost:8888

# 查看后台程序
ps -aux|grep jupyter
```

