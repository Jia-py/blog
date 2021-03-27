---
layout:     post
title:      "jupyter使用conda虚拟环境 & conda install 超时解决"
subtitle:   "Use conda env in jupyter & Solve CondaHTTPError"
date:       2021-03-23 16:37:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - Conda
    - Jupyter
---

# 背景

在Conda中创建了Pytorch的虚拟环境，那么如何在jupyter中运用呢？在conda中`conda install`时有时会遇到`CondaHTTPError`的错误，下文展开如何解决

## 方法

1. 解决虚拟环境的问题。

   * 简单粗暴的方法就是装conda插件`conda install nb_conda`即可，在jupyter notebook中选择kernel时选择你的虚拟环境就可以了。

   * **比较好的方法**是如下

   * ```python
     #创建虚拟环境,pytorch为环境名
     conda create -n pytorch python=3.7
     #在该环境中安装ipykernel
     conda install -n pytorch ipykernel
     #添加kernel，第一个pytorch为环境名；第二个为在jupyter中显示的名字，可修改
     python -m ipykernel install --user --name pytorch --display-name "pytorch"
     ```

     

2. 解决conda install超时问题。

   添加镜像

   ```
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
   conda config --set show_channel_urls yes
   ```

   如果这时依然无效，则在`C:\Users\Administrator\.condarc`文件中将加入的channel的`https`都改为`http`即可。

