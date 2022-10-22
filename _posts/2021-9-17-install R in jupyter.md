---
layout:     post
title:      "Install R in jupyter"
date:       2021-09-17 17:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
# 生活，工作，笔记（个人理解消化心得），文档（方便后续查阅的资料整理），其他
tags:
    - 文档
---

# Background

For the need of FITE7410 Financial Fraud analytics, Students need to use R to analysis data. 

My environment

* installed: miniconda, jupyter

# Process

## method 1(recommended)

1. install R

click the website [The Comprehensive R Archive Network (r-project.org)](https://cran.r-project.org/) download and install R.

2. find location of R.exe (D:\Program Files\R\R-4.1.1\bin)
3. go to this location in anaconda prompt
4. run R in the location (directory) in the prompt
5. run the following codes in R started from the prompt

* install.packages('IRkernel')
* IRkernel::installspec()

complete!

## method 2

for I have not found a good way to use this R env in jupyter, the first method is recommended.

process in miniconda directly

```
conda create -n R4.1
conda activate R4.1
conda install r-base=4.1.1
```

complete!

conda安装R包有两种方式，一种是使用conda命令安装：conda install -c r package-name，需要注意的是conda下面的r包的名称与普通R包的名称不一样，具体名称可以在官网上面查询（http://docs.anaconda.com/anaconda/packages/r-language-pkg-docs/）；另外一种是直接进入conda下面的R交互界面，安装普通安装R包的方式进行安装，比如bioconductor或者install.packages方式。

