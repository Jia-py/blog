---
layout:     post
title:      "jupyter notebook插件推荐"
subtitle:   "jupyter notebook extensions recommendation"
date:       2021-06-04 21:32:00
author:     "Jpy"
header-img: "img/post-bg-arthur.png"
tags:
    - Jupyter
---
# 前言

jupyter插件是用nbextensions这个包实现的，具体的安装步骤可以参考网上资源。

# 推荐

## Scratchpad

在jupyter的应用过程中，我们有时会想着敲几行代码试着运行一下，但这些代码呢不是我们的最终版本。

那么在notebook里面写了还得删去非常麻烦，`scratchpad`可以解决这个问题，只需要在右下角点一下，或者用快捷键 `ctrl+b`，就有一块独立的空间用来敲代码可视化了。

## Variable Inspector

非常简单粗暴，可以看到所有变量的值

## Execute time

能够看到每一个单元格的运行时间

但我其实还是更喜欢用tqdm关于notebook的拓展，tqdm_notebook，因为对jupyter专门做了美化，非常美观。
