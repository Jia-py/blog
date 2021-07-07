---
layout:     post
title:      "pytorch 动态图实现思考"
subtitle:   "some thoughts about pytorch dynamical graph"
date:       2021-07-07 21:32:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - Pytorch
---

## 背景

暑期的科研涉及到了图网络的处理，想着使用动态图能够更好地反映人口流动的时间变化，所以研究了一下怎么在pytorch上实现动态图。

## 思考

主要的难点其实是在于，pytorch的一个batch是同时进行计算的，而我们的数据是每一条数据拥有一个新的图结构，如果在一个batch里让model生成多个图结构进行计算，这是最困难的地方。

一方面，首先考虑到可以使用pytorch geometric的mini batch试一试。在mini batch中，数据以data的方式存储，每一个data都拥有x与edge两个信息，也就是说，它将每条数据以及每一张图的结构进行了绑定，传入dataloader中的每一条数据都是有独立的图结构的。

另一方面，也可以考虑在原有的pytorch中的模型中进行改写，在forward函数中，根据batchsize的大小，对数据进行多次图卷积。

还有一种思考角度，是否就将batch_size改为1呢？直接随机梯度下降。但在训练中，随机梯度下降有时会引入噪音，很可能会使得模型不能收敛到局部最优解。

最终考虑到时间的紧迫与数据量的大小，还是决定不使用动态图结构了，希望能在以后的科研中尝试一下吧！