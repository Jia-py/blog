---
layout:     post
title:      "What is transformer?"
subtitle:   
date:       2021-12-28 23:00:00
updatedate:
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: true
tags:
    - CS
    - Deep Learning
---

# Abstract

目前大多模型做序列数据都是使用的循环神经网络或者卷积神经网络。

而这篇文章提出了Transformer，一个简单的结构，只依赖于注意力机制。

# Conclusion

Transformer是第一个做序列转录的模型，只依赖注意力。

# Introduction

lstm，rnn，encoder-decoder

RNN的缺陷：

1. 如果有100个词，我们必须计算100步，是不能并行的。
2. 历史信息是一步一步往后传递的，早期的时序信息在后期会被淡化。

Attention已经有了较多应用

# Background

用卷积神经网络对长序列难以建模，一般都是小窗口。但使用注意力机制的话，一次可以看全部像素。

multi-head attention，可以具有多输出通道。

transformer是第一个依赖于self-attention来做encoder到decoder的模型。

# Model Architecture

一般的encoder-decoder模型是首先用encoder将一个向量$$(x_1,...,x_n)$$转换为$$(z_1,...,z_n)$$，每个z是一个向量，如$$z_1$$向量表示$$x_1$$。然后再用解码器生成输出序列$$(y_1,...,y_m)$$。At each step the model is auto-regressive, 即y1输出后，当作输入去生成y2，然后再将y1与y2输入去生成y3。

## Encoder

<img src="https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211228234304311.png" alt="image-20211228234304311" style="zoom:67%;" />

在文章中设定了有6（N）个layer的堆叠，每个layer中有两个sub-layer。第一个sublayer叫multi-head self-attention，第二个sublayer是一个简单的MLP。均用了残差连接，即将子层的输入和输出结合，但followed了一个`layer normalization`。每个sub-layer的输出可以表示为$$LayerNorm(x+Sublayer(x))$$。

## LayerNorm

<img src="https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211228235456193.png" alt="image-20211228235456193" style="zoom:67%;" />

三张图依次为，上方，三维数据的batchnorm（蓝色）与layernorm（棕色），左下二维数据的batchnorm，右下，二维数据的layernorm。

## Decoder

<img src="https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211229130056795.png" alt="image-20211229130056795" style="zoom:67%;" />

除去encoder有的两个子层，还有第三个子层。也使用了残差连接，layernorm。

因为在t时刻我们只能看到t时刻之前的数据用于训练解码器，因此解码器中是`Masked Multi-Head Attention`，将t时刻之后的数据遮盖。

## Scaled Dot-Product Attention

一个attention function是a query and a set of key-value pairs to an output.

query与哪个key更相似，相应的value的权重会高一些，将value加权求和。

transformer中使用的注意力机制叫做Scaled Dot-Product Attention。

<img src="https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211229131134769.png" alt="image-20211229131134769" style="zoom: 67%;" />

首先Q与K作矩阵乘法，计算每个key与Q的相似度，然后通过除以根号dk（dk为Q与K的特征维度）进行scale，再通过softmax计算各个key赋多少权重，最后使V加权。

mask环节，将t时刻即以后的计算出的scale后的值都赋一个极大的负数，这样进入softmax后会得到接近于0的权重，那么kt,kt+1,...将对结果没影响。

## Multi-Head Attention

<img src="https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211229132023159.png" alt="image-20211229132023159" style="zoom:67%;" />

先将V,K,Q都投影到较低维度，经过注意力函数后聚合。

为什么要这样做呢，因为单使用Scaled Dot-Product Attention，没有什么可以学习的参数，所以引入一些线性层可以学习一些参数用来捕获不同的模式。

## Positional Encoding

如何引入时序呢？

把每个query标一个数字，即他们的顺序，1，2，3，把这个数字也当作输入引入。