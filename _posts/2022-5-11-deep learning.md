---
layout:     post
title:      "HKU notebook - deep learning"
subtitle:   ""
date:       2022-05-11 22:00:00
updatedate:
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: false
# 生活，工作，笔记（个人理解消化心得），文档（方便后续查阅的资料整理），项目，其他
tags:
    - 笔记
---

# 1A Motivation

P38 what triggered Deep Learning's success in recent years?

# 1B Linear Models

P6 when should you use ML?

P15 Linear Classification

P51 Logistic Regression

P70 multiclass classification & loss function

# 2A Artificial Neural Networks

P10 Perceptron

P13 activation functions

# 2B Artificial Neural Networks

P8 Backpropagation
$$
s_j^{(l)}=(\sum_{i}w_{ij}^{(l)}x_i^{(l-1)})+b \\
x_j^{(l)}=g(s_J^{(l)})
$$
P36 earlier layers 求导过程

第i层的梯度求法
$$
w_{ij}^{(l)} \rightarrow w_{ij}^{(l)} - \alpha x_i^{(l-1)} \delta _j^{(l)} \\
\frac{\partial J(W)}{\partial w_{ij}^{(l)}}=x_i^{(l-1)}\delta_{j}^{(l)} \\
\delta_{i}^{(l)} =  g'(s_i^{(l)})(\sum_j w_{ij}^{(l+1)}\delta_j^{(l+1)})
$$
P41 Output Layer 求导过程，常见**损失函数**求导
$$
w_{ij}^{(l)} \rightarrow w_{ij}^{(l)} - \alpha x_i^{(l-1)} \delta _j^{(l)} \\
\frac{\partial J(W)}{\partial w_{ij}^{(l)}}=x_i^{(l-1)}\delta_{j}^{(l)} \\
\delta_i^{(l)} = g'(s_i^{(L)})(\frac{\partial J}{\partial h(x)})
$$

# 3A Techniques to Improve Training

P10 梯度消失

P14 激活函数 not zero-centered 缺点

P16 Sigmoid, Tanh, ReLU优缺点

P21 Problem with initial Symmetry

P23 优化器

P43 split dataset

# 3B Further Techniques to Improve Training

regularization

dropout

early stopping

P32 Bias and Variance

# 4A CNN and Computer Vision

P29 CV发展史

P57 Convolutional Layer

P66 Padding

P79 计算parameters个数（CNN及dense layer）

P84 SIRENs

P89 why pooling

P102 计算维度，conv与pooling

conv，卷积后的维度为$$M \times M \times filter\_num$$
$$
M = ((N+2*padding)-F)/S + 1
$$
pooling与conv计算方式一致，只不过不改变通道数维度

P127 GoogleNet；计算operations；1*1卷积的作用, Inception

# 5A CNN Visualization+application

**P24 25 两个exam**

P34 Saliency（显著性） map and heat map，具体过程可见P48

P57 Object Localization

P64 Object Detection

P79 语义分割

# 5B DeepDream Adversarial Images

# 6A Adversarial Images

P22 Autoencoders

# 6B Deep Generative Models

P19 VAEs. given a distribution by comparing its input to its output 

VAE 与 Autoencoder的区别

P25 GAN

P38 when stop training

P39 Loss Function for Discriminator and Generator
$$
J = -[y \log h(x)+(1-y)\log (1-h(x))]
$$
Discriminator wants to minimize J, Generator wants to maximize J.

P42 Training of GAN

P43 Best Discriminator

P45 Problems in GAN

P58 VAEGAN C-GAN Pix2PixGAN

P66 不同的几种generator的架构；UNET Generator

P72 CycleGAN

# 7A RNNs

P43 RNN公式

P46 Pros and Cons of typical RNN

P69 Backpropagation in RNN

P75 LSTM
$$
\begin{gathered}
i_{t}=\sigma\left(W_{i} X_{t}+U_{i} h_{t-1}+b_{i}\right), \\
f_{t}=\sigma\left(W_{f} X_{t}+U_{f} h_{t-1}+b_{f}\right), \\
o_{t}=\sigma\left(W_{o} X_{t}+U_{o} h_{t-1}+b_{o}\right), \\
\tilde{c}_{t}=\operatorname{Tanh}\left(W_{c} X_{t}+U_{c} h_{t-1}\right), \\
c_{t}=f_{t} \odot c_{t-1}+i_{t} \odot \tilde{c}_{t}, \\
h_{t}=o_{t} \odot \operatorname{Tanh}\left(c_{t}\right) .
\end{gathered}
$$
P88 Bidirectional RNN

# 8 Machine Translation

P32 Machine Translation

P45 Alignment Probability

P57 Document vector

P65 Word2Vec

P74 Glove

P92 Evaluation of Word Vectors

P111 Transformer

# 9A Machine Translation(2)

P19 BERT

P23 Back translation

# 9B Deep Reinforcement Learning

P11 Reinforcement Learning

P32 Valued-based Learning 学习每个位置的value，根据value得到最优$$\pi$$
$$
\begin{aligned}
Q^{*}\left(s_{t}, a_{t}\right) &=E\left[r_{t}+\gamma \max _{a^{\prime}} Q^{*}\left(s_{t+1}, a^{\prime}\right)\right] \\
&=\sum_{s^{\prime}, r^{\prime}} P\left(s^{\prime}, r^{\prime} \mid s_{t}, a_{t}\right)\left[r^{\prime}+\gamma \max _{a^{\prime}} Q^{*}\left(s^{\prime}, a^{\prime}\right)\right] \\

v(s_t) &= \sum_{s'}\pi(s'|s_t) Q^{\pi}(s_t,a) \\

v^{*}\left(s_{t}\right) &=\max _{a} Q^{*}\left(s_{t}, a\right) \\
&=\max _{a} \sum_{s^{\prime}, r^{\prime}} P\left(s^{\prime}, r^{\prime} \mid s_{t}, a\right)\left[r^{\prime}+\gamma v^{*}\left(s^{\prime}\right)\right]
\end{aligned} \\

$$
P36 Policy Learning

Value Iteration，根据max函数计算出每个位置进行每个动作的Q值，通过找到max的Q值作为新的Value值。

Policy Iteration分为两步，先根据之前得到的policy，一个固定的路径，更新所有格子的value值。第二步，再根据不同的Value值更新Policy

![img](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204231128974.png)

P45 Q-learning

Exploitation: with probability 1-e use greedy action, i.e., 𝑎" = arg maxQ( s_t, 𝑎)
Exploration: with probability e use random action
$$
Q(s, a) \leftarrow {Q(s, a)+\alpha}\left[r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}\right)-Q(s, a)\right]
$$
P50 Deep Q-Network

P74 Double DQN

# 10A Deep Reinforcement Learning

# 10B Ethical Issues in AI