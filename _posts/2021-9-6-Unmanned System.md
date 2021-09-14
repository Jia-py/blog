---
layout:     post
title:      "notebook - Unmanned System"
date:       2021-09-06 19:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - HKU CS
---

# introduction

what is robots

# Transform

## Linear Algebra

unit vector: [0...1...0...] just one element equals one

identity matrix: 单位矩阵

Vectors: arrays of numbers, represent a point in a *n* dimensional space

linear (in)dependence: 如何b能被 
$$
\mathbf{b}=\sum_{i} k_{i} \cdot \mathbf{a}_{i}
$$
该式子表示，则为dependence

Linear Systems: 线性空间？一系列的等式，解一个矩阵。

## Coordinate transform

why? because sensor observation and robot control is in robot's local coordinate, but action takes in the global coordinate.

### 坐标系转换

将global coordinate的**坐标系**转换为local coordinate (I->B)，围绕Z轴旋转：
$$
{ }_{B}^{I} R(\theta)=\left[\begin{array}{ccc}\cos \theta & -\sin \theta & 0 \\ \sin \theta & \cos \theta & 0 \\ 0 & 0 & 1\end{array}\right]
$$

$$
\begin{equation}\left[\begin{array}{lll}X_{B} & Y_{B} & Z_{B}\end{array}\right]=\left[\begin{array}{lll}X_{I} & Y_{I} & Z_{I}\end{array}\right]{ }_{B}^{I} R(\theta)\end{equation}
$$

其中，B为local coordinate，I为global coordinate，**XI，YI，ZI均为三维的向量**。

![image-20210913203058961](https://cdn.jsdelivr.net/gh/Jia-py/blog_picture/21_9/image-20210913203058961.png)

以此类推可得到围绕X轴与Y轴旋转的转换矩阵。
$$
\begin{equation}R_{2}(\beta)=\left[\begin{array}{ccc}1 & 0 & 0 \\ 0 & \cos \beta & -\sin \beta \\ 0 & \sin \beta & \cos \beta\end{array}\right]\end{equation}
$$

$$
\begin{equation}R_{3}(\gamma)=\left[\begin{array}{ccc}\cos \gamma & 0 & \sin \gamma \\ 0 & 1 & 0 \\ -\sin \gamma & 0 & \cos \gamma\end{array}\right]\end{equation}
$$

### 点转换

$$
\begin{equation}{ }_{B}^{A} R \times{ }^{B} p={ }^{A} p\end{equation}
$$

### Rotate

how to rotate a vector? just like the point transform
$$
\begin{equation}{ }^{A} p^{\prime}={ }_{B}^{A} R \times{ }^{A} p\end{equation}
$$
**位置坐标旋转变化时，相对于local coordinate，是右乘R；相对于global coordinate，是左乘R。**

