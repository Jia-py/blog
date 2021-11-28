---
layout:     post
title:      "notebook - Visualization and visual analysis"
date:       2021-09-03 19:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - HKU CS
---

# introduction

data literacy : ability to convert data to information

people can understand the image data **parallelly**

地理可变单元问题(高松老师之前提及过)，也是一个可视化问题

可视化图像的设计师可以决定图像所代表的观点，甚至是相反的观点。

Why Visualization is important?

* **comprehend** huge amounts of data
* new insight
* reveal problems quickly (**anomalies**)
* facilitates hypothesis formation

Lie factor: 图中表示长度、面积或其他表示数值的变化 / 实际数值的变化

Computer Vision 与 Computer Graphics difference? maybe the same as the difference between information and data?

--

# Perception & Color

what is perception? the process we interpret the world.

try not to use too much conjunctions

find a symbol with a line is easier than without a line.

opponent color: black-white red-green yellow-blue

**additive** color,RGB ;**subtractive** color, CMY

for example, C(品红) in CMY can absorb Green and Blue, and reflect Red. 

**Visit https://www.zhihu.com/question/22839343/answer/137530508**

stronger pre-attentive cues by text than by color

color scales:

* qualitative scalle: 不同类别的颜色
* sequential scale: 单一颜色，不同深浅
* diverging scale: two sequential scale

color blindness

* rule1: maintain sufficient value contrast
* rule2: reinforce color encoding with position, size, shape

# Temporal, Geospatial & Multivariate Data

temporal data: set of values that change over time

Dimension Reduction

Dimension Ordering

## Geospatial data

map projection: must have distortions

- Conformal: 等角的
- Equal area:等面积
- Equidistance:等距的

# Trees

## graphs

* degree = the number of edges incident to it = the number of neighboring vertices
* adjacency matrix is O(n^2) space
* there are two ways to represent a graph, adjacency list(1--> 2,3) and edge list(1,2,weight).

## Trees

directed acyclic graph

In general, tree layout can be done efficiently in O(n) or O(nlogn) time

* Problem of Node-Link Diagrams
  * Tree breadth may grow exponentially
  * easily run out of space
* solution
  * filtering, clustering, interactions, distortion

Hyperbolic Layout

* A distorted view of a tree so that the region in focus has more space for layout
* 如果我们想获取到另一个地区比较准确的投影，那么要滑动原来的draw的点，其实其含义就是地理投影中有的要分区，不同的区使用不同的投影，不断转动地球，也就是滑动地区的意思。

