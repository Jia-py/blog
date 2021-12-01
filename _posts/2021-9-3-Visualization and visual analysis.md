---
layout:     post
title:      "notebook - Visualization and visual analysis"
date:       2021-09-03 19:00:00
updatedate: 
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
latex:      true
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

Lie factor: （图中表示长度、面积或其他表示数值的变化） / 实际数值的变化

Computer Vision 与 Computer Graphics difference? maybe the same as the difference between information and data?

# Perception & Color

what is perception? the process we interpret the world.

some features has a **pop-out effect**

conjunctive search: find a target with two features, generally not pre-attentive

stronger effects: Color, orientation, size, contrast, motion/ blinking

opponent color: black-white red-green yellow-blue

**additive** color,RGB ;**subtractive** color, CMY

for example, C(品红) in CMY can absorb Green and Blue, and reflect Red. 

**Visit https://www.zhihu.com/question/22839343/answer/137530508**

## Lightness VS Luminance

Lightness: 与周围物体比较，目标有多亮，一般用百分之几来表示

Luminance：是一个客观值，表示目标发光亮度

**stronger pre-attentive cues by text than by color**, 文本比颜色更容易引起注意

color scales:

* qualitative scale: 不同类别的颜色,for labeling
* sequential scale: 单一颜色，不同深浅, for indicating quantity
* diverging scale: two sequential scale

color blindness 色盲

* rule1: maintain sufficient value contrast
* rule2: reinforce color encoding with position, size, shape

# Temporal, Geospatial & Multivariate Data

## Basic Plots

* Bar Charts/ Histograms: 柱状图，x轴是一个个bin，代表一个区间
* Box Plots：高效地展示quantitative distribution of 1D data. Can highlight outliers.
* 2D Bar Charts: to show the joint distribution of the values of two variables.
* Line Graphs: 折线图
* Scatter Plots: 散点图 show the relationship between `two variables`
* Scatter Plot Matrices
* Contour Plots: 等高线图，用于绘制高程图，温度图，降雨图

## Time-Series Data

set of values that change over time

* index charts: 有一条可互动的直线，可以比较某一个x值下的各个y值

![image-20211129151350236](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211129151350236.png)

* stacked graphs：堆叠图。优势：useful for showing summation of time-series values. 局限：不支持负值；很难准确分析趋势；对于一些数据类型没啥意义（温度）
* Horizon Graphs：高于平均的为正值，低于平均的为负值。局限：不直观，有学习成本。
* Spiral Graphs：时间轴为螺旋型的图。可以展示数据的周期性结构

## Multivariate Data

* Heat Maps：一般列代表不同的样本，行代表一个基因等，观察样本在不同特征下的集聚情况。
* Parallel Coordinates：present n axes of n dimensions on a 2D plane. axes的顺序非常重要，可以帮助找出规律。临近的dimension的规律比远的更容易发现。

![image-20211129210337478](C:\Users\JPY\AppData\Roaming\Typora\typora-user-images\image-20211129210337478.png)

## Geospatial data

map projection: must have **distortions**

- Conformal: 等角的
- Equal area:等面积
- Equidistance:等距的

---

* Cylindrical Projection圆柱投影，equator附近没有distortion，poles附近severely distorted
  * Equirectangular Projection：等距矩形投影。经纬度都被投影成等间隔的。
  * Lambert Cylindrical Projection：Area Preserving

---

* Choropleth Maps：Use color to encode values for a region, 不同颜色代表value的大小
* Cartograms：地图的要素大小可以根据值的大小而变动
* Graduated Symbol Maps：分级符号图。在不同value的地块上有一个不同大小或颜色的符号

## Volume Data

体数据，(x,y,z)

# Trees

## graphs

Graphs are best for representing **relational** structures

* G = (Vertices, Edges)
* degree = the number of edges incident to it = the number of neighboring vertices. In_degree = 指向它的箭头, out_degree = 它指出的箭头
* adjacency matrix is O(n^2) space
* there are two ways to represent a graph, adjacency list(1--> 2,3) and edge list(1,2,weight).

---

Graph Drawing Requirements

* Drawing conventions：一些具体的绘制规则
  * straight-line drawing；polyline drawing；orthogonal drawing；planar drawing；grid drawing；upward drawing
* Aesthetics：提高readability
  * minimize crossings，total area，total edge length...
  * maximize angular resolution（两条边之间的最小角度）
* Constraints：adding knowledge about semantics（语义）to enhance readability
  * place a vertex close to the middle

## Trees

directed acyclic graph （有向无环图）

In general, tree layout can be done efficiently in O(n) or O(nlogn) time

---

树布局的方法

* Indentation 缩进：如电脑的文件夹展开列表

* Node-Link Diagrams：

  * 其中一种是radial trees，root在圆心的树
  * Cone Trees：一种3D树，子节点以圆柱形散开
  * Balloon Trees：子节点在父节点周围圆形散开，像Cone Trees的压扁版本，但没有重叠。
  * Problem of Node-Link Diagrams
    * Tree breadth may grow exponentially
    * easily run out of space

  * solution
    * filtering, clustering, interactions, distortion（在不同regions使用不同的aspect ratios）

* Hyperbolic Layout：双曲线布局，使关注的区域有更多的空间

* Layered Diagrams

![image-20211129224905915](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211129224905915.png)

* Enclosure/Containment Diagrams：在layered Diagrams基础上压缩了空间，用不同颜色表示子节点与父节点的关系
* Treemaps：递归地把区域割分为长方形的子区域
  * 优势：give a single overview of a tree; large or small nodes are easily identified
  * 问题：difficult to perceive hierarchical structure, 难以察觉层级关系
  * 解决方法：使用边框分隔不同节点，多使用长方形而不是正方形。

绘制Treemaps的方法 - Squarified Treemaps

**Aspect ratio：max(height/width, width/height)**

<img src="https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211129231707347.png" alt="image-20211129231707347" style="zoom: 67%;" />

在长边加入一块，再不断推入新的块分割短边，直到新的元素的aspect ratio不再变小，则到另一块空间重复以上行为。

Squarified Treemap 问题：一个数据集中的小改动会造成可视化的巨变。orders not preserved.

* Circular Treemaps：use circles instead of rectangles
  * advantages: easy to compare size; hierarchical structure is clear
  * problems: not as space efficient

* Voronoi Treemaps
  * Use arbitrary polygons to fill up an arbitrary space
  * weighted centroidal Voronoi tessellation(CVT): 在一个cell里的点，与该cell的seed point最近
  * CVT：all seed points are the centroid of their respective cell
  * **How to compute a CVT for k seed?** 初始化点，计算点与临近点的连线的垂直平分线，根据这些线绘制出图。将点移动到图形的centroid。递归直到convergence。

# Networks

极易出现crossings and cluttering

## Optimization methods：

1. Force Directed Layout (力引导布局)

   1. attractive force: 胡克力，F = k\*x; repulsive force: 库仑力 F = -k\*q1\*q2/x^2

   2. Fruchterman-Reingold Algorithm:

      ![image-20211130141049867](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211130141049867.png)

      area = 画布面积 \|V\| = 点的个数

2. Filtering

3. Clustering

4. Edge Bundling: Clustering of edges instead of nodes

5. Hierarchical Edge Bundles: 将点按照层次分级来优化布局

## Network metrics

### Graph Metrics

**Max geodesic distance**: aka diameter, the distance between two nodes that are farthest spart.

**Average geodesic distance**: average distance from one node to another node through the graph edges.

**Graph density**: 1. undirected graph: 2E / (V \* (V - 1)) 2. directed graph: E/ (V \* (V - 1))

### Node Metrics

**Degree**

**Betweenness centrality**:  $$C(v)=\sum_{s, t \neq v \in V} \frac{\sigma_{s t}(v)}{\sigma_{s t}}$$, $$\sigma_{st}(v)$$ 代表node s与node t间的最短路径经过node v的数量，$\sigma_{st}$代表由node s到node t的所有最短路径数量。相当于计算节点之间的最短路径，通过v的占比，作为重要程度。

**Closeness Centrality**: $C(v)=\frac{1}{\sum_{u \neq v \in V} d(u, v)}$ ，1除以v到其他所有点的距离之和。==这里需要最短距离吗？==

**Clustering Coefficient**: measures how well a person's friends are connected to each other

![image-20211130153234125](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211130153234125.png)

**Eigenvector Centrality**: measures the influence of a node in a network

![image-20211130153617552](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211130153617552.png)

**Page Rank**: Score of a page = the probability of being brought to a page after many clicks.  
$$
x(u)=p \sum_{v \in V} a_{v, u} \frac{x(v)}{L(v)}+\frac{1-p}{N}
$$

p: probability of following a link

N: total number of nodes, 1-p / N = 随机跳到了这个网页的概率

L(v): out degree of v

# Text & Document

## Levels of Text Analysis

1. Lexical level 句法等级：word stems，phrases， word n-grams：sequence of n words
2. Syntactic level 语法级：grammatical category：名词，形容词... ；Tense 时态；...
3. Semantic level 语义层次： extract meaning

## Vector Space Model

每个维度对应一个item，一个文件对应一条向量。

Pre-processimg of documents: 

* **Filtering**: remove **stop words**. 
* **Stemming**: group inflected forms of a word.把同一个意思的不同词语改为一样的表达，如robot, robots...

### Document Retrieval

Term Frequency of term t in document d is defined as the number of times that t occurs in d.

Given a query q, the score of a document is defined as:
$$
\mathrm{S}_{q, d}=\sum_{t \in q} \mathrm{tf}_{t, d}
$$
The summation of the term frequencies of all words appearing in the query

但相关性不与term frequency直接相关，可以修改上式为
$$
\mathrm{S}_{q, d}=\Sigma_{t \in q} \log \left(1+\mathrm{tf}_{t, d}\right)
$$
但上式的计算对待每个item是一样的，没有引入权重。

引入Inverse Document Frequency（IDF）

$df_t$是document frequency of term t is defined as the number of documents that contain t.

Inverse document frequency: 
$$
\mathrm{idf}_{t}=\log \left(N / \mathrm{df}_{t}\right)
$$
可以将IDF当作term frequency的权重代入，得到了tf-idf公式
$$
\operatorname{tf-idf}_{t, d}=\log \left(1+t f_{t, d}\right) * \log \left(N / d f_{t}\right)
$$
tf-idf是针对每个document中每个item的，再返回到之前的查询问题

我们可以用新的tf-idf指数来替换简单的tf指数了
$$
\mathrm{S}_{q, d}=\sum_{t \in q} \mathrm{tf-idf}_{t, d}
$$

### Term Vector Similarity

* Euclidean distance
* Cosine Similarity

### Visualization

* Word Tree: root is user-specified words, branches are phrases that appear after root
* Arc Diagrams: 把重复的部分用圆弧连接起来，适用于可视化乐谱
* ThemeRiver
* Textflow：展示主题之间的分割和融合

# Interaction

* changing data representation

* data transformation

* data selection and filtering

* relating multiple graphical views: linking and brushing

* focusing and getting details

  * overview + detail 小地图+原始地图
  * zooming 放大缩小
    * zoom 有比例尺的缩放
    * pan 平移

  ![image-20211130184125702](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211130184125702.png)

  * focus + context: focus wiithin surrounding context in a single view
    * focus + context with distortion
      * 空间可以被扭曲 stretched and squeezed, e.g., fisheye, hyperbolic tree
      * problems: 1. not suitable if spatial judgment is needed 2. difficult for target acquisition , e.g., dock in macOS
    * focus + context without distortion
      * mixed resolution displays on map

# Visual Analytics

use tools and tech. that allow us to

* detect the expected and discover the unexpected

VA vs Information Visualization

* IV focus on the process of producing views
* VA focuses on making sense of user interactions on the data and refining parameters

Visual Analytiics is highly interdisciplinary 跨学科的。
