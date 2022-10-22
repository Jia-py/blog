---
layout:     post
title:      "HKU notebook - advanced topic in data science"
subtitle:   ""
date:       2022-05-06 22:00:00
updatedate:
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: true
# 生活，工作，笔记（个人理解消化心得），文档（方便后续查阅的资料整理），项目，其他
tags:
    - 笔记
---

# 01 data

## relational database

## types of data

p14

record data: 有固定的特征。

* data matrix mapping, 多维数据可以共同映射为points in a multi-dimensional space
* document data: unstructured to structured data, 'term' vector
* set-valued data: transaction data

graph

ordered data:有先后顺序的数据，文本，时序

unstructured data

semistructured data: XML, JSON, 既有结构化部分，也有非结构化的文本数据

## Types of Attributes

Nominal: just names, ID

Ordinal: rankings

Interval: calendar dates， 转换，因为零值位置不同？如华氏度和摄氏度的转换

Ratio: length, time 比例，用乘除来比较

## Discrete and Continuous Attributes

p37

## Asymmetric Attributes

Binary attributes

## How to handle non-record data

p39

## Work with Data

database search(SQL), Information Retrieval(检索) search(keywords search), similarity search, data mining

## Data Preprocessing

Data quality, noise, outliers, missing values, Inconsistent or Duplicate data

solutions: L1P51

Aggregation, Sampling, Dimensionality Reduction, Feature subset selection, Feature creation, Discretization and Binarization, Attribute Transformation

## Similarity and Distance

Similarity/Dissimilarity for **Simple Attributes**

Euclidean Distance, Minkowski Distance

how a distance becomes a metric? L1P67

Similarity Between **Binary Vectors**: SMC, Jaccard

**Document data**: Cosine Similarity

**Combining Similarities**: L1P72 思路为，输入为两个向量，在每个维度上计算相似度$$s_k$$，同时计算该维度的indicator value，若该维度为asymmetric attribute且两个objects的value均为0，或其中有objects为缺失值，则indicator value为0，反之所有情况为1。最终的similarity计算公式为
$$
similarity(p,q)=\frac{\sum_{k=1}^{n}indicator_k \times s_k}{\sum_{k=1}^{n}indicator_k}
$$
why similarity / distance?

# 02 spatial

Topological Relationships: disjoint, intersects(overlaps), equals, inside(A,B) A在B内, contains(A,B) B在A内, adjacent邻接

## Spatial Queries

Range query, Nearest neighbor query, Spatial join

processing: 1. Filter step 2. Refinement step

Z-score

Grid Indexing - 每个网格有一个索引

k-d tree

Point-region(PR) quadtree 四叉树

**R-tree**, Group object MBRs to disk blocks hierarchically. To avoid one object clipping in a grid. 防止一个物体同时在多个网格中。

## ***Range Query***

L2P41

```python
def range_query(query w, node n):
	if n is not leaf:
        for n_ in n.child:
            if intersects(w,n_):
                range_query(w,n_)
    else:
        for n_ in n.child:
            if intersects(w, n_):
                report n_.object
```

## R-tree Construction

***加粗的为四种主要方法***

Method 1: iteratively insert rectangles into an initially empty tree (**R*-tree insertion**)

* tree reorganization is slow

* more space occupied for the tree

* 与B-tree的构建很相似，一开始初始化一个root节点，然后开始插入rectangles，然后我们根据rectangles的center进行分裂，只不过和B-tree不同的是我们存储的每个数据是一个指针而不是一个值。（下图是B-tree的建树过程）

* ```
  [3,7, , ] - >insert 9 ->[3,7,9, ] -> insert 23 -> [3,7,9,23] ->insert45 -> 
           [9]
         /     \
  [1,3,7, ]   [23,45, , ]
  ```

 Method 2: **bulk-load** the rectangles into the tree using some fast (sort or hash-based) process

* R-tree is built fast
* good space utilization

* 1. 以一个轴排序，比如X轴，基于各个rectangles的中心点排序。每M(max children number)个连续的矩形组成一个叶节点，自下而上建树。(**x-sorting**)

<img src="C:\Users\JPY\AppData\Roaming\Typora\typora-user-images\image-20220502205924062.png" alt="image-20220502205924062" style="zoom:50%;" />

* 2. 针对上面的改进，可以对rectangles进行space-filling curve的index，在空间上使得临近的index更接近，比如Hilbert index。(**Hilbert sorting**)
  3. Sort using one axis first and then groups of sqrt(n) rectangles using the other axis。大概意思是在X商按照排序先选一个范围，在该范围内按照Y的排序选sqrt(n)个rectangles组合起来。这个方法一般是最优的建树方法。(**Sort-tile recursive**)

## ***Nearest Neighbor Queries***

* Depth-first NN search using an R-tree

```python
# 初始化O_NN = None, dist(q,O_NN) = inf
def DFNN(query_point q, root n, point O_NN):
    if n is not leaf:
        for entry e in n:
            if dist(q,e.MBR)<dist(q,O_NN):
                DFNN(q,e.ptr,O_NN)
    else:
        for entry e in n:
            if dist(q,e)<dist(q,O_NN):
                O_NN = e
```

相当于不断比较`dist(q,e.MBR)`与`dist(q,O_NN)`，减枝。

* Best-first NN search

```python
def BFNN(query_point q, root):
	add all entries of root into min-heap Q, key = dist(q, entry.MBR)
	while Q:
		node = heapq.heappop(Q)
		if node is leaf entry:
			return node
		else:
			for entry in node:
				heapq.heappush(Q,dist(q,entry.MBR))
```

## Why incremental NN search?

意思是先找到最近的，再看是否满足一些条件，若不符合，则继续找下一个最近的点，查看是否满足条件，循环。

# 03 dense multidimensional

## Similarity search

1. range similarity search, find `dist(q,o)<=epsilon`
2. KNN similarity query, find k nearest objects

Some problems about using R-tree in multidimensional data. L3P15

*以下两种方法中，$$q$$为query，$$p$$为point，$$S$$为set，$$s$$为search point*

## two-step processing of range similarity queries

step1: 降维，convert 目标点$$q \rightarrow q'$$，在降维向量上应用R-tree range search找到$$S'\subseteq S$$，$$S$$是原向量集合，所有在$$S'$$中的点都满足$$D'(p',q')\leq\epsilon$$

step2: 升维重新映射，验证。

## two-step processing of nearest neighbor similarity queries

step1: 降维，在较低维度找到nearest neighbor $$p'$$，再升维成$$p$$，计算在高维距离$$D(q,p)$$。

​			重新在低维找到一个集合$$S'$$，使得其中所有的点$$s'$$满足$$D'(s',q')\leq D(q,p)$$。

step2: 升维$$S'$$中的点，计算$$D(s,q)$$，返回距离最小的点。

## Multi-step processing of nearest neighbor similarity queries

```python
1. convert q to q' using the same dimensionality reduction technique
NN = None, D(q,NN) = inf
while True:
	incremental R-tree nearest neighbor search find next nearest p' to q'
	if D'(q',p') < D(q,NN):
		if D(q,p)<D(q,NN):
			NN = p
	else:
		break
```

其实multi-step与two-step的区别只在于，在已获得$$D(q,p)$$的情况下，multi-step是一边找小于该距离的点并更新NN，而two-step是先找出所有符合该距离的点再一起验证。

## Compression-based indexing VA-file

step1：filter，upper bound和lower bound是网格的四个角至q的最大最小值，先维护upper bound的最小值，找出所有lower bound小于该upper bound最小值的点，作为candidates

step2：refine，把candidates内的点计算actual dist，update current actual NN and t

## M-tree range search

每个单元维护三个东西，点，与子树中的点的最大距离，与父节点的距离。

过程可见L3P39，想法为只有满足$$dist(q,o) \leq 与子树的最大距离+目标距离$$才可能构成三角形，注意这里不是一定可以构成三角形，目标是filter，因为只有能构成三角形的才能满足at most n from q的目标。根据该原则一路follow，到叶节点之后计算真实的distance，符合要求的加入答案。

## pivot-based distance

L3P41

pivots are effective if they are far from each other

主要思想是利用pivots来filter，循环每个point o，对于某一个$$pivot$$，根据三角定理$$dist(q,o) \geq |dist(p,o) - dist(p,q)| $$，$$dist(p,q)$$是可以提前计算存储的值，若$$|dist(p,o) - dist(p,q)| > \epsilon$$，那么可得$$dist(q,o) > \epsilon$$，不符合要求，减枝。

对于那些没有被减枝的点，计算真实距离并验证。

## iDistance index

L3P44

选一些pivots，把每个object都分配到最近的pivot $$p_i$$一组，并计算其value $$v(o)=maxd*i+dist(o,p_i)$$，$$maxd = max(maxd_i), maxd_i = max(dist(o,p_i))$$，这一步相当于是以maxd为间隔，将每个object插入到一个区间内。

**Range query**

```python
for p_i in pivots:
	# 距离离pivot太远了，prune
	if dist(q,p_i)-maxd_i > bound:
		continue
	else:
		o_lis = (maxd*i+dist(q,p_i)-bound,maxd*i+dist(q,p_i+bound))的符合要求元素列表
        for o in o_lis:
            if dist(o,q) <= bound:
                answer.append(o)
```

**K-NN query**

## Searching for similar time-series Subsequence Matching

L3P57

## DTW Dynamic Time Warping

类似于编辑距离中的动态规划解法，example L3P68

图像的含义在我看来是说较相似片段的连接

![image-20220503140742032](C:\Users\JPY\AppData\Roaming\Typora\typora-user-images\image-20220503140742032.png)

而后面的二维图表则为连接的两点的坐标值，比如红线的连接点值作为x轴，蓝线的作为y轴

# 04 sparse multidimentional

## Signature-based Indexing

example L4P13

signature-tree L4P17

## Inverted File

字典形式存储，key为words，value为一个list，list中保存存在该word的document_id

compression：在list中不直接存储每个document_id，而是前后两个id的差值，这样可以用较少位数encode each id

## Document similarity and Information Retrieval

### Document ranking

1. cosine similarity L4P25

2. using the inverted file for ranking L4P26 (log-frequency)

3. tf-idf: $$W_{t,d}=log(1+tf_{t,d}) \times \log_{10}{(N/df_t)}$$，总得分为该document中所有query的word的weight之和。

### Improving efficiency

L4P34

### Authority

文件的权威性$$g(d)$$也是一个考虑的指标，最后的$$score(q,d)=g(d)+similarity(q,d)$$

一种想法是在inverted file中排列documents时按照权威性降序排列，也可以在计算champion lists(L4P35,L4P39)时维护最大$$g(d)+tf-idf_{td}$$

## Phrase queries

1. Biword indexes, 将连续的pair of terms组合成新的词用于检索，较长短语也可以拆解
2. Positional indexes，对之前的inverted lists作一些改变，key为word，value为一个字典(key为document_id，value为word出现在document中的位置list)。L4P47

### Precision, Recall and F-score

L4P56

## Substring Search

### search for substrings

trie, 字典树，PPT中为后缀树

每个节点代表一个子字符串，数字代表position of the suffix compressed in the path

### Approximate search

**Edit distance** to measure the similarity between two strings, number of insert, delete, replace necessary to transform one string to the other.

```python
def edit_distance(s1,s2):
    # cal the matrix
    dp = [[0]*(len(s2)+1) for _ in range(len(s1)+1)]
    for i in range(len(s1)+1):
        for j in range(len(s2)+1):
            # initialization
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            # two cases
            elif s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]) + 1
```

返回的结果应该是最后一行与右下角值相同的所有path

## Recommender Systems

Pearson correlation coefficient L4P83

Rating predictions, **Collaborative Filtering**: 1. user-user collaborative 2. item-item collaborative; example L4P86 3. baseline estimates L4P91

matrix factorization for CF, 思想是定义Q和P两个矩阵，额外引入了一个factors的维度，使得$$Q \cdot P^{T}$$与原矩阵R的loss越小越好。

# 05 multidimensional queries

## Rank Aggregation

### TA

TA = Threshold Algorithm, L5P16

数据表为每个特征从大到小排序，遍历访问每一行，继续当前行的sum为T，**访问已出现元素的三个特征和**(random access, 费时)，并维护一个和最大的元素，若该元素的特征和大于当前行sumT，则终止，表示之后不论如何，T都不可能大于该元素特征和了。

### NRA

L5P20

不断维护 1. 各元素的upper bound和lower bound，upper bound即以当前行的值补缺失值，lower bound即以0补缺失值。2. $$W_k$$数组，是lower bound最大的k个元素

终止条件：若$$W_k$$中元素最小的lower bound，比任何不在$$W_k$$中的元素的upper bound都大，则终止。

### LARA

是针对NRA的改进，有三个observation，前两个observations思想为**growing phase**，维护$$W_k$$中的lower bound最小值t，T为当前行的和。当$$t<T$$时，未知的一行出现一个从未见过的元素依然可能是前k大的元素，或者，原本没在$$W_k$$内的元素也可能是前k大的元素。因此我们在$$t<T$$的阶段，是不可能确定哪些是前K大的元素的，因此在这个阶段只需维护元素的lower bound即可。observation3为**shrinking phase**，意思是当$$t \geq T$$时，任何还未出现的元素都不可能进入前K大。只有在shrinking phase我们再去维护元素的upper bound，且可以一个特征一个特征更新，减少计算量。

终止条件：若$$W_k$$中元素最小的lower bound，比任何不在$$W_k$$中的元素的upper bound都大，则终止。

example：L5P35

### Multi-dimensional index search

用R-tree建索引，但这里计算每个节点的函数$$f=ax+by+...$$，通过函数值放入最小堆，计算步骤使用Best-First search

example见L5P46

## Skyline Queries

dominate概念：A若在某一个特征上大于B，在其他特征上不小于B，则称A dominates B

1. Block-nested loops L5P60
2. Sort-Filter-Skyline L5P64 (按照sum升序排序，例子中是两个特征越小越好)
3. ==Branch-and-Bound== L5P71

## Data Warehousing and OLAP

Data transformation and data cleansing

## Temporal Databases

MVB-tree

Interval tree

## Cluster Analysis

K-means L5P157

PAM(partitioning around medoids) L5P162

DBSCAN L5P167

# 06 Streams

## Sampling from a stream

1. ==Naïve Approach== L6P12
2. Sample Users:  Use a hash function that maps the user name or user id uniformly to 10 numbers: 0..9

## Stream Filtering

1. First Cut Solution: 二进制数组B，利用hash函数把我们需要的s对应的B[hash(s)]=1。流数据进来，只需要比对B[hash(a)]是否等于1即可。但注意，等于1也只是有可能在原集合S中，不能保证。

$$
1-(1-1/n)^{n/(m/n)}=1-e^{-m/n} \ when \  n \rightarrow \infty
$$

​		该式为计算一个target被darts击中的概率，即B中某一个索引被hash function赋值为1的概率，也即B中1的fraction。原式应为$$1-(1-1/n)^{m}$$，为方便技术，引入n。

2. Bloom Filter：与First Cut Solution很相似，只不过在这里采用了多个hash function，当比对得到所用的B[hash_i(a)]均为1，那么declare that a is in S。

​		fraction of 1 in B: $$1-e^{-km/n}$$，false positive: 预测为in S，实际不在，prop = $$(1-e^{-km/n})^k$$

## Counting from a Stream

Flajolet-Martin Approach: 用一个hash函数映射element为一个数字，用二进制表示这个数字。另定义函数r(a)，计算从右往左数，第一个1前出现了多少个0。记录R=max(r(a))，大致记录的不同数字数即为$$2^R$$个。

## Estimating Moments

AMS Method: example L6P45 取多个采样点，计算采样点元素的**n(2*c-1)**，求平均拟合真实2nd moment value。其中n为整段stream的长度，c为从采样点往后有多少个该元素值。

一般来说$$Estimate=n(c^k-(c-1)^k)$$

### Reservoir Sampling

水池抽样，解决问题：**给定一个数据流，数据流长度N很大，且N直到处理完所有数据之前都不可知，请问如何在只遍历一遍数据（O(N)）的情况下，能够等概率选取出m个不重复的数据。**

```python
def reservoir_sampling(n,s):
    S = []
    # sampling
    for i in range(n):
        if i < s:
            S.append(i)
        else:
            j = random.randint(0,i)
            if j < s:
                S[j] = i
```

## Sliding Window Queries

解决最新的N个数据内有多少个1

simple solution: 用过去的分布估计当前数据 L6P60

DGIM：将这一段数据用多个2^n的间隔间隔开，前面的间隔永远大于后面的间隔。能够保证这样的数据结构是由于其特殊的update方法，见L6P68。不断将小的桶随着新数据的输入合并为大的桶。

## Exponentially Decaying windows

example L6P80，what are currently most popular movies

# 07 adaptive indexing

## Database Cracking

1. **Stantard database cracking**: QuickSort

Stantard database cracking（分割）: 如搜索10<key<=20，则先搜索10<key，利用quick sort，以10为pivot，将大于10的筛选出来；再同理筛选出<=20的。

不断地接受query，就会不断地将整段数据分割为更多段数据，每段与每段之间按大小排列。可以用二叉搜索树的形式整理数据。

simple, sort, cracking比较，L7P20

最优情况，最差情况图解，L7P22

2. **Adaptive Merging**: Merge-sort

example: P27 每次都将有可能的区间里找出本次query的答案，组成一个新的区间

3. Hybrid Adaptive Indexing

与Adaptive Merging相比，每个partition不再保持有序，而是根据quick-sort来进行分割

## Stochastic Cracking

1. DDC: Data Driven Center

先根据一些初试的crack（如数据的median值）将数据分割，再根据真实的query来分割数据。

2. DDR: Data Driven Random

先根据一些初试的pivot将数据分割，再根据真实的query来分割数据，避免了寻找median的overhead。

## Coarse Granular Index

首先将数据粗粒度地分割，保证前一个partition中所有的数据都小于后一个partition

## Multidimensional Adaptive Indexing

Adaptive KD-tree

# 08 Learned Indexes

使用一个机器学习模型，输入为key，输出为key position estimate，用于替代traditional index structures

Learned index for range searching P8

Learned index for value searching P15

**ALEX: Adaptive Learned Index**

support updates

# 09 Data Provenance

## Data Provenance（起源） Concepts

Data Lineage: data origin

Data Provenance: historical record of the data and its origin

Data Lineage is a simple type of why-provenance

Data Curation (数据管理): process of extracting important information

一些SQL语法

natural join: 两张表根据相同attributes连接，只保留共有的数据，并组合成新的数据。

## Database Provenance

**lineage** of a tuple t的意思是对结果有贡献的原始输入数据

**witness** of a tuple t 一系列的输入数据集合，每个集合的数据都足够产出t

**why-provenance** the set of witnesses of t

satisfaction-compatible: 有的attributes都满足，但在要求里的一些attributes在数据中不存在，则是satisfaction-compatible P76

unpicked P78

一个操作m 关于一个unpicked tuple t is picky的意思是：

1. t或t所属的lineage 在m的输入中
2. t或t所属的lineage都不在m的输出中

==Frontier picky==

==Why-not answer==

## Data Causality

$$D^x$$: Exogenous tuples, 是输出的原因候选

$$D^n$$: Endogenous tuples, 是除了Exogenous tuples的其他tuples

Counterfactual cause P91

Actual cause P92

Monotone, non-monotone P93