---
layout:     post
title:      "HKU notebook - big data management"
subtitle:   ""
date:       2022-05-12 22:00:00
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

by Prof. Reynold Cheng

# RelationalDB

**selection operation**: $$\sigma_p(r)$$

some relation: and $$\and$$, or $$\or$$, not $$\lnot$$

like $$\sigma_{a = 'curry' \and b >= 2}(account)$$

**project operation**: $$\Pi_{A1,A2,...,AK}(r)$$

只选择A1,A2,...,AK这些特征构建表

**Union operation**: $$r \cup s$$

r表和s表必须拥有相同的特征，才能进行union，重复的删去

**Intersection operation**: $$r \cap s$$

与union相反，只取重复的

**Set Difference Operation**: $$r-s$$

将r中，与s中的重复项删去

**Cartesian-Product Operation**: $$r \times s$$

得到的是行数为r的行数乘以s的行数，列数为r的列数加上s的列数

**Aggregate operation**: $$G1,G2,...,Gn \mathscr{g}_{F1(A1),F2(A2)...}(E) $$

G1,G2代表根据什么特征聚合，F1，F2代表聚合函数，A1，A2代表需要聚合的特征值。

![image-20220314180009119](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220314180009119.png)

# Indexes

two basic kinds of indices:

1. Ordered indices: search keys are stored in sorted order. Example: B+ Tree
2. Hash indices

Classification of Indexes:

Primary index: index的顺序和原始数据表顺序一致

Secondary index: index的顺序和原始数据表不一致，have to be dense

Dense index：所有的原表数据都在index中存在

Sparse index：部分的原表数据在index中存在

## B+ Tree

Advantage: automatically reorganizes itself with small local changes, in the face of insertions and deletions. 

Disadvantage: extra insertion and deletion overhead, space overhead.

each node has multiple children (between [n/2] and n)

**B+ Tree Insertion**

1. Key K is found in the leave node Bi, so data can be added  to the data file.
2. Key K is not found in Bi. K is inserted into Bi , and then to the data file.
3. If Bi is already full, a split of Bi is performed.

![image-20220314210959223](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220314210959223.png)

**B+ Tree Deletion**

Example remove Brown

![image-20220314211854846](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220314211854846.png)

![image-20220314211906335](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220314211906335.png)

## Hash

An ideal hash function is **uniform**, i.e., each bucket is assigned the same number of search-key values from the set of all possible values.
Ideal hash function is **random**, so each bucket will have the same number of records assigned to it irrespective of the actual distribution of search-key values in the file.

Handling of Bucket Overflows P41

# Spatial

Intersects (adjacent, contains, inside, equals), disjoint

contains与inside的区别在于主语不同

MBR(minimum bounding rectangle)

two steps in spatial query: 1. Filter step 2. Refinement step

Object clipping can be avoided if we allow the regions of object groups to overlap

![image-20220314214111012](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220314214111012.png)

**Optimization Criteria**

Nodes in tree should be filled as much as possible

Minimizes tree height and potentially decreases dead space

## Nearest Neighbor Search

1. Depth First P70

选取每个可能的node，即跟MBR的距离小于当前最短距离的node，计算，更新最短距离

2. Best First P85

选用优先队列保存所有已访问点与目标点的距离，每次计算距离最短的node

# Spatial Network

## Dijkstra's Shortest Path Search

![image-20220314234215982](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220314234215982.png)

![image-20220314234227565](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220314234227565.png)

## A* Search

不只根据已走过的路程衡量优先程度，还要加上预计还要走的路，详情可见machine learning的search笔记。

![image-20220314234435846](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220314234435846.png)
$$
f(p)=Dijkstra_{dist}(s,p)+Euclidean_{dist}(p,t)
$$

## Bi-directional search

不仅从起点开始找，也从重点开始找。但这需要同时维护两个优先队列，并且每个步骤选取两个优先队列中的最小值为当前操作对象。

![image-20220314234940506](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220314234940506.png)

在左侧的操作中，是仅对s优先队列操作或仅对t优先队列操作，当在两个优先队列中，我们同时操作了某点，则终止。最短路径为，从两头到该重复节点的最短路径。

# Ranking Queries

Topk query evaluation 1D ordering and merging lists.

Advantages and drawbacks: P14

前提：各特征都已经排序好

![image-20220316160557482](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220316160557482.png)

**TA方法(threshold)**见PPT P15

大概思想是，在循环第i次时，我们选取各特征排序的第i个值，计算和得到T。此时继续随机选取某个ID的某行数据，如ID为a的所有数据，ID为b的所有数据，在当前累积的所有选取数据的所有特征和中，若最大值大于T，则停止当前循环，该最大值所对应的ID即为我们想选的数据。

**NRA：No Random Accesses** 见PPT P20

大概思想是按照顺序遍历，若topk数组中的最小的lower bound都大于等于在topk数组外的数据的upper bound，那么就terminate

若当前特征中未出现，如'd'的数据，则通过前面的数据的均值来估计，加总时采用取整的方式。

**LARA** 见PPT P26

example见P35，大致意思是只统计lower bound，T为当前读到的排序后的一行的和，t为优先队列中第k大的数的lowerbound值，Wk为优先队列第k大的数。比较t与T，若t<T，则继续，若t>=T，则停止，进入shrinking phase。

**shrinking phase是如何操作的？**

# Uncertainty

Uncertainty in Applications

1. Sampling Accuracy
2. Uncertainty in Satellite
3. Measurement Errors
4. Text Extraction
5. Uncertain Graphs
6. Criminal Databases

Problems on Managing Uncertain Data

1. Modeling data uncertainty
2. Probabilistic Queries
3. Data Quality and Cleaning

Uncertainty Models



Two types of uncertainty

1. Attribute uncertainty (value uncertainty)
2. Tuple uncertainty (existential uncertainty)
