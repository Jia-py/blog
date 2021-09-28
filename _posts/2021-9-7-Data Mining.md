---
layout:     post
title:      "notebook - Data Mining"
date:       2021-09-07 09:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
latex: true
tags:
    - HKU CS
---


by prof. Ben C.M. Kao

# introduction

google trends, a website can see the search trends of a few words.

search recommendation why? because the user is usually not familiar with the problem, it's really hard to search the proper problem in only a few words.

wind shear: will leads to sudden change of the wind direction to a helicopter. it's dangerous.

**big data analysis**

* web data: google trends
* sensor data: wind shear
* scientific and medical data: patients data (privacy)
* document data: emails tweets weibo
* social network data: facebook
* customer data: product recommendation

**Nov 2 midterm exam**

some conferences: ACM SIGKDD, IEEE ICDM, ACM CIKM

KDD: Knowledge Discovery and Data mining, the discovery of knowledge( much human involvement)

data mining: the discovery of patterns (mechanical, less human involvement)



the simple process of summing, counting ... are not data mining. because data mining should **carry certain degree of predictive ability or descriptive ability**

* prediction: predict future behavior from past behavior
* description: summarize the underlying relationships in data and to describe the characteristics of data. (like the same features of cluster)



**OLAP: On-Line Analytical Processing**

allows users to view data in a data cube, selection, aggregation and summarization operations.

**classification**

**regression**

**clustering**

# Data

types of Attributes

* Binary: 0 or 1
* Nominal: some thing that looks like number, but should be treated as name. Examples: ID number
* Ordinal: just names, but human order these names. like grades {A,B,C,D}
* Numerical: numbers that can take operations

## Record

tables/relations, these are structured data

## Document data

a term vector

for example, term-frequency model (TF) , extract the number of words in one sentence, and take them as a vector.

TF-IDF model: Term frequency of t * Inverse document frequency of t 

$$
\begin{equation}
T F(t) \times I D F(t)
\end{equation}
$$

where
$$
\begin{equation}
I D F(t)=\log \frac{|D|}{\left|D_{t}\right|}
\end{equation}
$$
where D is the total number of documents in dataset, Dt is the number of documents containing t.

if the t frequency is very big, so the IDF(t) will be very small, it says that elements that appear frequently are not important.

## Set-valued Data

each record is a set of items, like a market-basket data.

one record is {"ID": 1 ,"items" : (Bread, Coke, Milk)}

比如计算顾客拿了milk，又会拿Bread的概率。则是P(Bread,Milk) / P(Milk)

## Graph Data

how to calculate two nodes' similarity?

1. personalized page rank: random walk, but start at a specific node

## Ordered Data

sequences of events

spatiotemporal data: trajectory data, 根据时间点获取的GPS数据

## how to detect outliers

outliers are the data may be noise or right data.

noise is incorrect data.

we can use cluster analysis to detect outliers

## Data Reduction and Transformation

* Aggregation. 
* Generalization. 
  * reduce the cardinality of an attribute's domain. like the bucket analysis. to group data based on their values.
  * rules: (1)more easily found (2) more concise (3) more interpretable
* Sampling.
  * because obtaining the entire set of data is too expensive
  * because processing the entire set of data is too expensive

## Dimensionality Reduction

Purpose:

* Avoid curse of dimensionality
* reduce amount of time and memory required by data mining algorithms
* Less complex rules -- more interpretable

Techniques:

* Feature creation: new attributes are derived from existing attributes, e.g., weight/height^2 = BMI

* Correlation analysis

  * $$
    \begin{equation}\operatorname{corr}(x, y)=\frac{\operatorname{coVariance}(x, y)}{s_{x} s_{y}}\end{equation}
    $$

    where
    $$
    \begin{equation}\operatorname{coVariance}(x, y)=\frac{1}{n-1} \sum_{k=1}^{n}\left(x_{k}-\bar{x}\right)\left(y_{k}-\bar{y}\right)\end{equation}
    $$

    $$
    \begin{equation}s_{x}=\sqrt{\frac{1}{n-1} \sum_{k=1}^{n}\left(x_{k}-\bar{x}\right)^{2}}\end{equation}
    $$

    

* Principal components analysis: capture enough data variability (at least 85%) on fewer dimensions

* Feature selection

# Classification

Supervised Learning: training set (for train), validation set (for parameter tuning), test set (for testing the model)

Entropy, this part we can look at my former post.

[DT-GBDT-XGB-Lightgbm - 我是憤怒 (jiapy.space)](https://blog.jiapy.space/2021/02/07/DT_GDBT_XGB_Lightgbm/)

larger entropy --> higher uncertainty

**entropy's calculation, plz follow the PPT carefully, it's very important!**

![](https://cdn.jsdelivr.net/gh/Jia-py/blog_picture/21_9/Snipaste_2021-09-28_11-08-27.png)

