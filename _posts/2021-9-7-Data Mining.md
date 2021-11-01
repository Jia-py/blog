---
layout:     post
title:      "notebook - Data Mining"
date:       2021-09-07 09:00:00
updatedate: 2021-11-01 18:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
latex: true
tags:
    - HKU CS
---


by prof. Ben C.M. Kao

# Chapter 1: introduction

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

## What is KDD and Data mining?

>  KDD: Knowledge Discovery in Databases, **the discovery of knowledge** from big collections of data( much human involvement)

> data mining: A step in the KDD process, **the discovery of patterns** (mechanical, less human involvement)

## What are Data, Pattern, and Knowledge?

Data: a collection of facts

Pattern: characteristics of data that are frequently observed

Knowledge: some general rules about the objects

## Why KDD is iterative and interactive?

iterative: some patterns or knowledges require multiple KDD processes to get

interactive: KDD process needs human involvement to monitor and modify the steps.

## Difference between Database systems and KDD

Database systems: store and get facts.

KDD: for knowledge discovery



the simple process of summing, counting ... are not data mining. because data mining should **carry certain degree of `predictive` ability or `descriptive` ability**

* prediction: predict future behavior from past behavior
* description: summarize the underlying relationships in data and to describe the characteristics of data. (like the same features of cluster)



## OLAP: On-Line Analytical Processing

allows users to view data in a data cube, **selection, aggregation and summarization** operations.

## KDD process

(1) Goal Identification (2) Data Collection and Selection (3) Data Cleaning and preprocessing (4) Data Reduction and Transformation (5) Data Mining (6) Result Evaluation (7) Knowledge Consolidation

# Chapter 2: Data

## Types of Attributes

* Binary: 0 or 1
* Nominal: some thing that looks like number, but should be treated as name. Examples: ID number
* Ordinal: just names, but human order these names. like grades {A,B,C,D}{tall,medium,short}
* Numerical: numbers that can take operations

## Types of data

### Record

tables/relations, these are structured data

### Document data

a term vector

for example, term-frequency model (TF) , extract the number of words in one sentence, and take them as a vector.

problem with TF-model: commonly occurring words dominate the vector.

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

> 有很多不同的数学公式可以用来计算TF-IDF。这边的例子以上述的数学公式来计算。词频 (TF) 是一词语出现的次数除以该文件的总词语数。假如一篇文件的总词语数是100个，而词语“母牛”出现了3次，那么“母牛”一词在该文件中的词频就是3/100=0.03。一个计算文件频率 (IDF) 的方法是测定有多少份文件出现过“母牛”一词，然后除以文件集里包含的文件总数。所以，如果“母牛”一词在1,000份文件出现过，而文件总数是10,000,000份的话，其逆向文件频率就是 log(10,000,000 / 1,000)=4。最后的TF-IDF的分数为0.03 * 4=0.12。
>
> 引自https://blog.csdn.net/dongtest/article/details/84814042

### Set-valued Data

each record is a set of items, like a market-basket data.

one record is {"ID": 1 ,"items" : (Bread, Coke, Milk)}

比如计算顾客拿了milk，又会拿Bread的概率。则是P(Bread,Milk) / P(Milk)

### Graph Data

how to calculate two nodes' similarity?

1. personalized page rank: random walk, but start at a specific node

### Ordered Data

sequences of events

spatiotemporal data: trajectory data, 根据时间点获取的GPS数据

## Data Quality

### Noise and Outliers

Noise in data occurs due to error in data collection

Outliers are data that deviate so much from the norm. can be useful or not.

Detecting Outliers: Cluster analysis

Handling Noise: perform outlier detection and then check manually whether the outliers are likely noise

### Missing Values

* Ignore records with missing values
* go find the missing data
* guess, e.g., fill in the missing values with averages
* use a special symbol, e.g., null

### Erroneous and Duplicate Data

Erroneous data: incorrect data

## Data Preprocessing

Data cleaning; data reduction and transformation

### Data cleaning

fill in missing data; resolve inconsistency; remove noisy data

### Data Reduction and Transformation

* Aggregation. 

* Generalization. 归纳
  * reduce the cardinality of an attribute's domain. like the bucket analysis. to group data based on their values.
  * rules: (1)more easily found (2) more concise (3) more interpretable
  
* Sampling.
  * because obtaining the entire set of data is too expensive
  * because processing the entire set of data is too expensive or time consuming
  
* Dimensionality Reduction

  * purpose: Avoid curse of dimensionality; reduce amount of time and memory required by data mining algorithms; Less complex rules -- more interpretable

  * techniques:

    * Feature creation: new attributes are derived from existing attributes, e.g., weight/height^2 = BMI
    * Correlation analysis (correlation measures between attributes x and y)

    $$
    \begin{equation}\operatorname{corr}(x, y)=\frac{\operatorname{coVariance}(x, y)}{s_{x} s_{y}}\end{equation}
    $$

    $$
    \begin{equation}\operatorname{coVariance}(x, y)=\frac{1}{n-1} \sum_{k=1}^{n}\left(x_{k}-\bar{x}\right)\left(y_{k}-\bar{y}\right)\end{equation}
    $$

    $$
    \begin{equation}s_{x}=\sqrt{\frac{1}{n-1} \sum_{k=1}^{n}\left(x_{k}-\bar{x}\right)^{2}}\end{equation}
    $$

    * Principal components analysis: capture as much information as possible using as few dimensions as possible
      * keep the minimum number of components that capture at least `85%` of the total variability in the data
    * Feature selection

* Discretization: Discretization: Convert a numerical attribute into an ordinal/nominal attribute

  * bucket (0-60; 60-80 ...)

* Binarization: Convert a numerical attribute into binary

* Normalization: convert numerical numbers to a common range

  * min-max:
  $$x'=(x-min)/(max-min)$$
  
  * z-score: $$x^{\prime}=(x-\bar{X}) / \sigma_{X}$$ 
  
  where
  
  $$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}}$$

## Similarity and Dissimilarity

Distance measures: 

* Euclidean Distance: $$\operatorname{dist}=\sqrt{\sum_{k=1}^{n}\left(p_{k}-q_{k}\right)^{2}}$$  , k is the number of dimension.
* Minkowski Distance:  $$\operatorname{dist}=\left(\sum_{k=1}^{n}\left\|p_{k}-q_{k}\right\|^{r}\right)^{\frac{1}{r}}$$  , r = 1, manhattan distance; r=2, euclidean distance

Metric: (1) d(p,q)>=0 (2) d(p,q) = d(q,p) (3) d(p,r) <= d(p,q) +d(q,r)

Similarity Between Binary Vectors:

![image-20211101144104738](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211101144104738.png)

Cosine Similarity: $$\cos \left(\mathrm{d}_{1}, \mathrm{d}_{2}\right)=\left(\mathrm{d}_{1} \bullet \mathrm{d}_{2}\right) /\left\|\mathrm{d}_{1}|| \mid \mathrm{d}_{2}\right\|$$



# Chapter 4: Classification 1

## Decision Tree

Supervised Learning: training set (for train), validation set (for parameter tuning), test set (for testing the model)

Entropy, this part we can look at my former post.
$$
H(S)=\sum_{i} p_{i} \log _{2} \frac{1}{p_{i}}
$$
[DT-GBDT-XGB-Lightgbm - 我是憤怒 (jiapy.space)](https://blog.jiapy.space/2021/02/07/DT_GDBT_XGB_Lightgbm/)

larger entropy --> higher uncertainty

**entropy's calculation, plz follow the PPT carefully, it's very important!**

![](https://cdn.jsdelivr.net/gh/Jia-py/blog_picture/21_9/Snipaste_2021-09-28_11-08-27.png)

## Splitting

### Nominal Attribues

Multi-way split; Binary split

### Ordinal Attributes

Multi-way split; Binary split

### Continuous Attributes

Discretization: transform it to an ordinal attribute

Binary Decision

## Impurity measures

* Entropy
  * $$Entropy(S)=\sum_{i} p_{i} \log _{2} \frac{1}{p_{i}}$$
  * disadvantage: tends to prefer splits that result in large number of partitions, each being small but pure.
* Gini Index
  * $$G I N I(S)=1-\sum_{i}\left(p_{i}\right)^{2}$$
* Classification error
  * $$\operatorname{Error}(S)=1-\max _{i} P_{i}$$
* Gain Ratio
  * $$GainRATIO = GAIN/SplitINFO$$
  * $$\operatorname{SplitINFO}=-\sum_{i=1}^{k} \frac{n_{i}}{n} \log _{2} \frac{n_{i}}{n}$$
  * 其实很好理解，就是在决策树中单纯使用Entropy的优化版本。使用splitinfo计算分支的entropy，用这个来限制分支的数量，解决单纯使用Entropy的disadvantage。

## Accuracy metrics

* Confusion Matrix

![image-20211101153017470](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211101153017470.png)

Accuracy = (TP+TN)/(TP+TN+FP+FN)

Error rate = 1 - accuracy

precision = TP/(TP+FP)

recall = TP/(TP+FN)

F measure = (2\*precision\*recall) / (precision + recall)

## Getting a (training set, test set) pair

* Holdout: reserve like 80% for taining and use the rest for testing
* Random subsampling: repeated holdout for k times, each with a different sample of the data as holdout. classifier's accuracy = average accuracy
* Cross validation: Divide data into k partitions, run classifier by using k-1 partitions as training data. Repeat for all combinations and measure average accuracy.
* Bootstrap:https://blog.csdn.net/Answer3664/article/details/100021968; 训练集是不断从原始数据集取数据，可能获得重复的数据样本。prob. an example is selected from dataset containing N samples, and we pick N times is $$1-\left(1-\frac{1}{N}\right)^{N} \approx 1-\frac{1}{e}=0.632$$

## Generalization error

expected error when model is applied to a future unseen record.

In general, generalization error > training error; we hope generalization error ≈ test error

## Handling overfitting

* pre-pruning (early stopping rule)

## Decision Tree Based Classification

* advantage: non-parametric; easy to construct; fast; easy to interpret
* disadvantages: subject to overfitting; decision boundaries are only rectilinear

# Chapter 5: Classification 2

## Nearest-neighbor classifiers

use class labels of K nearest neighbors to determine the class label of unknown record

choose K : too small, sensitive to noise; too large, may include points from other classes.

avoid model-building; search can be expensive

perform poorly in high-dimensional spaces; feature selection is important

## Bayesian classifiers

P(H\|X)=P(H,X) / P(X)

P(C 1 \| X ) = P( X \|C 1 ) * P(C 1 ) / P(X)

我们在比较P(C1\|X)与P(C2\|X)时，只需要比较P(X\|C1)\*P(C1)与P(X\|C2)\*P(C2)的大小即可。

![image-20211101161641239](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211101161641239.png)

![image-20211101161653675](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211101161653675.png)

**What about numerical attriburs?**

* Discretize the range into bins
* Probability density estimation

## SVM

classify line: $$\vec{w} \cdot \vec{x}+b=+1$$

we want to max margin = 2/ ||$\vec{w}$||

* if the problem is not linearly separable
  * introbuce slack variables. 相当于一定的分类错误容忍度。
  * transform the data to higher-dimensional space
  * kernel trick can help computation be more efficient

## Logistic regression

predict the probability of positive class from an independent variable x

可以看作是线性回归后加了一层sigmoid函数用来计算分类的概率。

## Ensemble methods

construct a set of classifiers from the training data. use voting to predict.

How to generate an ensemble of classifiers?

(1) use different training sets (2) use different attribute sets for input (3) use different partitions of class labels (3) use different learning algorithms

# Chapter 6: Association Analysis 1

N = number of samples

I = a set contains all items

sup({a,b,e}) = {a,b,e}这个item set出现的次数

**Association Rules: X==>Y, If X occurs then Y is likely to occur**

* support condition: $$\sup (X \cup Y) / N \geq \rho_{s}$$ (出现的频率不低)
* confidence condition: $$\sup (X \cup Y) / \sup(X) \geq \rho_{c}$$ (X出现时，Y出现的概率不低)

## How to find the rules?

1. find all frequent itemsets.  sup(S) >= N * $$\rho_{s}$$
2. generate rules from frequent itemsets. 遍历所有可能的关联组合，但要注意分出的关联组合需要满足$$X \cap Y=\emptyset ; X \cup Y=S$$，检查关联组合是否满足$$\sup (X \cup Y) / \sup(X) \geq \rho_{c}$$ ，如果满足，则找到了一条关联rule。

## Apriori Algorithm (帮助解决第一步, find frequent itemsets)

If there are m different items in a dataset, then there are $$2^m - 1$$ possible non-empty itemsets, too time-consuming

**If X is a frequent itemset, then any non-empty subset of X is also frequent. Conversely, if an itemset X is not frequent, then any superset of X must not be frequent either.**

1st iteration: find all frequent 1-itemsets

2st iteration: use frequent 1-itemsets to construct candidate 2-itemsets list C2(C represents candidate). just need to calculate the frequent of itemsets in C2. The itemsets are frequent go into L2, which denotes frequent 2-itemsets.

...

我们从小的itemsets到长度长的itemsets计算时有一个问题，我们只能保证X频率不高时，X的superset频率也不高。但不能保证，X频率高时，X的superset频率也高。因此，我们构建出X的superset后，要检验superset的全部的长度-1的子集是否为frequent，若有一个不是，则该superset也不是frequent set。

