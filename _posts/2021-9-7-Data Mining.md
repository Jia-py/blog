---
layout:     post
title:      "notebook - Data Mining"
date:       2021-09-07 09:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - HKU CS
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

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
I D F(t)=\log \frac{|D|}{\left|D_{t}\right|}
$$
where D is the total number of documents in dataset, Dt is the number of documents containing t.

if the t frequency is very big, so the IDF(t) will be very small, it says that elements that appear frequently are not important.

## Set-valued Data

each record is a set of items, like a market-basket data.

one record is {"ID": 1 ,"items" : (Bread, Coke, Milk)}

比如计算顾客拿了milk，又会拿Bread的概率。则是P(Bread,Milk) / P(Milk)

## Graph Data

