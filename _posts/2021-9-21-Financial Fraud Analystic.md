---
layout:     post
title:      "Financial Fraud Analytics"
date:       2021-09-21 14:45:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - HKU CS
---

# 1. Introduction

* Introduction to Financial Fraud
* The Fraud Cycle
* Fraud Detection and Prevention
* Getting the data

# 2. Processing Fraud Data

* EDA
* Imbalance Data Handling
* Data Pre-processing Data Cleaning
* Feature Engineering

# 3. Financial Fraud Analytics

* Autoencoder

  * 克服了PCA线性的限制

  * unsupervised learning. 是因为没有输出y所以被划分为非监督训练？训练模型输出和输入尽量相似，如何判断异常值呢？当fraud数据通过autoencoder后，输出的向量与原向量的squared Error会与非fraud数据的MSE有较大偏差。

  * > 异常检测(anomaly detection)通常分为有监督和无监督两种情形。在无监督的情况下，我们没有异常样本用来学习，而算法的基本上假设是异常点服从不同的分布。根据正常数据训练出来的Autoencoder，能够将正常样本重建还原，但是却无法将异于正常分布的数据点较好地还原，导致还原误差较大。
    >
    > [利用Autoencoder进行无监督异常检测(Python) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/46188296)

  * Confusion Matrix

  * ROC curve

    * 横坐标为FPR(false positive rate) = FP/(FP+TN)，即在所有真实为假的数据中，我们预测为真的比例
    * 纵坐标为TP/(TP+FN)，即Recall，在真实为真的数据中，我们预测为真的比例
    * 从左到右，其实是无数多个不同分类阈值的模型，当分类阈值为0时，不论什么数据进来，均分为0类，这时FPR与Recall均为0；当分类阈值为1时，不论什么数据，均分为1类，这时FPR与Recall均为1。在中间的部分，我们希望Recall更大，而FPR更小，因此我们希望曲线下的面积更大，则模型更优秀。

* Benford's Law

# 4. Techniques 1

* Linear Regression
* MAE MSE RMSE 
* R-squared: 越接近1代表模型拟合效果越好。但存在问题，it will either stay the same or increases with the addition of more variables, even if they do not have any relationship with the output variables. So, 引入**adjusted R-squared**.
* Logistic Regression
* clustering
  * distance metrics：Minkoski distance, Pearson correlation; Simple matching coefficient, Jaccard index
* performance evaluation
  * split training, validation, test set
  * K-fold cross-validation
  * confusion matrix
  * how good is a cluster? use SSE, Average silhouette method to evaluate. 

# 5. Techniques 2

* Decision Trees
* Ensemble Learning: bagging 思想，有放回抽样多次，可能出现重复样本
* Random Forest
  * Out-of-Bag(OOB) Sample
  * Variable Importance
* Neural Network
  * interpret neural network: variable selection, decompositional approach, pedagogical approach (use the neural network predictions as input to a white-box analytical technique, e.g., decision tree)

# 6. SVM and Social Network Analysis

* SVM
* Social Network Analysis
  * jaccard weight
  * metrics to measure impact of neighborhoods

# 7. Forensic Accounting & Fraud Investigation

* what is foresic accounting
* Fraud Prevention and Detection
* Red Flags of Fraud
* Fraud Risk Management
* Fraud Investigation
* Anti-fraud Challenges amid the covid-19 pandemic
