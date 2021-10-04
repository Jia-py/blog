---
layout:     post
title:      "R tutorial"
date:       2021-10-03 12:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - R
---

# install

```R
install.packages('')
```



# data exploratory analysis

```R
# 导入csv
dataset = read.csv(path)
# 查看数据
head(dataset)
str(dataset)
```

## 单变量探索

```R
summary(dataset)
table(dataset&column)
hist(dataset$column)
pie(table(dataset$column))
```

## 多变量探索

```R
# 协方差
cov(iris$sepal.length, iris$petal.length)
cov(iris[,1:4])
# 相关系数
cor(iris$sepal.length, iris$petal.length)
cor(iris[,1:4])
```



# Problems

1. 在vscode中使用jupyter插件写R时，发现不能在一个cell中写入多行代码，会报错。

