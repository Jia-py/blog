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

# Import

```R
library('')
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
colnames(df)
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

# Data Preprocessing

```R
# 删除几列

dataset = df[,c(-25,-40)]
# 将数据中的字符转其他格式

dataset = dataset %>% mutate_if(is.character,as.factor)
# 日期计算

dataset$policy_bind_date = as.Date(dataset$policy_bind_date) - as.Date("1990-01-08")
# 将表内所有数据转为numeric

df_num = as.data.frame(lapply(df_sampled,as.numeric))
```

## sample

```R
library('ROSE')
df_sampled <- ovun.sample(fraud_reported ~ ., data = dataset, method = "over",N = 1500)$data
```

# Model

## Split training set and test set

```R
dataset_xgb = data.frame(df_sampled)
train = sample(nrow(dataset_xgb), 0.7*nrow(dataset_xgb), replace = FALSE)
TrainSet = dataset_xgb[train,]
TestSet = dataset_xgb[-train,]
```



# Problems

1. 在vscode中使用jupyter插件写R时，发现不能在一个cell中写入多行代码，会报错。

   > 不知道为啥自动解决了

