---

layout:     post
title:      "DT-GBDT-XGB-Lightgbm"
date:       2021-02-07 21:19:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - CS
    - Machine Learning
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

### DT决策树

#### 什么是信息熵？

信息量和质量一样，也是可以被衡量的。熵衡量数据的不确定性。

质量中，我们采用把一个物体的质量规定为`1kg`，在此基础上衡量其他物体的质量。在信息论中，也选取一个参照事件（抛硬币等**只有两种等概率事件**组成的事件,称为`1bit`）的不确定性（信息），当`m`个参照事件的不确定性与待测事件的不确定性相等时，则用`m`来表示信息量。

但还有一点不同。

3kg重物体A与3个1kg的物体B一样重。但3个1/2的抛硬币事件与1个1/6等概率分布事件是不相等，这不是一个乘法的关系，而是$log_2x$的关系。也就是说，1个1/8等概率分布事件的信息量（不确定性）是`3bit`。

[![ydkQRf.md.jpg](https://s3.ax1x.com/2021/02/09/ydkQRf.md.jpg)](https://imgchr.com/i/ydkQRf)

则我们计算$\frac{1}{6}log_2m_A$只需要看作是1/6的6个等概率事件分布即可，即$\frac{1}{6}log_26$

图中的分布计算得到的最终信息量为：$\frac{1}{6}log_26 + \frac{1}{6}log_26+\frac{1}{2}log_22+\frac{1}{6}log_26=1.792$

所以得出计算一般分布的信息量（熵）： $\Sigma p_{i} \log 2 p i^{-1}$ 


参考资料：[为什么信息还有单位？如何计算信息量？](https://www.bilibili.com/video/BV1jt411b7o3)

#### 决策树原理

决策树有一个很强的假设：信息是可分的

* 按**任务**分：分类树与回归树
* 按**特征选择方法**分：ID3决策树、C4.5决策树、CART决策树

##### CART决策树

CART采用**二元切分**的方法

分类树特征选择采用**基尼系数**，回归树特征选择采用**平方误差**

**1. CART分类树**

参考链接：https://blog.csdn.net/Subin_/article/details/99681866

采用基尼系数(Gini index)衡量数据集的混乱程度，基尼系数越小说明数据不纯度（混乱程度）低，特征越显著。

Single_Gini：  

$$
\operatorname{Gini}(D)=1-\sum_{k=1}^{K}\left(\frac{\left|C_{k}\right|}{|D|}\right)^{2}
$$

在该特征下，共有K种分类，\|$C_k$\|代表第k种分类的样本个数。\|D\|代表总样本数。

Gini_Index: 在A特征下，分为D1与D2两类，在特征A下的基尼系数  

$$
\operatorname{Gini}(D, A)=\frac{\left|D_{1}\right|}{|D|} \operatorname{Gini}\left(D_{1}\right)+\frac{\left|D_{2}\right|}{|D|} \operatorname{Gini}\left(D_{2}\right)
$$

这里Gini(D1)指的是D1这一类再根据**所需预测的特征**的划分。

根据各个特征的基尼指数，选择基尼指数最小的为分类特征，将分类特征中的值分为两类，选取一类分支分出去。

例子：[决策树的基尼系数计算过程](https://blog.csdn.net/qq_35540187/article/details/111729115)  

在该例子中，计算`Good Blood Circulation`的基尼系数的公式为：  

$$
\frac{164}{297} \times\left(1-\left(\frac{37}{164}\right)^{2}-\left(\frac{127}{164}\right)^{2}\right)+\frac{133}{297}\left(1-\left(\frac{100}{133}\right)^{2}-\left(\frac{33}{133}\right)^{2}\right)
$$

---

对离散分布、且取值数目>=3的特征的处理：

正是因为CART树是二叉树，所以对于样本的有N>=3个取值的离散特征的处理时也只能有两个分支，这就要通过组合人为的创建二取值序列并取GiniGain最小者作为树分叉决策点。如某特征值具有[‘young’,’middle’,’old’]三个取值,那么二分序列会有如下3种可能性(空集和满集在CART分类中没有意义):
[((‘young’,), (‘middle’, ‘old’)), ((‘middle’,), (‘young’, ‘old’)), ((‘old’,), (‘young’, ‘middle’))]
采用CART算法，就需要分别计算按照上述List中的二分序列做分叉时的Gini指数，然后选取产生最小的GINIGain的二分序列做该特征的分叉二值序列参与树构建的递归。

原文链接：https://blog.csdn.net/jiede1/article/details/76034328

---

**2. CART回归树**

![CART算法](https://pic1.zhimg.com/v2-2650df8818dbc09e531ba2bfe9914ccc_r.jpg)

其中，第一步首先寻找最优切分点。切为左右两部分，分别找到左右两部分的$c_1$和$c_2$，使得$\sum_{x_{i} \in R(j, s)}\left(y_{i}-c\right)^{2}$最小。

第二步，根据找到的切分点s进行切分。注意这里的Nm为该部分的样本数，即这块求的是均值作为输出值。

**如何找到最佳的切分特征呢**

* 每个特征：  
	* 每个特征值：  
		* 将数据切分成两份  
		* 计算切分的误差  
		* 如果当前误差小于当前最小误差，那么将当前切分设定为最佳切分并更新最小误差  
* 返回最佳切分的特征和特征值  

切分后的最终结果示例：  

$$
f(x)=\left\{\begin{array}{ll}
5.63, & x<2.5 \\
5.72, & 2.5 \leq x<3.5 \\
6.24, & 3.5 \leq x<4.5 \\
6.75, & 4.5 \leq x<6.5 \\
7.31, & 6.5 \leq x<8.5 \\
8.91, & x \geq 8.5
\end{array}\right.
$$


具体例子可见[决策树的原理、推导、Python实现和Sklearn可视化](https://zhuanlan.zhihu.com/p/339380585)

##### ID3决策树

特征选择标准：信息增益

参考链接（含例子）：[决策树算法如何切分特征如何选择节点、信息增益、熵值计算](https://blog.csdn.net/weixin_44451032/article/details/100046855)

ID3决策树只能用于**分类问题**

下面为ID3进行一次分类的步骤：  

* 计算初始熵Initial_entropy：$\Sigma p_{i} \log 2 p i^{-1}$，其中，$p_i$为不同类别的占比（概率）。  **这里计算的是y(预测值)的熵**

* 初始化信息增益 entropy_gain = 0  

* 遍历每个特征：  
	* 根据该特征的类别数n,将样本分为n类,每类的概率为pk：  
		* 为n类计算熵**（根据y的分类）**，加总，得到切分后的总熵值 sum_entropy = $\sum_{k=1}^{n} P_{k} \sum p_{n i} \log _{2} P n i^{-1}$  
	* Tmp = Initial_entropy - sum_entropy   
	* if Tmp > entropy_gain: entropy_gain = Tmp  
* 选择信息增益最大时的分类特征和分类方法  

即，特征A对训练数据集D的信息增益g(D,A)，定义为集合D的经验熵H(D)与特征A给定条件下D的经验条件熵H(D丨A)之差：  

$$
g(D, A)=H(D)-H(D \mid A)
$$

缺点：1.存在偏向于选择取值较多的特征的问题 2. 只能处理分类属性 3. 对训练样本的质量的依赖性强

##### C4.5决策树

<br>

特征选择标准：信息增益比

其实与ID3非常相似，只是这里考虑的是**信息增益/训练集关于分类特征A的熵**

公式为：  

$$
g_{R}(D, A)=\frac{g(D, A)}{H_{A}(D)}
$$

其中，**如果A有n个取值**，则其中数据集D关于特征A的熵为：  

$$
H_{A}(D)=-\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{|D|} \log _{2} \frac{\left|D_{i}\right|}{|D|}
$$

#### 决策树小结



| 算法 | 支持模型   | 树类型 | 特征选择         | 连续值 | 缺失值 | 剪枝   | 样本量 |
| ---- | ---------- | ------ | ---------------- | ------ | ------ | ------ | ------ |
| ID3  | 分类       | 多叉树 | 信息增益         | 不支持 | 不支持 | 不支持 | 小样本 |
| C4.5 | 分类       | 多叉树 | 信息增益比       | 支持   | 支持   | 支持   | 小样本 |
| CART | 分类、回归 | 二叉树 | 基尼系数，方差和 | 支持   | 支持   | 支持   | 大样本 |

### GBDT 梯度提升决策树

参考链接：https://zhuanlan.zhihu.com/p/280222403

是一种Boosting(集成)算法，集成学习分为两种。

Bagging：个体学习器间不存在强依赖关系，可以同时生成的并行化方法。

Boosting：个体学习器间存在强依赖关系，必须串行生成的序列化方法。

提升树算法：

![](https://cdn.jsdelivr.net/gh/Jia-py/blog_picture/21_3/v2-fedb38d98fdc20eeaea35d966f085836_r.jpg)

其实就是构建多棵CART决策树，构建好了一棵后，让下一棵树去拟合预测值与真实值之间的残差。最终达到规定的树的数量时，停止循环。

计算预测值则是将每棵树的预测值相加即可。

GBDT算法：**GBDT和提升树的区别就在于GBDT用负梯度来近似模拟残差，而提升树直接用的是真实值与预测值的差。**

![](https://cdn.jsdelivr.net/gh/Jia-py/blog_picture/21_3/v2-52ac5f6a5418309fc519f9617d1681c1_r.jpg)

![](https://cdn.jsdelivr.net/gh/Jia-py/blog_picture/21_3/v2-d17d9e9982290288719f9433d58c7b0a_r.jpg)

如果损失函数是平方损失，则负梯度就是残差

### XGBoost

参考链接：XGBoost的原理、公式推导、Python实现和应用 - 刘启林的文章 - 知乎 https://zhuanlan.zhihu.com/p/162001079

**XGBoost**是一种**集成树**模型，最终的预测结果是每棵树预测结果之和。  

$$
\hat{\mathrm{y}}_{i}=\phi\left(x_{i}\right)=\sum_{k=1}^{K} f_{k}\left(x_{i}\right), f_{k} \in \mathrm{F}
$$

在这里$f_{k}\left(x_{i}\right)$是每一棵决策树，具体说是CART(Classification and regression tree)。

XGBoost是对XGBT的优化，主要有以下几点：

1. 利用二阶泰勒公式展开：优化损失函数，提高计算精确度
2. 利用正则项：简化模型，避免过拟合
3. 采用Blocks存储结构：可以并行计算等



