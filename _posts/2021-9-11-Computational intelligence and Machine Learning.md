---
layout:     post
title:      "notebook - Computational Intelligence and Machine Learning"
date:       2021-09-11 09:30:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - HKU CS
---

# assignment

## assignment 1

* Q1. write the graph search version of DFS
  * 根据老师上课给的ppt，将代码实现就好了，难度不大。
  * 注意用util内的stack去做，能省一些时间
  * 这里需要注意的是python的list直接赋值给另一个变量是分享的指针，共享一块内存。需要使用list.copy()进行深拷贝。
* Q2.  BFS
  * 更改一下所用的数据结构为Queue即可
* Q3. uniformcostsearch
  * 更换数据结构为priorityqueue
* Q4. A*
  * 与Q3的区别在于加上了h函数，把原来的cost替换为了cost+h
* Q5. representation for corners problem
  * 算是难度比较大的一题。问题在于，我们不仅要存储下走过的路径，还要存储我们已经经过了哪些corner的状态。
* Q6. cornersHeuristic
  * 主要想法是先走到最近的一个corner，再从该corner走到下一个最近的corner，直到走完全部的corner
* Q7. 
  * 可以试试老师写好的一个距离函数
* Q8. getnumberofattack
  * 计算当前棋盘的受攻击次数，计算方法很多。
* Q9. getbetterboard
  * 选择改动一个棋子后getnumberofattck最少的策略，更新棋盘
  * 修改停止策略，可以考虑随机在几个最优位置中选择一个位置移动，并设置最大优化搜索次数上限。

## Assignment 2

1. Q1. Reflex Agent：大致思路：计算与最近食物的距离，计算与ghost距离，若与ghost距离过近，给一个极大的惩罚
2. Q2. 使用递归。在这里树的深度的意思貌似是一层树包含了pacman的一次移动，以及，各个ghost的一次移动。写一个evaluate函数，用于计算state下的max or min，当移动的agent为pacman时，返回max，当移动的agent为ghost时，返回min。
3. Q3. 按照pdf给的写就好了，需要注意的时在最外层的循环中也得更新alpha
4. Q4. 要融合概率了，这里直接采用了最简单的等概率，可以过检查点
5. Q5. 考虑三个部分。当pacman吃到超级食物时，要尝试去捕捉ghost。当pacman离ghost很近时，要尽量远离ghost。pacman与最近的食物的距离。我在实验中发现采用欧几里得距离会表现比曼哈顿距离好。
6. Q6. 太难了太难了。一开始把终止条件看错了，原来不是三个棋盘赢的棋盘多的人获胜，而是最后一盘棋获胜的人胜利。用强化学习写了写不知道evaluation函数如何定义，于是用了论文中的办法。可以参考[Secrets of 3-Board Tic-Tac-Toe - Numberphile - YouTube](https://www.youtube.com/watch?v=h09XU8t8eUM&ab_channel=Numberphile)。

# Search

## Type of Search

* Uninformed search: 不知道终点在哪里, BFS, DFS, UCS
* Informed search: 用一个h值来模拟与goal的距离
* Local search:evaluate and modify a current state to move closer to a goal state
* Constraint Satisfaction Problems: search faster by understanding states better
* Adversarial Search: 有对手情况

## State Space

The set of all states reachable from the initial state by any sequence of actions

* state space graph: each state occurs only once
* state space tree: states may occur more than one

## Uninformed Search

Strategies: BFS, DFS, UCS

Algorithms: Tree search algorithm(TSA) 基于state space tree, Graph search algorithm(GSA) 基于state space graph

* BFS

  * TSA-BFS可能会造成死循环，因为没有记录访问过的节点。GSA-BFS解决了这个问题。

  * completeness (Always find a solution if one exists): True. 即使某一个分支进入死循环，但其他分支还在正常搜索。

  * Optimal (Always find a least-cost solution): False

  * 时间空间复杂度都较高
* DFS
  * completeness: False? dead loop exists. 如果dfs进入死循环，则之后都只在死循环里了。
* GSA VS TSA

  * GSA requires memory proportional to its time, avoids infinite loops

  * TSA could be stuck in infinite loops, less memory, easier to implement.
* UCS GSA

  * 相比bfs，dfs，差别在于将队列，栈替换为了`优先队列`，入队时将该路径的cost也一起入队，pop时pop出cost最小的路径，继续探索。

|      | complete | optimal | time complexity | space complexity |
| ---- | -------- | ------- | --------------- | ---------------- |
| UCS  | True     | True    |                 |                  |

## Informed Search

指示器是一个`h(n)`函数，a function estimates how close you are to the goal. `designed for a particular search problem`

Greedy best-first search, A\* search

* Greedy best-first search TSA
  * 与UCS其实非常类似，只不过维护的值不再是路径的cost，而是当前位置距离goal state的距离，比如地图中两点的直线距离（这个距离是提前估算的，因此不是准确的，可能存在两点直线距离很近，但中间隔了一条江，无法度过的情况，因此greedy best-first search选出的路径不一定是最优的）
* A\* TSA
  * 相当于USC+Greedy，在一个点时同时维护已走过的cost g(n)与距目标的距离h(n)
  * `f(n) = g(n) + h(n)`
  * 但h(n)的设置就非常重要 `0<=h(n)<=h*(n)`，h\*(n) is the true cost to the nearest goal
  * A* is optimal if an admissible heuristic is used
* Consistency of Heuristic - h(n)
  * h(a) - h(c) <= cost( a to c )
  * consequence of consistency: the f value along a path never decreases, h(a) <= cost(a to c) + h(c)
  * Can you prove that A* GSA is optimal if the f value never decreases? 如果fvalue不下降，则在Sbc之后会搜索Sac，会考虑到其他路径的可能。

![image-20211204152746271](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204152746271.png)

	## Local Search

evaluate and modify one current state rather than systematically explore paths from an initial state.

advantages: Require very little memory, often find reasonable solutions in large spaces.

Example: 8-Queens. select the cheapest neighbor as currentstate every step

## Constraint Satisfaction Problem

A problem is solved when each variable has a value that satisfies certain constraints

examples: 填色问题

* Backtracking Search
  * 每次填一格一种颜色，若该方案下不行则回退到可能的状态。`类似DFS`，但不是一口气把所有邻居节点都塞入，而是一次只塞一个，继续dfs，这个不行了再换下一个。
  * DFS with (1) only consider assignments to a single variable at each point (2) only allow legal assignments at each point
* Improving Backtracking
  * Forward checking(FC)
    * 维护一张表，存储每个domain还能存放的variables。从我的感觉来说，优化的是可以提前一步知道当前方案不行，不用再去填入具体颜色后判断相邻颜色相同才backtrack。`填入一个variable，将邻居中该variable删去。`
  * Constraint propagation(AC-3)
    * 维护一张表，存储每个domain还能存放的variables。但相比Forward checking，AC-3还会关注所有domain之间的constraints，提早筛选出不可能的variables删去，相当于又比forward checking提早了一步知道当前方案不行。`填入一个variable，把整张表扫一遍，把不可能的情况删去`
  * improving backtracking further
    * Minimum remaining values (MRV)
      * 优先选择剩余选项最少的domain
      * 优先选择拥有最多限制的domain做出选择（填色游戏中有最多的邻居）
    * Least constraining value (LCV)
      * 选择能使其他domain可选variables最少的当前variable
      * ![image-20211204163838142](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204163838142.png)

## Adversarial Search

Minimax Search

将对手也当作是高级智能体，会选择对我们最不利的情况。

要搞清楚的是，from bottom to top。

<img src="https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204170642616.png" alt="image-20211204170642616" style="zoom:50%;" />

* Complete：yes
* Optimal: In general no, yes against an optimal opponent
* time complexity: O(b^m)
* space complexity: O(bm)
* Minimax performs a complete DFS exploration of the tree

**Improve the Minimax algorithm**

* DLS
  * search only to a limited depth in the tree. 只能搜索到特定的深度
  * replace terminal utilities with an evaluation function. 用一个评估函数评估当前状态
* Game Tree Pruning
  * ![image-20211204175759084](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204175759084.png)
  * ![image-20211204175808590](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204175808590.png)
  * ![image-20211204180218718](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204180218718.png)
  * it is worth, O(b^m) --> O(b^(m/2))

**Expectimax Search**

Minimax太理想化，把对手想成了智能体；而把对手想成随机选择的又不妥。那么可以给每个路径赋weight，代表走的可能性。

层级为max-chance-max-chance...

![image-20211204182433306](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204182433306.png)

# Markov Decision Processes

* Compute optimal values: use `value iteration` or `policy iteration`

* Compute values for a particular policy: use `policy evaluation`

* Turn your values into a policy: use `policy extraction `(one-step lookahead)

**They are all variations of Bellman updates!**

Deterministic motion (确定的，朝某个方向的概率是100%) VS Stochastic motion(朝某个方向的概率是大概率，仍有概率走其他方向)

transitions are Markovian means the probability of reaching s' from s depends only on s and not on the history of earlier states.

MDP has `a Markovian transition model` and `additive rewards`

* Reward function: R(s,a,s'), 可以是生存的cost，太小或太大都不行
* Policy π: π(s) is the action recommended by the policy π for the state s
* Transition: T(s,a,s'), 一次action

comparing two state sequences, we prefer to get value sooner.

* Discounted rewards: 
  * ![image-20211204214254736](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204214254736.png)
  * Discount factor: define the value worth one step from now is $$\gamma$$, worth two step from now is $$\gamma^2$$. e.g., [1,2,3] and [3,2,1], let $$\gamma = 0.5$$, $$1 + 0.5* 2 + 0.5^2*3 < 3 + 0.5* 2 + 0.5^2*1$$
  * not infinite. 
* Additive rewards
  * ![image-20211204214306178](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204214306178.png)
  * will be infinite, like smart race game 

$$
U\left(\left[s_{0}, s_{1}, s_{2}, \ldots\right]\right)=\sum_{t=0}^{\infty} \gamma^{t} R\left(s_{t}\right) \leq \sum_{t=0}^{\infty} \gamma^{t} R_{\max }=R_{\max } /(1-\gamma)
$$

* V*(s): how good is state s, an expected value
* Q*(s,a): expected utility for having taken action a from s
* π*(s): is the optimal policy for state s, 告诉最优的走法

![image-20211204215154439](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204215154439.png)

## Bellman Equation

$$
\begin{aligned}
&V^{*}(s)=\max _{a} Q^{*}(s, a) \\
&Q^{*}(s, a)=\sum_{s^{\prime}} T\left(s, a, s^{\prime}\right)\left[R\left(s, a, s^{\prime}\right)+\gamma V^{*}\left(s^{\prime}\right)\right] \\
&V^{*}(s)=\max _{a} \sum_{s^{\prime}} T\left(s, a, s^{\prime}\right)\left[R\left(s, a, s^{\prime}\right)+\gamma V^{*}\left(s^{\prime}\right)\right]
\end{aligned}
$$

$$R\left(s, a, s^{\prime}\right)$$ is the reward taking action a from state s.

可以理解为R为增量，V为存量。

![Reward为0的简化版本](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204220351495.png)

这是一个R(s)=0的简化版本。其实每一个循环中，将每个格子都进行了计算，只不过其他格子计算过程全为0，在这里没有写出。

## Value Iteration

$$
V_{k+1}(s) \leftarrow \max _{a} \sum_{s^{\prime}} T\left(s, a, s^{\prime}\right)\left[R\left(s, a, s^{\prime}\right)+\gamma V_{k}\left(s^{\prime}\right)\right]
$$

$$V_{n}(d)$$表示在state d 走n个Value Iteration得到的Value值。

![image-20211204220655002](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204220655002.png)

## Policy Evaluation

### Policy Evaluation

$$
V^{\pi}(s)=\sum_{s^{\prime}} T\left(s, \pi(s), s^{\prime}\right)\left[R\left(s, \pi(s), s^{\prime}\right)+\gamma V^{\pi}\left(s^{\prime}\right)\right]
$$

action的选择不再凭借max，而是根据$$\pi(s)$$函数，用来评估策略？

### Policy Extraction

$$
\pi^{*}(s)=\arg \max _{a} Q^{*}(s, a)
$$

找在q-values中值最大的action，当作最优action

## Policy Iteration与Value Iteration的区别

**都是为了更好地评估每个状态的Value值的方法。**

Value Iteration，根据max函数计算出每个位置进行每个动作的Q值，通过找到max的Q值作为新的Value值。

Policy Iteration分为两步，先根据之前得到的policy，一个固定的路径，更新所有格子的value值。第二步，再根据不同的Value值更新Policy。

![image-20211204231128974](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211204231128974.png)

## RL与MDP的区别

MDP的action都是已知的，但RL的一些action是未知的。

# Reinforcement Learning

New to MDP: Don't know T(transition) or R(reward), i.e., we don't know what the actions do. Must actually try actions and states out to learn.

## Model-Based Learning

通过不断让agent尝试，得到估计的T(s,a,s')与R(s,a,s')，T就按照简单的出现概率估计，而R则可以求平均

再通过估计的T与R计算MDP中的V

## Model-Free Learning

## Direct Evaluation

![image-20211205172305332](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211205172305332.png)

上图相当于一个introduction，这是一个简单的估算，从某个state出发能得到的最终收益。但这种做法存在问题，所以我们引入了sample。

### 

$$
\begin{aligned}
\text { sample }_{1} &=R\left(s, \pi(s), s_{1}^{\prime}\right)+\gamma V_{k}^{\pi}\left(s_{1}^{\prime}\right) \\
\text { sample }_{2} &=R\left(s, \pi(s), s_{2}^{\prime}\right)+\gamma V_{k}^{\pi}\left(s_{2}^{\prime}\right) \\
\text {... } & \\
\text { sample }_{n} &=R\left(s, \pi(s), s_{n}^{\prime}\right)+\gamma V_{k}^{\pi}\left(s_{n}^{\prime}\right) \\
V_{k+1}^{\pi}(s) & \leftarrow \frac{1}{n} \sum_{i} \text { sample }_{i}
\end{aligned}
$$

既然我们不能得知T(s,a,s')，那么我们可以拿多次在$$\pi$$下sample的平均来模拟获得V，这样就跳过了T函数。

==计算某个state的V值需要一路走到exit，这是我们唯一确认的sample？==



### Temporal Difference Learning

与direct evaluation的区别是每在$$\pi$$下进行一次sample，就马上更新，而不是进行一批samples再求平均。
$$
Sample of \mathrm{V}(\mathrm{s}): \quad sample =R\left(s, \pi(s), s^{\prime}\right)+\gamma V^{\pi}\left(s^{\prime}\right)  \\
Update to \mathrm{V}(\mathrm{s}): \quad V^{\pi}(s) \leftarrow(1-\alpha) V^{\pi}(s)+(\alpha) sample  \\
Same update: \quad V^{\pi}(s) \leftarrow V^{\pi}(s)+\alpha\left(\right. sample \left.-V^{\pi}(s)\right)
$$

## Q-learning

不去迭代计算V，不去估计T与R，而迭代估计Q，这样我们还可以计算得到optimal policy。

sample-based Q-value iteration：
$$
Q_{k+1}(s, a) \leftarrow \sum_{s^{\prime}} T\left(s, a, s^{\prime}\right)\left[R\left(s, a, s^{\prime}\right)+\gamma \max _{a^{\prime}} Q_{k}\left(s^{\prime}, a^{\prime}\right)\right]
$$
Steps：

* Recieve a sample (s,a,s',r)
* Consider old estimate: Q(s,a)
* Consider new sample estimate: $$\text { sample }=R\left(s, a, s^{\prime}\right)+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}\right)$$
* Incorporate new estimate: $$Q(s, a) \leftarrow(1-\alpha) Q(s, a)+(\alpha)[\text { sample }]$$

![image-20211205191707949](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211205191707949.png)

Advantages: Q-learning converges to optimal policy

Limitations: 1. You have to explore enough 2. You have to eventually make the learning rate small enough 3. it doesn't matter how you select actions

## Exploration vs. Exploitation

Exploration是放弃已知奖励，去探索未知路径。Exploitation是走已知的道路，尝试最大化已知奖励。

整合这两步可以很简单，比如走每一步时设置一个random参数，有一定概率$$\epsilon$$去exploration，一定概率$$(1-\epsilon)$$exploitation。

Problem: 可能某一个时间点，learning is done，但我们会keep thrashing around。

Solution: lower $$\epsilon$$ or Exploration functions

![image-20211205193312518](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211205193312518.png)

## Approximate Q-Learning

Basic Q-Learning需要记录每个state的Q-values，需要占用很大的内存。

Approximate Q-Learning: 可以用不同的features来定义当前Q-state，比如distance to closest ghost。
$$
\begin{gathered}
V(s)=w_{1} f_{1}(s)+w_{2} f_{2}(s)+\ldots+w_{n} f_{n}(s) \\
Q(s, a)=w_{1} f_{1}(s, a)+w_{2} f_{2}(s, a)+\ldots+w_{n} f_{n}(s, a)
\end{gathered}
$$
![image-20211205200540597](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211205200540597.png)

![image-20211205201025403](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211205201025403.png)

# Perceptron & Adaline

## Terminology and Notations

$$x_{i}$$代表x的第i个feature，$$x^{j}$$代表x的第j个sample。

## Roadmap

Preprocessing: feature selection, extraction and scaling; dimensionality reduction; sampling

Learning: Model selection; Cross-validation; Performance metric; Hyperparameter optimization

Evaluation and Prediction

## Perceptron

$$
\mathrm{z}=w_{0} x_{0}+w_{1} x_{1}+\ldots+w_{m} x_{m}=w^{T} x \\
\phi(\mathrm{z})=\left\{\begin{aligned}
1 & \text { if } z \geq 0 \\
-1 & \text { otherwise }
\end{aligned}\right.
$$

update the weights:
$$
w_{j} :=w_j+\triangle w_j \\
\triangle w_j = \eta(y^{(i)}-\hat{y}^{(i)})x_{j}^{i}
$$
$$\eta$$ is the learning rate. $$\hat{y}^{(i)}$$ is the predicted class label.

**only guaranteed if two classes are linearly separable**, XOR problem cannot be separated by perceptron

![image-20211205213544130](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211205213544130.png)

## Adaline

ADAptive Linear Neuron: Adaline

Improvement on Perceptron algorithm: In Adaline the weights are updated based on linear activation function

![image-20211205214109407](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211205214109407.png)

cost function: SSE
$$
J(\boldsymbol{w})=\frac{1}{2} \sum_{i}\left(y^{(i)}-\phi\left(z^{(i)}\right)\right)^{2}
$$
Feature Scaling: gradient descent converges more quickly if data follows a standard distribution. mean value of 0, standard deviation of 1.
$$
w:=w+\Delta w \\
\Delta w=-\eta \nabla J(w) \\
\frac{\partial J}{\partial w_{j}} =-\sum_{i}\left(y^{(i)}-\phi\left(z^{(i)}\right)\right) x_{j}^{(i)}
$$
**Batch Gradient Descent**:
$$
\Delta w_{j}=-\eta \frac{\partial J}{\partial w_{j}}=\eta \sum_{i}\left(y^{(i)}-\phi\left(z^{(i)}\right)\right) x_{j}^{(i)}
$$
**Stochastic Gradient Descent**: update all the weights for each sample

* reaches convergences faster because more frequent weight updates
* can escape shallow local minima more readily
* can used in online learning as new training data arrives
* 训练集必须打乱

**Mini-Batch Learning**

* convergence is reached faster than batch gradient descent
* Furthermore, mini-batch learning allows us to replace the for loop over the training samples in stochastic gradient descent with vectorized operations? ==unclear==

# Logistic Regression, SVM, Decision Trees, KNN





