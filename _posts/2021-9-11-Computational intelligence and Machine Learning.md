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
