---
layout:     post
title:      "Leetcode Notebook"
subtitle:   
date:       2021-10-26 15:00:00
updatedate:
author:     "Jpy"
header-img: "img/leetcode.png"
no-catalog: False
onTop: false
latex: false
tags:
    - CS
    - LeetCode
---

[496. Next Greater Element I](https://leetcode-cn.com/problems/next-greater-element-i/)

暴力解题时间复杂度太高，这里需引入`单调栈`与`哈希表`

单调栈相当于是排序的栈，**哈希表**在python中可以用**字典**来实现<br>

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 哈希表，存储每个nums1中对应的返回值
        
        res = {}   
        # 用于构建单调栈
        
        stack = []   
        # 将nums2倒序遍历  
        
        for num in reversed(nums2): 
            # 若栈空或当前值大于栈顶的元素，则将栈顶的pop  
            
            while stack and num >= stack[-1]: 
                stack.pop()
            # 构建哈希，注意这里的num是nums2中的num，因为题目提到nums1的元素，nums2都有，所以可以提前构建哈希表，到时候直接遍历nums1读就行了
            
            res[num] = stack[-1] if stack else -1 
            # 将当前数字加入单调栈
            
            stack.append(num) 
        return [res[num] for num in nums1]
```

375. Guess Number Higher or Lower 2

接触到的第一道动态规划题目，废了好多时间理解。

精选题解：https://leetcode-cn.com/problems/guess-number-higher-or-lower-ii/solution/dong-tai-gui-hua-c-you-tu-jie-by-zhang-xiao-tong-2/

# 队列 & 栈

`python`的队列实现比较好的是用`collections.deque`实现，也可以使用列表简单实现，入队为list.append()，出队使用list.pop(0)。栈的简单实现也一样，入栈为list.append()，出栈为list.pop(-1)即可实现。

## 循环队列

即可以重复使用dequeue空出的空间的队列，拥有`head`与`tail`两个记录索引，在初始化时，`head`与`tail`均不指向数组内任何一个位置，可以初始化为-1，在dequeue队列中最后一个元素时，也需要再次将`head`与`tail`赋值为-1. [动画演示](https://leetcode-cn.com/leetbook/read/queue-stack/kgtj7/)

[622. 设计循环队列 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/design-circular-queue/)

```python
class MyCircularQueue:
    def __init__(self, k: int):
        "initialize parameters"
        self.length = k
        self.queue = [None] * k
        self.head = -1
        self.tail = -1
    def enQueue(self, value: int) -> bool:
        if self.Empty():
            self.queue[0] = value
            self.head, self.tail = 0, 0
            return True
        elif self.isFull(): return False
        else:
            self.tail = (self.tail+1)%self.length
            self.queue[self.tail] = value
            return True
    def deQueue(self) -> bool:
        if self.isEmpty(): return False
        elif self.head == self.tail:
            self.head, self.tail = -1, -1
            return True
        else:
            self.head = (self.head+1)%self.length
            return True
    def Front(self) -> int:
        if self.isEmpty(): return -1
        else: return self.queue[self.head]
    def Rear(self) -> int:
        if self.isEmpty(): return -1
        else: return self.queue[self.tail]
    def isEmpty(self) -> bool:
        if self.head == -1 and self.tail == -1: return True
        else: return False
    def isFull(self) -> bool:
        if (self.tail+1)%self.length == self.head: return True
        else: return False
```

## BFS

Template

```python
def bfs(start_node):
    queue = collections.deque([start_node])
    distance = {start_node : 0}
    while queue:
        node = queue.popleft()
        if node 是终点:
            break or return something
        for neighbor in node.get_neighbors():
            if neighbor in distance:
                continue
            queue.append(neighbor)
            distance[neighbor] = distance[node] + 1
    return distance
	return distance.keys()
	return distance[end_node]
```



