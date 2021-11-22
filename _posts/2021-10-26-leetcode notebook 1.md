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

## 栈

栈在leetcode解法里很多是直接使用了`list`，使用`list.append`与`list.pop()`即可实现栈的push与pop。

想与队列的实现对应起来的话，也可以使用`collections.deque()`来实现。

**单调栈**

参考leetcode [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

在入栈之前，比较当前元素与栈顶元素的大小，把大于或小于（视情况而定）当前元素的栈顶元素`都`pop掉，再将当前元素入栈，即可获得单调栈。（需要注意的是很多单调栈的题，栈中存的是index）

## DFS

**DFS的模板可以直接修改BFS模板中的数据结构为栈来实现**，也可以直接调**递归**，用系统中的隐式栈来实现。最糟糕的情况下空间复杂度为O(h)，h为最深深度。

递归：

```python
visited_nodes = [start_node]
def dfs(node_cur, node_target, visited_nodes):
    if node_cur == node_target: return true or 0(如计算步长)
    for neighbor in node.get_neighbors():
        if (neighbor not in visited_nodes):
            visited_nodes.append(neighbor)
            return true if dfs(neighbor, node_target, visited_nodes) == true
        	return 1 + dfs(neighbor, node_target, visited_nodes) (如计算步长)
```

# 树

## 二叉树的遍历

reference: [Link](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/solution/python3-er-cha-shu-suo-you-bian-li-mo-ban-ji-zhi-s/) 

### 递归

简单，但有堆栈溢出的风险

```python
def recursion(self, root:TreeNode) -> List[int]:
	if not root:
        return []
   	# 前序
    return [root.val] + self.recursion(root.left) + self.recursion(root.right)
	# 中序与后序只需要更换return中的顺序即可
```

通用版本

```python
def recursion(self, root:TreeNode) -> List[int]:
    def dfs(cur):
        if not cur:
            return
        # 前序递归
        res.append(cur.val)
        dfs(cur.left)
        dfs(cur.right)
        # 中序与后序只需要改变顺序
    res = []
    dfs(root)
    return res
```

### 迭代

前、中、后序通用模板（只需要一个栈的空间）

```python
def inorderTraversal(self, root:TreeNode) -> List[int]:
    res = []
    stack = []
    # add one more variable to save the current_node
    cur = root
    # inorderTraversal, first find the node in the bottom left corner
    # here use stack or cur, because stack will be empty when we move
    # to the top of the tree
    while stack or cur:
        # if cur(the right children node) is None, move to the upper layer
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        res.append(cur.val)
        cur = cur.right
        # # 前序，相同模板
        # while stack or cur:
        #     while cur:
        #         res.append(cur.val)
        #         stack.append(cur)
        #         cur = cur.left
        #     cur = stack.pop()
        #     cur = cur.right
        # return res
        
        # # 后序，相同模板
        # while stack or cur:
        #     while cur:
        #         res.append(cur.val)
        #         stack.append(cur)
        #         cur = cur.right
        #     cur = stack.pop()
        #     cur = cur.left
        # return res[::-1]
    return res
```





# 随机算法

## Fisher-Yates洗牌算法

给一个数组list，返回它的shuffle。思想很简单，返回的shuffle不能与原数组一样，将原数组的一项位置排除即可。

```python
def shuffle(lis) -> List[int]:
        for i in range(len(lis)):
            j = random.randrange(i, len(lis))
            lis[i], lis[j] = lis[j], lis[i]
        return lis
```



# Python

## 位运算

python中的位运算只能用于`int`类型

| 位运算符 | 说明     | 使用形式 | 举 例                            |
| -------- | -------- | -------- | -------------------------------- |
| &        | 按位与   | a & b    | 4 & 5                            |
| \|       | 按位或   | a \| b   | 4 \| 5                           |
| ^        | 按位异或 | a ^ b    | 4 ^ 5                            |
| ~        | 按位取反 | ~a       | ~4                               |
| <<       | 按位左移 | a << b   | 4 << 2，表示整数 4 按位左移 2 位 |
| >>       | 按位右移 | a >> b   | 4 >> 2，表示整数 4 按位右移 2 位 |

## Zip

zip函数可将可迭代的对象作为参数，将对象中对应元素打包成一个元组，然后返回由这些元组组成的列表。

利用`*`号操作符，可以将元组解压为列表，即解压前的状态

一些神奇的做法：

* 创建字典：`stu = dict(zip(names, scores))`
* 反向解压字符串列表：zip(*str)，可以用于判断最大相同前缀等

```python
str = ['flower','flight','flow']
list(zip(*str))
# output [('f', 'f', 'f'), ('l', 'l', 'l'), ('o', 'i', 'o'), ('w', 'g', 'w')]
```

## Collections library

* collections.deque()
* collections.Counter()
  * 支持用字典、字符串、list进行初始化，相当于生成一个记录count的字典。
  * 可以使用`sorted(counter)`对key值进行排序
  * 支持使用`counter.most_common(n)`返回最多的前n个item

## bisect library

* bisect_left(list,value,lo=0,hi=None)

  * 在列表list中搜索适合插入value的位置，但不会真正插入。当在list中出现value时，返回value左边位置的索引。lo为搜索的起始位置，默认为0。hi为搜索的结束位置，默认为len(a)。

  * ```python
    li = [1, 23, 45, 12, 23, 42, 54, 123, 14, 52, 3]
    li.sort()
    print(li)
    print(bisect.bisect_left(li, 3))
    # return
    # [1, 3, 12, 14, 23, 23, 42, 45, 52, 54, 123]
    # 1
    ```

* bisect_right(list,value,lo=0,hi=None)，同理。

* insort_left(list, value, lo=0, hi=None)  insort_right(list, value, lo=0, hi=None). 在列表list中搜索适合插入value的位置，并真正插入。其他与bisect相同。

## Random library

* random.shuffle(list)  随机打乱一个数组，注意该操作没有返回值，只是打乱list。
