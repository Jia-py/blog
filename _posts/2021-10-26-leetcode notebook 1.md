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

更新文档在[Leetcode Notebook - NoteBook (gitbook.io)](https://jia-pengyue.gitbook.io/notebook/algorithm/leetcode_notebook)

# 1. 队列 & 栈

`python`的队列实现比较好的是用`collections.deque`实现，也可以使用列表简单实现，入队为list.append()，出队使用list.pop(0)。栈的简单实现也一样，入栈为list.append()，出栈为list.pop(-1)即可实现。

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

## 优先队列

```python
import heapq
q = []
heapq.heappush(q,(rank,'code'))
# m
node = heapq.heappop(q)
```

# 2. 树

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
    'add one more variable to save the current_node'
    cur = root
    '''inorderTraversal, first find the node in the bottom left corner
    here use stack or cur, because stack will be empty when we move
    to the top of the tree'''
    while stack or cur:
        # if cur(the right children node) is None, move to the upper layer
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        res.append(cur.val)
        cur = cur.right
        '''
        # 前序，相同模板
        while stack or cur:
            while cur:
                res.append(cur.val)
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            cur = cur.right
        return res
        
        # 后序，相同模板
        while stack or cur:
            while cur:
                res.append(cur.val)
                stack.append(cur)
                cur = cur.right
            cur = stack.pop()
            cur = cur.left
        return res[::-1]
        '''
    return res
```

### 层序遍历 BFS

推荐还是使用第一种方法，比较容易理解

```python
'第一种方法，通过维护每层的size值，得知何时停止bfs，得知每层的信息'
def levelOrder(self,root:TreeNode):
    result = []
    if not root: return result
    q = deque([root])
    while q:
        size = len(q)
        level = [] # 存储该层的所有元素
        while size > 0:
            cur = q.popleft()
            level.append(cur.val)
            if cur.left: q.append(cur.left)
            if cur.right: q.append(cur.right)
            size -= 1
        result.append(level)
    return result

'第二种方法，把下一层的元素单独存在新列表中，再将新列表赋值给q'
def levelOrder(self,root:TreeNode):
    result = []
    if not root: return result
    q = [root]
    while q:
        level = [] # 存储该层的所有元素的val
        nxt = [] # 存储下一层的元素
        for node in q:
            level.append(node.val)
            if cur.left: nxt.append(cur.left)
            if cur.right: nxt.append(cur.right)
        result.append(level)
        q = nxt
    return result
```

# 3. 动态规划

环形数组问题一般可以分解为`[0:-1]`与`[1:]`两个子数组的问题，或者有环情况时，反向考虑中间的部分。

## 背包问题

![416.分割等和子集1](https://img-blog.csdnimg.cn/20210117171307407.png)

### 01背包

参考链接：[代码随想录 - 背包问题](https://programmercarl.com/背包理论基础01背包-2.html#一维dp数组-滚动数组)

**二维dp方法**

dp\[i\]\[j\] 表示从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少

递推公式： $d p[i][j]=\max (d p[i-1][j], d p[i-1][j- weight [i]]+\operatorname{value}[i])$

简而言之，是两种状态，一种为由i-1到i的过程，不拿第i个物品；一种为要拿第i个物品

```python
'外围两个循环的顺序可以更换，无所谓'
for i in range(len(weight)):
    for j in range(len(bagSize)):
        '如果当前袋子装不下当前物品，直接用上一轮的结果'
        if (j<weight[i]): dp[i][j] = dp[i-1][j];
        else: dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])
```

**一维dp方法**

思想是和二维dp一样的，只不过遍历下一个物品时，直接将dp的值在上一层的数组上覆盖，这样就只需要一维数组了。

递推公式：$\mathrm{d} p[\mathrm{j}]=\max (\mathrm{dp}[\mathrm{j}], \mathrm{dp}[\mathrm{j}- weight [\mathrm{i}]]+\operatorname{value}[\mathrm{i}])$

简单理解，也是两种状态，一种为不取第i个物品;一种为取第i个物品

```python
'这里最好先遍历物品，再遍历bagsize'
for i in range(len(weight)):
    '这一行里直接把bagsize<weight[i]的情况去除了'
    '这里一定得倒序遍历，正序遍历会使同一物品被使用多次'
    for j in range(bagSize, weight[i] - 1, -1):
        dp[i][j] = max(dp[j],dp[j-weight[i]] + value[i])
```

# 4. 搜索算法

## 二分搜索

基础写法

```python
def binary_search(nums, target):
    if not nums: return -1
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

这里采用的是最高级的写法，可以返回最后位置左右两个元素，以解决有些题目的需求

```python
def binary_search(nums, target):
	if not nums: return -1
    start, end = 0, len(nums) - 1
    # 使用start+1<end是为了防止死循环，如[1,1],1,同时，只有这样在下面的赋值时才可以将start与end直接赋值mid，否则应该参考简单写法
    while start + 1 < end:
        mid = start + (end - start) // 2
        if nums[mid] == target: return mid
        elif nums[mid] < target: start = mid
        else: end = mid
    if nums[start] == target: return start
    if nums[end] == target: return end
    return -1
```

# 5. 并查集

此处代码模板采用了==路径压缩==的优化方法，没引入关于秩的优化。

```python
class UnionFind:
	def __init__(self):
        self.father = {} # key-节点,value-父节点
        self.size_of_set = {} # key-父节点,value-size
        self.num_of_set = 0
    def add(self, x):
        # 点如果已经存在，操作无效
        if x in self.father: return
        # 初始化点的父节点，点所在集合大小，另外使集合数量+1
        self.father[x] = None
        self.num_of_set += 1
        self.size_of_set[x] = 1
    def merge(self, x, y):
        # 找到两个节点的根
        root_x, root_y = self.find(x), self.find(y)
        # 如果不是同一个根则连接
        if root_x != root_y:
            self.father[root_x] = root_y
            self.num_of_set -= 1
            self.size_of_set[root_y] += self.size_of_set[root_x]
	def find(self, x):
        root = x
        while self.father[root] != None:
            root = self.father[root]
        # 优化步骤，将路径上所有点指向根节点root
        while x != root:
            original_father = self.father[x]
            self.father[x] = root
            x = original_father
        return root
    def is_connected(self, x, y):
        return self.find(x) == self.find(y)
    def get_num_of_set(self):
        return self.num_of_set
    def get_size_of_set(self, x):
        return self.size_of_set[self.find(x)]
   	
```

# 6. 字符串算法

## KMP匹配子串算法

用于找到字符串中`s`的连续子串`l`的下标

相较于暴力解法，KMP可以在匹配到不相符的字符时跳过多格。

```python
class Solution:
    # haystack为字符串,needle为待匹配的子串
    def strStr(self, haystack: str, needle: str) -> int:
        if needle == "":
            return 0
        def getNxt(x):
            # 从长到短遍历
            for i in range(x,0,-1):
                if needle[0:i] == needle[x-i+1:x+1]:
                    return i
            return 0
        nxt = [getNxt(x) for x in range(len(needle))]
        tar = 0
        pos = 0
        while tar<len(haystack):
            if haystack[tar] == needle[pos]:
                tar+=1
                pos+=1
            elif pos:
                pos = nxt[pos-1]
            else:
                tar += 1
            if pos == len(needle):
                return (tar-pos)
                pos = nxt[pos-1]
        return -1
```

## LIS 最长递增子序列算法

这里的子序列可以是不连续的

1. 动态规划O(N^2)

```python
def lengthOfLIS(self, nums: List[int]) -> int:
    	# nums为数组list
        if not nums:
            return 0
        n = len(nums)
        # 初始化dp数组，dp[i]代表nums[0]到nums[i]中的最长递增子序列的长度
        dp = [1] * n
        # 递推方程为
        # 在i>j的情况下，筛选出并且符合nums[i] > nums[j]的j
        # 则dp[i] = max(dp[符合上述条件的j]) + 1 
        for i in range(n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i],dp[j] + 1)
        return max(dp)
```

1. 贪心算法，二分查找O(NlogN)

```python
def lengthOfLIS(self, nums: List[int]) -> int:
    	# 这里lis[i]为长度为i+1的子序列末尾元素的最小值
        # 通过证明我们可以得知lis序列是严格递增的
        lis = [nums[0]]
        # 循环每个nums元素
        for num in nums[1:]:
            # 如果该元素大于lis中最后一个元素，说明最长子序列可以延长
            '若题目要求非严格递增，则num>=lis[-1]，且使用bisect_right'
            if num > lis[-1]:
                lis.append(num)
            # 否则说明不可以延长
            # 找到lis中第一个大于num的元素，替换
            # 这样我们优化了该长度下序列末尾的最小值
            else:
                lis[bisect.bisect_left(lis, num)] = num
        # 最长非递减子序列的长度即为lis的长度
        return len(lis)
```

## Trie

Trie有两种不同实现方法，不同点在于如何表示当前节点是否为结束字符。

1. 每个节点维护一个end的变量，当当前节点为结束字符时`end==True`
2. 在词后插入一个字符，比如`#,$`等等，不会在正常字符串中出现的字符作为结束字符。

这里的模板采用的是第一种方式，较为普遍。

```python
class Trie:
    def __init__(self):
        # key为当前字符，value为以当前key为前缀的前缀树
        self.child = dict()
        # 当前节点是否为结束节点
        self.isend = False
    def insert(self,word):
        rt = self
        for w in word:
            if w not in rt.child:
                rt.child[w] = Trie()
            rt = rt.child[w]
        rt.isend = True
    # 搜索树中是否有完整单词word
    def search(self, word):
        rt = self
        for w in word:
            if w not in rt.child:
                return False
            rt = rt.child[w]
        return rt.isend == True
    # 搜索树中是否有前缀prefix
    def startsWith(self, prefix):
        rt = self
        for w in prefix:
            if w not in rt.child:
                return False
            rt = rt.child[w]
        return True
```

# A. 随机算法

## Fisher-Yates洗牌算法

给一个数组list，返回它的shuffle。思想很简单，返回的shuffle不能与原数组一样，将原数组的一项位置排除即可。

```python
def shuffle(lis) -> List[int]:
        for i in range(len(lis)):
            j = random.randrange(i, len(lis))
            lis[i], lis[j] = lis[j], lis[i]
        return lis
```



# B. Python

## inf

```python
float('inf')
float('-inf')
int(1e9)
```

## 基础数据结构

set 集合：集合是无序**不重复元素的序列，不支持索引**

* 增：添加一个元素`set.add()`，添加多个元素`set.update(list or set)`

* 删：`set.discard()`与`set.remove()`，但使用remove时元素本身不存在会报错。`set.clear()`清空集合

* 集合运算

  * 子集：

    ```python
    subset < set # return True or False
    subset.issubset(set) # return True of False
    ```

  * 并集: `A|B` or `A.union(B)`

  * 交集：`A&B` or `A.intersection(B)`

  * 差集：`A-B` or `A.difference(B)`

  * 对称差：只属于其中一个集合，不属于另一个集合的元素的集合。`A^B` or `A.symmetric_difference(B)`

tuple 元组: 只能查看，不能增删改。元组的连接可以使用`+`

## ASCII

`ord('a') == 97`

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

经典位运算操作：

1. 除以2 `number >> 1`
2. 一个2^n^的数字与另一个[2^n-1^,2^n^)的数字求和，可以使用`number1 | number2`

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

* 获取二维数组的一列

```python
col_maxes = [max(col) for col in zip(*two_D_list)]
```

## Collections library

* collections.deque()
* collections.Counter()
  * 支持用字典、字符串、list进行初始化，相当于生成一个记录count的字典。
  * 可以使用`sorted(counter)`对key值进行排序
  * 支持使用`counter.most_common(n)`返回最多的前n个item
  * counter是一个字典，其中的key是没有顺序的，两个不同的counter可以比较是否相同。

## bisect library

* bisect_left(list,value,lo=0,hi=None)

  * 在列表list中搜索适合插入value的位置，但不会真正插入。当在list中出现value时，返回value左边==空缺==位置的索引。lo为搜索的起始位置，默认为0。hi为搜索的结束位置，默认为len(a)。

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

## Sorted

```python
sorted(iterable, cmp=None, key=None, reverse=False)
```

* 返回排序数组的原索引列表 

```python
index = sorted(range(len(list)),key=lambda k:list[k])
```

* 根据数组第几列sort


```python
sorted(list,key = lambda a:a[2])
```

## map()

```python
map(function, iterable, ...)
```

将function用于list中每个元素，返回一个迭代器

```python
# 将list每个元素取绝对值
lis = list(map(abs,lis))
# 平方
lis = list(map(lambda x: x ** 2, lis))
```

## pow()

```python
# 返回a^b
pow(a,b)
# 返回(a^b)%c
pow(a,b,c)
```

## datetime

```python
import datetime
# 创建datetime对象
x = datetime.datetime(year,month,day,hour,minute,second...)
x = datetime.datetime.strptime('1999-04-21','%Y-%m-%d')
# 个位数显示删去前置0，添加'#‘
x = datetime.datetime.strptime(str,'%Y-%#m-%#d')
# 日期加减
now + datetime.timedelta(days=1)
(datetime1-datetime2) .days .seconds # 并b
# datetime转str,常用对象属性
x.strftime('%A') # Weekday,e.g.,Wednesday
x.strftime('%w') # weekday,e.g.,3
x.strftime('%d') # day,e.g.,31
x.strftime('%m') # month,e.g.,12
x.strftime('%Y') # year,e.g.,2021
x.strftime('%H') # hour,e.g.,23
x.strftime('%M') # minute,e.g.,41
x.strftime('%S') # second,e.g.,30
x.strftime('%j') # 一年中的天数,365
x.strftime('%W') # 第几周，每周第一天是周一
```

