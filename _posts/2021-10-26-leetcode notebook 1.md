---
layout:     post
title:      "Leetcode Notebook"
subtitle:   
date:       2021-10-26 15:00:00
updatedate:
author:     "Jpy"
header-img: "https://th.bing.com/th/id/R.d7d8b5a0c1be9005b4b000dd29bef530?rik=qdZkHLlkYgMC%2fg&pid=ImgRaw&r=0"
no-catalog: False
onTop: false
latex: false
tags:
    - CS
    - LeetCode
---

# 496. Next Greater Element I

[496. 下一个更大元素 I - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/next-greater-element-i/)

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
