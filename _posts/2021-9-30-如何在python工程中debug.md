---
layout:     post
title:      "How to debug in python project, especially need to input args in terminal"
subtitle:   "如何在需要使用命令行启动的python工程中debug"
date:       2021-09-30 22:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

# Background

在ML的课程作业中，我们需要运行多个`.py`文件的工程项目，并且在运行时需要在命令行中输入一些args用来控制不同的实验场景。在这种情况下，如何debug呢。

# Method

1. 打开VScode，将项目文件夹添加入工作区

2. 点击调试按钮，点击新建`launch.json`配置文件

3. 在文件中加入代码，使其符合以下格式

   ```json
   {
       // 使用 IntelliSense 了解相关属性。 
       // 悬停以查看现有属性的描述。
       // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Python: 当前文件",
               "type": "python",
               "request": "launch",
               "program": "${file}",
               "console": "integratedTerminal",
               "args": [ 
                   // "-p", "ReflexAgent",
                   // "-l", "testClassic",
                   "-q", "q5"
                ]
           }
       ]
   }
   ```

4. 如，我们要运行`python autograder.py -q q5`，我们就在args中填入q5，并且在我们想要debug的页面上打上断点，然后打开`autograder.py`文件，点击调试中上方调试按钮。即可单步调试。