---
layout:     post
title:      "Coursera-Scala环境配置"
subtitle:   "Coursera-Scala Environment Configuration"
date:       2021-01-24 17:46:00
author:     "Jpy"
header-img: "img/coursera.jpg"
tags:
    - Coursera
    - Scala
---

### Coursera中Scala课程推荐使用的环境为Intellij IDEA + sbt

> 按照默认的配置是不需要自行安装scala的,直接在Intellij IDEA中构建sbt项目即可

### 我的设备  
- Windows10 x64

### 软件背景:  
- IntelliJ IDEA
- SBT 

> 这两个直接按照课程中的安装文档一路一路安装下来即可  

### 写作业的步骤
1. 下载作业文件,解压
2. 右键-选择open folder as intellij IDEA project `这时会自动构建项目，这个过程一般来说两三分钟`
3. 打开左上角项目文件中的src-main-scala目录，作业文件在这里
4. 编写代码文件，想要测试某个函数，可以右键该文件-Scala REPL，输入`import file_name `，再直接调用函数，输入参数测试即可。
5. 提交文件：点击底部导航栏sbt shell。**这里需要注意弹出sbt版本有更新，是否override时一定不能点击update，而是使用原来的版本，若要使用新版本，请参考下面的链接进行操作，否则将无法提交作业**click👉[scala作业submit后报错](https://www.coursera.org/learn/progfun1/discussions/forums/KZzRnLHpEemRkwrWmYSNRA/threads/yYqUyC1aRqiKlMgtWvaouQ)
6. 在构建好的输入栏中输入【submit 你的邮箱 识别码】即可

