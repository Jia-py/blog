---
layout:     post
title:      "Create windows index for github folder"
subtitle:   "为github文件夹创建windows索引"
date:       2022-01-22 14:14:00
updatedate: 2022-01-22 14:14:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: false
# 生活，工作，笔记（个人理解消化心得），文档（方便后续查阅的资料整理），其他
tags:
    - 文档
---

# Background

When I use the `powertoys run`, which is a search tool offered by microsoft using windows index to open folders or apps quickly like the spotlight in MacOS, I found the github folders can never be added to the windows index.

After several times rebuilding indexes of whole folders, the problem is still unsolved.

# How to solve it?

By searching the internet, I found some feedback posts on the microsoft website. It's a normal problem due to the windows update. The main reason for that is the presence of hidden `.git` document in every repo folder.

One solution is like

![image-20220122142204198](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220122142204198.png)

It is important to note that we should add every hidden `.git` document to excluded, not just add the `/github/.git` document to excluded.

![image-20220122142522135](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220122142522135.png)

After that, you should find these documents can be searched from windows search or powertoys run.