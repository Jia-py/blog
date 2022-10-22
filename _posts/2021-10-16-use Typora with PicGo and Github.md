---
layout:     post
title:      "Use Typora with PicGo and Github"
subtitle:   "使用PicGo与Github实现Typora截图自动上传图床"
date:       2021-10-16 02:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
# 生活，工作，笔记（个人理解消化心得），文档（方便后续查阅的资料整理），其他
tags:
    - 文档
---

# 背景

每次在typora中插入图片都需要以下繁琐的步骤

1. 图片保存至本地github文件夹
2. push至github
3. 将url填入markdown中

# 实现过程

1. 根据typora的`图像设置`中的设置，在这里我采用的时APP版本的PicGo

![typora图像设置](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211016021027280.png)

2. 下载PicGo后，将exe文件路径填入
3. 在PicGo中如图设置

![PicGo设置](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211016021151531.png)

该处的`token`是来自于github用户`setting-development setting`中，可通过赋予token的权限，`generate new token`即可。

4. 全部设置完毕，可以点击typora设置中的验证图片上传选项验证一下，若弹出success，则以配置完毕。

# 使用方法

直接截图复制进入typora即可，会自动上传至图床，并且转为markdown语法写入md文件。
