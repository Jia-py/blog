---
layout:     post
title:      "CSGO auto-aim"
subtitle:   
date:       2022-01-06 23:00:00
updatedate:
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: true
# 生活，工作，笔记（个人理解消化心得），文档（方便后续查阅的资料整理），项目，其他
tags:
    - xiang'mu
---

# 尝试

总共时间花费了3天，尝试了三种不同的模型，感受到了从理论到应用的巨大挑战。

## YoloV3

第一次尝试yolov3直接使用的训练好的coco训练集中的person类，训练效果不错，但识别速度太慢了，无法使用在csgo中

## Yolo-FastestV2

速度非常快，即使是用cpu也能跑出30fps

但缺点也很明显，识别的准确度非常低，很多时候只有人正对视角时才可能识别出来。

在这之中，我与室友尝试了自己构造数据集用于训练

![image-20220106231153037](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220106231153037.png)

大概尝试了120张数据，数据的标注使用了[Computer Vision Annotation Tool (cvat.org)](https://cvat.org/tasks)，还是非常方便的，但经过训练得出的模型精度相差依旧甚远。不得已，决定将所有yolo模型比对一下，选择精度和运行速度都能够接受的模型。

## YoloALL

yoloall是一个可以方便地部署各版本yolo模型的软件，根据导入不同csgo实战视频查看效果，最终选择了YoloV5s6用作接下去的模型。

## YoloV5s6

部署的方式非常简单，可以通过torch.hub直接导入，关键在于自己调试拉枪的逻辑以及怎么开枪。

具体模型代码已开源到github [Jia-py/CSGO_AIM: demo on csgo auto-aim (github.com)](https://github.com/Jia-py/CSGO_AIM)

模型在我的笔记本3060显卡gpu的加速下，前向计算一次只需要大概0.02s，基本实现了实时的分类预测。

在其他博主的代码中，很多采用了pyautogui库，PIL库的函数，但经过测试，这些库函数的运行效率极低，推荐还是使用win32api的函数，运行效率能提高十几倍。

# 实战效果

尝试了一下实战效果，只能说一般般，目前的模型无法分辨队友与敌人。并且，有时墙上的涂鸦也会被识别为人，因此玩起来只能是娱乐了。想要提高模型的实战能力，可以进一步训练模型分辨警匪双方。

另外，因为脚本的性质，各位还是不要用于平台的匹配模式，给玩家带去不好的体验。