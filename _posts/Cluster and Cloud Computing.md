---
layout:     post
title:      "notebook - Cluster and Cloud Computing"
date:       2021-09-09 19:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - HKU CS
---

**by prof. Cho-Li Wang**

# Introduction

* two hands-on experience
* check the CPU model `cat/proc/cpuinfo`
* students need to repeat the spark and hadoop installation for 3 times.
* meeting record - screen shot of every meeting and write down the meeting result.
* **Trouble Shooting Report** : report technical problems and write down how you solve it
* report submitted 7 days before the deadline will be given 5% bonus

Why Cloud?

Cloud: Dynamic resource scaling, instead of provisioning for peak; without cloud, company should balance the capacity with demand carefully.

Virtualization techiniques: VMware Xen KVM

虚拟化技术帮助可以将一台服务器的硬件资源进行切分，但一般每切分出一个独立的单元都需要装一个系统，而系统是很臃肿的（What Hypervisor does, such as Xen, KVM, and VMWare）。所以，docker产生了，可以不用再切分出系统，而是在一个系统上可以运行不同的相互独立的环境 (under one Container Engine (Docker) )。

It's OK to deploy containers on virtual machines.

Cloud Deployment Models

* Public Cloud: Google, Amazon sell it, a rental basis
* Private Cloud: for exclusive use by a single organization of subscribers. more safe.
* Community Cloud: e.g., cryptocurrency mining
* Hybrid Cloud: e.g., private + public

# Cloud Service Models

* infrastructure as a service **Iaas**
  * network virtualization
  * 一个独立的主机，有独立的系统，a virtual machine
* container as a service **Caas**
  * 只是一个container，装一些软件，运行脚本
* platform as a service **Paas**
  * **deploy an entire application**
  * 多是基于云的平台，用户只需要直接写代码就可以或写文字就可以。如印象笔记、spotify。
* function as a service **Faas**
  * **just deploy a single function**： 比如用户模糊手机中的图片，或旋转图片，都是一个function。
  * event-driven: 比如事件为获取倒今天股市走势很糟糕，则运行代码发送伤心的表情。
  * AWS Lambda: write your functions in python..., uploaded as a zip file
* software as a service **SaaS**
  * 给用户使用的软件，比如各种apps，云盘软件，云游戏

# Problem Shooting

* Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?
  * maybe low version of apt-get or unstable internet make this mistake, run `sudo apt-get clean` and run `sudo apt-get update` and redownload it again with like `sudo apt-get -y install sysbench`
* How to run the command in each terminal simultaneously?
  * just use the function of xshell, use the tool-发送键输入到所有会话

