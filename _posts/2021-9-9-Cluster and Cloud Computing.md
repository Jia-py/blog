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