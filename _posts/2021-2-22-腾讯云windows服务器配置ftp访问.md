---
layout:     post
title:      "腾讯云windows服务器配置ftp访问"
subtitle:   "FTP access with CVM"
date:       2021-02-22 14:26:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - CS	
    - Cloud
---

## 前言

之前配置了jupyter notebook server，但是在实际使用过程中发现并不能上传本地笔记本上的文件，于是怎么能够方便地文件互通成了一个问题。

在这里我选择了FTP。

## 配置过程

* 云服务器端的配置服务主要参考[腾讯云文档](https://cloud.tencent.com/document/product/213/10414#.E6.9C.8D.E5.8A.A1.E5.99.A8.E8.AF.81.E4.B9.A6.E5.88.B6.E4.BD.9C.3Cspan-id.3D.22ssl.22.3E.3C.2Fspan.3E)

  > 但配置完成后发现并不能成功连接上。出于偶然，我又在IIS管理器中建立了一个IP为外网的FTP服务，这时候去连接ftp://外网IP：端口，发现成功了！

* 注意在腾讯云控制台需要打开设备的端口，可能在安全组、防火墙这些地方开启TCP的访问端口。

## Mac使用ftp

mac直接在find里面连接发现并不能直接拖拽文件，估计是系统设置的问题。为了更方便一些，下载了软件`FileZilla`。

但在使用过程中需要注意，新建站点时，加密方式需要选用`只使用普通FTP（不安全）`，才能够拖拽上传和下载文件。

## 成功

整套体系应该是正常完成了，可以比较方便使用notebook server了。

但我的99包年的云服务器卡是真的有点卡，希望以后能整台配置好一些的。