---
layout:     post
title:      "github pages与coding pages双线部署提高全球访问速度"
subtitle:   "Dual deployment of GitHub Pages and Coding Pages improves global access speed"
date:       2021-02-21 17:39:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - CS	
    - Cloud
---

## 背景

github pages虽然部署非常方便，但有两个非常严重的问题。

1. 国内访问速度较慢。虽然大部分时间都是比较稳定的，但是是稳定的慢！😭大部分时间的访问延迟都在100ms左右。
2. 国内百度爬虫等无法访问到github pages的博客，导致使用百度搜索的同学是看不到github pages的博客的。虽然，百度的搜索广告漫天，但在国内占有量还是很大的。

## 实现的方式

那么如何能解决上面的两个问题呢？

我想到的有以下几种方法。

1. 创建国内的镜像站。

   比如`coding`，`gitee`。gitee没能配置成功，这里就不说了。coding pages的配置我有之前的文章详细说过，链接在这里[Coding静态网站部署及添加自定义域名](https://jiapy.space/2021/02/20/Coding%E9%9D%99%E6%80%81%E7%BD%91%E7%AB%99%E9%83%A8%E7%BD%B2%E5%8F%8A%E6%B7%BB%E5%8A%A0%E8%87%AA%E5%AE%9A%E4%B9%89%E5%9F%9F%E5%90%8D/)。coding的配置可以选择服务器，有`境外`和`境内`两种。如果要修改自定义域名，境内服务器是需要备案的，而境外的服务器不需要备案。香港的服务器大家就不要去用了，除非你可以忍受那么一长串的默认域名，如果要修改为个人域名，则必须使用腾讯云的cdn服务。但是！境外的腾讯云的cdn服务也太垃圾了，甚至都没有我裸连github快。**推荐使用境内的服务器，然后备案后修改自定义域名**

2. 使用国外的`netlify`等网站构建。

   不知道为什么，我按照网上的教程导入我的github库之后，创建起来的网站都是没有样式的。我也无法在短时间内解决这个问题，于是放弃了。有知道的朋友欢迎在最下方留言告诉我呀。

3. 使用cdn加速访问github pages。

   这个想法就是购买腾讯云阿里云等等平台的cdn的服务，用cdn的加速让github pages上的静态资源加载更快一些。但缺点就是会有一定的费用产出，作为个人博客的话也不贵。

4. 👍**（推荐）**单部署github pages，使用DNSPOD指向github pages

   这是我在配置过程中发现的，新域名直接指向github pages居然很多节点是通过香港腾讯云接入的，在DNSPOD里只需要添加两行信息就足够了。

   | 主机记录 | 记录类型 | 线路类型 | 记录值             |
   | -------- | -------- | -------- | ------------------ |
   | www      | CNAME    | 默认     | github pages的网址 |
   | @        | A        | 默认     | github pages的IP   |

   估计是腾讯云的DNSPOD是自动会根据访客IP来选择更好的访问途径的，毕竟在响应IP中一些是直接的美国Github，一些又是香港腾讯云。

   这种方法配置也比较简单，还是很推荐的。

   ![](https://cdn.jsdelivr.net/gh/Jia-py/blog_picture/21_3/yTn0zD.md.jpg)

   总体看来，提升的效果还是有的。

5. 👍👍👍**（推荐）双线同步coding pages及github pages，使用DNS解析指向不同IP**

   这就是最终我选择的方式！这样的方式我目前认为是最好的了。其实就是根据访客ip的不同，我们为他指向不同的网址。**对于境内的访客，我们指向coding pages，对于境外的访客，我们指向github pages。**

## 配置过程

1. 在`DNSPOD`上配置我们的自定义域名。

| 主机记录 | 记录类型 | 线路类型 | 记录值             |
| -------- | -------- | -------- | ------------------ |
| @        | CNAME    | 境内     | cdn链接            |
| @        | CNAME    | 境外     | github pages的网址 |

> @代表的是域名无前缀直接访问，如XXX.com；www则代表带www前缀访问。同时还有*，代表其他所有前缀访问。mail，为mail.XXX.com访问。
>
> 记录类型一般只会用到A和CNAME，A表示填入IP，CNAME表示填入网址。
>
> 线路类型则是访客的访问地址类型。

2. 为github pages和coding pages添加自定义域名。

   在github上部署我们的博客，找到我们的github pages仓库。点击setting，找到domain customize设置，输入我们的自定义域名即可，github会自动为我们申请证书以及创建CNAME文件。

   在coding上部署我们的博客，当然服务器选择的是境内的服务器。然后为我们的域名备案，为coding上的网站添加自定义域名。

   至此，所有操作已经完成。

3. 注意检查cdn加速区域
4. 注意需要修改`_config.yml`文件中的baseurl为`baseurl: ""`，否则在coding自定义域名后会出现点击除主页外其他页`404 notfound`错误

## 后记

这里简单比较一下结果

单纯用github的节点：

![](https://cdn.jsdelivr.net/gh/Jia-py/blog_picture/21_2/Snipaste_2021-02-25_22-08-15.jpg)

![](https://cdn.jsdelivr.net/gh/Jia-py/blog_picture/21_2/纯github.jpg)

境内使用cdn加速指向coding pages，境外指向github pages：

![](https://cdn.jsdelivr.net/gh/Jia-py/blog_picture/21_2/Snipaste_2021-02-25_22-06-33.jpg)

![](https://cdn.jsdelivr.net/gh/Jia-py/blog_picture/21_2/Snipaste_2021-02-25_22-09-15.jpg)

可以看到效果非常明显！

整个思考的流程花费了我大概一个下午，不断搜集资料，尝试配置。总之最终能想到也还是非常开心的一件事情！当然有更好的方式的朋友也非常欢迎留言！

