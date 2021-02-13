---

layout:     post
title:      "Crash Course Computer Science"
date:       2021-02-13 00:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - Study
    - CS
---

[Github 中文字幕](https://github.com/1c7/crash-course-computer-science-chinese)

* [1.计算机早期历史 2.电子计算机](#1)



<h2 id="1">计算机早期历史-Early Computing</h2>

人类在很早就拥有了计算的需求，计算机最早的形式是算盘，用珠子来表示不同的数字，不同行代表不同的位数。

> Charles Babbage：随着知识的增长和新工具的诞生，人工劳力会越来越少。

”computer“一开始代表的是计算能力很强的人，是一种职业。

机械计算机是非常成功的，一直用了有三个世纪，但还是十分耗时费力。

美国1890年的人口普查的需求，按照以往的方法，需要13年才能完成，但人口普查10年必须进行一次，这极大推动了计算机的发明。

<h2>电子计算机-Electronic Computing</h2>

20世纪，战争、科研、探月等都带来了巨大的数据，需要更多自动化、更强的计算能力。

* 哈弗马克计算机：因为继电器的故障总是因为有虫子卡死在齿轮中，所以叫“bugs”
* 真空管的发明，一秒钟可以实现近千次开关。
* 晶体管，每秒开关10000次。由半导体控制电流是否流通。

<h2>布尔逻辑和逻辑门-Boolean Logic & Logic Gates</h2>

计算机使用二进制，Boolean值可以很容易地用二极管来表示，布尔运算AND,OR,NOT也可以用不同的电路来完成。

因为二极管也可以控制通过电流的大小，因此当时有些计算机也采用过三进制、五进制，但因为电流大小很容易受到环境影响很难判别到底是哪个数字，因此二进制得以发展。

具体如何用电路来表示AND,OR,NOT,XOR(异或)：

[![ysuVxg.jpg](https://s3.ax1x.com/2021/02/13/ysuVxg.jpg)](https://imgchr.com/i/ysuVxg)

Not门是AND和OR的基础，当Current为通电状态，Input为true时，因为下侧电路接通接地，将没有电流从output流出，因此为False，实现了NOT的功能。

[![ysueMQ.jpg](https://s3.ax1x.com/2021/02/13/ysueMQ.jpg)](https://imgchr.com/i/ysueMQ)

[![ysumrj.jpg](https://s3.ax1x.com/2021/02/13/ysumrj.jpg)](https://imgchr.com/i/ysumrj)



[![ysuERS.md.jpg](https://s3.ax1x.com/2021/02/13/ysuERS.md.jpg)](https://imgchr.com/i/ysuERS)

## 二进制-Representing Numbers and Letters with Binary

二进制是怎么表示的

一般8位我们称为一个bit，一个字节。也就是8位的二进制数字，取值范围为0~255。

计算机的32位，64位的含义是 计算机一块一块地处理，最小单元为32位，64位，这是一个非常大的数字。

* 如何表示正负？ 一般首位为1代表正数，首位为0代表负数
* 如何表示浮点数？ 首位依旧表示正负，接下的8位用来表示10的指数次方，其余的位数代表有效数字。这种表示其实是科学计数法。
* 如何表示文字呢？ 上世纪60年代，美国发明了`ASCII`，是一个7位的码，可以保存128个不同的字符，因此足以保存大小写英文字母以及标点符号。但对于中国和日本来说，即使8位的码也很难表示所有的文字，所以会采用多字节的方式来显示文字，但往往伴随着乱码的问题。`1992`年，`unicode`横空出世，设计了统一的规则。最常见的Unicode是16位的，可以存放一百多万个字符。100多种语言所有字母表加起来占了12万个位置，还有很多空余，甚至可以放得下emoji表情。

