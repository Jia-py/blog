---
layout:     post
title:      "IDEA配置java和scala"
subtitle:   ""
date:       2022-06-14 14:30:00
updatedate:
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: false
# 生活，工作，笔记（个人理解消化心得），文档（方便后续查阅的资料整理），项目，其他
tags:
    - 文档
---

# 下载IDEA

这里通过官网下载community版本即可，公司通过内部平台下载

## 配置java

IDEA提供了方便的内部下载JDK方案

点击`File-Project Structure - Project`，在SDK一栏中可以直接下载对应版本的JDK。

可以在project中的src文件夹中新建java类，输出`hello world`验证。

```java
public class helloworld {
    public static void main(String[] args){
        System.out.println("hello world!");
    }
}
```

## 配置scala

首先，需要下载IDEA中的scala插件，使IDEA支持使用scala语言。

点击`File-Setting-Plugins`，选中scala插件安装即可。

接下去需要配置scala对应版本的Scala SDK，在对应project上右键，依次点击`Add Framework Support - Scala - Create `，选择对应版本下载即可。

```scala
object helloworld {
  def main(args: Array[String]): Unit = {
    print("hello world!")
  }
}
```

