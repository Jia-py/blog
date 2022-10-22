---
layout:     post
title:      "jsDelivr加速静态网页"
subtitle:   "Accelerate static web pages with JsDelivr"
date:       2021-02-22 08:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
# 生活，工作，笔记（个人理解消化心得），文档（方便后续查阅的资料整理），其他
tags:
    - 文档
---

## 什么是[jsDelivr](https://www.jsdelivr.com/?docs=gh)

jsDelivr是一个开源的网站，提供免费的cdn加速加载资源。

比如网站中要用的js文件，jpg等图片，mp3等文件都可以通过免费的cdn加速。

cdn加速即在访问一个资源时，通过离目标IP最近的服务器进行访问，加快访问速度。

cdn加速访问github文件有默认的格式，可以先按照下面的格式在浏览器中访问试一试。

`https://cdn.jsdelivr.net/gh/user/repo@version/file`

## 如何在Jeckyll中配置

我是在`header.html`的头部添加代码，因为每个页面都会用到header

```html
<!--jsdelivr-cdn加速-->
<!--定义assets_base_url为baseurl，若开启cdn加速，则替换资源链接-->
{% assign assets_base_url = site.baseurl %} 
{% if site.jsdelivr-cdn-enable %} 
    {% assign assets_base_url = "https://cdn.jsdelivr.net/gh/Jia-py/Jia-py.github.io" %} 
{% endif %}
```

然后将引用资源的链接中`site.baseurl`都替换为`assets_base_url`

注意一些跳转链接不要修改，可能会造成跳转错误。比如这样的样式。

```html
<a href>
```

但是我发现这样修改后`index.html`和`about.html`依然不会使用cdn加速

于是我在`page.html`（也就是`index.html`和`about.html`所使用的的`layout`）的头部加入上面的代码，发现就可以了。难道这个创建的变量不能通过两次`layout`吗？算是瞎猫碰见死耗子了。

## 效果

配置完的效果很好！

可以通过`F12`后打开`网络`里面看到访问各个网络资源所需要的时间，注意提交一次网站修改后需要`shift+F5`刷新，否则清不掉缓存。

我原先访问主页背景慢的时候需要7秒左右，而现在还没有出现超过100ms的情况！

jsDelivr🐂🍺！

---

## JsDelivr缓存更新

2021.11.30 update

在修改了css等文件后，若采用了jsdelivr加速，可能会存在更新不及时的问题。

这是只需要访问几次url`https://purge.jsdelivr.net/gh/user/repo@version/file`即可，注意有时需多访问几次，直到一个`bottle**`的参数为True。