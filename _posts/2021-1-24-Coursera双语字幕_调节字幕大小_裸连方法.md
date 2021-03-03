---
layout:     post
title:      "Coursera双语字幕设置及裸连方法"
subtitle:   "Coursera subtitle & connection settings"
date:       2021-01-24 17:47:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - Coursera
---

## coursera 双语字幕 & 字幕大小修改
### 双语字幕
- click👉下载[字幕插件](https://github.com/tamshadow/coursera-subtitle-translation)
- 下载插件为压缩包，解压缩
- 打开浏览器的拓展功能，添加解压缩的拓展包
- 打开视频网页后单击一次拓展图标即可
  
> 参考链接：https://www.zhihu.com/question/302709745/answer/954417676

### 字幕大小修改
- click👉下载[chrome插件](https://link.zhihu.com/?target=https%3A//chrome.google.com/webstore/detail/stylus/clngdbkpkpeebahjckkjfobafhncgmne/related%3Fhl%3Dzh-CN)
- 直接添加该插件到浏览器
- 新建一段代码 video::-webkit-media-text-track-display{font-size:60%;}
- 在下方可以设置代码应用的网址范围
- 保存即可

## Coursera 裸连方法
- 打开C:\Windows\System32\drivers\etc\hosts文件
- 在最下方添加`52.84.167.78 d3c33hcgiwev3.cloudfront.net`并保存
- 按住win+R打开运行对话框，框里输入cmd，点击确定
- 在闪烁的光标那里输入ipconfig/flushdns，清除DNS缓存
- 刷新coursera视频界面，完成！


