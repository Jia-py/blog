---
layout:     post
title:      "Arcgis Online要素图层设置时间特征"
subtitle:   ""
date:       2022-06-08 23:25:00
updatedate:
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: true
# 生活，工作，笔记（个人理解消化心得），文档（方便后续查阅的资料整理），项目，其他
tags:
    - 文档
    - GIS
---

# 1. 设置图层时间格式

预处理文件的时间特征，使得时间列的显示格式符合`YYYY/MM/DD HH:MM:SS`，中间的连接为横杠，或没有小时、分钟、秒的数据都可以

![image-20220608233029204](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220608233029204.png)

# 2. 设置图层的时间属性

双击图层名，在上图中的时间选项中完成设置

# 3. 导出shp并压缩为zip

这时上方工具栏会选中‘时间’按钮，需要将其取消选中，然后导出该图层为shp文件并压缩

# 4. 上传至Arcgis Online的content中

按照网页端的步骤上传，并托管图层

# 5. 设置图层时间属性

依次点击

![image-20220608233316689](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220608233316689.png)

![image-20220608233333017](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220608233333017.png)

# 6. 在Map Viewer中打开图层

![image-20220608233423138](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220608233423138.png)

发现时间滑块选项已激活
