---
layout:     post
title:      "Geopandas"
subtitle:   
date:       2021-11-4 17:00:00
updatedate:
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: false
tags:
    - CS
    - GIS
    - Python
---

official document: [GeoPandas 0.10.0 — GeoPandas 0.10.0 documentation](https://geopandas.org/index.html)

# 导入文件

```python
data = gdp.read_file('.gbd',layer='')
```

# 提取geometry属性

目前还不知道对单一一条数据赋geometry的操作，这肯定是能实现的，以后找到了更新。

目前直接提取到geometry字段，是这么一个东西。

![image-20211104170648803](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211104170648803.png)

但如果，通过行筛选后，去选择geometry属性，会变成这样。

![image-20211104170727675](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211104170727675.png)

发现莫名其妙多出个行索引在前面，这时候需要用`values`属性，才能提取到真正的geometry。

![image-20211104170819770](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20211104170819770.png)

# 构建Geodataframe

构建方式与dataframe非常相似，但多出了一个参数为geometry。

可以将geometry的列表读入即可。

```python
route_new = gpd.GeoDataFrame(route_csv,geometry=geometry_list)
```

# 投影

设置投影：`data.set_crs(epsg=4326)`

投影转换：`data.to_crs(epsg=4326)`

注意，转换的话需要赋值才生效。

```python
route = route.to_crs(epsg=4326)
```



# 保存

```python
data.to_file('.shp')
data.to_file('.geojson',driver='GeoJSON')
```

