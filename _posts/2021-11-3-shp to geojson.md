---
layout:     post
title:      "shp to geojson"
subtitle:   
date:       2021-11-3 23:50:00
updatedate:
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: false
tags:
    - CS
    - Python
---

# shp to Geojson

a good website, but has some error in transforming features.

[Demo page - shp2geojson.js (gipong.github.io)](http://gipong.github.io/shp2geojson.js/)

or, you can use geopandas

```python
import geopandas as gpd

def shp2gj(input_file, output_file):
    data = gpd.read_file(input_file)
    data.to_file(output_file, driver="GeoJSON", encoding='utf-8') # 指定utf-8编码，防止中文乱码
    print('Success: File '+input_file.split('\\') [-1] + ' conversion completed')

```

# The best way

well , the best way I find is using QGIS's `save as`.

It's really convenient and precise