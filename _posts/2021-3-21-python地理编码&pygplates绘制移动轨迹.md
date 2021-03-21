---
layout:     post
title:      "python地理编码&Gplates绘制移动轨迹"
subtitle:   "Python Geocoding & Pygplates Drawing Motion Tracks"
date:       2021-03-21 15:22:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - GIS
    - Python
---

## Python地理编码

地理编码的实现主要基于`Geopy`包的功能，能够实现通过文本地名获得经纬度的功能。

主要实现代码：

```python
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="specify_your_app_name_here")
location = geolocator.geocode("175 5th Avenue NYC")
print(location.address)
>>Flatiron Building, 175, 5th Avenue, Flatiron, New York, NYC, New York, ...
print((location.latitude, location.longitude))
>>(40.7410861, -73.9896297241625)
print(location.raw)
>>{'place_id': '9167009604', 'type': 'attraction', ...}
```

需要注意的是`Nominatim`是免费的服务，通过OSM的搜索来确定经纬度，投影为WGS 1984，但每天有限制次数，对于小项目用用完全足够了。其他的如百度地图等的服务就需要注册百度地图开放平台的服务了，需要用到ID和密钥。

## Gplates绘制移动轨迹

通过[GPlates: Download GPlates and Data](https://www.gplates.org/download.html)下载

首先，将上一步中的经纬度数据生成`.shp`数据导出，可用`geopandas`

```python
df = pandas.read_excel('unique_loc_lat&long_delete0.xlsx')
latitude = df['latitude']
longitude = df['longitude']
loc_arr = []
for i in range(len(latitude)):
    loc_arr.append(geometry.Point(longitude[i],latitude[i]))
point_list = geopandas.GeoSeries(loc_arr,
                         crs='EPSG:4326',)  # 指定坐标系为WGS 1984 
point_list.to_file('unique_loc_lat&long_delete0.shp',
           driver='ESRI Shapefile',
           encoding='utf-8')
```

完成后，将`shp`文件拖入`Gplates`中，将`SampleData\FeatureCollections\DynamicPolygons`中的两个文件也一并拖入，这里的数据是板块数据。

点击`Features-Asign Plates ID`,选择两个板块文件为`partitioning layers`，选择shp点数据为`to be partitioned`，设置完成后直接apply，即可。

此时拖动时间轴可以看到`shp`文件的运动。

