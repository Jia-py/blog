---
layout:     post
title:      "Install Geopandas & shp to Geojson"
subtitle:   
date:       2021-11-3 21:00:00
updatedate:
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: false
# 生活，工作，笔记（个人理解消化心得），文档（方便后续查阅的资料整理），其他
tags:
    - 文档
    - GIS
---

# Install Geopandas

```
conda create -n geopandas python=3.6
conda activate geopandas
```

the install method recommended

```
conda install --channel conda-forge geopandas
```

install the ipykernel

```python
#在该环境中安装ipykernel 

conda install -n geopandas ipykernel
#添加kernel，第一个pytorch为环境名；第二个为在jupyter中显示的名字，可修改 

python -m ipykernel install --user --name geopandas --display-name "geopandas"
```

delete conda envs

```
conda remove -n geopandas --all
```



## problems

the conda stucks in the solving environment step.

It is because the internet setting.

For I'm at Hong Kong now, but my channel in conda is the Tsing Hua source.

```
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
```

changing the channels slove the problem.

中间出现了只能在终端导入包，但在notebook中不能导入包的情况

而后发现解决方案是导入一下其他基础的包，比如pandas导入一下，再导入geopandas就可以了。

