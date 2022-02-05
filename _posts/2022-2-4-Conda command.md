---
layout:     post
title:      "Conda command"
subtitle:   "conda常用命令"
date:       2022-02-04 16:57:00
updatedate:
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: false
tags:
    - HKU CS
    - CS
---

# env

## create env

```
conda create -n name python=3.7 -y
```

## delete env

```
conda remove -n name --all
```

## view env list

```
conda list
```

## activate and deactivate env

```
conda activate name
conda deactivate name
```

# clean

```
conda clean --all
```

