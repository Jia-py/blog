---
layout:     post
title:      "Install R in jupyter"
date:       2021-09-17 17:00:00
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
tags:
    - Jupyter
    - R
---

# Background

For the need of FITE7410 Financial Fraud analytics, Students need to use R to analysis data. 

My environment

* installed: miniconda, jupyter

# Process

## method 1

1. install R

click the website [The Comprehensive R Archive Network (r-project.org)](https://cran.r-project.org/) download and install R.

2. find location of R.exe (D:\Program Files\R\R-4.1.1\bin)
3. go to this location in anaconda prompt
4. run R in the location (directory) in the prompt
5. run the following codes in R started from the prompt

* install.packages('IRkernel')
* IRkernel::installspec()

complete!

## method 2

process in miniconda directly

```
conda create -n R4.1
conda activate R4.1
conda install r-base=4.1.1
```

complete!
