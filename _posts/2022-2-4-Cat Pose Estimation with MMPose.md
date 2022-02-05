---
layout:     post
title:      "Cat Pose Estimation with MMPose"
subtitle:   "基于MMPose的猫动作估计"
date:       2022-02-04 15:22:00
updatedate:
author:     "Jpy"
header-img: "img/post-bg-2015.jpg"
no-catalog: False
onTop: false
latex: false
tags:
    - Deep Learning
    - HKU CS
    - CS
    - Pytorch
---

Please mention that I have installed Cuda And cuDNN before.

# Install MinGW on windows

**The first method does not work.**

## Download

Link: [MinGW - Minimalist GNU for Windows - Browse /MinGW at SourceForge.net](https://sourceforge.net/projects/mingw/files/MinGW/)

![image-20220204155124419](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220204155124419.png)

tick these five boxes to install gcc, gdb, make, and click `Installation-apply changes`, waiting for installation.

## ADD System Path

Click `my PC - properties - advanced setting - system path  `

- add `C:\MinGW\bin` into PATH　　
- add `C:\MinGW\include` into INCLUDE
- add `C:\MinGW\lib` into LIB

![image-20220204155851193](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220204155851193.png)

## use this method

follow this link [MinGW-w64 - for 32 and 64 bit Windows - Browse Files at SourceForge.net](https://sourceforge.net/projects/mingw-w64/files/) and find your wanted version, copy the download link and download the resource by other downloader.

just extract the tar file and set the system path.

# Install MMPose

```bash
conda create -n open-mmlab python=3.7 
conda activate open-mmlab
conda install pytorch torchvision torchaudio cudatoolkit=11.3

pip install intel-openmp
pip install opencv-python

# mention your cuda version and your torch version
# here I use the 11.3 cuda and the latest torch 1.10.2
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.2/index.html

git clone https://github.com/open-mmlab/mmpose.git

cd xtcocoapi
python setup.py install

cd mmpose
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
pip install mmdet

conda activate open-mmlab
python -m ipykernel install --user --name open-mmlab --display-name "open-mmlab"
```

# Trouble Shooting

1. OSError: [WinError 182] when import torch

![image-20220204195009771](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220204195009771.png)

solution: `pip install intel-openmp `

reference: [FileNotFoundError - caffe2_detectron_ops.dll on Windows source build if Python 3.8 used · Issue #35803 · pytorch/pytorch (github.com)](https://github.com/pytorch/pytorch/issues/35803)

2. 安装`mmcv-full`时报错Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/

solution: follow the link and install the C++ build tools.

![image-20220204195916575](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220204195916575.png)

3. 安装MMPose时出现xtcocotools无法安装的情况

![image-20220204211518862](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220204211518862.png)

solution: ![image-20220204212025034](https://raw.githubusercontent.com/Jia-py/blog_picture/master/img/image-20220204212025034.png)

reference: [FAQ — MMPose 0.22.0 documentation](https://mmpose.readthedocs.io/en/latest/faq.html)

4. 安装mmcv-full报错cv2没有`__version__`属性

solution：`pip install opencv-python`

5. 模型训练时报错`BrokenPipeError: [Errno 32] Broken pipe `

solution: 修改trainmodel中的参数worker_num = 0
