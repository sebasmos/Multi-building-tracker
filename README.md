<a href="https://github.com/sebasmos/Building.predictor"> 
 
    
<img src="./.github/TRDP.png" width="550"></a>
 

<p align="left">
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.8-ff69b4.svg" /></a>
    <a href= "https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-1.8-2BAF2B.svg" /></a>
    <a href= "https://github.com/sebasmos/Building.predictor/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
</p>

<hr/>

 

By [Sebastian Cajas](https://www.linkedin.com/in/sebasmos777/) and [Julián Salazar](https://www.linkedin.com/in/julian-norberto-salazar-vidal-ipcv/)

This repository contains the implementation of Multi-target building tracker for satellite images using deep learning.
 
Proponent: Juan Carlos SanMiguel 

Supervisor: Fabien Baldacci

## Summary

The automatic analysis of *satellite imagery* has a wide range of applications within the field of urban planning, including fair distribution of resources, effective disaster response, updating of real-time maps and epidemiological vector-borne diseases control. Furthermore, it poses compelling technical challenges that even today are not completely solved. A  system  for  multi-target  building  tracking  using  satellite images  has  been  developed  following  the  guidelines  pro-posed in the SpaceNet 7 Multi-Temporal Urban Development Challenge  and  as  a  continuation  of  a  previous  theoretical exploration of the problem. The system was implemented by considering  each  individual  block:  a  preprocessing  stage,  a neural  network  for  *semantic  segmentation*,  and  an  algorithm for data assignment as a tracker for static targets. Even thoughthe dataset provides only images with moderate resolution and includes regions with high variability, crowded scenes, and a high number of targets, the model is able to segment correctly most  of  the  buildings  and  maintain  their  identities  along  the sequence with a 62% of Intersection over Union (IoU). The system is able to locate correctly the buildings in the image and to determine accurately their borders with the exception of those too close to each other. Most importantly, the system reacts well to changes, which is an important factor of concern for urban planning purposes. The tracker reaches a MOTA of *0.647* and a F-score of *0.805* on the testing set. 

## Environment

The code is developed and tested under the following configurations.

- Hardware: GeForce RTX 3060 Mobile/Max-Q at 16G GPU memory.
- Software: Ubuntu 21.10 (Core), ***CUDA>=11.1, Python>=3.8, PyTorch>=1.9.0***

## Installation

Create a new conda environment and install PyTorch:

```shell
conda create -n py_mos python=3.8
source activate py_mos
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
```

Download and install the package:

```shell
https://github.com/sebasmos/Building.predictor.git
cd Building.predictor
pip install --upgrade pip
pip install -U albumentations
pip install patchify
```

