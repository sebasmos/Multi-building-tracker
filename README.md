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
 
**Proponent**: Juan Carlos SanMiguel 

**Supervisor**: Fabien Baldacci

**Report**: [Download](https://drive.google.com/file/d/1dytXRhjk3wIuqjNQ9rs0ukJLhGslj5AG/view?usp=sharing)

<hr/>

## Summary

<div align="justify">
 <p>
The automatic analysis of satellite imagery has a wide range of applications within the field of urban planning, including fair distribution of resources, effective disaster response, updating of real-time maps and epidemiological vector-borne diseases control. Furthermore, it poses compelling technical challenges that even today are not completely solved. A  system  for  multi-target  building  tracking  using  satellite images  has  been  developed  following  the  guidelines  pro-posed in the SpaceNet 7 Multi-Temporal Urban Development Challenge  and  as  a  continuation  of  a  previous  theoretical exploration of the problem. The system was implemented by considering  each  individual  block:  a  preprocessing  stage,  a neural  network  for  semantic  segmentation,  and  an  algorithm for data assignment as a tracker for static targets. Even thoughthe dataset provides only images with moderate resolution and includes regions with high variability, crowded scenes, and a high number of targets, the model is able to segment correctly most  of  the  buildings  and  maintain  their  identities  along  the sequence with a 62% of Intersection over Union (IoU). The system is able to locate correctly the buildings in the image and to determine accurately their borders with the exception of those too close to each other. Most importantly, the system reacts well to changes, which is an important factor of concern for urban planning purposes. The tracker reaches a MOTA of 0.647 and a F-score of 0.805 on the testing set. This repository has been developed as a tutored research project of the University of Bordeaux, the Autonomous University of Madrid and the Pázmány Catholic Peter's University under MIT license. 
</p>
</div> 
 
<hr/>

## Environment

The code is developed and tested under the following configurations.

- Hardware: GeForce RTX 3060 Mobile/Max-Q at 16G GPU memory.
- Software: Ubuntu 21.10 (Core), ***CUDA>=11.1, Python>=3.8, PyTorch>=1.9.0***

<hr/>

## Installation

Create a new conda environment and install PyTorch:

```shell
conda create -n py_mos python=3.8
source activate py_mos
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
```

Download and install the package:

```shell
https://github.com/sebasmos/Multi-building-tracker.git
cd Multi-building-tracker
pip install --upgrade pip
pip install -U albumentations
pip install patchify
```
<hr/>

## Datasets 

1. Training data [download](https://drive.google.com/drive/folders/1VrNtC48kYpMFgNIscw_g3bM6K_VjZkIY?usp=sharing)
1. Testing data [download](https://drive.google.com/drive/folders/1ijxwQH5cpihxXFvj4eMavDjNUMDpX2vf?usp=sharing)


<hr/>

## Pre-processing - Data augmentation

<div align="justify">
 <p>
The  data  augmentation  performed  in  the  scope  of  this project  can  be  grouped  into  two  categories:  color  augmen-tations  and  geometrical  augmentations.  Both  contribute  to alleviate the small size of the dataset, but color augmentationsare specifically intended to add generalization ability regardingthe high  variability in the  vegetation due to  the geographical differences  and  seasonal  changes.  The  processing  outline  for the two types of augmentations is the same: the whole dataset is  traversed,  and  each  augmentation  is  applied  to  an  image with a probability of 20%
 </p></div>

<img src="https://github.com/sebasmos/Building.predictor/blob/main/Media/augmented.gif" width="400" height="400" />

Types of augmentation using *Albumentations* package and customized augmetation:

* *Geometric*: Rotations, flips

* *Color*: RGB shift, CLAHE

* Offline process

* Probability of 20%

<hr/>

## Patching

Considering this, the  training  images  were  split  into  smaller  pieces  that  arefed into the network individually, since these are independentand contain by themselves enough information to represent anurban  area.  At  the  end,  the  patch’s  size  was  set  to  256x256.Given that the images are originally 1023x1023 pixels, it wasnecessary to add zero padding to guarantee an exact divisionby 256.

![](https://github.com/sebasmos/Building.predictor/blob/main/Media/patching_imgs.gif)
![](https://github.com/sebasmos/Building.predictor/blob/main/Media/patching_mask.gif)

*Patching image (left) - mask patched image (right)*

<hr/>

## Segmentation

<img src="https://github.com/sebasmos/Building.predictor/blob/main/Media/label.gif" width="250"></a>
<img src="https://github.com/sebasmos/Building.predictor/blob/main/Media/pred.gif" width="250"></a>

*Mask image (left) - segmented image (right)*

<div align="justify"><p>
UNET  with  transfer  learning  was selected for the best results in three scenarios were implemented. In all of them the input size was set to 256x256, the binarization threshold to 0.6, the  number  of  epochs  to  100,  and  the  initial  learning  rate  to 1x10−4. However, learning rate decay was found to be useful in the three models. It is an strategy to train neural networks that  improves  both  optimization  and  generalization,  allowing a  faster  convergence  of  the  loss  function  to  a  minimum and  avoiding  oscillation.  To  do  so,  the  learning  rate  is decreased  to 1x10−5 in  epoch  30  and  to 1x10−6 in  epoch 65.  Regarding  the  particular  configurations,  model  6  is  the basic  configuration  (UNET+VGG16).  Model  7  introduces  a new loss function (hence forth mixed loss) proposed by one of the winners in a previous version of the challenge [21], whose implementation can be found in [23]. It corresponds to the sum of the focal loss and the dice loss. Since the ground truth of the dataset for which it was designed had several channels, it was computed  as  a  weighted  average  over  the  different  channels.For  this  case,  being  a  binary  class  task,  it  corresponds  to  a simple sum. For tuning methods and other models trained please refer to 
<a href="https://drive.google.com/file/d/1Rbc202UVYKSnIZx3vauio32_Ty-SJmIy/view?usp=sharing">docs</a>.
 </p></div> 

 
<hr/>

## Tracking

<img src="https://github.com/sebasmos/Building.predictor/blob/main/Media/tracking.gif" width="350"></a>



<div align="justify"><p>
The problem previously stated corresponds exactly to a data assignment problem. To solve it, the Hungarian Algorithm is used (sometimes referred as Munkres or Kuhn-Munkres algorithm). It is a strategy frequently employed in multiple object tracking systems [12] to find the optimal assignment between "workers" and "tasks" (in this case between the buildings detected in step $t$ and those detected in step t+1) by minimizing a cost matrix. It has been proven that the algorithm solves the problem in a polynomial time, which is fast enough for the intended application. 
 </p></div>
The process can be summarized as follows: 

    * From the segmentation results extract the footprints (buildings' contours) for all time steps. 
    
    * Assign initial labels for the footprints in step t=0.
    
    * Compare each footprint in $t$ with each footprint in t+1, using Intersection Over Union (IOU) as the comparison criterion. 
    
    * Compute the cost by subtracting the IOU from 1. This way, if the IOU between a pair of footprints is large enough, the cost of assignment will be very small.

<hr/>

## Tracking optimization

<div align="justify">
<p> The initial method is clearly suboptimal, especially considering the high number of targets in each image. The first and most relevant improvement is to narrow down the estimation of the cost by considering only the neighborhood of the footprint currently analysed. For this, a bounding box is generated for each footprint, considering its borders plus an offset of 20 pixels. The IOU for a pair of detections (in two consecutive frames) is computed only if the bounding boxes intercept. Otherwise, it will be set to 0. A second operation was done to the reduce the size of the matrices used. To explain this, let us consider the case when the footprints are located in the bottom right corner of the image. Originally, the IOU is computed with the entire mask (1024x1024) to preserve the spatial information. Nevertheless, it is possible to preserve such information using a smaller matrix size: first, the minimum x and y coordinates for the pair of detections is subtracted from every coordinate to reset the minimum as the origin. Then, the mask is created from the point (0,0) until the new x and y maximum for both cases. This allows to operate with considerably smaller matrices and, as a result, the processing time decreases.</p>

<p>An experiment was performed on a sample sequence of the dataset to measure the processing time and the effect of the optimizations. Only one time step was considered (i.e., the tracking is done for two frames only). The first image has 1500 targets and the second one 1520. The results can be seen in the report.</p>
 
 </div>


## Acknowledgement

This project is built upon numerous previous projects. Especially, we'd like to thank the contributors of the following github repositories:

- [Spacenet](https://github.com/SpaceNetChallenge): SpaceNet.
- [A Merii](https://www.kaggle.com/amerii): Spacenet 7 Utility functions.
- [Markus Rosenfelder ROSENFELDER](https://rosenfelder.ai/Instance_Image_Segmentation_for_Window_and_Building_Detection_with_detectron2/): Utility functions

## License

This project is licensed under the MIT License and the copyright belongs to Sebastián Cajas & Julián Salazar- see the [LICENSE](https://github.com/sebasmos/Building.predictor/blob/main/LICENSE) file for details.

## Citation

For a detailed description of our framework, please read this [technical report](https://drive.google.com/file/d/1dytXRhjk3wIuqjNQ9rs0ukJLhGslj5AG/view?usp=sharing).


