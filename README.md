<p align="center">
<img src="docs/figs/header.png" height="160" alt="Header">
</p>

#  The SpaceNet 7 Baseline Algorithm 

The [SpaceNet 7](https://spacenet.ai/sn7-challenge/) dataset contains ~100 data cubes of monthly Planet 4 meter resolution satellite imagery taken over a two year time span, with attendant building footprint labels.  The goal of the SpaceNet 7 Challenge is to identify and track building footprints and unique identifiers through the multiple seasons and conditions of the dataset.  

To address this problem we propose the SpaceNet 7 Baseline algorithm.  This algorithm is a multi-step process that refines a deep learning segmentation model prediction mask into building footprint polygons, and then matches building identifiers (i.e. addresses) between time steps. See [The DownLinQ](https://medium.com/the-downlinq/the-spacenet-7-multi-temporal-urban-development-challenge-algorithmic-baseline-4515ec9bd9fe) for further details.  While the goal is akin to traditional video object tracking, the semi-static nature of building footprints and extremely small size of the objects introduces unique challenges. 

There are a few steps required to run the algorithm, as detailed below.

-----------
I. Download Data and Create Environment

1. Download SpaceNet 7 Data
    
    The SpaceNet data is freely available on AWS, and all you need is an AWS account and the [AWS CLI](https://aws.amazon.com/cli/) [installed](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) and [configured](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html). Once you’ve done that, simply run the command below to download the training dataset to your working directory (e.g. `/local_data/sn7/aws_download/`):
   
        cd /local_data/sn7/aws_download/
        aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz .
        aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz .

    Extract data from the tarballs:
    
         cd /local_data/sn7/aws_download/
         tar -xvf SN7_buildings_train.tar.gz
         tar -xvf SN7_buildings_test_public.tar.gz


2. Download SpaceNet 7 baseline code to the desired location (e.g. `/path_to_baseline/`):
    
        cd /path_to_baseline/
        git clone https://github...
 
3. Instead (It worked like this in Ubuntu 20.10): 
Install anaconda https://docs.anaconda.com/anaconda/install/linux/ 
Install solaris: https://solaris.readthedocs.io/en/latest/installation.html 
Install Nvidia-docker https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html  
Install docker: https://docs.docker.com/get-docker/  

After this, follow the next command to create a new environment called "Solaris" in ANaconda:

git clone https://github.com/cosmiq/solaris.git
cd solaris
conda env create -f environment-gpu.ymlv

conda activate solaris
pip install .

From console, always activate this option manually through``` conda activate solaris ``` or manually tipe ```anaconda-navigator``` from terminal and change ```root``` to ```solaris```, install Jupyter in this environment +  requirements.txt using ```pip install -y requirements.txt```. 


5. (OPTIONAL, for me this is not necesary if you have 1 GPU - Intel) Build and launch the docker container, which relies upon [Solaris](https://solaris.readthedocs.io/en/latest/) (a GPU-enabled machine and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) are recommended):

        nvidia-docker build -t sn7_baseline_image /path_to_baseline/docker 
        NV_GPU=0 nvidia-docker run -it -v /local_data:/local_data  -ti --ipc=host --name sn7_gpu0 sn7_baseline_image
        conda activate solaris
       
4. Execute commands

    1. Either use the jupyter notebook:
    
            cd /path_to_baseline/
            jupyter notebook --ip 0.0.0.0 --port=9111 --no-browser --allow-root &
    
        Locally in your browser, visit:
    
            http://localhost:9111
        
    
    2. Or simply copy the relevant commands into a terminal within the docker container
    

-------
II. Prepare the data for training and inference (see `notebooks/sn7_data_prep.ipynb`)

-------
III. Train (see `notebooks/sn7_baseline.ipynb`, or use the pretrained weights in `models`

-------
IV. Infer (see `notebooks/sn7_baseline.ipynb`)

Output will consist of:

1. Inference masks:

<p align="left">
<img src="docs/figs/im_mask.png" height="400" alt="Header">
</p>

2. Building footprints:

<p align="left">
<img src="docs/figs/footprints.png" height="400" alt="Header">
</p>

3. Building footprints matched by ID across the data cube:

<p align="left">
<img src="docs/figs/footprints_matched.png" height="400" alt="Header">
</p>


## Paper Spacenet 7

### Evaluation metrix 

**SCOT : spacenet change and object tracking** 

‌       1. **Tracking term**: Ability to measure whether the building stays the same over time.
        Evaluates the IOU between different buildings with its corresponding ID, from months to months
        * Penalizes incorrect ids across time (mismatch)

        Mismatch(mm): if one ID has been assign to another ID proposal more than once recently 
                
        Mismatches are not counted as TP. Each mismatch decreases TP by one. This effectively divorces 
        the ground truth footprint from its mismatched proposal footprint, creating an additional false 
        negative and an additional false positive
        
        `tp -> tp - mm`
        `fp -> fp + mm`
        `fn -> fn + mm`
‌       
       2. **Change detection term** 
        Detects whether there is a change on the picture. New buildings, missed ones (?)  This is the term that identifies new buildings detection or construction. 
        That is, ground truth or proposal footprints with identifier numbers making their first chronological appearance. It will be zero because there are no new proposals after the first month (which is ignored) even for small new constructions
        
        Comparing with MOTA: “MOTA scores are mathematically unbounded, making them less intuitively interpretable for challenging low-score scenarios, and sometimes even yielding negative scores.” [1]
        Since under-standing time-dependence is usually a primary purpose of time series data, this is a serious drawback. SCOT’s change
        detection term prevents this. In fact, many such approaches to “gaming” the SCOT metric by artificially increasing one
        term will decrease the other term 
        
**Multiple Object Tracking Accuracy (MOTA) metric**: Yield negative scores. This is the reason why we are using SCOt metric, because MOTA does not have time dependence, and this can be really unuseful for time series data
        
        
**Idea**: match footprint GT vs month to month footprint. 

There is match if iou <= 0.25 (comes from imagenet)

The idea of using these 2 terms (change and tracking terms) generalizes the F1 score over time series upon appearance of new buildings or its disappearance.

**Advantages and disadvantages of SCOT**: While the coarser the building positions, the better the prediction. While if they are clustered together, it will be worst.  Buildings with more than 4pxs squared are evaluated, with around 4 m area. 

‌SCOT will yield bad results if there is not good overlap, so that's why for clustered regions it is very complicated to good results, and may be crossed IOU, which creates mismatches and those are penalized ‌despite of adding multiple imagery for representing the time component on training set, future and past generations. This however could be useful for implementing LSTM or RNN models. While testing SCOTT results, there were not differences between vgg16 & efficientnet


MUDS dataset = spacenet dataset 
ond its relevance for disaster response, disease preparedness, and environmental monitoring, time series analysis of satellite imagery poses unique technical challenges often unaddressed by existing methods.

### Spacenet 4 <a href="https://medium.com/the-downlinq/a-deep-dive-into-the-spacenet-4-winning-algorithms-8d611a5dfe25">[Ref]</a> 

Results showed that:

1. **Augmentation strategies**: Flipping or rotating worsen results because of distorted images and inplaced labeling (shattered by shadows). Then, algorithms need to know the right labeling positions

2. **Loss functions**: Combination of Dice coefficient + Focal loss yielded best results than Jaccard. (top 2 competitors [Focal loss] (https://arxiv.org/abs/1708.02002)) which is basically a binary cross-entropy variant that penalizes low-confidence predictions more strongly. These loss functions combined with the competitors’ advanced segmentation objective masks yielded high-fidelity building footprint extraction.

3. **Objective masks**: NN for segmentation create 0-1 prob masks (*semantig segmentation*), but Spacenets asks additionally for *instance segmentation task*, which is creating polygons labeling, what this means is that the crafted polygons must fit accurately or in consistent shapes into the Ground Truth footprints. The winners solution in this case proposed using 3 channel join method: Each color corresponds to a different objective mask channel. Blue marks building footprints; pink outlines the buildings; green denotes points where buildings are closely juxtaposed.

**Conclusions of Spacenet 4**: Competitors used their loss functions, learning objectives, and augmentation strategy to address the unique challenges posed by the SpaceNet Off-Nadir Buildings Challenge task and data. They cut common image augmentations (rotation and flipping) from their pipeline so their algorithms could learn offset. They used loss functions optimized for a low foreground-to-background class ratio to ensure algorithms learned to find the relatively uncommon building pixels. Finally, they used advanced learning objectives to effectively separate buildings for this instance segmentation task





### Modeling Architecture walkthrough XD_XD

1. Neural net architectures: Semantic segmentation with U-net and VGG16 encoder (pre-trained on Imagenet), which builds an *effective building footprint detection model*

U-net: Originally developed for biomedicine and uses upsampling technique instead of pooling and consits of a encoder and a decoder, so XD_XD replaces the encoder by the VGG16 and the decoder remains as the u-net. **Training time: 7 days**.


**Tutorials**: AWS: <a href="https://registry.opendata.aws/spacenet/">[Ref]</a>

https://aws.amazon.com/es/blogs/machine-learning/extracting-buildings-and-roads-from-aws-open-data-using-amazon-sagemaker/ 
 
https://medium.com/the-downlinq/getting-started-with-spacenet-data-827fd2ec9f53

https://medium.com/the-downlinq/announcing-solaris-an-open-source-python-library-for-analyzing-overhead-imagery-with-machine-48c1489c29f7

to check:
https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-38dad1cf7571

one of the competitors winners won the 1m dollars deep face
https://www.kaggle.com/c/deepfake-detection-challenge

spacenet winners

https://medium.com/the-downlinq/spacenet-7-results-arxiv-paper-and-winning-model-release-8da36df16aa5

### NOTES

Next Spacenet challenge owner: https://www.maxar.com/ 

How to build a custom baseline model https://solaris.readthedocs.io/en/latest/tutorials/notebooks/api_training_spacenet.html

About solaris toolkit:
https://www.linkedin.com/in/roshanr11/


[Roshan11](https://github.com/roshanr11?tab=repositories) 

