from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import rasterio as rio
from rasterio import features
from pathlib import Path
import pathlib
import geopandas as gpd
from descartes import PolygonPatch
from PIL import Image
import itertools
import re
from tqdm.notebook import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#import imgaug
import random
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


# Albumentations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Pytorch
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
from torch.utils.data import Dataset, DataLoader, Sampler # custom dataset handling
import torch.autograd.profiler as profiler # to track model inference and detect leaks
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torchvision import datasets, transforms, models
import torchvision.transforms as T
from torchvision import transforms, utils
from torch.autograd import Variable
from torch.nn.modules.padding import ReplicationPad2d
import torchvision.models as models
from torch import optim
from collections import OrderedDict
import segmentation_models_pytorch as smp #semantic segmentation models and utils
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast




def set_seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #imgaug.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
# Load split over training data
train_dir = Path('./DATASET/dataset_pruebas/train')
test_dir = Path('./DATASET/dataset_pruebas/validation')
sample_dir = Path('./DATASET/SN7_buildings_train_sample')

class Dataloader_trdp(Dataset):
    """SpaceNet 7 Multi-Temporal Satellite Imagery Dataset"""
    
    def __init__(self,csv_file, root_dir, no_udm=True, transform=None, chip_dimension=None):
        """
        Args:
            csv_file (Path): Path to the csv file with annotations
            root_dir (Path): Parent directory containing all other directories.
            no_udm (bool): Specifies whether the dataset will load UDM images or not.
            transform (callable, optional): Optional transform to be applied on a sample. This will be used for data augmentation
            chip_dimension (int, optional): Specifies the dimensions of the chip being generated.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.no_udm = no_udm
        self.transform = transform
        self.total_images_index = self.total_number_of_images()
        self.chip_dimension = chip_dimension
        if self.chip_dimension is not None:
            self.chip_generator = self.__ChipGenerator(chip_dimension = self.chip_dimension)
            # this will be replaced later with an abstracted version
            # returns number of chips per image assuming all images are 1024
            # We will obtain the total number of chips as a patching manual method
            self.n_chips = ((1024 - 1) // self.chip_dimension + 1)**2
    '''
    __len__ gets the total number of patches as the number of images available multiply by the number of splits n_chips
    '''
    def __len__(self):
        if self.chip_dimension is not None: # Total # images = No clouds + Not none
            return len(self.total_images_index)*self.n_chips
        else:
            # if no patching is perform, image remain constant
            return len(self.total_images_index)
    '''
    __getitem__ receives the index idx for each element of the dataset, interatevely.
    raster_idx is the index for patches: number of idx available over the number of chips
    chips_idx is the module of index with number of chips
    '''
    def __getitem__(self,idx):
        if self.chip_dimension is not None:
            raster_idx = idx//self.n_chips
            chip_idx = idx%self.n_chips
        else:
            raster_idx = idx
            
        if torch.is_tensor(raster_idx):
            raster_idx = raster_idx.tolist()
        # Obtain the corresponding index to the desired iteration of the raster
        idx1 = self.total_images_index[raster_idx]
        # paths where the images are stored: containing UDM mask
        img1_path = self.root_dir/self.annotations.loc[idx1,'images_masked']
        # paths where the corresponding true building footprints 
        labels1_path = self.root_dir/self.annotations.loc[idx1,'labels_match_pix']
        # read rasters using imported rasterio library
        with rio.open(img1_path) as r1:
            raster1 = r1.read()[0:3]  
        raster_diff = raster1 
        # get the dates for the images
        date1 = tuple(self.annotations.loc[idx1,['month','year']])
        # read geojson files for each of the satellite images into a geodataframe
        gdf1 = gpd.read_file(labels1_path).set_index('Id').sort_index()
        gdf_diff = gdf1  # self.__geo_difference(labels1_path,labels2_path)
        # get the corresponding rasterized image of the geodataframes
        mask_diff = self.__rasterize_gdf(gdf_diff,out_shape=raster1.shape[1:3])
        
        if self.chip_dimension:
            raster_diff_dict = self.chip_generator(raster_diff)
            mask_diff_dict = self.chip_generator(mask_diff)

            sample = {'raster_diff':raster_diff_dict['chip'][chip_idx],'date1':date1,
          'mask_diff':mask_diff_dict['chip'][chip_idx],'im_dir':str(img1_path.parent.parent),'blank_label':mask_diff_dict['blank'][chip_idx]}
        
        else:
            sample = {'raster_diff':raster_diff,'date1':date1, 'mask_diff':mask_diff,'im_dir':str(img1_path.parent.parent)}
        
        if self.transform is not None:
            # get the individual images and mask from the output sample
            raster1 = np.moveaxis(np.uint8(sample['raster_diff'][:3]),0,-1)
            #raster2 = np.moveaxis(np.uint8(sample['raster_diff'][3:6]),0,-1)
            mask = np.moveaxis(np.uint8(sample['mask_diff']),0,-1)
            import random
            seed = random.randint(0,1000)
            set_seed(seed)
            
            # apply transform on first image and mask
            transformed = self.transform(image=raster1,mask=mask)
            raster1 = transformed['image']
            mask_diff = transformed['mask']
            
            set_seed(seed)
            
            # concatenate input images
            raster_diff = raster1
            # update sample dictionary paramters after transformation
            if not isinstance(raster_diff,np.ndarray):
                sample['raster_diff'] = raster_diff
                mask_diff = mask_diff.permute(2,0,1)
                sample['mask_diff'] = mask_diff
                
                
            else:
                sample['raster_diff'] = raster_diff
                mask_diff = np.moveaxis(mask_diff,-1,0)
                sample['mask_diff'] = mask_diff
            
        return sample
    
    '''
    total_number_of_images obtains the indexes for the images_masks, this can be changed by updating the groupby. 
    If UDM filter is available to discard cloudy images.
    OUTPUT: indexes for selected folder. 
    '''
    def total_number_of_images(self):
            # we need to change it to obtain all the possible images we are going to use
            total_images = []
            # group by satellite image location
            location_groups = self.annotations.groupby('images')
            for i,location in enumerate(location_groups):
                # get the dataframe in the group
                loc_frame = location[1]
                # make sure that list does not contain images with unidentified masks
                condition = (loc_frame['has_udm'] == False)
                # return a list of the indices in the location dataframe
                l = list(loc_frame[condition].index)
                total_images.extend(l)
            return total_images 
    

    
    def __rasterize_gdf(self,gdf,out_shape):
        # if geodataframe is empty return empty mask
        if len(gdf)==0:
            return np.zeros((1,*out_shape))
            
        mask = features.rasterize(((polygon, 255) for polygon in gdf['geometry']),out_shape=out_shape)
        
        return np.expand_dims(mask,axis=0)
    
    class __ChipGenerator():   
        def __init__(self, chip_dimension=256,return_raster=False):  
            self.chip_dimension = chip_dimension
            self.return_raster = return_raster
            self.chip_dict = {'chip':[],'x':[],'y':[], 'blank':[]}

        def __call__(self,raster):
            np_array = self.__read_raster(raster)
            # get number of chips per colomn
            n_rows = (np_array.shape[1] - 1) // self.chip_dimension + 1
            # get number of chips per row
            n_cols = (np_array.shape[2] - 1) // self.chip_dimension + 1
            # segment image into chips and return dict of chips and metadata
            chip_dict = {'chip':[],'x':[],'y':[], 'blank':[]}

            for r in range(n_rows):
                for c in range(n_cols):
                    start_r_idx = r*self.chip_dimension
                    end_r_idx = start_r_idx + self.chip_dimension

                    start_c_idx = c*self.chip_dimension
                    end_c_idx = start_c_idx + self.chip_dimension
                    
                    chip = np_array[:,start_r_idx:end_r_idx,start_c_idx:end_c_idx]

                    chip_dict['chip'].append(chip)
                    chip_dict['x'].append(start_r_idx)
                    chip_dict['y'].append(start_c_idx)
                    
                    # Check if the chip is an empty chip
                    if chip.mean() == 0 and chip.sum() == 0:
                        chip_dict['blank'].append(1)
                    else:
                        chip_dict['blank'].append(0)

            return chip_dict

        def __read_raster(self,raster):
            # check whether raster is a path or array
            if isinstance(raster,(pathlib.PurePath,str)):
                    with rio.open(raster) as r:
                        # convert raster into np array
                        np_array = r.read()
                    return np_array

            elif isinstance(raster,np.ndarray):
                return raster
            else:
                raise ValueError(f"Expected Path or Numpy array received: {type(raster)}")  
                
                
import skimage    

class TorchDataset(Dataset):
    """Dataset class
    Args:
        root_folder: Path object, root directory of picture dataset
        csv: pandas.DataFrame, untidy df with all data relationships
        aug: albumentations dictionary
        preproc: callable, preprocessing function related to specific encoder
        grayscale: boolean, preprocessing condition to grayscale colored rasters
    Return:
        image, mask tensors"""
    
    def __init__(self, root_folder, df, aug = None, preproc = None, grayscale = True):
        self.root_folder = root_folder
        self.csv = df
        self.aug = aug
        self.preproc = preproc
        self.grayscale = grayscale
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        chip_path = self.root_folder + self.csv.loc[idx,'chip_path']
        # read chip into numpy array
        chip = skimage.io.imread(root_folder + self.csv.loc[idx,'chip_path']).astype('float32')
        if self.grayscale:
            gray1 = np.dot(chip[:,:,0:3], [0.2989, 0.5870, 0.1140])
            gray2 = np.dot(chip[:,:,3:], [0.2989, 0.5870, 0.1140])
            chip = np.divide(np.stack((gray1, gray2),axis = 2),255).astype('float32')
        # get target for corresponding chip
        mask = np.abs(np.divide(skimage.io.imread(root_folder + self.csv.loc[idx,'mask_path']),255)).astype('float32')
        # apply augmentations
        if self.aug:
            sample = self.aug(image=chip, mask=mask)
            image, mask = sample['image'], sample['mask']
            mask = mask.unsqueeze(0)
            if self.grayscale:
                sample = {'I1':image[0,:,:].unsqueeze(0),'I2':image[1,:,:].unsqueeze(0), 'label':mask}
            else: 
                sample = {'I1':image[0:3,:,:],'I2':image[3:,:,:], 'label':mask}
            del(image,mask,chip,gray1,gray2)
            return sample
        else:
            image = torch.Tensor(np.moveaxis(chip, 2, 0))
            mask = torch.Tensor(mask).unsqueeze(0)
            if self.grayscale:
                sample = {'I1':image[0,:,:].unsqueeze(0),'I2':image[1,:,:].unsqueeze(0), 'label':mask}
            else: 
                sample = {'I1':image[0:3,:,:],'I2':image[3:,:,:], 'label':mask}
            del(mask,chip,gray1,gray2)
            return sample
    

    
    


class BalancedSampler(Sampler):
    """Balancer for torch.DataLoader to adjust chips loading"""
    
    def __init__(self, dataset, percentage = 0.5):
        """
        dataset: custom torch dataset
        percentage: float number between 0 and 1, percentage of change containing pictures in batch
        """
        assert 0 <= percentage <= 1,'percentage must be a value between 0 and 1'
        
        self.dataset = dataset
        self.pct = percentage
        self.len_ = len(dataset)
    
    def __len__(self):
        return self.len_
    
    def __iter__(self):
        # get indices for chips containing change and blank ones
        change_chip_idxs = np.where(self.dataset.csv['target'] == 1)[0]
        blank_chip_idxs = np.where(self.dataset.csv['target'] == 0)[0]
        # randomly sample from the incides of each class according to percentage value
        change_chip_idxs = np.random.choice(change_chip_idxs,int(self.len_ * self.pct), replace=True)
        blank_chip_idxs = np.random.choice(blank_chip_idxs,int(self.len_ * (1 - self.pct))+1, replace=False)
        # stack and shuffle of sampled indices
        all_idxs = np.hstack([change_chip_idxs,blank_chip_idxs])
        np.random.shuffle(all_idxs)
        return iter(all_idxs)

from sklearn.metrics import confusion_matrix

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    """Fast enough iou calculation function"""
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    #outputs = outputs.detach()
    #labels = labels.detach()
    
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean() # to get a batch average

def segmentation_report(running_preds, running_labels):
    """Function to get a closer look to a confusion metrics and related metrics"""
    rp = running_preds.flatten()
    rl = running_labels.flatten()
    tn, fp, fn, tp = confusion_matrix(rl, rp, labels=[0,1]).ravel()
    px_accuracy = (tp+tn) / (tp+fp+tn+fn)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    #calculating intersection over union
    intersection = np.logical_and(rl, rp)
    union = np.logical_or(rl, rp)
    iou_score = np.sum(intersection) / np.sum(union)
    fmeasure = 2 * precision * recall / (precision + recall)
    #making report
    report = { 'tp/tn/fp/fn' : (tp,tn,fp,fn),
              'px_accuracy': px_accuracy,
              'precision': precision,
              'recall': recall,
              'iou_score': iou_score,
              'fmeasure':fmeasure
             }
    return report

import time

def log_batch_statistics(batch_number,batch_labels,batch_preds,phase,loss,since,num_batches,period=500):
    if batch_number % period == 0:
        iou_score = segmentation_report(batch_preds,batch_labels)
        time_elapsed = time.time() - since
'''
        if phase == 'train':
            telegram_bot_sendtext('TRAINING BATCH')
        else:
            telegram_bot_sendtext('VALIDATION BATCH')
        
        telegram_bot_sendtext('-'*50)
        telegram_bot_sendtext(f'\n{batch_number}/{num_batches-1}:')
        telegram_bot_sendtext(f'Total Time Elapsed: {time_elapsed/60:.2f} mins')
        telegram_bot_sendtext(f'Batch Loss: {loss.item():.4f}\n')
        telegram_bot_sendtext(f"``` IoU_score:{iou_score}\n ```")
        telegram_bot_sendtext('-'*50)
'''

import sys 
def break_time_limit(start_time,time_limit=28080):
    time_elapsed = time()-start_time
    if time_elapsed > time_limit:
        sys.exit()
        
def test():
    x = torch.randn((3, 1, 161, 161))
    # Hyperparameters
    in_channel = 3
    num_classes = 1
    learning_rate = 1e-4
    batch_size = 1
    num_epochs = 2
    chip_dimension = None
    LOAD_MODEL = True
    TRAIN = False
    TEST = False
    SAVE = False
    # define threshold to filter weak predictions
    THRESHOLD = 0.5
    
    
    train_dir = Path('../../DATASET/archive/train_final')
    
    #train_dir  = Path('../../DATASET/dataset_pruebas/train')
    test_dir  = Path('../../DATASET/dataset_pruebas/validation')
    #test_dir  = Path('../../DATASET/archive/val_final')
    
    csv_file = Path('./output_csvs/df_train_untidy.csv')
    #csv_file = Path('./registros/output_csvs_dataset_prueba/df_train_untidy.csv')
    csv_file_test = Path('./registros/output_csvs_dataset_prueba/df_test_untidy.csv')
    
    #csv_file_test = Path('./output_csvs/df_test_untidy.csv')
    
    root_dir  = train_dir
    
    df = pd.read_csv(csv_file)
    df_test = pd.read_csv(csv_file_test)
    
    
    sample_dir = Path('../../DATASET/SN7_buildings_train_sample')

    
    train_set   = Dataloader_trdp(root_dir=train_dir,csv_file=csv_file)
    
    testing_set = Dataloader_trdp(root_dir=test_dir,csv_file=csv_file_test,chip_dimension=chip_dimension,transform=transform)
    
    #train_set, val_set = torch.utils.data.random_split(train_set, [round(0.7*len(train_set)),round(0.3*len(train_set))])
    
    train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle = False)
    #val_loader   = DataLoader(dataset = val_set, batch_size=batch_size, shuffle = False)
    test_loader  = DataLoader(dataset = testing_set, batch_size=batch_size, shuffle = False)
    
    print(f"Train : {len(train_loader)} - Test: {len(test_loader)}")

    ###############
    
    chip_dimension = 64
    augs = {
        'train': A.Compose([
            A.PadIfNeeded(min_height=chip_dimension,min_width=chip_dimension,value=0,p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2() #apparently doesn't work properly with smp Unet, included in get_preprocessing function
        ]),
        'test': A.Compose([
            A.PadIfNeeded(min_height=chip_dimension,min_width=chip_dimension,value=0,p=1),
            ToTensorV2()
        ]), 
        'valid': A.Compose([
            A.PadIfNeeded(min_height=chip_dimension,min_width=chip_dimension,value=0,p=1),
            ToTensorV2()
        ]),
    }
    
    augs['train']
    
    #learning policy params
    grayscale = True
    if grayscale == True:
        in_channels = 2
    else: in_channels = 6
        
    N_EPOCHS = 25
    
    # turning on GPU if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    
    
    # mean percentage of positives is 6.5% from the frame, median is 3.4%, so weights for bce loss are required.
    weights = torch.Tensor([28]).to(device) 
    
    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight = weights) 
    criterion = IoULoss()
    
    #defining datasets, samplers and dataloaders
    datasets = {x:TorchDataset(root_folder = root_dir,df = csv_file[x],aug = None, preproc = None, grayscale = grayscale) for x in ['train','test','valid']}


if __name__ == "__main__":
    test()