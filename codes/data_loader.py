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