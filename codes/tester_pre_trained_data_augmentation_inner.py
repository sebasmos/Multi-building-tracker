from __future__ import print_function, division

import sys
import subprocess
import pkg_resources

required = {'opencv-python'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing], stdout=None)
    
import matplotlib.pyplot as plt
    
from patchify import patchify, unpatchify
from pathlib import Path
import pathlib
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib
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

import rasterio as rio
# Albumentations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from model import UNET, UNET_manual

from data_loader import Dataloader_trdp
from utils import (
    load_checkpoint,
    load_checkpoint_pspnetlite,
    save_checkpoint,
    #get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    predict,
    store_predictions,
    store_predictions_with_patching,
    store_predictions_unet_improved
)


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
# Clean graphics memory
import gc
gc.collect()
torch.cuda.empty_cache()

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
    
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

force_cudnn_initialization()
'''

# Hyperparameters
in_channel = 3
num_classes = 1
learning_rate = 1e-4# 0.00001 is ideal, 0.0000001 is tooo much
batch_size = 1
num_epochs = 100
chip_dimension = 256
LOAD_MODEL = True
TRAIN = False
TEST = False
SAVE = False
# define threshold to filter weak predictions
THRESHOLD = 0.5
PREDICT = True

filename = "unet_24_dic_iou_DATASET_full_100_epochs_aug_defailt.pth" # my_checkpoint.pth.tar

'''
train_dir  = Path('../../DATASET/dataset_pruebas/train')
test_dir  = Path('../../DATASET/dataset_pruebas/validation')


train_dir = Path('../../DATASET/archive/train_final')
test_dir  = Path('../../DATASET/archive/val_final')


csv_file = Path('./output_csvs/df_train_untidy.csv')
csv_file_test = Path('./output_csvs/df_test_untidy.csv')


csv_file = Path('./registros/output_csvs_dataset_prueba/df_train_untidy.csv')
csv_file_test = Path('./registros/output_csvs_dataset_prueba/df_test_untidy.csv')
'''

#DATASET TOTAL
#train_dir = Path('../../DATASET/archive/train_final')
#csv_file = Path('./output_csvs_all/df_train_untidy.csv')

# DRAFT DATASET 
train_dir  = Path('../../DATASET/dataset_pruebas/train')
csv_file = Path('./registros/output_csvs_dataset_prueba/df_train_untidy.csv')

test_dir  = Path('../../DATASET/dataset_pruebas/validation')
csv_file_test = Path('./registros/output_csvs_dataset_prueba/df_test_untidy.csv')
#test_dir  = Path('../../DATASET/archive/val_final')
#csv_file_test = Path('./output_csvs_all/df_test_untidy.csv')

root_dir  = train_dir

df = pd.read_csv(csv_file)

df_test = pd.read_csv(csv_file_test)


sample_dir = Path('../../DATASET/SN7_buildings_train_sample')

# Reconstructiontest_loader
img_size = 1024
# Chip size given batch_size
chip_dim = ((img_size -1)//batch_size + 1)*2
# number of Columns per chip
columns = img_size / chip_dim
# Needed patches to reconstruct original image
patches_total = int(columns**2)




# https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions/blob/6a9c3962d987d985384d0d41a187f5fbfadac82c/2-MaksimovKA/train/losses.py#L14 
class FocalDiceLoss(torch.nn.Module):
    def __init__(self, coef_focal=1.0, coef_dice=1.0, weights=(1.0, 0.1, 0.5)):
        super().__init__()
       
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.weights = weights

        self.coef_focal = coef_focal
        self.coef_dice = coef_dice

    def forward(self, outputs, targets):
        loss = 0.0

        for i in range(outputs.shape[1]):
            dice = self.weights[i]*self.dice_loss(outputs[:, i, ...], targets[:, i, ...])
            focal = self.weights[i]*self.focal_loss(outputs[:, i, ...], targets[:, i, ...])
            loss += self.coef_dice * dice + self.coef_focal * focal

        return loss

    
from torch.nn.modules.loss import _Loss

class DiceLoss(_Loss):

    def __init__(self, per_image=False):
        super(DiceLoss, self).__init__()
        self.per_image = per_image

    def forward(self, y_pred, y_true):
        """
        :param y_pred: NxCxHxW
        :param y_true: NxCxHxW
        :return: scalar
        """
        per_image = self.per_image
        y_pred = y_pred.sigmoid()
        
        batch_size = y_pred.size()[0]
        eps = 1e-5
        if not per_image:
            batch_size = 1
        
        dice_target = y_true.contiguous().view(batch_size, -1).float()
        dice_output = y_pred.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
        loss = (1 - (2 * intersection + eps) / union).mean()

        return loss
    
    
class FocalLoss(_Loss):

    def __init__(self, ignore_index=255, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index


    def forward(self, y_pred, y_true):
        """
        :param y_pred: NxCxHxW
        :param y_true: NxCxHxW
        :return: scalar
        """
        y_pred = y_pred.sigmoid()
        gamma = self.gamma
        ignore_index = self.ignore_index

        outputs = y_pred.contiguous()
        targets = y_true.contiguous()
        eps = 1e-8
        non_ignored = targets.view(-1) != ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** gamma * torch.log(pt)).mean()



class Dataloader_trdp_pre_trained(Dataset):
    """SpaceNet 7 Multi-Temporal Satellite Imagery Dataset"""
    
    def __init__(self,csv_file, root_dir, no_udm=True, transform=None, chip_dimension=None, 
            preprocessing=None):
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
        self.preprocessing = preprocessing
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
        # apply preprocessing
        if self.preprocessing is not None:
            raster1 = np.moveaxis(np.uint8(sample['raster_diff'][:3]),0,-1)
            mask = np.moveaxis(np.uint8(sample['mask_diff']),0,-1)
            sample = self.preprocessing(image=raster1, mask=mask)
            image= sample['image']
            mask = sample['mask']
            sample['raster_diff'] = image
            sample['mask_diff'] = mask
            
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
                
                
          
###################################
import torch
import numpy as np
import segmentation_models_pytorch as smp

# https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
ENCODER = 'vgg16'#'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['Buildings']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
                
             
###############################################################################

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
A.Normalize(mean=mean,std=std)
#A.Rotate(limit=(-360, 360), interpolation=4, border_mode=4,p=1),


transform = A.Compose(
    [
        #A.Resize(height=256, width=256),
        
        A.PadIfNeeded(min_height=chip_dimension,min_width=chip_dimension,value=0,p=1),
        #A.RandomRotate90(p=1.0),
        ToTensorV2()
    ]
)
transform_testing = A.Compose(
    [
        #A.Resize(height=256, width=256),
        
        A.PadIfNeeded(min_height=chip_dimension,min_width=chip_dimension,value=0,p=1),
        A.RandomRotate90(p=1.0),
        ToTensorV2()
    ]
)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

train_set   = Dataloader_trdp_pre_trained(root_dir=train_dir,csv_file=csv_file,
                              chip_dimension=chip_dimension,
                              preprocessing=get_preprocessing(preprocessing_fn),
                              transform=transform)

testing_set = Dataloader_trdp_pre_trained(root_dir=test_dir,
                              csv_file=csv_file_test,
                              chip_dimension=chip_dimension,
                              preprocessing=get_preprocessing(preprocessing_fn),transform=transform)

#train_set, val_set = torch.utils.data.random_split(train_set, [round(0.7*len(train_set)),round(0.3*len(train_set))])

train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle = False, num_workers=12)
#val_loader   = DataLoader(dataset = val_set, batch_size=batch_size, shuffle = False)
test_loader  = DataLoader(dataset = testing_set, batch_size=batch_size, shuffle = False, num_workers=4)

print(f"Train : {len(train_loader)} - Test: {len(test_loader)}")

#########################################3

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    print(f"Number of images to plot: {n}")
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap='gray')
    plt.show()
    
sample = train_set[4] # get some sample

image = sample["raster_diff"]
mask = sample["mask_diff"]
print(f"{image.shape} - {mask.shape}")
'''
visualize(
    image=image[1,:,:].squeeze(), 
    Building_mask=mask.squeeze(),
)
'''


###############################################################################

import torchvision.transforms.functional as TF
import random
import numpy as np
print('PyTorch version:', torch.__version__)

class configuration:
    def __init__(self):
        self.experiment_name = "trdp1.001"
        self.pre_load = "True" ## Load dataset in memory
        self.pre_trained = "True"
        self.num_classes = num_classes
        self.ignore_label = 255
        self.lr = learning_rate  # 0.001 if pretrained. 0.1 if scratch
        self.M = [30,65] ##If training from scratch, reduce learning rate at some point - https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html
        self.batch_size = batch_size  # Training batch size
        self.test_batch_size = 4  # Test batch size
        self.epoch = num_epochs ## Number of epochs
        self.train_root = "./VOC"
        self.download = False
        self.seed = 271828


## Create arguments object
args = configuration()

# Make sure to enable GPU acceleration!
device = "cuda" if torch.cuda.is_available() else "cpu"


# Set random seed for reproducability
torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
torch.manual_seed(args.seed)  # CPU seed
torch.cuda.manual_seed_all(args.seed)  # GPU seed
random.seed(args.seed)  # python seed for image transformation
np.random.seed(args.seed)


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

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
    
    return thresholded.mean() # to get a batch averageimport pdb


def iou_fun(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    union = np.logical_or(im1, im2)
    im_sum = union.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return intersection.sum() / im_sum

def train_SemanticSeg(args, model, device, train_loader, optimizer, epoch, criterion):
    # switch to train mode
    model.train()

    train_loss = []
    counter = 1

    gts_all, predictions_all = [], []

    for batch_idx, (data) in enumerate(train_loader):
        
        images = data["raster_diff"].float()
        
        mask = data["mask_diff"].float().squeeze()
        
        images, mask = images.to(device), mask.to(device)

        
        if torch.cuda.is_available():
            model.cuda()
            
        
        #Forward pass
        outputs = model(images).squeeze()
                
        #Aggregated per-pixel loss
        #loss = criterion(outputs, mask)
        
        loss = criterion(outputs.unsqueeze(0).unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))
        train_loss.append(loss.item())

        iou = iou_fun(mask.data.cpu().numpy(), outputs.data.cpu().numpy())#iou_pytorch(mask.unsqueeze(0).int(), outputs.int())
        
        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        
        optimizer.step()
        
        if counter % 10000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, int(counter * len(images)), len(train_loader.dataset),
                100. * counter / len(train_loader), loss.item(),
                optimizer.param_groups[0]['lr']))
            print(f"Iou: {iou}")
        counter = counter + 1
        
    return sum(train_loss) / len(train_loss)#, mean_iu
        
        
def testing(args, model, device, test_loader, criterion):

    # switch to train mode
    model.eval()
    loss_per_batch = []
    test_loss = 0
    
    gts_all, predictions_all = [], []
    with torch.no_grad():
        for batch_idx, (data) in enumerate(test_loader):
            
            print(f" {batch_idx}/{len(test_loader)} ")
            if batch_idx <= 992:
            
                images = data["raster_diff"].float()
                mask = data["mask_diff"].float()#type(torch.LongTensor)
                images, mask = images.to(device), mask.to(device).squeeze()
                #Forward pass
                    
                if torch.cuda.is_available():
                    model.cuda()
                outputs = model(images).squeeze()
                #outputs = outputs.clone().detach().cpu().numpy()
        #        outputs = outputs.cpu().numpy()
                outputs = ((outputs - outputs.min())/(outputs.max()-outputs.min()))
                #outputs = np.round(outputs)

                outputs[outputs>THRESHOLD] = 1 
                outputs[outputs<=THRESHOLD] = 0
                #outputs = torch.round(outputs)

                #Aggregated per-pixel loss
                #loss = criterion(outputs, mask)
                
                loss = criterion(outputs.unsqueeze(0).unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))
                loss_per_batch.append(loss.item())

                # Convert to numpy
                outputs = outputs.detach().cpu().numpy()
                outputs = outputs.astype("int64")
                #predictions = np.round(predictions)

                mask = mask.data.cpu().numpy()
                mask = mask.astype("int64")

                '''
                import matplotlib.pyplot as plt
                plt.imshow(outputs)
                np.unique(outputs)

                import seaborn as sns
                sns.histplot(data = outputs)

                preds = outputs.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()

                mask = mask.data.cpu().numpy()
                mask = mask.astype("int64")
                plt.imshow(mask)

                assert np.unique(mask) == np.unique(outputs*255)
                '''
                gts_all.append(mask)
                predictions_all.append(255*outputs)

                if SAVE:

                    images = "outputs"
                    labels = images+"/predictions"
                    true = images+"/labels"

                    if not os.path.exists(images):
                        os.makedirs(images)
                    if not os.path.exists(labels):
                            os.makedirs(labels)
                    if not os.path.exists(true):
                            os.makedirs(true)

                    matplotlib.image.imsave(f"{labels}/Pred_{batch_idx}.png", outputs , cmap='gray')

                    matplotlib.image.imsave(f"{true}/True_{batch_idx}.png", mask, cmap='gray')
            else:
                break


    #test_loss /= len(test_loader.dataset)
    loss_per_epoch = [np.average(loss_per_batch)]
    

    ##Compute Mean Intersection over Union (mIoU)
    ##mIoU: Mean (of all classes) of intersection over union between prediction
    ##and ground-truth

    # Improved: https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686 
    iou_scores = []
    for lp, lt in zip(predictions_all, gts_all):
        # Convert lp to cpu
        #lp = lp.detach().cpu().numpy()
        intersection = np.logical_and(lp.flatten(), lt.flatten())  
        union = np.logical_or(lp.flatten(), lt.flatten())  
        iou_score = np.sum(intersection) / np.sum(union)
        iou_scores.append(iou_score)
    
    mean_iu = np.average(iou_scores)
    
    print('\nTest set ({:.0f}): Average loss: {:.4f}, mIoU: {:.4f}\n'.format(
        len(test_loader.dataset), loss_per_epoch[-1], mean_iu)) 
    
    ###########################################################################
    # Store predictions

    return loss_per_epoch, mean_iu, predictions_all, gts_all
    


criterion = FocalDiceLoss()#nn.BCEWithLogitsLoss()
 
milestones = args.M
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

import os
import time
loss_train_epoch = []
loss_test_epoch = []
acc_train_per_epoch = []
acc_test_per_epoch = []
new_labels = []

cont = 0

res_path = "./metrics_" + args.experiment_name


model_path = "./models" # + filename

if not os.path.exists(model_path):
    os.makedirs(model_path)
if LOAD_MODEL:
        load_checkpoint_pspnetlite(torch.load("models/" + filename), model)
    

if not os.path.isdir(res_path):    os.makedirs(res_path)
    


for epoch in range(1, args.epoch + 1):
        st = time.time()
        scheduler.step()    
        # train for one epoch
        if TRAIN:

            print("Unet improved --> training, epoch " + str(epoch))
    
            loss_per_epoch = train_SemanticSeg(args, model, device, train_loader, optimizer, epoch, criterion)
    
            loss_train_epoch += [loss_per_epoch]
            
            if epoch == args.epoch:
                torch.save(model.state_dict(), "models/" + filename)
    
            np.save(res_path + '/' + 'LOSS_epoch_train.npy', np.asarray(loss_train_epoch))
        
        # TESTING
        if TEST:
            print("Unet improved ==> testing, epoch " + str(epoch))
            # test
            loss_per_epoch_test, acc_val_per_epoch_i, a,b = testing(args, model, device, test_loader, criterion)
    
            loss_test_epoch += loss_per_epoch_test
            acc_test_per_epoch += [acc_val_per_epoch_i]
    
    
            if epoch == 1:
                best_acc_val = acc_val_per_epoch_i
    
            else:
                if acc_val_per_epoch_i > best_acc_val:
                    best_acc_val = acc_val_per_epoch_i
    
    
            np.save(res_path + '/' + 'LOSS_epoch_val.npy', np.asarray(loss_test_epoch))
    
            # save accuracies:
            np.save(res_path + '/' + 'accuracy_per_epoch_val.npy', np.asarray(acc_test_per_epoch))
            
            cont += 1

    
# PREDICT

if PREDICT: 
    
    y_pred, true_pred = [], []
    
    loss_per_epoch_test, acc_val_per_epoch_i, y_pred, true_pred = testing(args, model, device, test_loader, criterion)
    
    y_pred = np.array(y_pred)
    true_pred = np.array(true_pred)
    
    print(f"y_pred sizes: {y_pred.shape} - true_pred sizes: {true_pred.shape}")
    print(f"y_pred pixel values: {np.unique(y_pred)}")

    store_predictions_with_patching(y_pred, true_pred, 16)


##Accuracy
acc_test = np.load(res_path + '/' + 'accuracy_per_epoch_val.npy')

#Loss per epoch
loss_train = np.load(res_path + '/' + 'LOSS_epoch_train.npy')
loss_test = np.load(res_path + '/' + 'LOSS_epoch_val.npy')

numEpochs = len(acc_test)
epochs = range(numEpochs)

plt.figure(1)
plt.plot(epochs, acc_test, label='Test, max acc: ' + str(np.max(acc_test)))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

plt.figure(2)
plt.plot(epochs, loss_test, label='Test, min loss: ' + str(np.min(loss_test)))
plt.plot(epochs, loss_train, label='Train, min loss: ' + str(np.min(loss_train)))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.show()

