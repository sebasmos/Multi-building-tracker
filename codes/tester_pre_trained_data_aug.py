from __future__ import print_function, division

import sys
import subprocess
import pkg_resources

import os
import time

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
import scipy.io as sio
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
import torch.utils.data as data
import torchvision.transforms.functional as TF
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
from torch.nn.modules.loss import _Loss
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
LOAD_MODEL = False
TRAIN = True
TEST = False
SAVE = False
# define threshold to filter weak predictions
THRESHOLD = 0.5
PREDICT = True
filename = "unet_24_aug_dataset.pth" # my_checkpoint.pth.tar

class configuration:
    def __init__(self):
        self.experiment_name = "trdp1.001"
        self.pre_load = "True" ## Load dataset in memory
        self.pre_trained = "True"
        self.num_classes = num_classes
        self.ignore_label = 255
        self.lr = learning_rate  # 0.001 if pretrained. 0.1 if scratch
        self.M = [40, 60, 90] ##If training from scratch, reduce learning rate at some point
        self.batch_size = batch_size  # Training batch size
        self.test_batch_size = 4  # Test batch size
        self.epoch = num_epochs ## Number of epochs
        self.train_root = "../../DATASET/draft_dataset"#"../../DATASET/draft_dataset"
        self.download = False
        self.seed = 271828


## Create arguments object
args = configuration()

# Make sure to enable GPU acceleration!
device = 'cuda'

# Set random seed for reproducability
torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
torch.manual_seed(args.seed)  # CPU seed
torch.cuda.manual_seed_all(args.seed)  # GPU seed
random.seed(args.seed)  # python seed for image transformation
np.random.seed(args.seed)

###############################################################################
###############################################################################
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
###############################################################################
################################################################################
class TRDP_0_0_0(Dataset):

    def __init__(self, csv_images, csv_masks, transform=None, chip_dimension=1024, preprocessing=None):
        """
        Args:
            csv_images: images path with format
            csv_masks: masks path with format
            transform: transform 
            preprocessing: if transfer learning
        """
        self.images = pd.read_csv(csv_images, header=None, index_col=0)
        self.masks= pd.read_csv(csv_masks, header=None, index_col=0)
        self.transform = transform
        self.preprocessing = preprocessing
        self.total_images_index = self.total_number_of_images()


    def __len__(self):
            # if no patching is perform, image remain constant
            return len(self.total_images_index)
        
    def __getitem__(self,idx):

        if torch.is_tensor(idx):
            raster_idx = idx.tolist()
        

        #print(f"total_images {self.total_images_index} - idx: {idx}")
        image = self.images.iloc[idx, 0]
        image = io.imread(str(image))
        #image = np.moveaxis(image, -1, 0) # w,h,c
        #print(image.shape)
        #image = image[:,:,:] # why there is a third channel?
        
        mask = self.masks.iloc[idx, 0]
        mask = io.imread(str(mask), as_gray=True)
        mask = np.array([mask])
        mask = np.round(mask)
        sample = {'image': image, 'mask': mask}
        
        #print(f"0-{sample['image'].shape} - {sample['mask'].shape}")

        if self.transform is not None:
            
            transformed = self.transform(image = sample["image"], mask = sample["mask"])
            image = transformed['image']
            mask = transformed['mask']
            if not isinstance(image, np.ndarray): # if image not np.array-->
                sample['image'] = image
                #mask = mask.permute(2,0,1)
                sample['mask'] = mask
                
                
            else:
                sample['image'] = image
                mask = np.moveaxis(mask,-1,0)
                sample['masks'] = mask

        # apply preprocessing
        if self.preprocessing is not None:
            image = np.moveaxis(np.uint8(sample['image'][:3]),0,-1)
            masks = np.moveaxis(np.uint8(sample['mask']),0,-1)
            sample = self.preprocessing(image=image, mask=masks)
            image= sample['image']
            mask = sample['mask']
            sample['image'] = image
            sample['mask'] = mask
        
        return sample
    
    def total_number_of_images(self):
            # we need to change it to obtain all the possible images we are going to use
            return  self.images 
                
    
train_transform_train = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


transform = A.Compose(
    [
        ToTensorV2()
    ]
)

# Train_set should have h,w,c at this point
train_set = TRDP_0_0_0(csv_images='csv_train_images_aug_patched.csv',
                              csv_masks='csv_train_masks_aug_patched.csv',
                              preprocessing=get_preprocessing(preprocessing_fn),
                              transform=transform
                              )
testing_set = TRDP_0_0_0(csv_images='csv_test_images_aug_patched.csv',
                              csv_masks='csv_test_masks_aug_patched.csv',
                              preprocessing=get_preprocessing(preprocessing_fn),
                              transform=transform,
                              )
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
    
sample = train_set[5] # get some sample

image = sample["image"]
masks = sample["mask"]
print(f"{image.shape} - {masks.shape}") #-> (b,c,h,w) 

'''
img_Test = np.moveaxis(np.uint8(sample['image'][:3]),0,-1)
visualize(
    image=img_Test,  
    masks=masks.squeeze(),
)
'''

train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle = False, num_workers=12)
#val_loader   = DataLoader(dataset = val_set, batch_size=batch_size, shuffle = False)
test_loader  = DataLoader(dataset = testing_set, batch_size=batch_size, shuffle = False, num_workers=4)

print(f"Train : {len(train_loader)} - Test: {len(test_loader)}")
################################################################################
############################TRAINING############################################
################################################################################

import torchvision.transforms.functional as TF
import random
import numpy as np
print('PyTorch version:', torch.__version__)
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
        
        images = data["image"].float()
        
        mask = data["mask"].float().squeeze()
        
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
        
        if counter % 1000 == 0:
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
            
                images = data["image"].float()
                mask = data["mask"].float()#type(torch.LongTensor)
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

            print("VGG16 + UNET  --> training, epoch " + str(epoch))
    
            loss_per_epoch = train_SemanticSeg(args, model, device, train_loader, optimizer, epoch, criterion)
    
            loss_train_epoch += [loss_per_epoch]
            
            if epoch == args.epoch:
                torch.save(model.state_dict(), "models/" + filename)
    
            np.save(res_path + '/' + 'LOSS_epoch_train.npy', np.asarray(loss_train_epoch))
        
        # TESTING
        if TEST:
            print("VGG16 + UNET  ==> testing, epoch " + str(epoch))
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

