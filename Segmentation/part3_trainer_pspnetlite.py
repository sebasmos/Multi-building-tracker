'''
PSPNetLite - test 
'''

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
import time
import rasterio as rio

# Albumentations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from model import UNET, UNET_manual

from data_loader import Dataloader_trdp, Dataloader_trdp_pre_trained
from losses import CombinedLoss_TRDP


from utils import (
    load_checkpoint,
    load_checkpoint_pspnetlite,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
    predict,
    store_predictions,
    store_predictions_with_patching,
    store_predictions_unet_improved,
)

# Pytorch libraries
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
import torchvision.transforms.functional as TF

print('PyTorch version:', torch.__version__)

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
    
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import segmentation_models_pytorch as smp
import random

# Set seeds
def set_seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
'''
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

force_cudnn_initialization()
'''

###############################################################################
##########################Hyperparameters######################################
###############################################################################

in_channel = 3
num_classes = 1
learning_rate = 1e-5
batch_size = 1
num_epochs = 1
chip_dimension = 512
LOAD_MODEL = False
TRAIN = False
TEST = False
SAVE = True
# define threshold to filter weak predictions
THRESHOLD = 0.5



###############################################################################
#########################Set paths to dataset##################################
###############################################################################
'''
# Draft-dataset paths 
train_dir  = Path('../../DATASET/dataset_pruebas/train')
test_dir  = Path('../../DATASET/dataset_pruebas/validation')
csv_file = Path('./registros/output_csvs_dataset_prueba/df_train_untidy.csv')
csv_file_test = Path('./registros/output_csvs_dataset_prueba/df_test_untidy.csv')

# Full dataset paths
#test_dir  = Path('../../DATASET/archive/val_final')
#train_dir = Path('../../DATASET/archive/train_final')
#csv_file = Path('./output_csvs_all/df_train_untidy.csv')
#csv_file_test = Path('./output_csvs_all/df_test_untidy.csv')

'''
# Draft-dataset paths 
train_dir  = Path('../../DATASET/dataset_pruebas/train')
test_dir  = Path('../../DATASET/dataset_pruebas/validation')
csv_file = Path('./registros/output_csvs_dataset_prueba/df_train_untidy.csv')
csv_file_test = Path('./registros/output_csvs_dataset_prueba/df_test_untidy.csv')


t_dir  = train_dir

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


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
A.Normalize(mean=mean,std=std)
#A.Rotate(limit=(-360, 360), interpolation=4, border_mode=4,p=1),


transform = A.Compose(
    [
        A.PadIfNeeded(min_height=chip_dimension,min_width=chip_dimension,value=0,p=1),
        ToTensorV2()
    ]
)

train_set   = Dataloader_trdp(root_dir=train_dir,csv_file=csv_file,chip_dimension=chip_dimension,transform=transform)
testing_set = Dataloader_trdp(root_dir=test_dir,csv_file=csv_file_test,chip_dimension=chip_dimension,transform=transform)
train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle = False)
test_loader  = DataLoader(dataset = testing_set, batch_size=batch_size, shuffle = False)

print(f"Train : {len(train_loader)} - Test: {len(test_loader)}")


###############################################################################
###############################Training########################################
###############################################################################

class configuration:
    def __init__(self):
        self.experiment_name = "trdp1.001"
        self.pre_load = "True" ## Load dataset in memory
        self.pre_trained = "True"
        self.num_classes = num_classes
        self.ignore_label = 255
        self.lr = learning_rate  # 0.001 if pretrained. 0.1 if scratch
        self.M = [] ##If training from scratch, reduce learning rate at some point
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


class PSPNetLite(nn.Module):
    def __init__(self, args, num_classes, pretrained=True, use_aux=True):
        super(PSPNetLite, self).__init__()
        self.use_aux = use_aux
        
        #### TO FILL: define pytorch default resnet-18 architecture (pretrained and not) 
        if pretrained=="True":
            resnet = models.resnet18(pretrained=True) #
        else:
            resnet = models.resnet18(pretrained=False) #

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)

        ##Pooling module: simplification of Pyramid Pooling Module of PSPnet
        self.pm = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, momentum=.95),
            nn.ReLU(inplace=True)
        )

        ## Final classifier to get per-pixel predictions
        self.final = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95), #
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, 1, kernel_size=1)
        )
     
    #### To fill: write the forward pass function:
    #### layer0 --> layer1 --> layer2 --> layer3 --> layer4--> pm --> final
    def forward(self, x):
        x_size = x.size()

        x = self.layer0(x) #layer0
        x = self.layer1(x) #layer1
        x = self.layer2(x) #layer2
        x = self.layer3(x) #layer3
        
        x1 = self.layer4(x)
        x2 = self.pm(x1)

        # Concatenate layer4 features with upsampled Pooling Module features
        x = self.final(torch.cat((x1, F.interpolate(x2, x1.size()[2:], mode='bilinear')), dim=1)) #
        ##return prediction after bilinear upsampling to original size
        return F.interpolate(x, x_size[2:], mode='bilinear')



def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def train_SemanticSeg(args, model, device, train_loader, optimizer, epoch):
    # switch to train mode
    model.train()

    train_loss = []
    counter = 1

    criterion = nn.BCEWithLogitsLoss()
    gts_all, predictions_all = [], []

    for batch_idx, (data) in enumerate(train_loader):
        
        images = data["raster_diff"].float()
        
        mask = data["mask_diff"].float()
        
        images, mask = images.to(device), mask.to(device).squeeze()

        #Forward pass
        outputs = model(images).squeeze()

        
        #Aggregated per-pixel loss
        loss = criterion(outputs, mask)
        train_loss.append(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        
        optimizer.step()
        
        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, int(counter * len(images)), len(train_loader.dataset),
                100. * counter / len(train_loader), loss.item(),
                optimizer.param_groups[0]['lr']))
        counter = counter + 1
        
    return sum(train_loss) / len(train_loss)#, mean_iu
        
        
def testing(args, model, device, test_loader):

    # switch to train mode
    model.eval()
    loss_per_batch = []
    test_loss = 0

    ##We ignore index 255, i.e. object contours labeled with 255 in the val GT
    criterion = nn.BCEWithLogitsLoss()
    gts_all, predictions_all = [], []
    with torch.no_grad():
        for batch_idx, (data) in enumerate(test_loader):
            print(f" {batch_idx}/{len(test_loader)} ")
            images = data["raster_diff"].float()
            mask = data["mask_diff"].float()#type(torch.LongTensor)
            images, mask = images.to(device), mask.to(device).squeeze()
            
            
            #Forward pass
            outputs = model(images).squeeze()
            #outputs = outputs.clone().detach().cpu().numpy()
    #        outputs = outputs.cpu().numpy()
            outputs = ((outputs - outputs.min())/(outputs.max()-outputs.min()))
            #outputs = np.round(outputs)
            outputs[outputs>THRESHOLD] = 1 
            outputs[outputs<=THRESHOLD] = 0
            outputs = torch.round(outputs)
            
            #Aggregated per-pixel loss
            loss = criterion(outputs, mask)
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
                    
            

    #test_loss /= len(test_loader.dataset)
    loss_per_epoch = [np.average(loss_per_batch)]
    

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
    

model = PSPNetLite(1, num_classes=1, pretrained=args.pre_trained).to(device)

print('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    
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


filename = "psplite_9_dic_lr_-4.pth" # my_checkpoint.pth.tar
model_path = "./models" # + filename

if not os.path.exists(model_path):
    os.makedirs(model_path)
if LOAD_MODEL:
        load_checkpoint_pspnetlite(torch.load("models/" + filename), model)
    

if not os.path.isdir(res_path):
    os.makedirs(res_path)

for epoch in range(1, args.epoch + 1):
        st = time.time()
        scheduler.step()    
        # train for one epoch
        if TRAIN:

            print("Unet improved --> training, epoch " + str(epoch))
    
            loss_per_epoch = train_SemanticSeg(args, model, device, train_loader, optimizer, epoch)
    
            loss_train_epoch += [loss_per_epoch]
            
            if epoch == args.epoch:
                torch.save(model.state_dict(), "models/" + filename)
    
            np.save(res_path + '/' + 'LOSS_epoch_train.npy', np.asarray(loss_train_epoch))
        
        # TESTING
        if TEST:
            print("Unet improved ==> testing, epoch " + str(epoch))
            # test
            loss_per_epoch_test, acc_val_per_epoch_i, a,b = testing(args, model, device, test_loader)
    
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

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
# PREDICT

y_pred, true_pred = [], []

loss_per_epoch_test, acc_val_per_epoch_i, y_pred, true_pred = testing(args, model, device, test_loader)
