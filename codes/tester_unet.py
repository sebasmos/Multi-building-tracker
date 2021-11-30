# Imports

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
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torchvision import datasets, transforms, models

import torchvision.transforms as T
from pathlib import Path
import pathlib
from PIL import Image
import pandas as pd
import numpy as np
import os
# Albumentations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from model import UNET

from data_loader import Dataloader_trdp
from utils import (
    load_checkpoint,
    save_checkpoint,
    #get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    predict,
    store_predictions
)

# Clean graphics memory
import gc
gc.collect()
torch.cuda.empty_cache()

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


###############################################################################

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
TAM = 1024
in_channel = 3
num_classes = 1
LEARNING_RATE = 1e-4
batch_size = 1 # 64 patches to reconstruct img of 1024
num_epochs = 1
chip_dimension = 1024
TRAIN = True
LOAD_MODEL = False

'''

train_dir = Path('../../DATASET/dataset_pruebas/train')
test_dir  = Path('../../DATASET/dataset_pruebas/validation')

train_dir = Path('../../DATASET/archive/train_final')
test_dir  = Path('../../DATASET/archive/_final')


csv_file = Path('./output_csvs/df_train_untidy.csv')
csv_file_test = Path('./output_csvs/df_test_untidy.csv')


csv_file = Path('./registros/output_csvs_dataset_prueba/df_train_untidy.csv')
csv_file_test = Path('./registros/output_csvs_dataset_pruebadf_test_untidy.csv')
'''

train_dir = Path('../../DATASET/archive/train_final')
test_dir  = Path('../../DATASET/archive/_final')

csv_file = Path('./output_csvs/df_train_untidy.csv')
csv_file_test = Path('./output_csvs/df_test_untidy.csv')


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

train_set   = Dataloader_trdp(root_dir=train_dir,csv_file=csv_file,chip_dimension=chip_dimension,transform=transform)

testing_set = Dataloader_trdp(root_dir=test_dir,csv_file=csv_file_test,chip_dimension=chip_dimension,transform=transform)

#train_set, val_set = torch.utils.data.random_split(train_set, [round(0.7*len(train_set)),round(0.3*len(train_set))])

train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle = False)
#val_loader   = DataLoader(dataset = val_set, batch_size=batch_size, shuffle = False)
test_loader  = DataLoader(dataset = testing_set, batch_size=batch_size, shuffle = False)

print(f"Train : {len(train_loader)} - Test: {len(test_loader)}")



def train(loader, model, optimizer, loss_fn, scaler, epoch, num_epochs, TAM):
    
    size = len(loader.dataset)
    
    train_loss, train_acc = 0,0    
    
    num_correct = 0
    
    num_pixels = 0
    
    loop = tqdm(loader, )
    
    model.train()
    
    for batch_idx, (data) in enumerate(loop):
        
        images = data["raster_diff"].float()
    
        targets = data["mask_diff"].float()
        

        # Cropper
        cropper  = transforms.CenterCrop(300)
        images = cropper(images)
        targets = cropper(targets)
        
        '''
        # Resize 100, 100
        #transf = transforms.Resize(size = (TAM, TAM), interpolation = transforms.InterpolationMode.BILINEAR)
               
        images = transf(images)
        
        targets = transf(targets)
        
        '''
        
        
        '''
        # PATCHIFY
        true = images
        true = true.cpu().data.numpy()
        true = true[:,1,:,:].squeeze()
        true = np.reshape(true, (4,4,100 ,100))
        reconstructed_true = unpatchify(true, (400,400))
            # To store, save_image requires 4 dims
        
        plt.imsave(f"true_patchify.png", reconstructed_true, cmap='gray')
        '''
        
        images = images.to(device=DEVICE)
        targets = targets.float().squeeze().to(device=DEVICE) #

        # forward
        with torch.cuda.amp.autocast():
            pred = model(images).squeeze()
            loss = loss_fn(pred, targets)

        # Backpropagation
        optimizer.zero_grad()
        
        scaler.scale(loss).backward()
        
        #gradient descent update step/adam step
        scaler.step(optimizer)
        
        scaler.update()
        
        
        # Accuracy
        
        preds = torch.sigmoid(pred)
        
        preds = (preds > 0.5).float()
    
        aux = (preds == targets).sum()
        
        aux = aux.cpu().numpy()
        
        num_correct +=aux
        
        num_pixels += torch.numel(pred)
        

        if batch_idx % 100 == 0:
            print(f" Acc: {(num_correct/num_pixels)*100:.2f}")
        
        # Loss
        train_loss += loss_fn(pred, targets).item()
        
        # update tqdm loop: 
            
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())
                
    train_loss /= size
    train_acc = num_correct/num_pixels
    
    print(f"Train Error: \n Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>8f} \n")
  
    return train_acc, train_loss
        

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

torch.backends.cudnn.enabled = False

filename = "my_checkpoint.pth.tar" # my_checkpoint.pth.tar
model_path = "./models" # + filename
    
if not os.path.exists(model_path):
    os.makedirs(model_path)
if LOAD_MODEL:
        load_checkpoint(torch.load("models/" + filename), model)

#check_accuracy(test_loader, model, device=DEVICE)
scaler = torch.cuda.amp.GradScaler()

train_accs,train_losss = [], []

test_accs,test_losss = [], []

if TRAIN:
    
    for epoch in range(num_epochs):
        
            acc_train, loss_train = train(train_loader, model, optimizer, loss_fn, scaler, epoch, num_epochs, TAM)
    
            train_accs.append(acc_train)
            train_losss.append( loss_train)
    #        check_accuracy(train_loader, model, device=DEVICE)
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, "my_checkpoint.pth.tar")
            

    print(f"Cumulative results are ready!\nTrain acc: {100*np.array(train_accs).sum()/num_epochs} \
          Test acc: {100*np.array(test_accs).sum()/num_epochs}")


# compute predictions on the validation set

y_pred, true_pred = predict(train_loader, model, TAM)

print(f"y_pred sizes: {y_pred.shape} - true_pred sizes: {true_pred.shape}")
print(f"y_pred pixel values: {np.unique(y_pred)}")

#store_predictions(y_pred, true_pred, patches_total)
store_predictions(y_pred, true_pred)

'''
for epoch in range(num_epochs):
        
    
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)
        
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy+
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
'''



