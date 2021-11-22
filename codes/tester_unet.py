# Imports

import sys
import subprocess
import pkg_resources

required = {'opencv-python'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing], stdout=None)
    
    
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

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
in_channel = 3
num_classes = 1
learning_rate = 1e-3
batch_size = 8 # 64 patches to reconstruct img of 1024
num_epochs = 10
chip_dimension = 256
TRAIN = True

train_dir = Path('../../DATASET/dataset_pruebas/train')
test_dir  = Path('../../DATASET/dataset_pruebas/validation')
sample_dir = Path('../../DATASET/SN7_buildings_train_sample')

root_dir  = train_dir
csv_file = Path('./output_csvs/df_train_untidy.csv')
csv_file_test = Path('./output_csvs/df_test_untidy.csv')
df = pd.read_csv(csv_file)
df_test = pd.read_csv(csv_file_test)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
A.Normalize(mean=mean,std=std)
A.Rotate(limit=(-360, 360), interpolation=4, border_mode=4,p=1),


transform = A.Compose(
    [
        A.Resize(height=100, width=100),
        A.PadIfNeeded(min_height=chip_dimension,min_width=chip_dimension,value=0,p=1),
        A.RandomRotate90(p=1.0),
        ToTensorV2()
    ]
)

train_set   = Dataloader_trdp(root_dir=train_dir,csv_file=csv_file,chip_dimension=chip_dimension,transform=transform)

testing_set = Dataloader_trdp(root_dir=test_dir,csv_file=csv_file_test,chip_dimension=chip_dimension,transform=transform)

#train_set, val_set = torch.utils.data.random_split(train_set, [round(0.7*len(train_set)),round(0.3*len(train_set))])


train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle = True)
#val_loader   = DataLoader(dataset = val_set, batch_size=batch_size, shuffle = False)
test_loader  = DataLoader(dataset = testing_set, batch_size=batch_size, shuffle = False)

print(f"Train : {len(train_loader)} - Test: {len(test_loader)}")

# Hyperparameters etc.
LEARNING_RATE = 1e-4
PIN_MEMORY = True
LOAD_MODEL = True


def train(loader, model, optimizer, loss_fn, scaler, epoch, num_epochs):
    
    size = len(loader.dataset)
    
    train_loss, train_acc = 0,0    
    
    num_correct = 0
    num_pixels = 0
    
    loop = tqdm(loader)
    
    for batch_idx, (data) in enumerate(loop):
        
        images = data["raster_diff"].float()
        targets = data["mask_diff"].float()
      
        
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
        
        # Loss
        train_loss += loss_fn(pred, targets).item()
        
        # Accuracy
        
        preds = torch.sigmoid(pred)
        preds = (preds > 0.5).float()
    
        aux = (preds == targets).sum()
        
        aux = aux.cpu().numpy()
        
        num_correct +=aux
        
        num_pixels += torch.numel(pred)
        

        if batch_idx % 10 == 0:
            print(f" Acc: {(num_correct/num_pixels)*100:.2f}")
        
        # update tqdm loop: 
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())
                
    train_loss /= size
    train_acc = num_correct/size
    
    print(f"Train Error: \n Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>8f} \n")
  
    return train_acc, train_loss
        

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

torch.backends.cudnn.enabled = False

if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

#check_accuracy(test_loader, model, device=DEVICE)
scaler = torch.cuda.amp.GradScaler()

train_accs,train_losss = [], []

test_accs,test_losss = [], []


if TRAIN:
    
    for epoch in range(num_epochs):
        
            
            acc_train, loss_train = train(train_loader, model, optimizer, loss_fn, scaler, epoch, num_epochs)
    
            train_accs.append(acc_train)
            train_losss.append( loss_train)
    #        check_accuracy(train_loader, model, device=DEVICE)
            
            '''
            
            # print some examples to a folder
            
            # save model
            
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
            '''
    print(f"Cumulative results are ready!\nTrain acc: {100*np.array(train_accs).sum()/num_epochs} \
          Test acc: {100*np.array(test_accs).sum()/num_epochs}")


# compute predictions on the validation set
y_pred, true_pred = predict(train_loader, model)

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

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
'''



