# Imports
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
from data_loader import Dataloader_trdp
# Albumentations

import albumentations as A
from albumentations.pytorch import ToTensorV2

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
batch_size = 16 # 64 patches to reconstruct img of 1024
num_epochs = 2

chip_dimension = 256
#chip_dimension = 512

train_dir = Path('../../DATASET/dataset_pruebas/train')
test_dir = Path('../../DATASET/dataset_pruebas/validation')
sample_dir = Path('../../DATASET/SN7_buildings_train_sample')

root_dir  = train_dir
csv_file = Path('../output_csvs/df_train_untidy.csv')
csv_file_test = Path('../output_csvs/df_test_untidy.csv')
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
train_set   = Dataloader_trdp(root_dir=root_dir,csv_file=csv_file,chip_dimension=chip_dimension,transform=transform)

testing_set = Dataloader_trdp(root_dir=root_dir,csv_file=csv_file_test,chip_dimension=chip_dimension,transform=transform)

train_set, val_set = torch.utils.data.random_split(train_set, [round(0.7*len(train_set)),round(0.3*len(train_set))])


train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_set, batch_size=batch_size, shuffle = False)
test_loader = DataLoader(dataset = testing_set, batch_size=batch_size, shuffle = False)
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    #get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)



# Hyperparameters etc.
LEARNING_RATE = 1e-4
PIN_MEMORY = True
LOAD_MODEL = True

def train_fn(loader, model, optimizer, loss_fn, scaler):
    
    loop = tqdm(loader)
    
    for batch_idx, (data) in enumerate(loop):
        
        images = data["raster_diff"].float()
        targets = data["mask_diff"].float()#type(torch.LongTensor)
       # images, mask = images.to(device), mask.to(device).unsqueeze(1)
      
        
        images = images.to(device=DEVICE)
        #targets = targets.float().unsqueeze(1).to(device=DEVICE) #
        targets = targets.float().squeeze().to(device=DEVICE) #

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(images).squeeze()
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop: 
        loop.set_postfix(loss=loss.item())
        

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

torch.backends.cudnn.enabled = False

if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

#check_accuracy(test_loader, model, device=DEVICE)
scaler = torch.cuda.amp.GradScaler()

num_epochs = 10

for epoch in range(num_epochs):
        #train_fn(train_loader, model, optimizer, loss_fn, scaler)


        #check_accuracy(train_loader, model, device=DEVICE)
        
        
        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)
        
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        '''
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



