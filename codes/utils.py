import torch
import torchvision
from torch.utils.data import DataLoader
from patchify import patchify, unpatchify
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import matplotlib

import os
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
            for idx, data in enumerate(loader):
                x = data["raster_diff"].float()
                y = data["mask_diff"].float()
                x = x.to(device)
                y = y.to(device).squeeze()
                #print(f"x.shape : {x.shape}, y.shape : {y.shape}")
                preds = torch.sigmoid(model(x).squeeze())
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
                )

    print(
            f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
        )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    
def unpatchify_custom(img_patches, block_size):
    B0, B1 = block_size
    N = 1024 #np.prod(img_patches.shape[1::2])
    patches2D = img_patches.transpose(0,2,1,3).reshape(-1,N)

    m,n = patches2D.shape
    row_mask = np.zeros(m,dtype=bool)
    col_mask = np.zeros(n,dtype=bool)
    row_mask[::B0]= 1
    col_mask[::B1]= 1
    row_mask[-B0:] = 1
    col_mask[-B1:] = 1
    return patches2D[np.ix_(row_mask, col_mask)]

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    
    
    for idx, data in enumerate(loader):
        x = data["raster_diff"].float()
        y = data["mask_diff"].float()
        x = x.to(device)
        y = y.to(device).squeeze()
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        # Unpatch predictions
        
        cp = preds.cpu().data.numpy()
        block_size = (1024, 1024)
        reconstructed = unpatchify_custom(cp, block_size)
        reconstructed = torch.from_numpy(reconstructed)
        
        # To store, save_image requires 4 dims
        torchvision.utils.save_image(
            reconstructed, f"{folder}/pred_{idx}.png"
        )
        
        # Unpatch labels
        y = y.unsqueeze(1) # add 1 D
        labels = y.cpu().data.numpy() #convert to cpu numpy
        labels = unpatchify_custom(labels, block_size) # return 1024x1024
        labels = torch.from_numpy(labels) # return torch tensor
        torchvision.utils.save_image(labels, f"{folder}/label_{idx}.png")

    model.train()

def store_predictions(y_pred, true_pred):

    
    images = "outputs"
    labels = images+"/predictions"
    true = images+"/labels"
    
    if not os.path.exists(images):
        os.makedirs(images)
    if not os.path.exists(labels):
            os.makedirs(labels)
    if not os.path.exists(true):
            os.makedirs(true)

    p = 0

    for i in y_pred.squeeze():
      
        matplotlib.image.imsave(f"{labels}/Pred_{p}.png", i, cmap='gray')

        p+=1
        
    p = 0 
    for j in true_pred.squeeze():
        matplotlib.image.imsave(f"{true}/True_{p}.png", j, cmap='gray')
        p+=1


def predict(data_loader, model):
    
    print("Calculating segmentation with trained model..")
    model.eval()

    # save the predictions in this list
    y_pred = []
    y_true = []
    
    
    loop = tqdm(data_loader, leave = False)


        # go over each batch in the loader. We can ignore the targets here
    for batch_idx, data in enumerate(loop):
            print(f" - Batch index [{batch_idx}/{len(data_loader)}]")
            batch = data["raster_diff"].float()
            y = data["mask_diff"].float()
            batch = batch.to(device)
            y = y.to(device) # no squeeze for storing as same format
                # Move batch to the GPU
            batch = batch.to(device)

            # predict probabilities of each class
            
            with torch.no_grad():
                predictions = torch.sigmoid(model(batch))
                predictions = (predictions > 0.5).float()

            
            # move to the cpu and convert to numpy
            predictions = predictions.cpu().numpy()

            # save pred
            y_pred.append(predictions)
            # save respective labels
            
            labels = y.cpu().data.numpy() #convert to cpu numpy
            y_true.append(labels)
            
            '''
            if batch_idx == 2:
                break
            '''
    # stack predictions into a (num_samples, img size) array
    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)
    
    model.train()
    return y_pred, y_true
