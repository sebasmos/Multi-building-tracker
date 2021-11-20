import torch
import torchvision
from torch.utils.data import DataLoader
from patchify import patchify, unpatchify
import numpy as np

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
    print("ENTROOOO")
    with torch.no_grad():
            for idx, data in enumerate(loader):
                x = data["raster_diff"].float()
                y = data["mask_diff"].float()
                x = x.to(device)
                y = y.to(device).squeeze()
                print(f"x.shape : {x.shape}, y.shape : {y.shape}")
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
