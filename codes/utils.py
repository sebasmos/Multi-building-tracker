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
    m,n,r,q = img_patches.shape
    shp = m + r - 1, n + q - 1

    p1 = img_patches[::B0,::B1].swapaxes(1,2)
    p1 = p1.reshape(-1,p1.shape[2]*p1.shape[3])
    p2 = img_patches[:,-1,0,:]
    p3 = img_patches[-1,:,:,0].T
    p4 = img_patches[-1,-1]

    out = np.zeros(shp,dtype=img_patches.dtype)
    out[:p1.shape[0],:p1.shape[1]] = p1
    out[:p2.shape[0],-p2.shape[1]:] = p2
    out[-p3.shape[0]:,:p3.shape[1]] = p3
    out[-p4.shape[0]:,-p4.shape[1]:] = p4
    return out

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
            print(preds.shape)
        
       # preds = preds.squeeze() # get rid of 1d and leave chxNXN
       
        cp = preds.cpu().data.numpy()
        
        block_size = (1024, 1024)
        
        rec = unpatchify_custom(cp, block_size)
        
        # To store, save_image requires 4 dims
        torchvision.utils.save_image(
            rec, f"{folder}/pred_{idx}.tiff"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
