import torch
import torchvision
from torch.utils.data import DataLoader

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
            preds = torch.sigmoid(model(x).squeeze())
            preds = (preds > 0.5).float()
            print(preds.shape)
    
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.tiff"
        )
        torchvision.utils.save_image(y.squeeze(), f"{folder}{idx}.png")

    model.train()
