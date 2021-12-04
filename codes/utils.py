import torch
import torchvision
from torch.utils.data import DataLoader
from patchify import patchify, unpatchify
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import matplotlib.pyplot as plt
from torchvision import  transforms
import os
import matplotlib
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    model_path = "./models" # + filename
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(state, model_path + "/"+filename)
    
    return print(f"{filename} has been stored in {model_path}")

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def load_checkpoint_pspnetlite(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint)
    print("Model loaded successfully")

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


def store_predictions_with_patching(y_pred, true_pred, num_patches = 16):

    if len(y_pred) % num_patches == 0:
        print("Data size will be reconstructed successfully.")
        fixed_batch = len(y_pred)
    else:
        print("Wrong number of batches for original image reconstruction \n lowest boundery will be taken")
        fixed_batch = round(len(y_pred)/(num_patches))
        
    images = "outputs"
    labels = images+"/predictions"
    true_path = images+"/labels"
    
    if not os.path.exists(images):
        os.makedirs(images)
    if not os.path.exists(labels):
            os.makedirs(labels)
    if not os.path.exists(true_path):
            os.makedirs(true_path)
            
    
    p = 0
    dimen = int(np.sqrt(num_patches))  # number of columns or rows to reconstruct
    length = int(y_pred.shape[2]) # size of image, e.g. 256
    
    #x = torch.randn((64, 1, 100, 100))
    
    # len(y_pred) should be divisible by 16 or 2*batch_size if batch is 8
    for i in range(0, fixed_batch, num_patches):
            print(f"Prediction ({p})")
            #
            #
            # Create sliding window to reconstruct original image of size window x window
            pred = y_pred[ p*num_patches : p*num_patches + num_patches ]
            # Delete additional dimension
            prediction = pred.squeeze()
            # Take np.sqrt(window) to obtain real size. e.g: window is 16, then recons
            # requires 16 patches, which must go through the path 4x4
            
            prediction = np.reshape(prediction, (dimen, dimen, length, length))
            
            reconstructed = unpatchify(prediction, (dimen*length, dimen*length))
            
            # print(reconstructed.shape)
                # To store, save_image requires 4 dims        
            plt.imsave(f"{labels}/Pred_{p}.png", reconstructed, cmap='gray')
            p+=1
        
    p = 0 
    
    for i in range(0, fixed_batch, num_patches):
            print(f"True label ({p})")
            
            true = true_pred[p*num_patches:p*num_patches+num_patches] 
            
            prediction = true.squeeze()
            #print(prediction.shape)
            prediction = np.reshape(prediction, (dimen, dimen, length, length))
            
            reconstructed = unpatchify(prediction, (dimen*length, dimen*length))
            
           # print(reconstructed.shape)
                # To store, save_image requires 4 dims        
            plt.imsave(f"{true_path}/True_{p}.png", reconstructed, cmap='gray')
            p+=1
    return print("Ended segmentations..")


def predict(data_loader, model, TAM):
    
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
            
            # Cropper 
            
            # Cropper
            cropper  = transforms.RandomCrop(224, 224)
            batch = cropper(batch)
            y = cropper(y)
            
            '''
            # Resize 100, 100
            transf = transforms.Resize(size = (TAM, TAM), interpolation = transforms.InterpolationMode.BILINEAR)
            
            batch = transf(batch)
            y = transf(y)
            '''
            
            batch = batch.to(device)
            y = y.to(device) # no squeeze for storing as same format
            # Move batch to the GPU

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
            
            #if batch_idx == 5: # 1, 3, 5
            #    break
            
    # stack predictions into a (num_samples, img size) array
    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)
    

    '''
    
    # PATCHIFY
    prediction = y_pred.squeeze()
    prediction = np.reshape(prediction, (4,4,100,100))
    reconstructed = unpatchify(prediction, (400,400))
        # To store, save_image requires 4 dims
    
    plt.imsave(f"prueba_patchify.png",reconstructed, cmap='gray')

    # PATCHIFY
    true = y_true.squeeze()
    true = np.reshape(true, (4,4,100,100))
    reconstructed_true = unpatchify(true, (400,400))
        # To store, save_image requires 4 dims
    
    plt.imsave(f"true_patchify.png", reconstructed_true, cmap='gray')
    '''

    model.train()
    return y_pred, y_true
