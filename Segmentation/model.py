import torch as F
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.nn.modules.padding import ReplicationPad2d
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(#  in, out, kernel, stride
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # bias is false bc it is deleted since using batchnorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024], 
    ): # features contains the values of the resilting convs in the encoder of unet arch
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()# list storing the conv layers, bc we want to use model.eval and model lists makes things easier
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # changing level in between 

        # Down part of UNET
        for feature in features: # for all possibnle feature sizes
            self.downs.append(DoubleConv(in_channels, feature))# add layer
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):# start from 512
            self.ups.append(
                nn.ConvTranspose2d(# use feature*2 bc on UNET arch, we have 1024 = 2*512
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

	# boottleneck is the down part during concatenation.
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

	# start with highest resolution and add poll after each conv of downard part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

	#bottleneck: take 512 neuros, convert to 2*512, we want to go backwards now..  for doing that we need to reverse the list of previous convolutions (skip_connections), so for that we use the command skip_connections[::-1]
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

	# We will use use a step = 2, because for each level, we want to add the double convs.. so the 2, adds 2 intermediate steps per layer
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # for each level..
            skip_connection = skip_connections[idx//2] # add skip connection (integer division by 2), which we want to do linear - 1 step as we are going 2 steps on for-loop

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
	# connect skip connection on the same level
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)# run over second conv

        return self.final_conv(x)

class UNET_manual(nn.Module):
    """EF segmentation network."""

    def __init__(self, input_nbr, label_nbr):
        super(UNET_manual, self).__init__()

        self.input_nbr = input_nbr

        self.conv11 = nn.Conv2d(input_nbr, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=0.2)


        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(16, label_nbr, kernel_size=3, padding=1)

        self.sm = nn.Sigmoid()

    def forward(self, x):
        
        #x = torch.cat((x1, x2), 1)

        """Forward method."""
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x))))
        x12 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43, kernel_size=2, stride=2)


        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43.size(3) - x4d.size(3), 0, x43.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), x43), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33.size(3) - x3d.size(3), 0, x33.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), x33), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22.size(3) - x2d.size(3), 0, x22.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), x22), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12.size(3) - x1d.size(3), 0, x12.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), x12), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return self.sm(x11d)
###############################################################################3
import segmentation_models_pytorch as smp

def make_model(model_name='unet_resnet34',
               weights='imagenet',
               n_classes=2,
               input_channels=4):

    if model_name.split('_')[0] == 'unet':
        
        model = smp.Unet('_'.join(model_name.split('_')[1:]), 
                         classes=n_classes,
                         activation=None,
                         encoder_weights=weights,
                         in_channels=input_channels)

    elif model_name.split('_')[0] == 'fpn':
        model = smp.FPN('_'.join(model_name.split('_')[1:]),
                        classes=n_classes,
                        activation=None,
                        encoder_weights=weights,
                        in_channels=input_channels)

    elif model_name.split('_')[0] == 'linknet':
        model = smp.Linknet('_'.join(model_name.split('_')[1:]),
                            classes=n_classes,
                            activation=None,
                            encoder_weights=weights,
                            in_channels=input_channels)
    else:
        raise ValueError('Model not implemented')
    
    return model

#####################################################################################

def double_block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace = True),
        
        )
    return conv
    

class unet_bas(nn.Module):
    def __init__(self):
        super(unet_bas, self).__init__()
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.contractor_1 = double_block(3, 64)
        self.contractor_2 = double_block(64, 128)
        self.contractor_3 = double_block(128, 256)
        self.contractor_4 = double_block(256, 512)
        self.contractor_5 = double_block(512, 1024)
        
        # Decoder
        self.up_trans_1 = nn.ConvTranspose2d(in_channels = 1024, out_channels=512, kernel_size = 2, stride=2)
        self.up_conv_1 = double_block(1024, 512)
        
                
        self.up_trans_2 = nn.ConvTranspose2d(in_channels = 512, out_channels=256, kernel_size = 2, stride=2)
        self.up_conv_2 = double_block(512, 256)
        
                
        self.up_trans_3 = nn.ConvTranspose2d(in_channels = 256, out_channels=128, kernel_size = 2, stride=2)
        self.up_conv_3 = double_block(256, 128)
        
                
        self.up_trans_4 = nn.ConvTranspose2d(in_channels = 128, out_channels=64, kernel_size = 2, stride=2)
        self.up_conv_4 = double_block(128, 64)
        
        self.out = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1)
        
    def forward(self, image):
        # Encoder
        # image (1, 3, 256, 256)
        x1 = self.contractor_1(image) # (1, 64, 252, 252)
        x2 = self.max_pool_2x2(x1) #(1, 128, 126, 126)
        x3 = self.contractor_2(x2)# (1, 128, 122, 122)
        x4 = self.max_pool_2x2(x3)# (1, 128, 61, 61)
        x5 = self.contractor_3(x4)# (1, 256, 57, 57)
        x6 = self.max_pool_2x2(x5)# (1, 256, 28, 28) -> 28.5
        x7 = self.contractor_4(x6)# 
        x8 = self.max_pool_2x2(x7)
        x9 = self.contractor_5(x8)
        
        #Decoder

        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        if y.shape != x.shape:
        	y = TF.resize(y, x.shape[2:])
#        print(f"{y.size()} - {x.size()}")
        g = torch.cat([x,y],1)
 #       print("conc: ", g.size())
        x = self.up_conv_1(torch.cat([x,y],1))
        
	# connect skip connection on the same level
        x = self.up_trans_2(x)
        y = crop_img(x5, x)#torch.Size([1, 256, 25, 25])
        if y.shape != x.shape:
        	y = TF.resize(y, x.shape[2:])
         
        x = self.up_conv_2(torch.cat([x,y],1))
        
        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        if y.shape != x.shape:
        	y = TF.resize(y, x.shape[2:])
         
        x = self.up_conv_3(torch.cat([x, y],1))
        
        
        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        if y.shape != x.shape:
        	y = TF.resize(y, x.shape[2:])
         
        x = self.up_conv_4(torch.cat([x,y],1))
        
        
        # machetazo
        
        x = TF.resize(x, (256,256))
        
        x = self.out(x)
        
        return x
def crop_img(tensor, target_tensor):
	'''
	From original paper: take image of size 64 from 4th stage and crop it to fit with
	upconvoluted image of size 56. 
	e.g.
	target_size = 56
	tensor_size = 64
	delta = 8 // 2 # 4 
	
	#output
	tensor[:,:,delta:tensor_size-delta, delta:tensor_size-delta]
	tensor[:,:,4:64-4, 4:64-4]
	tensor[:,:,4:60, 4:60]
	
	# as a result, cropping is centered
	
	'''
	target_size = target_tensor.size()[2]
	
	tensor_size = tensor.size()[2]
	
	delta = tensor_size - target_size

	delta = delta//2
	

	x = tensor[:,:,delta:tensor_size-delta, delta:tensor_size-delta]
	
	# fix when image size is not even, we need to resize.. or zero-pad
	#if x.size()[2] != target_tensor.size()[2]:
#		x = TF.resize(x, size=(target_tensor.size()[2],target_tensor.size()[2]))
	
	return  x
        




def test():
    test_mask = torch.randn((1, 3, 256, 256)) # RGB image
    ## MODEL 1
    #model = UNET(in_channels=1, out_channels=1)v
    #model = UNET_manual(3, 1)
    model = unet_bas()
    preds = model(test_mask)
    print("final image size: ", preds.size())
    #assert preds.shape == (1,1,256,256), "sizes are different "
    
    '''
    model = make_model(
    model_name="",
    weights=None,
    n_classes=1,
    input_channels=3)
    '''
    
           
           
        
if __name__ == "__main__":
    test()

