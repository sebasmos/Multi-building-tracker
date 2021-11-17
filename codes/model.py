import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

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
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], 
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

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape
    print(preds.shape)
    print(x.shape)

if __name__ == "__main__":
    test()

