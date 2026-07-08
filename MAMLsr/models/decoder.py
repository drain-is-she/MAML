import torch 
import torch.nn as nn 
from .layers import DeconvBlock ,UnFlatten

class Decoder(nn.Module):
    def __init__(self , latent_dim =128,image_channels=3):
        super().__init__()
        self.feature_dim = 256*8*8 
        self.fc = nn.Linear(latent_dim ,self.feature_dim)
        self.unflatten = UnFlatten(channels = 256 , height = 8 ,width = 8)

        self.decoder = nn.Sequential(DeconvBlock(256,128),
                                     DeconvBlock(128 , 64), 
                                     DeconvBlock(64 , 32), 
                                    nn.ConvTranspose2d(
                                        in_channels= 32 , 
                                        out_channels= image_channels,
                                        kernel_size=4 , 
                                        stride = 2, 
                                        padding = 1
                                    ),
                                    nn.Sigmoid()

        )

    def forward(self , z):
        x = self.fc(z)
        x = self.unflatten(x)
        x = self.decoder(x)

        return x 
