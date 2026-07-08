import torch 
import torch.nn as nn 
from .layers import ConvBlock , Flatten 

class Encoder(nn.Module):
    def __init__(self,image_channels = 3 , latent_dim = 128 ):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(image_channels,32),
            ConvBlock(32,64),
            ConvBlock(64,128),
            ConvBlock(128,256),
        )
        self.flatten = Flatten()
        self.feature_dim = 256*8*8
        self.fc_mu = nn.Linear(self.feature_dim , latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim,latent_dim )

    def forward(self,x):
        x = self.features(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu , log_var
