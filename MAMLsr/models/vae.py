import torch 
import torch.nn as  nn 
from .encoder import Encoder 
from .decoder import Decoder

class VAE(nn.Module):
    def __init__(self , image_channels = 3 , latent_dim = 128 ):
        super().__init__()
        self.encoder = Encoder(image_channels=image_channels ,latent_dim = latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim,image_channels=image_channels)
    
    def reparameterize(self,mu , log_var):
        std = torch.exp(0.5*log_var)
        epsilon = torch.randn_like(std)
        z = mu + epsilon*std 
        return z 
    def forward(self , x):
        mu , log_var = self.encoder(x)
        z = self.reparameterize(mu ,log_var)
        reconstruction = self.decoder(z)
        return reconstruction , mu, log_var 



