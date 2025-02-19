import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim, image_size):
        
        super(AutoEncoder, self).__init__()
        self.image_size = image_size
        self.flatten_dim = 512 * ((image_size // 64) ** 2)
        
        
        self.encoder = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.Flatten(),                                          
            nn.Linear(self.flatten_dim, latent_dim)                
        )
        
       
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_dim),            
            nn.Unflatten(1, (512, image_size // 64, image_size // 64)),  
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),   
            nn.Tanh() 
        )
        
    def forward(self, x):
       
        latent = self.encoder(x)  #
        reconstructed = self.decoder(latent)
        return reconstructed, latent
