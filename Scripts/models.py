import torch
import torch.nn as nn

class AutoEncoder1(nn.Module):
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

class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(BasicResBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class UpBasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super(UpBasicResBlock, self).__init__()
        self.upsample = upsample
        if upsample:
            self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=2, stride=2, bias=False)
        else:
            self.up = nn.Identity()
            
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # For the skip connection: if upsampling or channel change is needed
        if upsample or in_channels != out_channels:
            if upsample:
                self.skip = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        out = self.up(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class AutoEncoder(nn.Module):
    def __init__(self, image_size, latent_dim):
        self.image_size = image_size
        super(AutoEncoder, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
       
        self.encoder_block1 = BasicResBlock(64, 128, downsample=True)
        self.encoder_block2 = BasicResBlock(128, 256, downsample=True)
        self.encoder_block3 = BasicResBlock(256, 512, downsample=True)
        self.encoder_block4 = BasicResBlock(512, 512, downsample=True)
        self.encoder_block5 = BasicResBlock(512, 512, downsample=True)
        
        
        final_spatial = image_size // 64  # since 2^6 = 64
        self.flatten_dim = 512 * final_spatial * final_spatial
        
        self.fc_enc = nn.Linear(self.flatten_dim, latent_dim)
        
    
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder_block1 = UpBasicResBlock(512, 512, upsample=True)
        self.decoder_block2 = UpBasicResBlock(512, 512, upsample=True)
        self.decoder_block3 = UpBasicResBlock(512, 256, upsample=True)
        self.decoder_block4 = UpBasicResBlock(256, 128, upsample=True)
        self.decoder_block5 = UpBasicResBlock(128, 64, upsample=True)
        self.final_up = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, bias=False)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        
        x = self.initial(x)
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        x = self.encoder_block3(x)
        x = self.encoder_block4(x)
        x = self.encoder_block5(x)
        x = x.view(x.size(0), -1)
        latent = self.fc_enc(x)
        
        # Decoder
        x = self.fc_dec(latent)
       
        #x = x.view(x.size(0), 512, int(x.size(1) // 512**0.5), int(x.size(1) // 512**0.5))
        x = x.view(x.size(0), 512, self.image_size // 64, self.image_size // 64)
        x = self.decoder_block1(x)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)
        x = self.decoder_block4(x)
        x = self.decoder_block5(x)
        x = self.final_up(x)
        out = self.tanh(x)
        return out,latent

