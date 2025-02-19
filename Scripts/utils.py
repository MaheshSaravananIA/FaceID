import torch.nn as nn
import torch
from config import Config

config = Config()
def weights_init_normal(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

def load_model(model):  
    
    model.encoder.load_state_dict(torch.load(config.Encoder_path))
    model.decoder.load_state_dict(torch.load(config.Decoder_path))
    
    print(f"Model weights loaded...")
    return model
    