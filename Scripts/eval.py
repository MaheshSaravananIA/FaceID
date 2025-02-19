import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

import torch
from train import evaluate
import torchvision.transforms as transforms

from config import Config
from utils import load_model


import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images")


config = Config()

def run_evaluation(model, test_loader, criterion, device):
    model = load_model(model)
    test_loss = evaluate(model, test_loader, criterion, device,plot=True,n = 5, print_flag = True)
    print(f"Test Loss: {test_loss:.4f}")

def self_eval(model):
    model = load_model(model)
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),  
    ])
    
    image = Image.open(config.image_path).convert("RGB")  
    image_tensor = transform(image).unsqueeze(0).to(config.device)
    recon = model(image_tensor)[0].squeeze().detach().cpu().numpy().transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.array(image));axes[0].set_title("Before (Original)");axes[0].axis("off")
    axes[1].imshow(recon);axes[1].set_title("After (Reconstructed)");axes[1].axis("off")
    
    plt.savefig((os.path.join(config.project_dir,"TestFolder", f"After_Reconstruction.png")), bbox_inches="tight", pad_inches=0, dpi=300)

    



