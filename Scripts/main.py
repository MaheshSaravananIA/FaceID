import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from dataset import get_dataloaders
from models import AutoEncoder
from train import train, evaluate
from eval import run_evaluation,self_eval
from utils import weights_init_normal
import kagglehub

import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images")



def main():
    parser = argparse.ArgumentParser(description="AutoEncoder Training and Evaluation")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Evaluate on test set")
    parser.add_argument("--self", action="store_true", help="Run self-evaluation")
    args = parser.parse_args()

    config = Config()
    device = config.device
    print("Using device:", device)
    if config.data_dir is None:config.data_dir  =kagglehub.dataset_download("ashwingupta3012/human-faces")
    

    # Load Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        config.data_dir,
        config.image_size,
        config.batch_size,
        config.num_workers
    )

    model = AutoEncoder(config.latent_dim, config.image_size).to(device)
    model.apply(weights_init_normal)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    
    if args.train: train(model, train_loader, val_loader, criterion, optimizer, device, config.num_epochs, config.checkpoint_path)

    if args.test:  run_evaluation(model, test_loader, criterion, device)

    if args.self: self_eval(model)

if __name__ == "__main__":
    main()
