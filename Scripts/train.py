import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.optim as optim
from utils import load_model
from config import Config
config = Config()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch,num_epoch):
    model.train()
    epoch_loss_bin = []
    running_loss = 0.0
    for c,(images, _ )in enumerate(dataloader):
        images = images.to(device)
        optimizer.zero_grad()
        recon, _ = model(images)
        loss = criterion(recon, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        #Progress bar
        progress_length = 50  
        progress = int((c / len(dataloader)) * progress_length)
        bar = "=" * progress+">" + " " * (progress_length - progress)
        print(f"\rEpoch:{epoch+1}/{num_epoch} [{c+1}/{len(dataloader)}] [{bar}] Loss: {loss.item():.6f}", end="")
        epoch_loss_bin.append(loss.item()) 

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss,epoch_loss_bin

def evaluate(model, dataloader, criterion, device,plot=False,n=2,print_flag = False):
    model.eval()
    running_loss = 0.0
    
    process_tensor = lambda x: x.detach().cpu().numpy().squeeze().transpose(0,2,3,1)
    plot_scale = lambda x: (x + 1) / 2


    with torch.no_grad():
        for c, (images,_) in enumerate(dataloader):

            images = images.to(device)
            recon, _ = model(images)
            loss = criterion(recon, images)
            running_loss += loss.item() * images.size(0)

            if print_flag: 
                progress_length = 50  
                progress = int((c / len(dataloader)) * progress_length)
                bar = "=" * progress+">" + " " * (progress_length - progress)
                print(f"\r [{c+1}/{len(dataloader)}] [{bar}] Loss: {loss.item():.6f}", end="")

    epoch_loss = running_loss / len(dataloader.dataset)
    
    if plot:
        batch = torch.empty(0) 
        required_met = False
        while not required_met:
            batch = torch.concat((batch, next(iter(dataloader))[0])) 
            if len(batch)>n: required_met = True

        batch = batch[torch.randint(0, batch.shape[0], (n,))]
        recon,_ = model(batch.to(device))
        
        
        
        before= process_tensor(batch)
        after = process_tensor(recon)
        
        before= plot_scale(before)
        after = plot_scale(after)
        

        _, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 25))  
        for i in range(n):
            axes[i, 0].imshow(before[i]);axes[i, 0].axis("off"),axes[i, 0]
            axes[i, 1].imshow(after[i]);axes[i, 1].axis("off");axes[i, 1]
        plt.tight_layout()
        save_path = os.path.join(config.project_dir,"log", "test_result.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    return epoch_loss


def plot_loss(loss_bin, epoch, log_folder="log"):
    
    os.makedirs(log_folder, exist_ok=True) 

    plt.plot(loss_bin, color='red')
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.title(f"Epoch {epoch}")
    save_path = os.path.join(config.project_dir,log_folder, f"loss_epoch.png")
    plt.savefig(save_path)
    plt.close()



def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path=None,plot_loss_flag = True):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    loss_bin = []
    os.makedirs( save_path, exist_ok=True) 

    print(f"training on {device}")
    if config.Load_model: model = load_model(model)
    
    for epoch in range(num_epochs):
        train_loss,e_bin = train_epoch(model, train_loader, criterion, optimizer, device,epoch,num_epochs)
        
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        loss_bin= loss_bin+e_bin
        
        if config.SaveInterval%(epoch+1):
            if save_path:
                torch.save(model.encoder.state_dict(), os.path.join(save_path, f"encoder.pth"))
                torch.save(model.decoder.state_dict(), os.path.join(save_path, f"decoder.pth"))
        
        if plot_loss_flag:plot_loss(loss_bin, epoch)

