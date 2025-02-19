import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, image_size, batch_size, num_workers, split_ratios=(0.85, 0.10, 0.05)):
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    total = len(full_dataset)
    
    train_len = int(total * split_ratios[0])
    val_len   = int(total * split_ratios[1])
    test_len  = total - train_len - val_len 
    
    #print(f"Total images: {total}")
    #print(f"Train images: {train_len}, Validation images: {val_len}, Test images: {test_len}")
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
