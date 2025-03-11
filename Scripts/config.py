import os
import torch

class Config:

    project_dir = r"C:\\Users\\DELL\\Documents\\FaceRec"
    
    data_dir = None#os.path.join(project_dir,"Dataset")

    image_size = 512    
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.0002
    latent_dim = 512
    num_workers = 0
    checkpoint_path =os.path.join(project_dir,"checkpoints")
    SaveInterval = 2
    image_path = os.path.join(project_dir,"TestFolder/test1.jpg")

    Load_model = False
    Encoder_path = os.path.join(project_dir,"checkpoints","encoder.pth")
    Decoder_path = os.path.join(project_dir,"checkpoints","decoder.pth")
   
    split_ratios = (0.90, 0.10, 0.0)
    device ="cuda" if torch.cuda.is_available() else "cpu"

    denorm = lambda x: (x + 1) / 2


