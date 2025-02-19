import os
import torch

class Config:

    project_dir = r"C:\\Users\\DELL\\Desktop\\FaceRec"
    
    data_dir = None#os.path.join(project_dir,"Dataset")

    image_size = 512    
    batch_size = 2
    num_epochs = 50
    learning_rate = 0.0002
    latent_dim = 512
    num_workers = 4
    checkpoint_path =os.path.join(project_dir,"checkpoints")
    SaveInterval = 2
    image_path = os.path.join(project_dir,"TestFolder/test1.jpg")

    Load_model = False
    Encoder_path = os.path.join(project_dir,"checkpoints","encoder.pth")
    Decoder_path = os.path.join(project_dir,"checkpoints","decoder.pth")
   
    split_ratios = (0.55, 0.45, 0.0)
    device ="cuda" if torch.cuda.is_available() else "cpu"

    denorm = lambda x: (x + 1) / 2
    
