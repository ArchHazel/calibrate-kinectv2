import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
from unet.model import UNet  # Make sure you have a UNet model defined in unet.py
from unet.dataset import UNetDataset  # Replace with your dataset class
from apply_RT import visualize_depth
import os
from tqdm import tqdm

# Configuration
checkpoint_path = './unet_epoch_10.pth'  # Path to your checkpoint
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader
dataset = UNetDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Load model and checkpoint
model = UNet()  # Adjust arguments if needed
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

os.makedirs('data/unet_output', exist_ok=True)

with torch.no_grad():
    for i, (inputs, targets, valid_mask) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs + inputs  # Adjust if your model outputs need to be added to inputs
        
        outputs = outputs.cpu().numpy().squeeze()  # Convert to numpy and remove batch dimension
        # diff = outputs - targets.numpy()
        # visualize the  output
        visualize_depth(outputs,'data/unet_output', i, factor=0.03)



# Optionally, concatenate all results
all_results = torch.cat(results, dim=0)
print("Inference complete. Results shape:", all_results.shape)