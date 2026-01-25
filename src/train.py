import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import tifffile
import numpy as np
import os
import glob
import random
import argparse
from tqdm import tqdm  # <--- Progress Bar
from model import UNet

class BeadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load
        image = tifffile.imread(self.image_paths[idx]).astype(np.float32)
        mask = tifffile.imread(self.mask_paths[idx]).astype(np.float32)
        
        # Normalize Image
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)
        
        # Handle Mask (Auto-detect 0-1 vs 0-255)
        if np.max(mask) > 1.0:
            mask = mask / 255.0

        # To Tensor (C, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        # --- DATA AUGMENTATION ---
        # Apply same transforms to BOTH image and mask
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # Random 90-degree Rotation
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        return image, mask

def train(epochs, batch_size, augment):
    # Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): DEVICE = torch.device("mps")
    
    DATA_DIR = "data/images"
    MASK_DIR = "data/masks"
    SAVE_PATH = "models/bead_unet.pth"
    os.makedirs("models", exist_ok=True)

    # Initialize Dataset with Augmentation Flag
    dataset = BeadDataset(DATA_DIR, MASK_DIR, augment=augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Loss: weighted for class imbalance
    pos_weight = torch.tensor([10.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"🚀 Training on {DEVICE} | Augmentation: {augment} | Epochs: {epochs}")
    print(f"Found {len(dataset)} images.")

    # Loop with Progress Bar
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # tqdm creates the visual progress bar
        with tqdm(loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
            
            for images, masks in tepoch:
                images, masks = images.to(DEVICE), masks.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # Update progress bar with current loss
                tepoch.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Training complete. Saved to {SAVE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()
    
    train(args.epochs, args.batch_size, args.augment)