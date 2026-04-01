import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except Exception:
    HAS_TENSORBOARD = False
import torchvision.transforms.functional as TF
import tifffile
import numpy as np
import os
import glob
import random
import argparse
from datetime import datetime
from tqdm import tqdm
from model import UNet

SUPPORTED_EXTENSIONS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def load_image(path):
    """Load an image from TIFF, PNG, or JPEG."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return tifffile.imread(path).astype(np.float32)
    else:
        from PIL import Image
        img = np.array(Image.open(path)).astype(np.float32)
        if img.ndim == 3 and img.shape[2] == 3:
            img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        return img


class BeadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False, crop_size=512):
        self.image_paths = sorted(
            p for ext in SUPPORTED_EXTENSIONS
            for p in glob.glob(os.path.join(image_dir, ext))
        )
        self.mask_paths = sorted(
            p for ext in SUPPORTED_EXTENSIONS
            for p in glob.glob(os.path.join(mask_dir, ext))
        )
        self.augment = augment
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        mask = load_image(self.mask_paths[idx])

        # Normalize image to [0, 1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)

        # Handle mask (auto-detect 0-1 vs 0-255)
        if np.max(mask) > 1.0:
            mask = mask / 255.0

        # To tensor (C, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Random crop (applied to both image and mask)
        if self.crop_size and min(image.shape[-2:]) >= self.crop_size:
            i, j, h, w = T.RandomCrop.get_params(
                image, output_size=(self.crop_size, self.crop_size)
            )
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        # Data augmentation
        if self.augment:
            # Geometric: flip + rotate (applied to both)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # Elastic deformation (applied to both via channel concat)
            if random.random() > 0.5:
                try:
                    from torchvision.transforms.v2 import ElasticTransform
                    elastic = ElasticTransform(alpha=50.0, sigma=5.0)
                    combined = torch.cat([image, mask], dim=0)  # (2, H, W)
                    combined = elastic(combined)
                    image = combined[0:1]
                    mask = (combined[1:2] > 0.5).float()
                except ImportError:
                    pass

            # Intensity augmentation (image only)
            if random.random() > 0.5:
                # Brightness jitter
                image = image * (0.8 + 0.4 * random.random())
            if random.random() > 0.3:
                # Gaussian noise
                image = image + torch.randn_like(image) * 0.03
            if random.random() > 0.7:
                # Gaussian blur
                image = TF.gaussian_blur(image, kernel_size=3)

            image = image.clamp(0, 1)

        return image, mask


def dice_loss(pred, target):
    """Differentiable soft Dice loss."""
    pred_soft = torch.sigmoid(pred)
    intersection = (pred_soft * target).sum()
    return 1.0 - (2.0 * intersection + 1e-6) / (pred_soft.sum() + target.sum() + 1e-6)


def dice_score(pred, target, threshold=0.5):
    """Compute Dice coefficient."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    return (2.0 * intersection + 1e-6) / (pred_bin.sum() + target.sum() + 1e-6)


def iou_score(pred, target, threshold=0.5):
    """Compute Intersection over Union (Jaccard index)."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def train(epochs, batch_size, lr, augment, val_split, depth, base_features, crop_size,
          resume=None):
    # Device setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")

    DATA_DIR = "data/images"
    MASK_DIR = "data/masks"
    SAVE_PATH = "models/bead_unet.pth"
    BEST_PATH = "models/best_model.pth"
    os.makedirs("models", exist_ok=True)

    # TensorBoard (optional)
    writer = None
    if HAS_TENSORBOARD:
        log_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")
    else:
        print("TensorBoard not available, skipping logging.")

    # Build dataset
    full_dataset = BeadDataset(DATA_DIR, MASK_DIR, augment=augment, crop_size=crop_size)
    n = len(full_dataset)

    # Train / validation split
    indices = list(range(n))
    random.shuffle(indices)
    n_val = max(1, int(n * val_split)) if val_split > 0 else 0
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_loader = DataLoader(Subset(full_dataset, train_indices),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_indices),
                            batch_size=batch_size, shuffle=False) if n_val > 0 else None

    # Model, optimizer, scheduler, loss
    model = UNet(depth=depth, base_features=base_features).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    pos_weight = torch.tensor([10.0]).to(DEVICE)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint
    if resume and os.path.exists(resume):
        checkpoint = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from {resume} (epoch {start_epoch}, best_val_loss={best_val_loss:.4f})")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"Training on {DEVICE} | Augmentation: {augment} | Epochs: {epochs}")
    print(f"Model: depth={depth}, base_features={base_features} ({n_params:.2f}M params)")
    print(f"Dataset: {n} images ({len(train_indices)} train, {n_val} val)")
    print(f"LR: {lr} | Batch size: {batch_size} | Crop: {crop_size}")
    print(f"Loss: BCE + Dice | Scheduler: CosineAnnealing (eta_min=1e-6)")

    for epoch in range(start_epoch, epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        with tqdm(train_loader, unit="batch", leave=False) as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{epochs} [train]")
            for images, masks in tepoch:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = bce_criterion(outputs, masks) + dice_loss(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        scheduler.step()
        avg_train_loss = train_loss / max(len(train_loader), 1)
        if writer:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # --- Validation ---
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_dice = 0
            val_iou = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(DEVICE), masks.to(DEVICE)
                    outputs = model(images)
                    val_loss += (bce_criterion(outputs, masks) + dice_loss(outputs, masks)).item()
                    val_dice += dice_score(outputs, masks).item()
                    val_iou += iou_score(outputs, masks).item()

            n_batches = len(val_loader)
            avg_val_loss = val_loss / n_batches
            avg_dice = val_dice / n_batches
            avg_iou = val_iou / n_batches

            current_lr = optimizer.param_groups[0]['lr']

            if writer:
                writer.add_scalar("Loss/val", avg_val_loss, epoch)
                writer.add_scalar("Metrics/dice", avg_dice, epoch)
                writer.add_scalar("Metrics/iou", avg_iou, epoch)
                writer.add_scalar("LR", current_lr, epoch)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f} | "
                  f"LR: {current_lr:.1e}")

            # Best-model checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'depth': depth,
                    'base_features': base_features,
                }, BEST_PATH)
                print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")

    if writer:
        writer.close()

    # Always save final model (full checkpoint for resuming)
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'depth': depth,
        'base_features': base_features,
    }, SAVE_PATH)
    print(f"Training complete. Final model saved to {SAVE_PATH}")
    if val_loader is not None:
        print(f"Best model (val_loss={best_val_loss:.4f}) saved to {BEST_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--base_features', type=int, default=64)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g. models/best_model.pth)')
    args = parser.parse_args()

    train(args.epochs, args.batch_size, args.lr, args.augment, args.val_split,
          args.depth, args.base_features, args.crop_size, args.resume)
