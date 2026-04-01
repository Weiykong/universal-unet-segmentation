"""Generate a side-by-side comparison PNG: Input | Ground Truth | Prediction | Overlay."""
import torch
import torch.nn.functional as F
import tifffile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import UNet
import os, glob

# --- Config ---
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"
MODEL_PATH = "models/best_model.pth"
OUTPUT_PATH = "comparison.png"
N_SAMPLES = 6  # how many images to show
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

SUPPORTED_EXTENSIONS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return tifffile.imread(path).astype(np.float32)
    else:
        from PIL import Image
        img = np.array(Image.open(path)).astype(np.float32)
        if img.ndim == 3 and img.shape[2] == 3:
            img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        return img


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        depth = checkpoint.get('depth', 4)
        base_features = checkpoint.get('base_features', 64)
        model = UNet(depth=depth, base_features=base_features).to(DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = UNet().to(DEVICE)
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict(model, image_np):
    """Run inference on a single image, handling padding."""
    img = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-6)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    h, w = tensor.shape[2], tensor.shape[3]
    align = 2 ** model.depth if hasattr(model, 'depth') else 16
    pad_h = (align - (h % align)) % align
    pad_w = (align - (w % align)) % align
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad():
        out = torch.sigmoid(model(tensor))

    if pad_h > 0 or pad_w > 0:
        out = out[:, :, :h, :w]

    return out.squeeze().cpu().numpy()


def make_overlay(image_np, mask_np, pred_np, threshold=0.5):
    """Create RGB overlay: green = TP, red = FP, blue = FN."""
    img_norm = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-6)
    rgb = np.stack([img_norm] * 3, axis=-1) * 0.6

    gt = mask_np > 0.5
    pr = pred_np > threshold

    tp = gt & pr
    fp = (~gt) & pr
    fn = gt & (~pr)

    rgb[tp] = [0, 1, 0]       # green = correct detection
    rgb[fp] = [1, 0, 0]       # red = false positive
    rgb[fn] = [0, 0.3, 1]     # blue = missed

    return np.clip(rgb, 0, 1)


def main():
    model = load_model(MODEL_PATH)

    image_paths = sorted(
        p for ext in SUPPORTED_EXTENSIONS
        for p in glob.glob(os.path.join(IMAGE_DIR, ext))
    )
    mask_paths = sorted(
        p for ext in SUPPORTED_EXTENSIONS
        for p in glob.glob(os.path.join(MASK_DIR, ext))
    )

    # Pick evenly spaced samples
    n = len(image_paths)
    indices = np.linspace(0, n - 1, min(N_SAMPLES, n), dtype=int)

    fig, axes = plt.subplots(len(indices), 4, figsize=(20, 5 * len(indices)))
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Ground Truth", "Prediction", "Overlay (G=TP, R=FP, B=FN)"]

    for row, idx in enumerate(indices):
        image_np = load_image(image_paths[idx])
        mask_np = load_image(mask_paths[idx])
        if mask_np.max() > 1.0:
            mask_np = mask_np / 255.0

        pred_np = predict(model, image_np)
        pred_bin = (pred_np > THRESHOLD).astype(np.float32)
        overlay = make_overlay(image_np, mask_np, pred_np, THRESHOLD)

        # Compute per-image metrics
        gt_bin = (mask_np > 0.5).astype(np.float32)
        intersection = (pred_bin * gt_bin).sum()
        dice = (2 * intersection + 1e-6) / (pred_bin.sum() + gt_bin.sum() + 1e-6)
        iou = (intersection + 1e-6) / (pred_bin.sum() + gt_bin.sum() - intersection + 1e-6)

        fname = os.path.basename(image_paths[idx])

        # Normalize input for display
        img_disp = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-6)

        axes[row, 0].imshow(img_disp, cmap='gray')
        axes[row, 0].set_ylabel(fname, fontsize=10, rotation=0, labelpad=100, va='center')

        axes[row, 1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)

        axes[row, 2].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_title(f"Dice={dice:.3f}  IoU={iou:.3f}", fontsize=10)

        axes[row, 3].imshow(overlay)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    # Column titles on top row
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold', pad=10)
    # Re-set the prediction title for row 0 (combine both)
    pred_np_0 = predict(model, load_image(image_paths[indices[0]]))
    mask_np_0 = load_image(mask_paths[indices[0]])
    if mask_np_0.max() > 1: mask_np_0 /= 255.0
    gt0 = (mask_np_0 > 0.5).astype(np.float32)
    pr0 = (pred_np_0 > THRESHOLD).astype(np.float32)
    inter0 = (pr0 * gt0).sum()
    d0 = (2 * inter0 + 1e-6) / (pr0.sum() + gt0.sum() + 1e-6)
    i0 = (inter0 + 1e-6) / (pr0.sum() + gt0.sum() - inter0 + 1e-6)
    axes[0, 2].set_title(f"Prediction\nDice={d0:.3f}  IoU={i0:.3f}", fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
