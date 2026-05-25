"""Generate a zoomed portfolio PNG showcasing bead segmentation results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import os, sys

sys.path.insert(0, 'src')
from model import UNet

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model = UNet(depth=checkpoint.get('depth', 4),
                     base_features=checkpoint.get('base_features', 64)).to(DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = UNet().to(DEVICE)
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict(model, img_np):
    img = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    h, w = t.shape[2], t.shape[3]
    align = 2 ** model.depth
    ph = (align - h % align) % align
    pw = (align - w % align) % align
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    with torch.no_grad():
        out = torch.sigmoid(model(t))
    if ph or pw:
        out = out[:, :, :h, :w]
    return out.squeeze().cpu().numpy()


def enhance(img, pct=1):
    lo, hi = np.percentile(img, pct), np.percentile(img, 100 - pct)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)


def get_crop(arr, cy, cx, size):
    h, w = arr.shape[:2]
    s = size // 2
    y1, y2 = max(0, cy - s), min(h, cy + s)
    x1, x2 = max(0, cx - s), min(w, cx + s)
    return arr[y1:y2, x1:x2]


# ---- Load model ----
model = load_model('models/best_model.pth')

# ---- Load one good sample with mask ----
img_raw = tifffile.imread('data/images/ch000_f001.tif').astype(np.float32)
mask_raw = tifffile.imread('data/masks/ch000_f001_mask.tif').astype(np.float32)
if mask_raw.max() > 1:
    mask_raw /= 255.0
img_n = (img_raw - img_raw.min()) / (img_raw.max() - img_raw.min() + 1e-6)
pred = predict(model, img_raw)

# ---- Inference sample ----
inf_raw = tifffile.imread('data/inference_input/ch04_f023.tif').astype(np.float32)
inf_n = (inf_raw - inf_raw.min()) / (inf_raw.max() - inf_raw.min() + 1e-6)
inf_pred = predict(model, inf_raw)

# ---- Metrics ----
gt = mask_raw > 0.5
pr = pred > 0.5
inter = (gt & pr).sum()
dice = (2 * inter + 1e-6) / (gt.sum() + pr.sum() + 1e-6)
iou = (inter + 1e-6) / (gt.sum() + pr.sum() - inter + 1e-6)

# ================================================================
BG = '#0d1117'
BG2 = '#161b22'
TEXT = '#c9d1d9'
MUTED = '#8b949e'
BORDER = '#30363d'
GREEN = '#3fb950'
PURPLE = '#a371f7'
BLUE = '#58a6ff'
CYAN = '#39d2e0'

fig = plt.figure(figsize=(18, 20), facecolor=BG)

# Title
fig.text(0.5, 0.975, 'Universal U-Net Segmentation (Zoomed)', fontsize=32, fontweight='bold',
         ha='center', va='top', color='white', fontfamily='sans-serif')
fig.text(0.5, 0.955, 'Detailed Bead Detection  |  109 training images  |  '
         f'Dice {dice:.3f}  |  IoU {iou:.3f}  |  7.7M params',
         fontsize=12, ha='center', va='top', color=MUTED, fontfamily='sans-serif')

gs = gridspec.GridSpec(4, 6, figure=fig, top=0.935, bottom=0.06, left=0.03, right=0.97,
                       hspace=0.35, wspace=0.15)

# ================================================================
# ROW 1: Zoom 1 - Input | GT overlay | Pred overlay
# ================================================================
cy1, cx1 = 600, 600
sz = 300
z1_img = enhance(get_crop(img_n, cy1, cx1, sz), 0.5)
z1_mask = get_crop(mask_raw, cy1, cx1, sz)
z1_pred = get_crop(pred, cy1, cx1, sz)

ax1 = fig.add_subplot(gs[0, 0:2])
ax1.imshow(z1_img, cmap='gray')
ax1.set_title('Zoom: Input Image', fontsize=14, color=TEXT, fontweight='bold', pad=8)
ax1.axis('off')

# GT overlay
gt1_rgb = np.stack([z1_img] * 3, axis=-1) * 0.5
gt1_rgb[z1_mask > 0.5] = [0.2, 1.0, 0.3]
ax2 = fig.add_subplot(gs[0, 2:4])
ax2.imshow(gt1_rgb)
ax2.set_title('Zoom: Ground Truth', fontsize=14, color=TEXT, fontweight='bold', pad=8)
ax2.axis('off')

# Pred overlay
pr1_rgb = np.stack([z1_img] * 3, axis=-1) * 0.5
pr1_rgb[z1_pred > 0.5] = [0.2, 0.8, 1.0]
ax3 = fig.add_subplot(gs[0, 4:6])
ax3.imshow(pr1_rgb)
ax3.set_title('Zoom: Prediction', fontsize=14, color=TEXT, fontweight='bold', pad=8)
ax3.axis('off')

# ================================================================
# ROW 2: Zoom 2 - Input | GT | Error overlay
# ================================================================
cy2, cx2 = 400, 800
z2_img = enhance(get_crop(img_n, cy2, cx2, sz), 0.5)
z2_mask = get_crop(mask_raw, cy2, cx2, sz)
z2_pred = get_crop(pred, cy2, cx2, sz)

ax4 = fig.add_subplot(gs[1, 0:2])
ax4.imshow(z2_img, cmap='gray')
ax4.set_title('Zoom: Region B', fontsize=14, color=CYAN, fontweight='bold', pad=8)
ax4.axis('off')

# Zoomed GT
z2_gt_rgb = np.stack([z2_img] * 3, axis=-1) * 0.4
z2_gt_rgb[z2_mask > 0.5] = [0.2, 1.0, 0.3]
ax5 = fig.add_subplot(gs[1, 2:4])
ax5.imshow(z2_gt_rgb)
ax5.set_title('Zoom: GT Region B', fontsize=14, color=CYAN, fontweight='bold', pad=8)
ax5.axis('off')

# Zoomed error overlay: green=TP, red=FP, blue=FN
z2_err = np.stack([z2_img] * 3, axis=-1) * 0.35
z2_gt_b = z2_mask > 0.5
z2_pr_b = z2_pred > 0.5
z2_err[z2_gt_b & z2_pr_b] = [0.1, 0.95, 0.3]      # TP green
z2_err[(~z2_gt_b) & z2_pr_b] = [1.0, 0.15, 0.15]   # FP red
z2_err[z2_gt_b & (~z2_pr_b)] = [0.2, 0.4, 1.0]     # FN blue
ax6 = fig.add_subplot(gs[1, 4:6])
ax6.imshow(np.clip(z2_err, 0, 1))
ax6.set_title('Zoom: Error Map (G=TP R=FP B=FN)', fontsize=14,
             color=CYAN, fontweight='bold', pad=8)
ax6.axis('off')

# ================================================================
# ROW 3: Inference Zoom - Input | Probability | Detection overlay
# ================================================================
cy3, cx3 = 512, 512 # Center of 1024x1024
z3_inf_n = get_crop(inf_n, cy3, cx3, sz)
z3_inf_e = enhance(z3_inf_n, 0.5)
z3_inf_p = get_crop(inf_pred, cy3, cx3, sz)

ax7 = fig.add_subplot(gs[2, 0:2])
ax7.imshow(z3_inf_e, cmap='gray')
ax7.set_title('Inference Zoom (ch04_f023)', fontsize=14, color=TEXT,
             fontweight='bold', pad=8)
ax7.axis('off')

ax8 = fig.add_subplot(gs[2, 2:4])
ax8.imshow(z3_inf_p, cmap='inferno', vmin=0, vmax=1)
ax8.set_title('Zoom: Probability Map', fontsize=14, color=TEXT, fontweight='bold', pad=8)
ax8.axis('off')

# Zoom overlay
z3_inf_rgb = np.stack([z3_inf_e] * 3, axis=-1) * 0.6
z3_inf_rgb[:, :, 1] = np.clip(z3_inf_rgb[:, :, 1] + z3_inf_p * 0.5, 0, 1)
ax9 = fig.add_subplot(gs[2, 4:6])
ax9.imshow(np.clip(z3_inf_rgb, 0, 1))
ax9.set_title('Zoom: Detection Overlay', fontsize=14, color=TEXT,
             fontweight='bold', pad=8)
ax9.axis('off')

# ================================================================
# ROW 4: Stats panels (Remains same for consistency)
# ================================================================

# Metrics panel
ax_m = fig.add_subplot(gs[3, 0:2])
ax_m.set_facecolor(BG2)
ax_m.axis('off')
ax_m.text(0.5, 0.82, f'{dice:.3f}', transform=ax_m.transAxes,
         fontsize=52, fontweight='bold', color=GREEN,
         ha='center', va='center', fontfamily='sans-serif')
ax_m.text(0.5, 0.60, 'Dice Score', transform=ax_m.transAxes,
         fontsize=13, color=MUTED, ha='center', va='center')
ax_m.text(0.5, 0.35, f'{iou:.3f}', transform=ax_m.transAxes,
         fontsize=52, fontweight='bold', color=PURPLE,
         ha='center', va='center', fontfamily='sans-serif')
ax_m.text(0.5, 0.13, 'IoU Score', transform=ax_m.transAxes,
         fontsize=13, color=MUTED, ha='center', va='center')
ax_m.set_title('Performance', fontsize=12, color=TEXT, fontweight='bold', pad=8)

# Architecture panel
ax_a = fig.add_subplot(gs[3, 2:4])
ax_a.set_facecolor(BG2)
ax_a.axis('off')
arch = (
    "  Architecture\n"
    "    U-Net  |  depth=4  |  64 base\n"
    "    7.70M parameters\n"
    "    Dropout 0.2 bottleneck\n\n"
    "  Training\n"
    "    Loss: BCE + Dice\n"
    "    LR: Cosine Annealing\n"
    "    Augment: elastic, flip,\n"
    "      rotate, noise, blur\n\n"
    "  Dataset\n"
    "    109 image-mask pairs\n"
    "    1 um fluorescent beads\n"
    "    200 epochs, batch=4"
)
ax_a.text(0.05, 0.92, arch, transform=ax_a.transAxes,
         fontsize=10, color=TEXT, fontfamily='monospace',
         va='top', ha='left', linespacing=1.35)
ax_a.set_title('Configuration', fontsize=12, color=TEXT, fontweight='bold', pad=8)

# Features panel
ax_f = fig.add_subplot(gs[3, 4:6])
ax_f.set_facecolor(BG2)
ax_f.axis('off')
features = [
    ("Configurable depth", "2-6 encoder levels"),
    ("Multi-format input", "TIFF, PNG, JPEG"),
    ("BCE + Dice loss", "Direct overlap optimization"),
    ("Rich augmentation", "Elastic, flip, rotate, noise"),
    ("Resume training", "Full checkpoint restore"),
    ("Best-model save", "Auto checkpoint on val loss"),
    ("Cross-platform", "CUDA, Apple MPS, CPU"),
    ("TensorBoard", "Real-time training curves"),
]
for i, (feat, desc) in enumerate(features):
    y = 0.92 - i * 0.115
    ax_f.text(0.06, y, feat, transform=ax_f.transAxes,
             fontsize=10, color=GREEN, fontweight='bold', fontfamily='sans-serif')
    ax_f.text(0.06, y - 0.045, desc, transform=ax_f.transAxes,
             fontsize=8.5, color=MUTED, fontfamily='sans-serif')
ax_f.set_title('Features', fontsize=12, color=TEXT, fontweight='bold', pad=8)

# Footer
fig.text(0.5, 0.02,
         'github.com/Weiykong/universal-unet-segmentation  |  MIT License  |  Python 3.10+',
         fontsize=11, ha='center', color='#484f58', fontfamily='monospace')

plt.savefig('portfolio_zoomed.png', dpi=150, bbox_inches='tight', facecolor=BG,
            edgecolor='none', pad_inches=0.3)
print('Saved portfolio_zoomed.png')
