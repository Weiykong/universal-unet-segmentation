"""Generate a portfolio PNG showcasing bead segmentation results."""
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


def crop_center(arr, size):
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2
    s = size // 2
    return arr[cy - s:cy + s, cx - s:cx + s]


# ---- Load model ----
model = load_model('models/best_model.pth')


def load_with_mask(img_path, mask_path):
    img = tifffile.imread(img_path).astype(np.float32)
    mask = tifffile.imread(mask_path).astype(np.float32)
    if mask.max() > 1:
        mask /= 255.0
    img_n = (img - img.min()) / (img.max() - img.min() + 1e-6)
    pred = predict(model, img)
    return img_n, mask, pred


def metrics(mask, pred, thr=0.5):
    gt = mask > 0.5
    pr = pred > thr
    inter = (gt & pr).sum()
    d = (2 * inter + 1e-6) / (gt.sum() + pr.sum() + 1e-6)
    i = (inter + 1e-6) / (gt.sum() + pr.sum() - inter + 1e-6)
    return float(d), float(i)


# ---- PRIMARY: held-out sample ch04_f023 (has ground truth, matches README) ----
img_n, mask_raw, pred = load_with_mask(
    'data/inference_input/ch04_f023.tif',
    'data/masks/ch04_f023_mask.tif',
)
dice, iou = metrics(mask_raw, pred)

# ---- SECONDARY: training sample for row 3 comparison ----
tr_img_n, tr_mask, tr_pred = load_with_mask(
    'data/images/ch000_f018.tif',
    'data/masks/ch000_f018_mask.tif',
)
tr_dice, tr_iou = metrics(tr_mask, tr_pred)

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

fig = plt.figure(figsize=(18, 21), facecolor=BG)

# ---- Heading ----
fig.text(0.5, 0.982, 'Universal U-Net Segmentation',
         fontsize=34, fontweight='bold', ha='center', va='top',
         color='white', fontfamily='sans-serif')

# Accent tagline
fig.text(0.5, 0.962, 'Deep learning for fluorescent microscopy images',
         fontsize=15, style='italic', ha='center', va='top',
         color=CYAN, fontfamily='sans-serif')

# ---- Resume / summary paragraph ----
resume_text = (
    "A configurable PyTorch U-Net trained end-to-end on 109 fluorescence microscopy images "
    "to localize individual 1 um beads with pixel-level precision.\n"
    "Combined BCE + Dice loss, elastic augmentation, and cosine-annealed learning rate — "
    f"reaching Dice {dice:.3f} / IoU {iou:.3f} on the held-out sample "
    "after 200 epochs on Apple Silicon."
)
fig.text(0.5, 0.942, resume_text, fontsize=11.5, ha='center', va='top',
         color=TEXT, fontfamily='sans-serif', linespacing=1.5)

# ---- Stat chips row ----
chip_y = 0.902
chip_specs = [
    ('7.70M', 'parameters', BLUE),
    ('depth 4', 'U-Net levels', PURPLE),
    ('109', 'train images', GREEN),
    ('200', 'epochs', CYAN),
    (f'{dice:.3f}', 'Dice', GREEN),
    (f'{iou:.3f}', 'IoU', PURPLE),
]
n_chips = len(chip_specs)
chip_span = 0.82
chip_x0 = (1 - chip_span) / 2
for i, (val, label, col) in enumerate(chip_specs):
    x = chip_x0 + (i + 0.5) * (chip_span / n_chips)
    fig.text(x, chip_y, val, fontsize=18, fontweight='bold',
             ha='center', va='center', color=col, fontfamily='sans-serif')
    fig.text(x, chip_y - 0.022, label.upper(), fontsize=9,
             ha='center', va='center', color=MUTED,
             fontfamily='sans-serif', fontweight='bold')

gs = gridspec.GridSpec(4, 6, figure=fig, top=0.855, bottom=0.06, left=0.03, right=0.97,
                       hspace=0.35, wspace=0.15)

# ================================================================
# ROW 1: Full image — Input | GT overlay | Pred overlay
# ================================================================
img_e = enhance(img_n, 0.5)

ax1 = fig.add_subplot(gs[0, 0:2])
ax1.imshow(img_e, cmap='gray')
ax1.set_title('Held-out Sample (ch04_f023)', fontsize=12, color=TEXT,
             fontweight='bold', pad=8)
ax1.axis('off')
ax1.set_facecolor(BG)

# GT overlay
gt_rgb = np.stack([img_e] * 3, axis=-1) * 0.5
gt_rgb[mask_raw > 0.5] = [0.2, 1.0, 0.3]
ax2 = fig.add_subplot(gs[0, 2:4])
ax2.imshow(gt_rgb)
ax2.set_title('Ground Truth Overlay', fontsize=12, color=TEXT, fontweight='bold', pad=8)
ax2.axis('off')
ax2.set_facecolor(BG)

# Pred overlay
pr_rgb = np.stack([img_e] * 3, axis=-1) * 0.5
pr_rgb[pred > 0.5] = [0.2, 0.8, 1.0]
ax3 = fig.add_subplot(gs[0, 4:6])
ax3.imshow(pr_rgb)
ax3.set_title('Prediction Overlay', fontsize=12, color=TEXT, fontweight='bold', pad=8)
ax3.axis('off')
ax3.set_facecolor(BG)
ax3.text(0.98, 0.04, f'Dice {dice:.3f}  |  IoU {iou:.3f}',
        transform=ax3.transAxes, fontsize=10, color='white', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#238636', alpha=0.9))

# Draw zoom box on input
sz = 200
h, w = img_e.shape
cy, cx = h // 2 + 50, w // 2
from matplotlib.patches import Rectangle
for ax in [ax1, ax2, ax3]:
    rect = Rectangle((cx - sz, cy - sz), 2 * sz, 2 * sz,
                     linewidth=2, edgecolor=CYAN, facecolor='none', linestyle='--')
    ax.add_patch(rect)

# ================================================================
# ROW 2: Zoomed — Input | GT | Pred | Error overlay
# ================================================================
z_img = enhance(crop_center(np.roll(np.roll(img_n, 50, 0), 0, 1), 400), 0.5)
z_mask = crop_center(np.roll(np.roll(mask_raw, 50, 0), 0, 1), 400)
z_pred = crop_center(np.roll(np.roll(pred, 50, 0), 0, 1), 400)

ax4 = fig.add_subplot(gs[1, 0:2])
ax4.imshow(z_img, cmap='gray')
ax4.set_title('Zoom: Input', fontsize=12, color=CYAN, fontweight='bold', pad=8)
ax4.axis('off')
ax4.set_facecolor(BG)

# Zoomed GT
z_gt_rgb = np.stack([z_img] * 3, axis=-1) * 0.4
z_gt_rgb[z_mask > 0.5] = [0.2, 1.0, 0.3]
ax5 = fig.add_subplot(gs[1, 2:4])
ax5.imshow(z_gt_rgb)
ax5.set_title('Zoom: Ground Truth', fontsize=12, color=CYAN, fontweight='bold', pad=8)
ax5.axis('off')
ax5.set_facecolor(BG)

# Zoomed error overlay: green=TP, red=FP, blue=FN
z_err = np.stack([z_img] * 3, axis=-1) * 0.35
z_gt_b = z_mask > 0.5
z_pr_b = z_pred > 0.5
z_err[z_gt_b & z_pr_b] = [0.1, 0.95, 0.3]      # TP green
z_err[(~z_gt_b) & z_pr_b] = [1.0, 0.15, 0.15]   # FP red
z_err[z_gt_b & (~z_pr_b)] = [0.2, 0.4, 1.0]     # FN blue
ax6 = fig.add_subplot(gs[1, 4:6])
ax6.imshow(np.clip(z_err, 0, 1))
ax6.set_title('Zoom: Error Map (G=TP  R=FP  B=FN)', fontsize=12,
             color=CYAN, fontweight='bold', pad=8)
ax6.axis('off')
ax6.set_facecolor(BG)

# ================================================================
# ROW 3: Training sample (ch000_f018) — Input | GT overlay | Pred overlay
# ================================================================
tr_img_e = enhance(tr_img_n, 0.5)

ax7 = fig.add_subplot(gs[2, 0:2])
ax7.imshow(tr_img_e, cmap='gray')
ax7.set_title('Training Sample (ch000_f018)', fontsize=12, color=TEXT,
             fontweight='bold', pad=8)
ax7.axis('off')
ax7.set_facecolor(BG)

# Training-sample GT overlay
tr_gt_rgb = np.stack([tr_img_e] * 3, axis=-1) * 0.5
tr_gt_rgb[tr_mask > 0.5] = [0.2, 1.0, 0.3]
ax8 = fig.add_subplot(gs[2, 2:4])
ax8.imshow(tr_gt_rgb)
ax8.set_title('Ground Truth Overlay', fontsize=12, color=TEXT, fontweight='bold', pad=8)
ax8.axis('off')
ax8.set_facecolor(BG)

# Training-sample prediction overlay
tr_pr_rgb = np.stack([tr_img_e] * 3, axis=-1) * 0.5
tr_pr_rgb[tr_pred > 0.5] = [0.2, 0.8, 1.0]
ax9 = fig.add_subplot(gs[2, 4:6])
ax9.imshow(tr_pr_rgb)
ax9.set_title('Prediction Overlay', fontsize=12, color=TEXT, fontweight='bold', pad=8)
ax9.axis('off')
ax9.set_facecolor(BG)
ax9.text(0.98, 0.04, f'Dice {tr_dice:.3f}  |  IoU {tr_iou:.3f}',
        transform=ax9.transAxes, fontsize=10, color='white', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1f6feb', alpha=0.9))

# ================================================================
# ROW 4: Stats panels
# ================================================================

# Metrics panel
ax_m = fig.add_subplot(gs[3, 0:2])
ax_m.set_facecolor(BG2)
ax_m.axis('off')
for s in ax_m.spines.values():
    s.set_visible(True)
    s.set_color(BORDER)
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
for s in ax_a.spines.values():
    s.set_visible(True)
    s.set_color(BORDER)
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
for s in ax_f.spines.values():
    s.set_visible(True)
    s.set_color(BORDER)

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

plt.savefig('portfolio.png', dpi=150, bbox_inches='tight', facecolor=BG,
            edgecolor='none', pad_inches=0.3)
print('Saved portfolio.png')
