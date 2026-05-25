"""Generate a high-end landscape SVG portfolio poster using manual crops."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import os, sys, glob

sys.path.insert(0, 'src')
from model import UNet

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---- Colors & Styles ----
BG = '#0d1117'
BG2 = '#161b22'
TEXT = '#c9d1d9'
MUTED = '#8b949e'
BORDER = '#30363d'
GREEN = '#3fb950'
PURPLE = '#a371f7'
BLUE = '#58a6ff'
CYAN = '#39d2e0'
WHITE = '#ffffff'

def load_model(path):
    if not os.path.exists(path): return None
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
    ph, pw = (align - h % align) % align, (align - w % align) % align
    if ph or pw: t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    with torch.no_grad(): out = torch.sigmoid(model(t))
    return out[:, :, :h, :w].squeeze().cpu().numpy()

def enhance(img, pct=1):
    lo, hi = np.percentile(img, pct), np.percentile(img, 100 - pct)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

def get_crop(arr, cy, cx, size):
    h, w = arr.shape[:2]
    s = size // 2
    y1, y2, x1, x2 = max(0, cy-s), min(h, cy+s), max(0, cx-s), min(w, cx+s)
    return arr[y1:y2, x1:x2]

# Load Model
model = load_model('models/best_model.pth')

# Prepare Bead Data
try:
    bead_img = tifffile.imread('data/inference_input/ch04_f023.tif').astype(np.float32)
    bead_mask = tifffile.imread('data/masks/ch04_f023_mask.tif').astype(np.float32)
    if bead_mask.max() > 1: bead_mask /= 255.0
    bead_pred = predict(model, bead_img) if model else np.zeros_like(bead_mask)
except:
    bead_img, bead_mask, bead_pred = np.zeros((1024, 1024)), np.zeros((1024, 1024)), np.zeros((1024, 1024))

# Setup Figure
fig = plt.figure(figsize=(26, 14), facecolor=BG)
# Main Grid: Left(0.25), Middle(0.5), Right(0.25)
gs_main = gridspec.GridSpec(1, 3, figure=fig, left=0.03, right=0.97, top=0.88, bottom=0.1, 
                            width_ratios=[1, 2.2, 1.2], wspace=0.15)

# =============================================================================
# HEADER
# =============================================================================
fig.text(0.03, 0.96, 'Universal U-Net Segmentation', fontsize=48, fontweight='bold', color=WHITE)
fig.text(0.03, 0.92, 'Deep Learning Framework for Precision Microscopy Image Analysis', 
         fontsize=20, color=CYAN, style='italic')

chips = [('Dice 0.937', 'Huh7 Cells', GREEN), ('Dice 0.780', '1\u00b5m Beads', PURPLE),
         ('7.70M', 'Params', BLUE), ('MPS/CUDA', 'Compute', CYAN)]
for i, (val, label, col) in enumerate(chips):
    x = 0.97 - (3 - i) * 0.11
    fig.text(x, 0.95, val, fontsize=26, fontweight='bold', color=col, ha='right')
    fig.text(x, 0.925, label.upper(), fontsize=11, color=MUTED, fontweight='bold', ha='right')

# =============================================================================
# COLUMN 1: ARCHITECTURE & SPECS
# =============================================================================
gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0], hspace=0.25)

# Architecture
ax_arch = fig.add_subplot(gs_left[0])
ax_arch.set_facecolor(BG2)
ax_arch.set_title('Configurable U-Net', color=TEXT, fontsize=18, fontweight='bold', pad=15)
layers = 5
for i in range(layers):
    y, w = 0.8 - i * 0.15, 0.3 - i * 0.05
    ax_arch.add_patch(Rectangle((0.1, y), w, 0.08, facecolor=BLUE, alpha=0.5, edgecolor=BLUE))
    ax_arch.add_patch(Rectangle((0.9 - w, y), w, 0.08, facecolor=PURPLE, alpha=0.5, edgecolor=PURPLE))
    if i < layers - 1:
        ax_arch.annotate('', xy=(0.9 - w, y + 0.04), xytext=(0.1 + w, y + 0.04),
                         arrowprops=dict(arrowstyle='->', color=BORDER, linestyle=':'))
ax_arch.text(0.5, 0.05, 'Encoder-Decoder with Skip Connections', color=MUTED, ha='center', fontsize=11)
ax_arch.axis('off')

# Features
ax_feat = fig.add_subplot(gs_left[1])
ax_feat.set_facecolor(BG2)
ax_feat.set_title('Platform Features', color=TEXT, fontsize=18, fontweight='bold', pad=15)
features = [("Dynamic Depth", "Scalable from 2 to 6 levels"),
            ("Optimized Loss", "Combined BCE + Soft-Dice"),
            ("Data Augmentation", "Elastic, Rotation, Jitter"),
            ("Multi-GPU Support", "Apple Silicon (MPS) & NVIDIA"),
            ("Deployment Ready", "Pure PyTorch implementation")]
for i, (f, d) in enumerate(features):
    y = 0.85 - i * 0.16
    ax_feat.text(0.05, y, f"• {f}", color=CYAN, fontsize=15, fontweight='bold')
    ax_feat.text(0.08, y - 0.06, d, color=MUTED, fontsize=12)
ax_feat.axis('off')

# =============================================================================
# COLUMN 2: CELL SEGMENTATION (6 MANUAL CROPS)
# =============================================================================
gs_mid = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs_main[1], hspace=0.15, wspace=0.08)

# Layout: 
# Input 1 | GT 1 | Pred 1
# Input 2 | GT 2 | Pred 2
# [Bottom row for labels or more space]
cell_files = [
    ('input1.png', 'Input Sample 1'), ('groundtruth1.png', 'Ground Truth'), ('prediction1.png', 'Model Prediction'),
    ('input2.png', 'Input Sample 2'), ('groundtruth2.png', 'Ground Truth'), ('predication2.png', 'Model Prediction')
]

for i, (fname, title) in enumerate(cell_files):
    path = os.path.join('cropped', fname)
    ax = fig.add_subplot(gs_mid[i // 3, i % 3])
    if os.path.exists(path):
        img = plt.imread(path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f'Missing:\n{fname}', ha='center', color=MUTED)
    
    if i < 3: ax.set_title(title, color=MUTED, fontsize=14, pad=10)
    ax.axis('off')
    ax.add_patch(Rectangle((0,0), 1, 1, transform=ax.transAxes, color=BORDER, fill=False, linewidth=1.5))

# Label for the section
fig.text(0.48, 0.84, 'Cell Segmentation Performance (Fluo-C2DL-Huh7)', 
         fontsize=20, color=WHITE, fontweight='bold', ha='center')

# =============================================================================
# COLUMN 3: BEAD DETECTION
# =============================================================================
gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[2], hspace=0.1)

# Full view
ax_b1 = fig.add_subplot(gs_right[0])
ax_b1.imshow(enhance(bead_img, 0.5), cmap='gray')
ax_b1.set_title('1 \u00b5m Bead Detection', color=TEXT, fontsize=18, fontweight='bold')
ax_b1.axis('off')

# Zoom GT
cy, cx, sz = 600, 600, 300
z_img = enhance(get_crop(bead_img, cy, cx, sz), 0.5)
z_mask = get_crop(bead_mask, cy, cx, sz)
z_rgb = np.stack([z_img]*3, axis=-1) * 0.5
z_rgb[z_mask > 0.5] = [0.2, 1.0, 0.3]
ax_b2 = fig.add_subplot(gs_right[1])
ax_b2.imshow(z_rgb)
ax_b2.set_title('Zoom: Ground Truth', color=TEXT, fontsize=14)
ax_b2.axis('off')

# Zoom Pred
z_pred = get_crop(bead_pred, cy, cx, sz)
z_prgb = np.stack([z_img]*3, axis=-1) * 0.5
z_prgb[z_pred > 0.5] = [0.2, 0.8, 1.0]
ax_b3 = fig.add_subplot(gs_right[2])
ax_b3.imshow(z_prgb)
ax_b3.set_title('Zoom: Model Prediction', color=TEXT, fontsize=14)
ax_b3.axis('off')

# =============================================================================
# FOOTER
# =============================================================================
fig.text(0.5, 0.05, 'TECHNOLOGY STACK', fontsize=11, color=MUTED, fontweight='bold', ha='center')
techs = ['Python', 'PyTorch', 'NumPy', 'Apple MPS', 'NVIDIA CUDA', 'TensorBoard', 'Scikit-Image']
for i, tech in enumerate(techs):
    x = 0.5 + (i - (len(techs)-1)/2) * 0.09
    fig.text(x, 0.03, tech, fontsize=13, color=TEXT, ha='center', fontweight='bold')

fig.text(0.5, 0.01, 'github.com/Weiykong/universal-unet-segmentation  |  MIT License  |  \u00a9 2024',
         fontsize=10, color=BORDER, ha='center', fontfamily='monospace')

plt.savefig('portfolio.svg', format='svg', bbox_inches='tight', facecolor=BG)
plt.savefig('portfolio_landscape.png', dpi=200, bbox_inches='tight', facecolor=BG)
print("Created high-end portfolio.svg and portfolio_landscape.png")
