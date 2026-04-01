import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import tifffile


IMAGE_PATH = "data/images/ch04_f023.tif"
MASK_PATH = "data/masks/ch04_f023_mask.tif"
PROB_PATH = "output/prob_ch04_f023.tif"
OUTPUT_PATH = "output/ch04_f023_overlay_comparison.png"


def normalize_image(image):
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-6)


def normalize_mask(mask):
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return mask


def normalize_prob(prob_map):
    prob_map = prob_map.astype(np.float32)
    if prob_map.max() > 1.0:
        prob_map = prob_map / 255.0
    return np.clip(prob_map, 0.0, 1.0)


def find_zoom_box(mask, prob_map, window_size=320, stride=32):
    density = mask + prob_map
    h, w = mask.shape
    box_w = min(window_size, w)
    box_h = min(window_size, h)

    best_score = None
    best_box = None

    for y0 in range(0, max(1, h - box_h + 1), stride):
        for x0 in range(0, max(1, w - box_w + 1), stride):
            score = float(density[y0:y0 + box_h, x0:x0 + box_w].sum())
            if best_score is None or score > best_score:
                best_score = score
                best_box = (x0, x0 + box_w, y0, y0 + box_h)

    if best_box is None:
        return 0, w, 0, h

    x0, x1, y0, y1 = best_box
    if x1 < w and (w - box_w) % stride != 0:
        tail_score = float(density[y0:y0 + box_h, w - box_w:w].sum())
        if tail_score > best_score:
            best_box = (w - box_w, w, y0, y0 + box_h)
            best_score = tail_score

    if y1 < h and (h - box_h) % stride != 0:
        tail_score = float(density[h - box_h:h, x0:x0 + box_w].sum())
        if tail_score > best_score:
            best_box = (x0, x0 + box_w, h - box_h, h)

    return best_box


def draw_gt_overlay(ax, image, mask, zoom_box=None):
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)

    green = np.zeros((*mask.shape, 4), dtype=np.float32)
    green[..., 1] = 1.0
    green[..., 3] = mask * 0.75
    ax.imshow(green)

    if zoom_box is not None:
        x0, x1, y0, y1 = zoom_box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=2))

    ax.set_xticks([])
    ax.set_yticks([])


def draw_prob_overlay(ax, image, prob_map, zoom_box=None):
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)

    alpha = np.clip(prob_map ** 0.85, 0.0, 1.0) * 0.85
    red = np.zeros((*prob_map.shape, 4), dtype=np.float32)
    red[..., 0] = 1.0
    red[..., 3] = alpha
    ax.imshow(red)

    if zoom_box is not None:
        x0, x1, y0, y1 = zoom_box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=2))

    ax.set_xticks([])
    ax.set_yticks([])


def main():
    if not (os.path.exists(IMAGE_PATH) and os.path.exists(MASK_PATH) and os.path.exists(PROB_PATH)):
        raise FileNotFoundError("Missing input image, mask, or probability map.")

    image = normalize_image(tifffile.imread(IMAGE_PATH))
    mask = normalize_mask(tifffile.imread(MASK_PATH))
    prob_map = normalize_prob(tifffile.imread(PROB_PATH))
    zoom_box = find_zoom_box(mask, prob_map)

    x0, x1, y0, y1 = zoom_box
    image_zoom = image[y0:y1, x0:x1]
    mask_zoom = mask[y0:y1, x0:x1]
    prob_zoom = prob_map[y0:y1, x0:x1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle("ch04_f023: Ground Truth Overlay vs Prediction Probability Overlay", fontsize=16, fontweight="bold")

    draw_gt_overlay(axes[0, 0], image, mask, zoom_box=zoom_box)
    axes[0, 0].set_title("Ground Truth Overlay")

    draw_prob_overlay(axes[0, 1], image, prob_map, zoom_box=zoom_box)
    axes[0, 1].set_title("Prediction Probability Overlay")

    draw_gt_overlay(axes[1, 0], image_zoom, mask_zoom)
    axes[1, 0].set_title("Zoom: Ground Truth Overlay")

    draw_prob_overlay(axes[1, 1], image_zoom, prob_zoom)
    axes[1, 1].set_title("Zoom: Prediction Probability Overlay")

    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
