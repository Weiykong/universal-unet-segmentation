"""Preprocess Cell Tracking Challenge dataset for U-Net training.

Converts Fluo-C2DL-Huh7 instance-labeled SEG masks into binary masks
and copies images + masks into data/images/ and data/masks/.

Uses SEG (full cell body) annotations, NOT TRA (centroid-only) annotations.
"""

import os
import shutil
import glob
import numpy as np
import tifffile

SRC_DIR = os.path.expanduser("~/Downloads/Fluo-C2DL-Huh7_trainning")
DST_IMG = "data/images"
DST_MASK = "data/masks"


def main():
    os.makedirs(DST_IMG, exist_ok=True)
    os.makedirs(DST_MASK, exist_ok=True)

    count = 0
    for seq in ["01", "02"]:
        img_dir = os.path.join(SRC_DIR, seq)
        seg_dir = os.path.join(SRC_DIR, f"{seq}_GT", "SEG")

        masks = sorted(glob.glob(os.path.join(seg_dir, "man_seg*.tif")))
        for mask_path in masks:
            # man_seg015.tif -> "015"
            mask_fname = os.path.basename(mask_path)
            frame_id = mask_fname.replace("man_seg", "").replace(".tif", "")
            img_path = os.path.join(img_dir, f"t{frame_id}.tif")

            if not os.path.exists(img_path):
                print(f"Skipping {mask_fname} (no image found)")
                continue

            # Copy image as-is
            out_name = f"seq{seq}_f{frame_id}.tif"
            shutil.copy2(img_path, os.path.join(DST_IMG, out_name))

            # Binarize mask: any label > 0 becomes 1.0
            mask = tifffile.imread(mask_path).astype(np.float32)
            binary_mask = (mask > 0).astype(np.float32)
            tifffile.imwrite(os.path.join(DST_MASK, out_name), binary_mask)

            count += 1

    print(f"Done. {count} image-mask pairs written to {DST_IMG}/ and {DST_MASK}/")


if __name__ == "__main__":
    main()
