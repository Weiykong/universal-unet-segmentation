# Universal U-Net Segmentation Engine 🔬

A lightweight, high-performance deep learning pipeline designed to transform raw microscopy images into high-fidelity **Probability Maps**.

## Overview
This project provides a robust, device-agnostic U-Net implementation. Unlike standard segmenters that produce binary masks, this engine outputs continuous probability values (0.0 to 1.0). This approach preserves sub-pixel information and intensity gradients, making it ideal for downstream analysis like particle tracking with **TrackPy**, centroid detection, or high-precision feature segmentation.



## Key Features
* **Universal Training**: Optimized for any raw image + binary mask pairs (beads, cells, vesicles, etc.).
* **Random Crop Training**: Train on large frames (e.g., 1024x1024) using random 512x512 patches to improve generalization and memory efficiency.
* **Hardware Acceleration**: Automatic support for Apple Silicon (**MPS**), NVIDIA (**CUDA**), and CPU.
* **Stable Learning**: Includes Batch Normalization and weighted loss functions (`pos_weight`) to prevent model collapse on sparse datasets.
* **32-bit Output**: Probability maps are saved as 32-bit TIFFs to maintain high numerical precision for scientific analysis.

## Project Structure
```text
.
├── data/
│   ├── images/           # Training Raw TIFFs
│   ├── masks/            # Training Binary Masks
│   └── inference_input/  # Unseen raw data for prediction
├── src/
│   ├── model.py          # U-Net Architecture
│   ├── train.py          # Training & Augmentation logic
│   └── inference.py      # Probability Map generation
├── output/               # Resulting Probability Maps
├── run.py                # Main pipeline entry point
└── requirements.txt      # Dependencies
```

## Installation
Bash
pip install -r requirements.txt
Usage
The entire pipeline is managed via run.py.

## 1. Training
Place your raw data in data/images and masks in data/masks.

Bash
python run.py --mode train --augment --epochs 100
--augment: Enables random flips and 90-degree rotations.

--epochs: Recommended 100+ for small datasets to ensure convergence.

## 2. Inference
Place raw images or films in data/inference_input.

Bash
python run.py --mode inference
The resulting maps will be saved in the output/ folder with a prob_ prefix.

## Why Probability Maps?
By outputting a 0.0–1.0 range instead of a 0/1 binary mask, this engine allows you to:

Perform sub-pixel localization of centers.

Filter objects by confidence/intensity using downstream thresholds.

Reduce "edge artifacts" common in binary segmentation.