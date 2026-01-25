# 🔬 Universal U-Net Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Weiykong/universal-unet-segmentation/blob/main/demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

A lightweight, universal U-Net implementation designed for high-performance biological image segmentation. This project is optimized for **TIFF stacks** and supports acceleration on **Apple Silicon (M1/M2/M3)** as well as standard CUDA GPUs.

## 📂 Project Structure

```text
universal-unet-segmentation/
├── data/                  # Data directory (ignored by Git)
│   ├── images/            # Place raw input TIFF images here
│   ├── masks/             # Place ground truth binary masks here
│   └── inference_input/   # Place new images to segment here
├── models/                # Saved model weights (.pth)
├── output/                # Segmentation probability maps
├── src/                   # Core implementation
│   ├── model.py           # U-Net architecture
│   ├── train.py           # Training loop
│   └── inference.py       # Prediction logic
├── run.py                 # Main execution script (Entry point)
└── requirements.txt       # Strict dependency versions
```

## 🚀 Quick Start
1. Installation
Clone the repository and install the dependencies:

Bash
git clone [https://github.com/Weiykong/universal-unet-segmentation.git](https://github.com/Weiykong/universal-unet-segmentation.git)
cd universal-unet-segmentation
pip install -r requirements.txt

## 2. Try the Demo (No Installation Required)
Click the "Open in Colab" badge at the top of this README to launch a live Jupyter Notebook. This will:

Clone the code on a remote cloud server.

Generate synthetic test data.

Run the model and visualize the results instantly.

## 🖥️ Usage Guide
### Mode 1: Training a New Model
To train the U-Net on your own dataset:

Place your raw images (TIFF) in data/images/.

Place your corresponding binary masks in data/masks/.

Run the training command:

Bash
python run.py --mode train --epochs 50 --batch_size 4 --lr 0.001
Models will be saved automatically to the models/ folder.

### Mode 2: Inference (Segmentation)
To use a trained model to segment new images:

Place your new images in data/inference_input/.

Run the inference command:

Bash
python run.py --mode inference --model_path models/best_model.pth
Results will be saved as probability maps in output/.

## 🛠️ Requirements & Compatibility
This project uses strict version pinning for reproducibility.

Python: 3.10+

PyTorch: >= 2.0.0 (Required for MPS acceleration on Mac)

Key Libraries: tifffile, numpy, pandas, trackpy, pystackreg

To install the exact environment used in development:

Bash
pip install -r requirements.txt

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author
Weiyuan Kong

GitHub: @Weiykong