import torch
import torch.nn.functional as F
import tifffile
import numpy as np
import os
import glob
from model import UNet
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "data/inference_input"
OUTPUT_DIR = "output"
MODEL_PATH = "models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available(): DEVICE = torch.device("mps")

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


def load_model(model_path):
    """Load model, auto-detecting checkpoint format (dict with hyperparams or raw state_dict)."""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        depth = checkpoint.get('depth', 4)
        base_features = checkpoint.get('base_features', 64)
        norm = checkpoint.get('norm', 'batch')
        out_channels = checkpoint.get('out_channels', 1)
        model = UNet(depth=depth, base_features=base_features,
                     norm=norm, out_channels=out_channels).to(DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model: depth={depth}, base_features={base_features}, "
              f"norm={norm}, out_channels={out_channels}")
    else:
        model = UNet().to(DEVICE)
        model.load_state_dict(checkpoint)
        print("Loaded model (legacy format, using defaults)")
    return model


def predict_folder():
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created input folder: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}.")
        return

    model = load_model(MODEL_PATH)
    model.eval()

    image_paths = sorted(
        p for ext in SUPPORTED_EXTENSIONS
        for p in glob.glob(os.path.join(INPUT_DIR, ext))
    )

    if not image_paths:
        print("No images found.")
        return

    print(f"Starting inference on {len(image_paths)} files...")

    for img_path in tqdm(image_paths, desc="Inference Progress", unit="img"):
        filename = os.path.basename(img_path)

        raw = load_image(img_path)
        p_low, p_high = np.percentile(raw, (1, 99))
        img = np.clip(raw, p_low, p_high)
        img = (img - p_low) / (p_high - p_low + 1e-6)

        input_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # Pad to multiple of 2^depth (16 for depth=4)
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        align = 2 ** model.depth if hasattr(model, 'depth') else 16
        pad_h = (align - (h % align)) % align
        pad_w = (align - (w % align)) % align
        if pad_h > 0 or pad_w > 0:
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            output = model(input_tensor)

        # Crop padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h, :w]

        multiclass = model.out_channels > 1
        if multiclass:
            # Save integer label map (0..N-1)
            result = output.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
            save_path = os.path.join(OUTPUT_DIR, f"labels_{filename}")
            tifffile.imwrite(save_path, result)
        else:
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            prob_map = (prob_map * 255.0).astype(np.float32)
            save_path = os.path.join(OUTPUT_DIR, f"prob_{filename}")
            tifffile.imwrite(save_path, prob_map)

if __name__ == "__main__":
    predict_folder()
