import torch
import torch.nn.functional as F
import tifffile
import numpy as np
import os
import glob
from model import UNet, LegacyUNet
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


def _is_legacy_state_dict(sd):
    """Detect pre-residual checkpoints: encoder keys are flat (encoders.0.0.weight)
    rather than nested under .block. (encoders.0.block.0.weight)."""
    return any('.block.' not in k and k.startswith('encoders.') for k in sd)


def _infer_arch(sd):
    """Infer depth and base_features from state dict key shapes."""
    enc_indices = sorted({int(k.split('.')[1]) for k in sd if k.startswith('encoders.')})
    depth = len(enc_indices)
    # base_features = out_channels of first encoder conv
    first_conv = next(v for k, v in sd.items() if k.startswith('encoders.0.') and k.endswith('.weight') and v.dim() == 4)
    base_features = first_conv.shape[0]
    out_channels = sd['final.weight'].shape[0]
    return depth, base_features, out_channels


def load_model(model_path):
    """Load model, auto-detecting checkpoint format."""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    sd = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

    if _is_legacy_state_dict(sd):
        depth, base_features, out_channels = _infer_arch(sd)
        model = LegacyUNet(depth=depth, base_features=base_features,
                           out_channels=out_channels).to(DEVICE)
        model.load_state_dict(sd)
        print(f"Loaded legacy model: depth={depth}, base_features={base_features}, out_channels={out_channels}")
    else:
        depth        = checkpoint.get('depth', 4)         if isinstance(checkpoint, dict) else 4
        base_features= checkpoint.get('base_features', 64) if isinstance(checkpoint, dict) else 64
        norm         = checkpoint.get('norm', 'batch')     if isinstance(checkpoint, dict) else 'batch'
        out_channels = checkpoint.get('out_channels', 1)   if isinstance(checkpoint, dict) else 1
        model = UNet(depth=depth, base_features=base_features,
                     norm=norm, out_channels=out_channels).to(DEVICE)
        model.load_state_dict(sd)
        print(f"Loaded model: depth={depth}, base_features={base_features}, norm={norm}, out_channels={out_channels}")

    return model


def predict_folder(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR,
                   model_path=MODEL_PATH, threshold=0.5):
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created input folder: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}.")
        return

    model = load_model(model_path)
    model.eval()

    image_paths = sorted(
        p for ext in SUPPORTED_EXTENSIONS
        for p in glob.glob(os.path.join(input_dir, ext))
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
            result = output.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
            save_path = os.path.join(output_dir, f"labels_{filename}")
            tifffile.imwrite(save_path, result)
        else:
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            binary = (prob_map > threshold).astype(np.uint8) * 255
            save_path = os.path.join(output_dir, f"prob_{filename}")
            tifffile.imwrite(save_path, prob_map * 255.0)
            tifffile.imwrite(os.path.join(output_dir, f"mask_{filename}"), binary)

if __name__ == "__main__":
    predict_folder()
