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
MODEL_PATH = "models/bead_unet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available(): DEVICE = torch.device("mps")

def predict_folder():
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"⚠️ Created input folder: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}.")
        return

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")) + 
                         glob.glob(os.path.join(INPUT_DIR, "*.tiff")))
    
    if not image_paths:
        print("No images found.")
        return

    print(f"🔍 Starting inference on {len(image_paths)} files...")

    # Wrap loop with tqdm for progress bar
    for img_path in tqdm(image_paths, desc="Inference Progress", unit="img"):
        filename = os.path.basename(img_path)
        
        raw = tifffile.imread(img_path).astype(np.float32)
        img = (raw - np.min(raw)) / (np.max(raw) - np.min(raw) + 1e-6)
        
        input_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Pad to multiple of 16
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        pad_h = (16 - (h % 16)) % 16
        pad_w = (16 - (w % 16)) % 16
        if pad_h > 0 or pad_w > 0:
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            output = model(input_tensor)
            prob_map = torch.sigmoid(output)

        # Crop padding
        if pad_h > 0 or pad_w > 0:
            prob_map = prob_map[:, :, :h, :w]

        # Post-process
        prob_map = prob_map.squeeze().cpu().numpy()
        # Scale to 32-bit float range (optional gain)
        prob_map = prob_map * 255.0 
        
        save_path = os.path.join(OUTPUT_DIR, f"prob_{filename}")
        tifffile.imwrite(save_path, prob_map.astype(np.float32))

if __name__ == "__main__":
    predict_folder()
