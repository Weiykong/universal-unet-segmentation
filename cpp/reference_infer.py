"""Python reference for the C++ ONNX deployment, used for parity testing.

Runs the SAME unet.onnx through onnxruntime-Python with the SAME preprocessing
as unet_infer.cpp, so the C++ mask can be validated against it pixel-for-pixel.
"""
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort

ALIGN = 16


def infer(model_path, image_path, threshold=0.5):
    img = np.array(Image.open(image_path).convert("L")).astype(np.float32)
    h, w = img.shape
    p_low, p_high = np.percentile(img, (1, 99))
    img = np.clip(img, p_low, p_high)
    img = (img - p_low) / (p_high - p_low + 1e-6)

    pad_h = (ALIGN - (h % ALIGN)) % ALIGN
    pad_w = (ALIGN - (w % ALIGN)) % ALIGN
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")

    x = img[None, None].astype(np.float32)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    logits = logits[0, 0, :h, :w]
    prob = 1.0 / (1.0 + np.exp(-logits))
    return (prob > threshold).astype(np.uint8) * 255


if __name__ == "__main__":
    model, image, out = sys.argv[1], sys.argv[2], sys.argv[3]
    mask = infer(model, image)
    Image.fromarray(mask).save(out)
    print(f"wrote {out} ({mask.shape[1]}x{mask.shape[0]})")
