"""Export the trained U-Net to ONNX and verify ONNX Runtime parity with PyTorch.

The exported graph has dynamic height/width (input must be a multiple of 2^depth,
which the deployment pre-pads). This is the foundation of the C++ edge-deployment
path: if ORT does not match PyTorch here, nothing downstream is trustworthy.
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import UNet, LegacyUNet  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
CKPT = os.path.join(REPO, "models", "best_model.pth")
OUT = os.path.join(os.path.dirname(__file__), "unet.onnx")
OPSET = 17


def _is_legacy(sd):
    return any('.block.' not in k and k.startswith('encoders.') for k in sd)


def _infer_arch(sd):
    enc = sorted({int(k.split('.')[1]) for k in sd if k.startswith('encoders.')})
    first = next(v for k, v in sd.items()
                 if k.startswith('encoders.0.') and k.endswith('.weight') and v.dim() == 4)
    return len(enc), first.shape[0], sd['final.weight'].shape[0]


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    if _is_legacy(sd):
        d, bf, oc = _infer_arch(sd)
        m = LegacyUNet(depth=d, base_features=bf, out_channels=oc)
    else:
        d = ckpt.get('depth', 4) if isinstance(ckpt, dict) else 4
        bf = ckpt.get('base_features', 64) if isinstance(ckpt, dict) else 64
        norm = ckpt.get('norm', 'batch') if isinstance(ckpt, dict) else 'batch'
        oc = ckpt.get('out_channels', 1) if isinstance(ckpt, dict) else 1
        m = UNet(depth=d, base_features=bf, norm=norm, out_channels=oc)
    m.load_state_dict(sd)
    m.eval()
    print(f"Loaded model: depth={m.depth}, out_channels={m.out_channels}")
    return m


def main():
    model = load_model(CKPT)
    dummy = torch.randn(1, 1, 256, 256)  # multiple of 2^depth

    torch.onnx.export(
        model, dummy, OUT,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {2: "H", 3: "W"}, "logits": {2: "H", 3: "W"}},
        opset_version=OPSET, do_constant_folding=True,
    )
    # torch exports weights as an external sidecar; consolidate into a single
    # self-contained file so the C++ edge deployment ships exactly one artefact.
    import onnx
    m = onnx.load(OUT)  # resolves external data
    onnx.checker.check_model(m)
    onnx.save(m, OUT, save_as_external_data=False)
    sidecar = OUT + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)
    print(f"Exported + checked (single file) -> {OUT} "
          f"({os.path.getsize(OUT) / 1e6:.1f} MB)")

    # Parity: PyTorch vs ONNX Runtime on several random sizes (all mult. of 16)
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    max_diff = 0.0
    min_mask_agree = 1.0
    for (h, w) in [(256, 256), (128, 320), (512, 256)]:
        x = torch.randn(1, 1, h, w)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(["logits"], {"input": x.numpy()})[0]
        d = float(np.abs(ref - got).max())
        # What actually ships is the thresholded mask, not raw logits.
        mask_ref = (1 / (1 + np.exp(-ref)) > 0.5)
        mask_got = (1 / (1 + np.exp(-got)) > 0.5)
        agree = float((mask_ref == mask_got).mean())
        max_diff = max(max_diff, d)
        min_mask_agree = min(min_mask_agree, agree)
        print(f"  {h}x{w}: max|logit diff| = {d:.3e}   mask agreement = {agree*100:.4f}%")
    print(f"\nMAX logit diff: {max_diff:.3e}   MIN mask agreement: {min_mask_agree*100:.4f}%")
    # Logit diff ~1e-4 is normal float32 divergence between PyTorch and ORT conv
    # kernels; the shipped artefact is the thresholded mask, which must match.
    assert max_diff < 1e-3, "ONNX Runtime logits diverge from PyTorch!"
    assert min_mask_agree > 0.9999, "ONNX Runtime mask diverges from PyTorch!"
    print("PARITY OK")


if __name__ == "__main__":
    main()
