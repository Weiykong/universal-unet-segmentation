"""Parity test: C++ ONNX deployment vs Python reference.

Runs the compiled unet_infer binary and reference_infer.py on the same image
through the same unet.onnx, and asserts the output masks are identical.

    python cpp/test_parity.py
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "unet.onnx")
IMAGE = os.path.join(HERE, "samples", "t000.png")
BINARY = os.path.join(HERE, "build", "unet_infer")
ORT_LIB = subprocess.run(["brew", "--prefix", "onnxruntime"],
                         capture_output=True, text=True).stdout.strip() + "/lib"


def main():
    for path, what in [(MODEL, "model"), (IMAGE, "sample image"), (BINARY, "C++ binary")]:
        if not os.path.exists(path):
            print(f"SKIP: {what} missing ({path}). Run export_onnx.py / cmake build first.")
            return 0

    cpp_out = os.path.join(HERE, "samples", "_parity_cpp.png")
    py_out = os.path.join(HERE, "samples", "_parity_py.png")

    env = dict(os.environ, DYLD_LIBRARY_PATH=ORT_LIB)
    subprocess.run([BINARY, MODEL, IMAGE, cpp_out], check=True, env=env)

    sys.path.insert(0, HERE)
    from reference_infer import infer
    Image.fromarray(infer(MODEL, IMAGE)).save(py_out)

    a = np.array(Image.open(cpp_out))
    b = np.array(Image.open(py_out))
    agree = float((a == b).mean())
    print(f"pixel agreement: {agree * 100:.4f}%  ({(a != b).sum()} differ / {a.size})")
    assert a.shape == b.shape, "shape mismatch"
    assert agree == 1.0, "C++ and Python masks differ!"
    print("PARITY OK — C++ deployment matches Python reference exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
