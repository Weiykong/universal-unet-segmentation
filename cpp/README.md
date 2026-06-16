# C++ Edge Deployment (ONNX Runtime)

A self-contained C++ inference path for the trained U-Net, with **no Python or
PyTorch runtime**. Intended for deployment on the edge / in compute-constrained
environments where shipping a multi-gigabyte Python + PyTorch stack is not an
option.

The C++ binary reproduces the exact pipeline of [`../src/inference.py`](../src/inference.py):

1. load a grayscale image
2. percentile (1, 99) clip + normalise
3. reflect-pad to a multiple of `2^depth` (16)
4. ONNX Runtime forward pass
5. crop padding, sigmoid, threshold → binary mask

## Why this exists (and why it isn't a "speed" play)

Both this binary and Python use the **same** ONNX Runtime engine, so inference
latency is comparable — this is **not** a performance optimisation. The value is
**deployment footprint and runtime independence**:

| | Python path | This C++ path |
|---|---|---|
| Runtime deps | Python + PyTorch + torchvision (~GBs) | `libonnxruntime` + a small native binary |
| Artefacts | checkpoint + source tree | one `unet.onnx` (30 MB) + `unet_infer` |
| Portable to edge / constrained HW | hard | yes |

## Build

```bash
# 1. Export the model from the trained checkpoint (one-time)
python cpp/export_onnx.py          # writes cpp/unet.onnx, verifies ORT parity

# 2. Build the C++ binary (requires ONNX Runtime)
brew install onnxruntime           # macOS; or pass -DONNXRUNTIME_ROOT=...
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
```

## Run

```bash
./cpp/build/unet_infer cpp/unet.onnx input.png mask.png [threshold]
```

`stb_image` handles PNG/JPEG input. (Microscopy TIFFs should be converted to
PNG first; 16-bit TIFF is out of scope for the single-header loader.)

## Correctness

The C++ output is validated **pixel-for-pixel** against a Python reference that
runs the same ONNX model with the same preprocessing:

```bash
python cpp/test_parity.py
# pixel agreement: 100.0000%  (0 differ / 1048576)
# PARITY OK — C++ deployment matches Python reference exactly
```

Parity holds across ONNX Runtime versions (built against 1.26, reference on
1.21): the thresholded masks are identical.

## Files

| File | Role |
|---|---|
| `export_onnx.py` | PyTorch → ONNX export + PyTorch/ORT parity check |
| `unet_infer.cpp` | Standalone C++ inference (preprocess → ORT → mask) |
| `CMakeLists.txt` | Build config; finds ONNX Runtime via Homebrew or `ONNXRUNTIME_ROOT` |
| `reference_infer.py` | Python reference using the same ONNX model |
| `test_parity.py` | Asserts C++ and Python masks are identical |
| `third_party/stb_image*.h` | Single-header PNG/JPEG I/O (public domain) |

## Troubleshooting

**`fatal error: 'cmath' file not found`** — a macOS Command Line Tools install
whose libc++ headers aren't on the default search path. Point the compiler at
the SDK explicitly:

```bash
SDK=$(xcrun --show-sdk-path)
cmake -B cpp/build -S cpp \
  -DCMAKE_CXX_FLAGS="-isysroot $SDK -isystem $SDK/usr/include/c++/v1 -isystem $SDK/usr/include"
```
