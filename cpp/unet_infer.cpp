// Standalone C++ U-Net inference using ONNX Runtime.
//
// Reproduces the exact pipeline of ../src/inference.py with no Python runtime:
//   1. load grayscale image
//   2. percentile(1, 99) clip + normalise
//   3. reflect-pad to a multiple of 2^depth (16)
//   4. ONNX Runtime forward pass
//   5. crop padding, sigmoid, threshold -> binary mask
//
// Target use case: deploying the segmentation model on the edge / in a
// compute-constrained environment where a Python + PyTorch stack is not viable.
//
//   ./unet_infer <model.onnx> <input_image> <output_mask.png> [threshold]

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

namespace {

constexpr int kAlign = 16;  // 2^depth, depth = 4

// numpy.percentile with default linear interpolation, on a copy of the data.
float Percentile(std::vector<float> v, double pct) {
  if (v.empty()) return 0.0f;
  std::sort(v.begin(), v.end());
  const double rank = (pct / 100.0) * (static_cast<double>(v.size()) - 1.0);
  const auto lo = static_cast<size_t>(std::floor(rank));
  const auto hi = static_cast<size_t>(std::ceil(rank));
  const double frac = rank - static_cast<double>(lo);
  return static_cast<float>(v[lo] * (1.0 - frac) + v[hi] * frac);
}

// Reflect-pad the bottom and right edges (matches torch F.pad mode='reflect').
// Reflection excludes the edge pixel itself: row H maps to row H-2, etc.
std::vector<float> ReflectPad(const std::vector<float>& img, int h, int w,
                              int pad_h, int pad_w, int& out_h, int& out_w) {
  out_h = h + pad_h;
  out_w = w + pad_w;
  std::vector<float> out(static_cast<size_t>(out_h) * out_w);
  for (int r = 0; r < out_h; ++r) {
    int sr = r < h ? r : 2 * h - 2 - r;  // reflect without repeating edge
    for (int c = 0; c < out_w; ++c) {
      int sc = c < w ? c : 2 * w - 2 - c;
      out[static_cast<size_t>(r) * out_w + c] =
          img[static_cast<size_t>(sr) * w + sc];
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0]
              << " <model.onnx> <input_image> <output_mask.png> [threshold]\n";
    return 1;
  }
  const std::string model_path = argv[1];
  const std::string input_path = argv[2];
  const std::string output_path = argv[3];
  const float threshold = argc > 4 ? std::stof(argv[4]) : 0.5f;

  // --- 1. load image as single-channel float ---
  int w = 0, h = 0, ch = 0;
  unsigned char* pixels = stbi_load(input_path.c_str(), &w, &h, &ch, 1);
  if (!pixels) {
    std::cerr << "failed to load image: " << input_path << "\n";
    return 1;
  }
  std::vector<float> img(static_cast<size_t>(h) * w);
  for (size_t i = 0; i < img.size(); ++i) img[i] = static_cast<float>(pixels[i]);
  stbi_image_free(pixels);

  // --- 2. percentile(1, 99) clip + normalise ---
  const float p_low = Percentile(img, 1.0);
  const float p_high = Percentile(img, 99.0);
  const float denom = (p_high - p_low) + 1e-6f;
  for (float& v : img) {
    v = std::min(std::max(v, p_low), p_high);
    v = (v - p_low) / denom;
  }

  // --- 3. reflect-pad to a multiple of kAlign ---
  const int pad_h = (kAlign - (h % kAlign)) % kAlign;
  const int pad_w = (kAlign - (w % kAlign)) % kAlign;
  int hp = h, wp = w;
  std::vector<float> padded =
      (pad_h || pad_w) ? ReflectPad(img, h, w, pad_h, pad_w, hp, wp) : img;

  // --- 4. ONNX Runtime forward pass ---
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "unet");
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(static_cast<int>(std::thread::hardware_concurrency()));
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  Ort::Session session(env, model_path.c_str(), opts);

  Ort::AllocatorWithDefaultOptions allocator;
  Ort::AllocatedStringPtr in_name = session.GetInputNameAllocated(0, allocator);
  Ort::AllocatedStringPtr out_name = session.GetOutputNameAllocated(0, allocator);
  const char* in_names[] = {in_name.get()};
  const char* out_names[] = {out_name.get()};

  std::array<int64_t, 4> shape{1, 1, hp, wp};
  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input = Ort::Value::CreateTensor<float>(
      mem, padded.data(), padded.size(), shape.data(), shape.size());

  auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1,
                             out_names, 1);
  const float* logits = outputs[0].GetTensorData<float>();

  // --- 5. crop padding, sigmoid, threshold -> mask ---
  std::vector<unsigned char> mask(static_cast<size_t>(h) * w);
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      const float logit = logits[static_cast<size_t>(r) * wp + c];
      const float prob = 1.0f / (1.0f + std::exp(-logit));
      mask[static_cast<size_t>(r) * w + c] = prob > threshold ? 255 : 0;
    }
  }

  if (!stbi_write_png(output_path.c_str(), w, h, 1, mask.data(), w)) {
    std::cerr << "failed to write mask: " << output_path << "\n";
    return 1;
  }
  std::cout << "wrote " << output_path << " (" << w << "x" << h
            << ", pad " << pad_w << "x" << pad_h << ")\n";
  return 0;
}
