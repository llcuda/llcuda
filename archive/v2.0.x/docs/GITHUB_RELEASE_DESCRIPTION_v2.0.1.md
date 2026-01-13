# llcuda v2.0.1 - Tesla T4 CUDA 12 Binaries

Official CUDA 12 binary release for llcuda v2.0.1 - optimized exclusively for **Tesla T4 GPU** (Google Colab).

## 📦 What's Included

**File:** `llcuda-binaries-cuda12-t4.tar.gz` (140 MB)

- **llama-server** (6.5 MB) - HTTP inference server with FlashAttention
- **libggml-cuda.so** (174 MB) - CUDA kernels optimized for T4 Tensor Cores
- **Supporting libraries** - Complete runtime dependencies

### Build Features
✅ **FlashAttention 2** - 2-3x faster for long contexts
✅ **Tensor Core Optimization** - FP16/INT8 acceleration
✅ **CUDA Graphs** - Reduced kernel launch overhead
✅ **All Quantization Types** - Q2_K through Q8_0 supported

## 🚀 Quick Start

### Installation
```bash
pip install llcuda
```

Binaries are automatically downloaded on first import (one-time, ~140 MB).

### Google Colab Example
```python
import llcuda

# Load model and run inference
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
```

## 📊 Performance (Tesla T4)

| Model | Speed | VRAM | Context |
|-------|-------|------|---------|
| Gemma 3-1B Q4_K_M | 45 tok/s | 1.2 GB | 2048 |
| Llama 3.2-3B Q4_K_M | 30 tok/s | 2.0 GB | 4096 |
| Qwen 2.5-7B Q4_K_M | 18 tok/s | 5.0 GB | 8192 |
| Llama 3.1-8B Q4_K_M | 15 tok/s | 5.5 GB | 8192 |

## 🎯 Requirements

- **GPU:** NVIDIA Tesla T4 (SM 7.5)
- **Python:** 3.11+
- **CUDA:** 12.x runtime
- **Platform:** Google Colab, Kaggle, or local T4

## 🔧 Manual Installation

If automatic download fails:

```bash
# Download
wget https://github.com/waqasm86/llcuda/releases/download/v2.0.1/llcuda-binaries-cuda12-t4.tar.gz

# Extract
mkdir -p ~/.cache/llcuda
tar -xzf llcuda-binaries-cuda12-t4.tar.gz -C ~/.cache/llcuda/

# Verify
ls ~/.cache/llcuda/bin/llama-server
```

## ✨ What's New in v2.0.1

- ✅ Updated to v2.0.1 release binaries
- ✅ Optimized package structure (140 MB vs 264 MB)
- ✅ Improved GPU detection and error messages
- ✅ Fixed silent mode bug in Google Colab
- ✅ Better library path detection

## 📚 Documentation

- **GitHub:** https://github.com/waqasm86/llcuda
- **PyPI:** https://pypi.org/project/llcuda/
- **Full Release Notes:** [RELEASE_NOTES_v2.0.1.md](https://github.com/waqasm86/llcuda/releases/download/v2.0.1/RELEASE_NOTES_v2.0.1.md)

## 🔐 Checksums

```
SHA256 (llcuda-binaries-cuda12-t4.tar.gz):
54bb904f213b38aec4a2c85baa2fab8256200966aa994bb9fa4a77640d699aa4
```

## 🛠️ Build Info

- **CUDA Version:** 12.4/12.6
- **Compute Capability:** SM 7.5 (Turing)
- **llama.cpp Version:** 0.0.7654
- **GGML Version:** 0.9.5
- **Build Date:** January 7, 2026

## ⚠️ GPU Compatibility

- ✅ **Tesla T4** (SM 7.5) - Fully supported and tested
- ⚠️ **RTX 20 series** (SM 7.5) - May work, not tested
- ❌ **Older GPUs** (SM < 7.5) - Use [llcuda v1.2.2](https://github.com/waqasm86/llcuda/releases/tag/v1.2.2)

## 📄 License

MIT License - Compatible with llama.cpp

---

**🎯 Target Platform:** Google Colab Tesla T4 (free tier)
**📦 Package Type:** Pre-compiled CUDA 12 binaries
**🔗 Python Package:** `pip install llcuda`
