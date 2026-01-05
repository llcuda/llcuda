# llcuda: CUDA-Accelerated LLM Inference

![Version](https://img.shields.io/badge/version-1.2.2-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Effortless, zero-configuration LLM inference with CUDA acceleration. Optimized for **Ubuntu 22.04 with NVIDIA GeForce 940M** and **Google Colab with Tesla T4**.

## 🎉 What's New in v1.2.2

- **GPU-Specific Binaries**: Automatic detection and download of optimized binaries
- **FlashAttention Support**: 2x faster inference on Tesla T4 in Google Colab
- **Maxwell GPU Support**: Optimized builds for GeForce 940M on Ubuntu 22.04
- **Smart Bootstrap**: Auto-selects appropriate binary bundle based on detected GPU
- **Bug Fixes**: Resolved stderr.read() issue in Google Colab

## 🚀 Quick Start

### Installation

**Ubuntu 22.04 (Local System):**
```bash
pip install llcuda
```

**Google Colab:**
```python
!pip install llcuda
```

**Requirements:**
- Python 3.11+
- NVIDIA GPU: GeForce 940M or Tesla T4
- CUDA 12.x runtime

### Basic Usage

```python
import llcuda

# Initialize inference engine (auto-downloads optimized binaries on first run)
engine = llcuda.InferenceEngine()

# Load a model (downloads from HuggingFace on first use)
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)

# Run inference
result = engine.infer("What is artificial intelligence?", max_tokens=100)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")
```

## 🎮 Supported GPUs

llcuda is specifically optimized for two environments:

| Environment | GPU | Compute Cap | Package Size | Features |
|-------------|-----|-------------|--------------|----------|
| **Ubuntu 22.04 (Local)** | NVIDIA GeForce 940M | 5.0 | 26 MB | cuBLAS optimized |
| **Google Colab** | NVIDIA Tesla T4 | 7.5 | 264 MB | **FlashAttention** (2x faster) |

### What Gets Downloaded?

On first `import llcuda`, the bootstrap will:
1. Detect your GPU using `nvidia-smi`
2. Download the appropriate binary package:
   - **GeForce 940M**: 26 MB package optimized with forced cuBLAS
   - **Tesla T4**: 264 MB package with FlashAttention (2x faster)
3. Extract binaries and configure paths automatically

**This is a one-time download.** Subsequent imports use cached binaries.

## 📊 Performance Benchmarks

### GeForce 940M (Ubuntu 22.04)
- **Package Size**: 26 MB
- **GPU Layers**: 10-15
- **Context Length**: 512-1024 tokens
- **Best Models**: 1-3B params (Q4_K_M quantization)
- **Speed**: 10-20 tokens/sec

**Example:**
```
Model: Gemma 3-1B Q4_K_M (769 MB)
Speed: 15 tok/s
Latency: 67ms per token
VRAM: ~1 GB
```

### Tesla T4 (Google Colab) with FlashAttention
- **Package Size**: 264 MB
- **GPU Layers**: 26-35
- **Context Length**: 2048-8192 tokens
- **Best Models**: 1-13B params (Q4_K_M/Q5_K_M)
- **Speed**: 25-60 tokens/sec (2x faster with FlashAttention)

**Example:**
```
Model: Gemma 3-1B Q4_K_M (769 MB)
Speed: 45 tok/s
Latency: 22ms per token
VRAM: ~1 GB
FlashAttention: Enabled
```

## 🛠️ Complete Examples

### Example 1: Ubuntu 22.04 with GeForce 940M

```python
import llcuda

# Initialize engine (downloads 26 MB package on first run)
engine = llcuda.InferenceEngine()

# Check GPU detection
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")  # GeForce 940M
print(f"Compute: {compat['compute_capability']}")  # 5.0

# Load small model (best for 940M)
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)

# Run inference
result = engine.infer("What is machine learning?", max_tokens=100)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")  # ~15 tok/s
```

### Example 2: Google Colab with Tesla T4

```python
# In Google Colab notebook
!pip install llcuda

import llcuda

# Initialize engine (downloads 264 MB package with FlashAttention)
engine = llcuda.InferenceEngine()

# Check GPU detection
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")  # Tesla T4
print(f"Compute: {compat['compute_capability']}")  # 7.5

# Load model (can handle larger models with T4)
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)

# Run inference with FlashAttention (2x faster)
result = engine.infer("Explain deep learning", max_tokens=200)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")  # ~45 tok/s
```

### Example 3: Custom Model Loading

```python
import llcuda

engine = llcuda.InferenceEngine()

# Load custom model from HuggingFace
# For 940M: Use smaller models (1-3B params)
# For T4: Can handle larger models (up to 13B params)
engine.load_model(
    model_name="TheBloke/Llama-2-7B-GGUF",
    filename="llama-2-7b.Q4_K_M.gguf",
    gpu_layers=30,  # Adjust based on GPU (940M: 10-15, T4: 26-35)
    context_size=4096,
    silent=True
)

result = engine.infer("Write a short poem about AI", max_tokens=200)
print(result.text)
```

## 📦 Pre-configured Models

| Model | Size | VRAM | 940M Speed | T4 Speed | Best For |
|-------|------|------|------------|----------|----------|
| Gemma 3-1B | 769 MB | ~1 GB | 15 tok/s | 45 tok/s | Fast inference, chat |
| Gemma 2-2B | 1.6 GB | ~1.5 GB | 10 tok/s | 35 tok/s | Balanced quality/speed |
| Llama 3.2-3B | 2.0 GB | ~2 GB | 8 tok/s | 30 tok/s | Higher quality |
| Qwen 2.5-1.5B | 1.0 GB | ~1.2 GB | 12 tok/s | 40 tok/s | Multilingual |

**Recommended for GeForce 940M**: Gemma 3-1B, Qwen 2.5-1.5B (1-3B parameter models)

**Recommended for Tesla T4**: All models above, plus larger 7B models

## 🌐 Platform Support

### Tested Environments
- ✅ **Ubuntu 22.04** with NVIDIA GeForce 940M
- ✅ **Google Colab** with Tesla T4 GPU

### Requirements
- **Python**: 3.11+
- **CUDA**: 12.x runtime (12.8 recommended)
- **VRAM**:
  - Minimum 1GB for small models (1-3B params)
  - 8GB+ for larger models on T4

## 🔧 Troubleshooting

### Check GPU Detection

```python
import llcuda
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"Compute Capability: {compat['compute_capability']}")
```

**Expected Output (Ubuntu 22.04):**
```
GPU: GeForce 940M
Compute Capability: 5.0
```

**Expected Output (Google Colab):**
```
GPU: Tesla T4
Compute Capability: 7.5
```

### Check Installed Binaries

```python
from pathlib import Path
import llcuda

binaries_dir = Path(llcuda.__file__).parent / "binaries" / "cuda12"
print(f"Binaries installed: {binaries_dir.exists()}")
print(f"llama-server: {(binaries_dir / 'llama-server').exists()}")
```

### Force Specific Bundle (if needed)

```bash
# Force 940M binaries
export LLCUDA_FORCE_BUNDLE="940m"
python your_script.py

# Force T4 binaries
export LLCUDA_FORCE_BUNDLE="t4"
python your_script.py
```

## 📚 Documentation

- **GitHub Repository**: https://github.com/waqasm86/llcuda
- **PyPI Package**: https://pypi.org/project/llcuda/
- **Changelog**: [CHANGELOG.md](https://github.com/waqasm86/llcuda/blob/main/CHANGELOG.md)
- **Build Guide**: [BUILD_GUIDE.md](https://github.com/waqasm86/llcuda/blob/main/BUILD_GUIDE.md)

## 🐛 Bug Reports

Report issues at: https://github.com/waqasm86/llcuda/issues

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [llama.cpp](https://github.com/ggerganov/llama.cpp)
- CUDA toolkit from NVIDIA
- FlashAttention implementation from Dao et al.

---

**Supported Platforms:**
- Ubuntu 22.04 with NVIDIA GeForce 940M
- Google Colab with Tesla T4

**Version**: 1.2.2
**Release Date**: January 4, 2025
**CUDA Version**: 12.x
**License**: MIT
