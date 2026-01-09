# llcuda v1.2.2 - CUDA 12 Support for GeForce 940M & Tesla T4

Official CUDA 12 binary release optimized for **Ubuntu 22.04 with GeForce 940M** and **Google Colab with Tesla T4**.

## 🎉 What's New

### GPU-Specific Binary Bundles
- ✅ **Automatic GPU Detection**: Detects your GPU and downloads the right binaries
- ✅ **GeForce 940M Package (26 MB)**: Optimized for Ubuntu 22.04 local systems
- ✅ **Tesla T4 Package (264 MB)**: FlashAttention support for 2x faster inference in Google Colab
- ✅ **Smart Selection**: Auto-selects based on detected GPU

### FlashAttention Support
- 🚀 **2x Faster Inference** on Tesla T4 in Google Colab
- ⚡ **Automatic**: Enabled when T4 is detected

### Critical Bug Fixes
- ✅ Fixed `AttributeError: 'NoneType' object has no attribute 'read'` in Google Colab silent mode
- ✅ Improved library path detection
- ✅ Better error messages and logging

## 📦 Binary Packages

### 🎮 GeForce 940M (26 MB) - Ubuntu 22.04
**File:** `llcuda-binaries-cuda12-940m.tar.gz`

**Target Platform:**
- **OS**: Ubuntu 22.04
- **GPU**: NVIDIA GeForce 940M
- **Compute Capability**: 5.0 (Maxwell architecture)

**Optimizations:**
- Forced cuBLAS for better compatibility
- CUDA graphs enabled
- Optimized for limited VRAM (~1-2GB)

**Performance:**
- **Speed**: 10-20 tokens/sec
- **Best For**: 1-3B parameter models (Q4_K_M quantization)
- **GPU Layers**: 10-15
- **Context**: 512-1024 tokens

**Example:**
```
Model: Gemma 3-1B Q4_K_M (769 MB)
Speed: 15 tok/s
Latency: 67ms per token
VRAM: ~1 GB
Platform: Ubuntu 22.04
```

### ☁️ Tesla T4 (264 MB) - Google Colab
**File:** `llcuda-binaries-cuda12-t4.tar.gz`

**Target Platform:**
- **OS**: Google Colab notebooks
- **GPU**: NVIDIA Tesla T4
- **Compute Capability**: 7.5 (Turing architecture)

**Optimizations:**
- **FlashAttention enabled** (2x faster than standard)
- Tensor cores utilized
- CUDA graphs enabled
- Custom CUDA kernels

**Performance:**
- **Speed**: 25-60 tokens/sec with FlashAttention
- **Best For**: 1-13B parameter models (Q4_K_M/Q5_K_M quantization)
- **GPU Layers**: 26-35
- **Context**: 2048-8192 tokens

**Example:**
```
Model: Gemma 3-1B Q4_K_M (769 MB)
Speed: 45 tok/s
Latency: 22ms per token
VRAM: ~1 GB
FlashAttention: Enabled
Platform: Google Colab
```

## 📥 Installation

### Ubuntu 22.04 (GeForce 940M)

```bash
pip install llcuda
```

On first import:
```python
import llcuda
# Detects: GeForce 940M (Compute 5.0)
# Downloads: llcuda-binaries-cuda12-940m.tar.gz (26 MB)
# Ready to use!
```

### Google Colab (Tesla T4)

```python
!pip install llcuda

import llcuda
# Detects: Tesla T4 (Compute 7.5)
# Downloads: llcuda-binaries-cuda12-t4.tar.gz (264 MB)
# FlashAttention enabled!
```

## 🚀 Quick Start

### Ubuntu 22.04 Example

```python
import llcuda

# Initialize engine (auto-downloads 940M binaries on first run)
engine = llcuda.InferenceEngine()

# Load model optimized for 940M
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)

# Run inference
result = engine.infer("What is artificial intelligence?", max_tokens=100)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")  # ~15 tok/s
```

### Google Colab Example

```python
!pip install llcuda

import llcuda

# Initialize engine (auto-downloads T4 binaries with FlashAttention)
engine = llcuda.InferenceEngine()

# Load model - T4 can handle larger contexts
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)

# Run inference with FlashAttention (2x faster)
result = engine.infer("Explain deep learning", max_tokens=200)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")  # ~45 tok/s
```

## 🎮 Supported Platforms

| Platform | GPU | Compute Cap | Package | Download Size | Features |
|----------|-----|-------------|---------|---------------|----------|
| **Ubuntu 22.04** | GeForce 940M | 5.0 | 940M | 26 MB | cuBLAS optimized |
| **Google Colab** | Tesla T4 | 7.5 | T4 | 264 MB | **FlashAttention** (2x faster) |

## 📊 Performance Comparison

### Gemma 3-1B Q4_K_M (769 MB model)

| Platform | GPU | Speed | Latency | GPU Layers | FlashAttention |
|----------|-----|-------|---------|------------|----------------|
| Ubuntu 22.04 | GeForce 940M | 15 tok/s | 67ms | 10-15 | ❌ |
| Google Colab | Tesla T4 | 45 tok/s | 22ms | 26-35 | ✅ |

### Llama 3.2-3B Q4_K_M (2 GB model)

| Platform | GPU | Speed | Latency | GPU Layers | FlashAttention |
|----------|-----|-------|---------|------------|----------------|
| Ubuntu 22.04 | GeForce 940M | 8-12 tok/s | 100ms | 10-12 | ❌ |
| Google Colab | Tesla T4 | 30 tok/s | 33ms | 26-30 | ✅ |

## 📋 Requirements

- **Python**: 3.11+
- **CUDA**: 12.x runtime (12.8 recommended)
- **Platform**:
  - Ubuntu 22.04 with GeForce 940M
  - Google Colab with Tesla T4
- **VRAM**:
  - Minimum 1GB for small models (1-3B params)
  - 2GB+ recommended for larger models on T4

## 🔧 What's Changed

### Added
- GPU-specific binary bundles for GeForce 940M and Tesla T4
- Automatic GPU detection using nvidia-smi
- FlashAttention support for Tesla T4 (2x faster inference)
- Smart binary selection based on detected GPU
- Platform detection for local/Colab environments

### Fixed
- **Critical**: Fixed `AttributeError` when reading stderr in silent mode (Google Colab)
- Library path detection for different build configurations
- Better error messages for missing binaries

### Changed
- Bootstrap now downloads GPU-specific binaries (26MB for 940M, 264MB for T4)
- Improved library path configuration
- Updated documentation to focus on supported platforms

## 📚 Documentation

- **GitHub**: https://github.com/waqasm86/llcuda
- **PyPI**: https://pypi.org/project/llcuda/
- **Changelog**: [CHANGELOG.md](https://github.com/waqasm86/llcuda/blob/main/CHANGELOG.md)

## 🔄 Upgrading from v1.1.x

**No breaking changes!** Simply upgrade:

```bash
pip install --upgrade llcuda
```

On first import after upgrade:
- Old binaries replaced with GPU-specific ones
- Appropriate package downloaded (940M or T4)
- All existing code continues to work

## 💡 Troubleshooting

### Check Your GPU
```python
import llcuda
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"Compute: {compat['compute_capability']}")
```

**Expected on Ubuntu 22.04:**
```
GPU: GeForce 940M
Compute: 5.0
```

**Expected on Google Colab:**
```
GPU: Tesla T4
Compute: 7.5
```

## 🙏 Acknowledgments

- Built on [llama.cpp](https://github.com/ggerganov/llama.cpp)
- CUDA toolkit from NVIDIA
- FlashAttention implementation from Dao et al.

## 📄 License

MIT License

---

**Supported Platforms:**
- Ubuntu 22.04 with NVIDIA GeForce 940M
- Google Colab with Tesla T4

**Full Changelog**: https://github.com/waqasm86/llcuda/compare/v1.1.9...v1.2.2

**Download from PyPI**: https://pypi.org/project/llcuda/1.2.2/
