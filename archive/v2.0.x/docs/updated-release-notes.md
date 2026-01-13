# llcuda v2.0.1 - Tesla T4 CUDA 12 Binaries

**Release Date**: January 7, 2026
**Type**: Maintenance + CUDA Binary Release
**Target**: Tesla T4 GPU (SM 7.5) - Google Colab

---

## 🎯 Overview

llcuda v2.0.1 includes **T4-optimized CUDA 12 binaries** with FlashAttention support, plus project cleanup and PyPI package optimization. The PyPI package is only ~70 KB - CUDA binaries (140 MB) are downloaded automatically on first import.

---

## 📦 What's Included

### CUDA 12 Binaries (NEW!)
**File**: `llcuda-binaries-cuda12-t4.tar.gz` (140 MB)

**Contents**:
- **llama-server** (6.5 MB) - HTTP inference server with FlashAttention
- **libggml-cuda.so** (174 MB) - CUDA kernels optimized for Tesla T4 Tensor Cores
- **Supporting libraries** - Complete runtime dependencies

**Features**:
✅ FlashAttention 2 (2-3x faster for long contexts)
✅ Tensor Core optimization (FP16/INT8)
✅ CUDA Graphs (reduced kernel launch overhead)
✅ All quantization types (Q2_K through Q8_0)

**Build Info**:
- CUDA: 12.4/12.6
- Compute: SM 7.5 (Turing)
- llama.cpp: 0.0.7654
- GGML: 0.9.5

### Package Optimization
- **PyPI package size: ~70 KB** (binaries excluded)
- Binaries downloaded on first import (one-time, ~140 MB)
- Repository cleaned up (~265 MB saved)

---

## 📊 Performance (Tesla T4)

| Model | Speed | VRAM | Context |
|-------|-------|------|---------|
| Gemma 3-1B Q4_K_M | **45 tok/s** | 1.2 GB | 2048 |
| Llama 3.2-3B Q4_K_M | **30 tok/s** | 2.0 GB | 4096 |
| Qwen 2.5-7B Q4_K_M | **18 tok/s** | 5.0 GB | 8192 |
| Llama 3.1-8B Q4_K_M | **15 tok/s** | 5.5 GB | 8192 |

---

## 🚀 Installation

```bash
pip install llcuda
```

Binaries download automatically on first import:

```python
import llcuda  # One-time download of CUDA binaries (~140 MB)
```

---

## 🔧 Quick Start (Google Colab)

### HTTP Server API
```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

### Native Tensor API
```python
from llcuda.core import Tensor, DType

A = Tensor.zeros([2048, 2048], dtype=DType.Float16, device=0)
B = Tensor.zeros([2048, 2048], dtype=DType.Float16, device=0)
C = A @ B  # cuBLAS with Tensor Cores
```

---

## 🎯 Requirements

- **GPU**: Tesla T4 (SM 7.5)
- **Python**: 3.11+
- **CUDA**: 12.x runtime
- **Platform**: Google Colab, Kaggle, or local T4

---

## ✨ What's Changed

### New in v2.0.1
- ✅ **T4-optimized CUDA 12 binaries** with FlashAttention
- ✅ **Automatic binary download** on first import
- ✅ **PyPI package optimized** to ~70 KB (binaries excluded)
- ✅ **Repository cleanup** - removed 265 MB of duplicate/obsolete files
- ✅ **Updated .gitignore** to prevent large file uploads

### Core Functionality (Unchanged)
- ✅ Native Tensor API works identically
- ✅ HTTP Server API works identically
- ✅ FlashAttention support
- ✅ CUDA Graphs optimization
- ✅ All performance benchmarks remain same

---

## 📦 Package Sizes

| Component | Size | Included in PyPI? |
|-----------|------|-------------------|
| PyPI wheel | 54 KB | ✅ Yes |
| Source tarball | 67 KB | ✅ Yes |
| CUDA binaries (T4) | 140 MB | ❌ Downloaded on first use |

**Total PyPI download**: ~70 KB
**First import download**: ~140 MB (one-time, cached)

---

## 🔐 Checksums

```
SHA256 (llcuda-binaries-cuda12-t4.tar.gz):
54bb904f213b38aec4a2c85baa2fab8256200966aa994bb9fa4a77640d699aa4
```

---

## ⚠️ Breaking Changes

**None** - This is a backward-compatible maintenance release.

---

## 📚 Migration from v2.0.0

No code changes needed! Simply upgrade:

```bash
pip install --upgrade llcuda
```

On first import after upgrade, binaries will be downloaded automatically.

---

## 🔗 Links

- **PyPI**: https://pypi.org/project/llcuda/
- **GitHub**: https://github.com/waqasm86/llcuda
- **Documentation**: https://github.com/waqasm86/llcuda#readme
- **Issues**: https://github.com/waqasm86/llcuda/issues

---

## 🙏 Acknowledgments

- Built on [llama.cpp](https://github.com/ggerganov/llama.cpp)
- FlashAttention from [Dao-AILab](https://github.com/Dao-AILab/flash-attention)
- Designed for [Unsloth](https://github.com/unslothai/unsloth) integration

---

**Version**: 2.0.1
**Release Type**: Maintenance + Binary Release
**Backward Compatible**: Yes
**License**: MIT
