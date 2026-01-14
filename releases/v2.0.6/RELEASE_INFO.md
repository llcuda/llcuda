# llcuda v2.0.6 Release Information

**Release Date:** January 10, 2026
**Distribution:** GitHub-Only

---

## Release Assets

### Binary Package

**File:** `llcuda-binaries-cuda12-t4-v2.0.6.tar.gz`
**Size:** 266 MB (278,892,158 bytes)
**SHA256:** `5a27d2e1a73ae3d2f1d2ba8cf557b76f54200208c8df269b1bd0d9ee176bb49d`
**Download:** https://github.com/llcuda/llcuda/releases/download/v2.0.6/llcuda-binaries-cuda12-t4-v2.0.6.tar.gz

**Contents:**
- llama-server (6.5 MB) - Main inference server
- llama-cli (4.2 MB) - Command-line interface
- llama-embedding (3.3 MB) - Embedding generator
- llama-bench (581 KB) - Benchmarking tool
- llama-quantize (434 KB) - Model quantization utility
- libggml-cuda.so (221 MB) - CUDA kernels with FlashAttention
- libllama.so (2.9 MB) - Llama core library
- libggml-cpu.so, libggml-base.so, libmtmd.so - Supporting libraries

### Python Packages

**Wheel:** `llcuda-2.0.6-py3-none-any.whl` (53 KB)
**Source:** `llcuda-2.0.6.tar.gz` (65 KB)

---

## Build Information

- **CUDA Version:** 12.x
- **Target GPU:** Tesla T4 (SM 7.5)
- **Build Date:** January 7, 2026
- **llama.cpp Version:** 0.0.7662
- **GGML Version:** 0.9.5
- **Platform:** Google Colab Tesla T4

### Features Enabled

- ✅ FlashAttention (GGML_CUDA_FA=ON)
- ✅ CUDA Graphs (GGML_CUDA_GRAPHS=ON)
- ✅ Tensor Core optimization
- ✅ All quantization types (Q2_K through Q8_0)
- ✅ Shared library support

---

## Installation

### Direct from GitHub (Recommended)

```bash
pip install git+https://github.com/llcuda/llcuda.git
```

Binaries will auto-download from this release on first import.

### Manual Binary Installation

```bash
# Download binary package
wget https://github.com/llcuda/llcuda/releases/download/v2.0.6/llcuda-binaries-cuda12-t4-v2.0.6.tar.gz

# Verify checksum
echo "5a27d2e1a73ae3d2f1d2ba8cf557b76f54200208c8df269b1bd0d9ee176bb49d  llcuda-binaries-cuda12-t4-v2.0.6.tar.gz" | sha256sum -c

# Extract
tar -xzf llcuda-binaries-cuda12-t4-v2.0.6.tar.gz -C ~/.cache/llcuda/
```

---

## Changelog

### v2.0.6 Changes

- **GitHub-only distribution** - Removed PyPI dependency
- Updated bootstrap to use GitHub Releases v2.0.6
- Added comprehensive GitHub installation guide
- Removed all PyPI references from code
- Improved version checking (uses GitHub API)
- Created automated build scripts
- Enhanced documentation

### Binary Compatibility

v2.0.6 uses the **same CUDA binaries as v2.0.3** (proven stable).

---

## Performance (Tesla T4)

| Model | Quantization | Speed | VRAM | Context |
|-------|--------------|-------|------|---------|
| Gemma 3-1B | Q4_K_M | 45 tok/s | 1.2 GB | 2048 |
| Llama 3.2-3B | Q4_K_M | 30 tok/s | 2.0 GB | 4096 |
| Qwen 2.5-7B | Q4_K_M | 18 tok/s | 5.0 GB | 8192 |
| Llama 3.1-8B | Q4_K_M | 15 tok/s | 5.5 GB | 8192 |

---

## Links

- **Repository:** https://github.com/llcuda/llcuda
- **Installation Guide:** https://github.com/llcuda/llcuda/blob/main/GITHUB_INSTALL_GUIDE.md
- **Issues:** https://github.com/llcuda/llcuda/issues

---

**Note:** The binary package is not stored in git due to its size (266 MB).
Only the checksum file is tracked. Download from GitHub Releases.
