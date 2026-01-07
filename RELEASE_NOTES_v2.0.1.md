# llcuda v2.0.1 Release Notes

**Release Date**: January 7, 2025
**Type**: Maintenance/Cleanup Release
**Target**: Tesla T4 GPU (SM 7.5) - Google Colab

---

## 🎯 Overview

llcuda v2.0.1 is a maintenance release focused on **project cleanup and PyPI package optimization**. This release ensures the PyPI package stays under 100 MB by excluding large binaries, which are now downloaded on first use.

---

## ✨ What's Changed

### Package Optimization
- **Excluded large binaries from PyPI wheel** (previously would have been 466+ MB)
- **PyPI package size reduced to ~70 KB** (from potential 500+ MB)
- Large binaries (llcuda_cpp.so, llcuda-binaries-cuda12-t4.tar.gz) are **downloaded on first import**
- Bootstrap mechanism handles binary downloads automatically

### Repository Cleanup (~265 MB saved)
- Removed duplicate backup files (`__init___backup.py`, `__init___pure.py`)
- Removed empty nested directory structure in `llcuda/` package
- Removed obsolete CMakeLists.txt and llcuda_py.cpp from package directory
- Removed 15+ obsolete documentation files from v1.x era
- Removed duplicate binary tarballs

### Configuration Improvements
- Updated .gitignore with comprehensive patterns to prevent large file uploads
- Added explicit exclusion patterns for *.so, *.gguf, *.tar.gz files
- Updated pyproject.toml with [tool.setuptools.exclude-package-data]
- Ensured no model files (.gguf) can be accidentally uploaded

---

## 📦 Installation

```bash
pip install llcuda==2.0.1
```

**First import** will download T4-optimized binaries (264 MB, one-time):
```python
import llcuda  # Triggers automatic binary download
```

Subsequent imports use cached binaries (~instant).

---

## 🚀 Quick Start

### Tensor API
```python
from llcuda.core import Tensor, DType

A = Tensor.zeros([2048, 2048], dtype=DType.Float16, device=0)
B = Tensor.zeros([2048, 2048], dtype=DType.Float16, device=0)
C = A @ B  # cuBLAS with Tensor Cores
```

### HTTP Server API
```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)
result = engine.infer("Hello, world!", max_tokens=50)
print(result.text)
```

---

## 📊 Package Sizes

| Component | Size | Included in PyPI? |
|-----------|------|-------------------|
| PyPI wheel | 54 KB | ✅ Yes |
| Source tarball | 67 KB | ✅ Yes |
| llcuda_cpp.so | 466 MB | ❌ No - Downloaded on first use |
| T4 binaries | 264 MB | ❌ No - Downloaded on first use |

**Total PyPI download**: ~70 KB (vs 730 MB if binaries were included)

---

## 🔧 What's NOT Changed

### Core Functionality (Unchanged)
- ✅ Native Tensor API works identically
- ✅ HTTP Server API works identically
- ✅ FlashAttention support unchanged
- ✅ CUDA Graphs optimization unchanged
- ✅ Tesla T4 targeting unchanged
- ✅ All performance benchmarks remain same

### Dependencies (Unchanged)
- Python 3.11+
- CUDA 12.x
- Tesla T4 GPU (SM 7.5)
- numpy>=1.24.0, requests>=2.31.0, huggingface_hub>=0.20.0, tqdm>=4.65.0

---

## 🐛 Known Issues

None specific to v2.0.1.

---

## ⚠️ Breaking Changes

**None** - This is a backward-compatible maintenance release.

---

## 📚 Migration from v2.0.0

No code changes needed! Simply upgrade:

```bash
pip install --upgrade llcuda
```

On first import after upgrade, binaries will be re-downloaded if needed.

---

## 🔗 Links

- **PyPI**: https://pypi.org/project/llcuda/2.0.1/
- **GitHub**: https://github.com/waqasm86/llcuda
- **Changelog**: https://github.com/waqasm86/llcuda/blob/main/CHANGELOG.md
- **Issues**: https://github.com/waqasm86/llcuda/issues

---

## 🙏 Acknowledgments

- Built on [llama.cpp](https://github.com/ggerganov/llama.cpp)
- FlashAttention from [Dao et al.](https://github.com/Dao-AILab/flash-attention)
- Designed for [Unsloth](https://github.com/unslothai/unsloth) integration

---

**Version**: 2.0.1
**Release Type**: Maintenance
**Backward Compatible**: Yes
**License**: MIT
