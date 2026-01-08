# llcuda v2.0.3 Release Notes

**Release Date:** January 8, 2026
**PyPI:** https://pypi.org/project/llcuda/2.0.3/
**GitHub:** https://github.com/waqasm86/llcuda/releases/tag/v2.0.3

---

## 🎉 Major Change: Bundled Binaries (PyTorch-Style Distribution)

llcuda v2.0.3 fundamentally changes how binaries are distributed - **following PyTorch's approach**.

### **What Changed**

| Aspect | v2.0.2 (Old) | v2.0.3 (New) |
|--------|--------------|--------------|
| **Installation** | `pip install` (~1 MB) + import downloads (~266 MB) | `pip install` (~270 MB) - everything included |
| **First Import** | Downloads binaries from GitHub | Instant - no downloads |
| **Offline Support** | ❌ Requires internet on first import | ✅ Works offline after install |
| **Distribution** | Two-stage (PyPI + GitHub) | Single-stage (PyPI only) |

### **User Experience**

**Old (v2.0.2):**
```bash
$ pip install llcuda           # ~1 MB, quick
$ python
>>> import llcuda              # Downloads 266 MB from GitHub
🎯 llcuda v2.0 First-Time Setup
📦 Downloading T4-optimized binaries (264 MB)...
```

**New (v2.0.3):**
```bash
$ pip install llcuda           # ~270 MB, standard
$ python
>>> import llcuda              # Instant, no downloads! ✅
```

---

## ✨ What's New

### 🎁 **Bundled Binaries**
- All CUDA binaries now included in PyPI wheel
- No runtime downloads from GitHub
- Works offline after installation
- Standard PyPI experience like PyTorch/TensorFlow

### 📦 **Package Structure**
- **PyPI wheel size**: ~270 MB (includes all binaries)
- **Git repo size**: ~1 MB (code only, binaries in .gitignore)
- **GitHub Release**: Binary tar.gz for manual download (optional)

### 🚀 **Improved Bootstrap**
- No longer downloads binaries on import
- Simply verifies binaries exist in package
- Better error messages if binaries missing

### 🔧 **Better Build Process**
- New `prepare_binaries.py` script
- Automated extraction of binaries into package
- Cross-platform (Windows/Linux/macOS)

---

## 🐛 Bug Fixes

### **Fixed: 404 Errors on Import**
- **Issue**: v2.0.2 tried to download `llcuda-binaries-cuda12-t4.tar.gz` (404)
- **Fix**: Binaries now bundled, no downloads needed

### **Fixed: GitHub API Rate Limits**
- **Issue**: Heavy users hit GitHub download limits
- **Fix**: No GitHub downloads, pure PyPI

### **Fixed: Two-Stage Installation Confusion**
- **Issue**: Users confused by pip install + import download
- **Fix**: Standard one-step pip installation

---

## 📝 Documentation Updates

- ✅ Updated README.md for v2.0.3
- ✅ New BUNDLED_BINARIES_GUIDE.md
- ✅ Updated CHANGELOG.md
- ✅ Clarified .gitignore for bundled approach
- ✅ Updated all version references

---

## 🔄 Migration from v2.0.2

Upgrading is seamless:

```bash
pip install --upgrade llcuda
```

**What happens:**
1. Old package (~1 MB) uninstalled
2. New package (~270 MB) downloaded
3. Binaries ready immediately
4. Old cached downloads can be removed:
   ```bash
   rm -rf ~/.cache/llcuda/llcuda-binaries-cuda12-t4*.tar.gz
   ```

---

## 📊 Package Comparison

| Package | Size | Distribution Method |
|---------|------|---------------------|
| **torch (CUDA 12.1)** | ~2.5 GB | Bundled binaries |
| **tensorflow-gpu** | ~500 MB | Bundled binaries |
| **jax[cuda12]** | ~400 MB | Bundled binaries |
| **llcuda v2.0.3** | ~270 MB | Bundled binaries ✅ |
| **llcuda v2.0.2** | ~1 MB + 266 MB | Download on import ❌ |

llcuda now follows industry standard! 🎯

---

## 🛠️ Technical Details

### **Binaries Included**

```
llcuda/binaries/cuda12/
├── llama-server (6.5 MB)    # Main inference server
├── llama-cli (4.2 MB)
├── llama-bench (581 KB)
├── llama-quantize (434 KB)
└── llama-embedding (3.3 MB)

llcuda/lib/
├── libggml-cuda.so (221 MB)  # FlashAttention enabled
├── libggml-base.so
├── libggml-cpu.so
├── libllama.so
└── libmtmd.so
```

**Total:** ~266 MB of binaries + libraries

### **Configuration Files Updated**

- `pyproject.toml`: Package data includes binaries
- `MANIFEST.in`: Includes binaries in wheel
- `bootstrap.py`: Verification only, no downloads
- `.gitignore`: Excludes binaries from git

---

## 🎯 Why This Change?

### **Problems with Download-on-Import**

1. ❌ Violates PyPI conventions
2. ❌ Confusing two-stage installation
3. ❌ GitHub rate limits affect users
4. ❌ Requires internet on first import
5. ❌ Inconsistent with other ML packages

### **Benefits of Bundled Binaries**

1. ✅ Standard PyPI experience
2. ✅ Single `pip install` command
3. ✅ Works offline after install
4. ✅ No GitHub dependencies
5. ✅ Consistent with PyTorch/TensorFlow/JAX

---

## 📦 Installation & Usage

### **Installation**

```bash
pip install llcuda
```

Downloads ~270 MB (one time), includes all binaries.

### **Quick Start**

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")
result = engine.infer("Hello!", max_tokens=50)
print(result.text)
```

**No setup, no downloads, just works!** ✨

---

## 🔗 Links

- **PyPI**: https://pypi.org/project/llcuda/
- **GitHub**: https://github.com/waqasm86/llcuda
- **Releases**: https://github.com/waqasm86/llcuda/releases
- **Documentation**: https://github.com/waqasm86/llcuda#readme

---

## 🙏 Credits

Built with:
- llama.cpp (GGUF inference)
- CUDA 12.x (NVIDIA)
- Python 3.11+

Optimized for Tesla T4 on Google Colab.

---

**Full Changelog**: https://github.com/waqasm86/llcuda/blob/main/CHANGELOG.md
