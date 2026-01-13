# llcuda v2.0.2 - Bundled Binaries Guide

## Overview

Starting with **llcuda v2.0.2**, CUDA binaries are **bundled directly in the PyPI package**, similar to how PyTorch distributes CUDA binaries. This means:

✅ **One command install**: `pip install llcuda` downloads everything (~270 MB)
✅ **No runtime downloads**: Import works instantly without downloading binaries
✅ **Offline ready**: Package works without internet after installation
✅ **PyPI standard**: Follows established patterns from PyTorch, TensorFlow, etc.

---

## What Changed from v2.0.1

### **Old Approach (v2.0.1)**
```bash
pip install llcuda           # Only ~1 MB downloaded
import llcuda                # Downloads 266 MB binaries from GitHub
```

**Problems:**
- ❌ Two-stage installation confusing for users
- ❌ GitHub rate limits affect installations
- ❌ Fails without internet during import
- ❌ Not standard PyPI practice

### **New Approach (v2.0.2)**
```bash
pip install llcuda           # Full 270 MB package downloaded
import llcuda                # Instant, no downloads
```

**Benefits:**
- ✅ Standard PyPI experience like PyTorch
- ✅ No GitHub dependencies
- ✅ Works offline after installation
- ✅ Cleaner user experience

---

## Package Structure

```
llcuda/
├── llcuda/
│   ├── __init__.py
│   ├── server.py
│   ├── _internal/
│   │   └── bootstrap.py        # Now just verifies binaries exist
│   ├── binaries/
│   │   └── cuda12/
│   │       ├── llama-server    # 6.5 MB - Main inference server
│   │       ├── llama-cli       # 4.2 MB
│   │       ├── llama-bench     # 581 KB
│   │       ├── llama-quantize  # 434 KB
│   │       └── llama-embedding # 3.3 MB
│   └── lib/
│       ├── libggml-cuda.so     # 221 MB - FlashAttention enabled
│       ├── libggml-base.so     # Base GGML
│       ├── libggml-cpu.so      # CPU fallback
│       ├── libllama.so         # llama.cpp library
│       └── libmtmd.so          # Multi-threading
├── pyproject.toml              # Includes binaries in package data
├── MANIFEST.in                 # Includes binaries in wheel
└── prepare_binaries.py         # Script to extract binaries
```

**Total Size**: ~270 MB wheel file

---

## Building and Publishing

### **Step 1: Prepare Binaries**

Extract CUDA binaries into the package structure:

```bash
cd llcuda
python prepare_binaries.py
```

**What it does:**
1. Extracts `llcuda-binaries-cuda12-t4-v2.0.2.tar.gz` from `build-artifacts/`
2. Copies binaries to `llcuda/binaries/cuda12/`
3. Copies libraries to `llcuda/lib/`
4. Verifies installation

**Output:**
```
✅ Binaries prepared successfully!

Binaries location: llcuda/binaries/cuda12/
Libraries location: llcuda/lib/
Total size: 266.0 MB
```

### **Step 2: Build the Package**

```bash
# Install build tools
pip install build twine

# Build wheel and source distribution
python -m build
```

**Expected output:**
```
Building wheel...
Successfully built llcuda-2.0.2-py3-none-any.whl (270 MB)
Successfully built llcuda-2.0.2.tar.gz (268 MB)
```

### **Step 3: Verify the Wheel**

```bash
# Check wheel size
ls -lh dist/

# Expected:
# llcuda-2.0.2-py3-none-any.whl  (~270 MB)
# llcuda-2.0.2.tar.gz            (~268 MB)

# Inspect wheel contents
unzip -l dist/llcuda-2.0.2-py3-none-any.whl | grep -E "(llama-server|libggml-cuda)"
```

**Should show:**
```
llcuda/binaries/cuda12/llama-server
llcuda/lib/libggml-cuda.so
```

### **Step 4: Test Locally**

```bash
# Install from local wheel
pip install dist/llcuda-2.0.2-py3-none-any.whl --force-reinstall

# Test import (should be instant)
python -c "import llcuda; print(llcuda.__version__)"
```

**Expected:**
```
2.0.2
```

No downloads, no setup messages - just instant import!

### **Step 5: Upload to PyPI**

```bash
# Upload to Test PyPI first (recommended)
python -m twine upload --repository testpypi dist/*

# Test install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ llcuda

# If all works, upload to real PyPI
python -m twine upload dist/*
```

---

## Configuration Files

### **pyproject.toml**

```toml
[tool.setuptools.package-data]
llcuda = [
    "*.py",
    "py.typed",
    "binaries/**/*",      # ← Include binaries
    "lib/**/*.so*",       # ← Include libraries
]

[tool.setuptools.exclude-package-data]
llcuda = [
    "*.tar.gz",           # Exclude source archives
    "*.gguf",             # Exclude models (downloaded on demand)
    "models/**/*",        # Exclude models directory
]
```

### **MANIFEST.in**

```
# INCLUDE binaries and libraries
recursive-include llcuda/binaries *
recursive-include llcuda/lib *.so*

# EXCLUDE models and build artifacts
global-exclude *.gguf
global-exclude *.tar.gz
recursive-exclude llcuda/models *
```

---

## Bootstrap Changes

The `bootstrap.py` no longer downloads binaries. It just verifies they exist:

```python
def bootstrap() -> None:
    """Verify binaries are bundled in package"""
    llama_server = BINARIES_DIR / "cuda12" / "llama-server"

    if llama_server.exists():
        return  # ✅ Binaries found

    # ❌ Missing binaries = package integrity issue
    raise RuntimeError(
        "llcuda binaries not found. "
        "Please reinstall: pip install --no-cache-dir --upgrade llcuda"
    )
```

---

## User Experience

### **Installation**

```bash
pip install llcuda
```

**Downloads:**
- `llcuda-2.0.2-py3-none-any.whl` (~270 MB)

**Time:**
- ~30-60 seconds depending on connection speed
- Similar to `pip install torch` with CUDA

### **First Import**

```python
import llcuda
```

**Result:**
- ✅ Instant import (no downloads)
- ✅ No setup messages
- ✅ Binaries ready to use

### **Using llcuda**

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")  # Downloads model (~700 MB, one-time)
result = engine.infer("Hello!", max_tokens=50)
print(result.text)
```

---

## Comparison with Other Packages

| Package | Size | Approach |
|---------|------|----------|
| **torch (CUDA)** | ~2.5 GB | Binaries in wheel |
| **tensorflow-gpu** | ~500 MB | Binaries in wheel |
| **jax[cuda]** | ~400 MB | Binaries in wheel |
| **llcuda v2.0.2** | ~270 MB | Binaries in wheel ✅ |
| **llcuda v2.0.1** | ~1 MB | Downloads at import ❌ |

llcuda now follows industry standard practice! 🎉

---

## PyPI Upload Limits

PyPI has a **100 MB default upload limit**, but provides exceptions for packages with binaries:

1. **Request Limit Increase**: https://pypi.org/help/#file-size-limit
2. **Upload Large Files**: https://twine.readthedocs.io/

**Note**: Many ML packages (PyTorch, TensorFlow) have 500MB-2GB wheels on PyPI. llcuda's 270 MB is well within acceptable range.

---

## Troubleshooting

### **Build Error: "Binaries not found"**

**Solution:**
```bash
python prepare_binaries.py
python -m build
```

### **Import Error: "Binaries not found"**

**Solution:**
```bash
pip install --no-cache-dir --upgrade llcuda
```

### **Large Wheel Warning**

This is expected! CUDA binaries are large. PyTorch's CUDA wheel is 2.5 GB.

---

## Migration from v2.0.1

Users on v2.0.1 can upgrade seamlessly:

```bash
pip install --upgrade llcuda
```

**What happens:**
1. Old 1 MB package uninstalled
2. New 270 MB package downloaded
3. Binaries ready immediately
4. Cached GitHub downloads can be deleted

**Cleanup old downloads:**
```bash
rm -rf ~/.cache/llcuda/llcuda-binaries-cuda12-t4*.tar.gz
```

---

## Summary

✅ **Binaries bundled in PyPI package** (like PyTorch)
✅ **One-step installation** with pip
✅ **Instant imports** without downloads
✅ **Offline ready** after installation
✅ **Industry standard** distribution method

**Total package size**: ~270 MB
**User experience**: Seamless, just like PyTorch! 🚀
