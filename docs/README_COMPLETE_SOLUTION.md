# Complete Solution: CUDA 12 Binaries for llcuda

## Executive Summary

This repository contains a complete solution for building and integrating CUDA 12 executables into your llcuda Python package, supporting both:
- **NVIDIA GeForce 940M** (Compute Capability 5.0) - Your local Xubuntu 22 system
- **NVIDIA Tesla T4** (Compute Capability 7.5) - Google Colab environment

## What Was Done

### 1. Bug Fixed ✅

**Issue in Google Colab:**
```
AttributeError: 'NoneType' object has no attribute 'read'
```

**Location:** `llcuda/llcuda/server.py:553`

**Fix Applied:** Added null check before reading stderr:
```python
if self.server_process.stderr is not None:
    stderr = self.server_process.stderr.read().decode("utf-8", errors="ignore")
else:
    raise RuntimeError("llama-server died. Run with silent=False for details.")
```

### 2. Complete Build System Created ✅

Created multiple build scripts with full CMake configurations:

#### A. Unified Build Script
- **[BUILD_AND_INTEGRATE.sh](BUILD_AND_INTEGRATE.sh)** - Smart script that:
  - Auto-detects your GPU
  - Guides through CMake configuration
  - Copies binaries to correct llcuda directories
  - Verifies integration
  - Creates test scripts

#### B. GPU-Specific Scripts
- **[build_cuda12_geforce940m.sh](build_cuda12_geforce940m.sh)** - Detailed guide for 940M
- **[build_cuda12_tesla_t4_colab.sh](build_cuda12_tesla_t4_colab.sh)** - Automated Colab build
- **[build_cuda12_unified.sh](build_cuda12_unified.sh)** - Multi-target build automation

### 3. Comprehensive Documentation ✅

Created detailed guides:

#### A. Build Guide
- **[BUILD_GUIDE.md](BUILD_GUIDE.md)** - Complete documentation:
  - Manual build instructions
  - CMake options explained
  - Troubleshooting section
  - Performance recommendations
  - 400+ lines of detailed guidance

#### B. Integration Guide
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Integration flow:
  - How llcuda finds llama-server
  - Path detection priority order
  - Directory structure requirements
  - Verification steps
  - Debugging procedures

#### C. Quick Start
- **[QUICK_START.md](QUICK_START.md)** - TL;DR version

---

## File Overview

### Build Scripts
| File | Purpose | Usage |
|------|---------|-------|
| `BUILD_AND_INTEGRATE.sh` | Complete build & integration | `./BUILD_AND_INTEGRATE.sh 940m` |
| `build_cuda12_geforce940m.sh` | Guide for GeForce 940M | Guide-style (read & follow) |
| `build_cuda12_tesla_t4_colab.sh` | Automated Colab build | `bash build_cuda12_tesla_t4_colab.sh` |
| `build_cuda12_unified.sh` | Multi-target builder | `./build_cuda12_unified.sh auto` |

### Documentation
| File | Content |
|------|---------|
| `BUILD_GUIDE.md` | Comprehensive build & troubleshooting guide |
| `INTEGRATION_GUIDE.md` | How llcuda detects and runs llama-server |
| `QUICK_START.md` | Fast reference guide |
| `README_COMPLETE_SOLUTION.md` | This file - overview of everything |

### Modified Source Files
| File | What Changed |
|------|--------------|
| `llcuda/llcuda/server.py:553` | Fixed stderr null check bug |

---

## How llcuda Detects llama-server

### Detection Flow (Priority Order)

1. **LLAMA_SERVER_PATH environment variable**
   - Manually set path takes highest priority
   - `export LLAMA_SERVER_PATH="/path/to/llama-server"`

2. **Package binaries directory** ⭐ PRIMARY PATH
   - `llcuda/llcuda/binaries/cuda12/llama-server`
   - This is where you copy your built binary
   - Auto-configured by `__init__.py`

3. **LLAMA_CPP_DIR environment variable**
   - `export LLAMA_CPP_DIR="/path/to/llama.cpp"`
   - Looks for `${LLAMA_CPP_DIR}/bin/llama-server`

4. **Cache directory** (for bootstrap downloads)
   - `~/.cache/llcuda/bin/llama-server`
   - `/content/.cache/llcuda/llama-server` (Colab)

5. **Project-specific paths**
   - Hardcoded paths for development

6. **System paths**
   - `/usr/local/bin/llama-server`
   - `/usr/bin/llama-server`

### Auto-Configuration on Import

When you `import llcuda`, it automatically:

```python
# llcuda/__init__.py
_LLCUDA_DIR = Path(__file__).parent
_BIN_DIR = _LLCUDA_DIR / "binaries" / "cuda12"
_LIB_DIR = _LLCUDA_DIR / "lib"

# Auto-configure LD_LIBRARY_PATH
if _LIB_DIR.exists():
    os.environ["LD_LIBRARY_PATH"] = f"{_LIB_DIR}:{existing}"

# Auto-configure LLAMA_SERVER_PATH
if (_BIN_DIR / "llama-server").exists():
    os.environ["LLAMA_SERVER_PATH"] = str(_BIN_DIR / "llama-server")
```

---

## Quick Start Guide

### For Xubuntu 22 (GeForce 940M)

```bash
# Step 1: Navigate to project
cd /media/waqasm86/External1/Project-Nvidia

# Step 2: Run integration script
./BUILD_AND_INTEGRATE.sh 940m

# Step 3: Follow prompts to run CMake commands
# (Script will guide you through each step)

# Step 4: Test integration
python3 test_llcuda_integration.py

# Step 5: Install llcuda
cd llcuda
pip install -e .
```

### For Google Colab (Tesla T4)

```bash
# In a Colab cell:
!git clone https://github.com/ggml-org/llama.cpp
!bash build_cuda12_tesla_t4_colab.sh

# This will:
# - Build llama.cpp for T4
# - Create tar.gz package
# - Auto-download for you
```

### Manual CMake Commands

#### GeForce 940M
```bash
cd /media/waqasm86/External1/Project-Nvidia/llama.cpp

cmake -B build_cuda12_940m \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="50" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA_FORCE_CUBLAS=ON \
    -DGGML_CUDA_FA=OFF \
    -DGGML_CUDA_GRAPHS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DBUILD_SHARED_LIBS=ON

cmake --build build_cuda12_940m --config Release -j$(nproc)
```

#### Tesla T4
```bash
cmake -B build_cuda12_t4 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DBUILD_SHARED_LIBS=ON

cmake --build build_cuda12_t4 --config Release -j$(nproc)
```

---

## Integration Steps (Manual)

After building with CMake:

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda

# 1. Create directories
mkdir -p binaries/cuda12
mkdir -p lib

# 2. Copy binaries
cp ../../llama.cpp/build_cuda12_940m/bin/llama-server binaries/cuda12/
cp ../../llama.cpp/build_cuda12_940m/bin/llama-cli binaries/cuda12/
chmod +x binaries/cuda12/*

# 3. Copy libraries
cp ../../llama.cpp/build_cuda12_940m/bin/*.so* lib/

# 4. Verify
ls -lh binaries/cuda12/
ls -lh lib/

# 5. Test
export LD_LIBRARY_PATH="$(pwd)/lib:$LD_LIBRARY_PATH"
./binaries/cuda12/llama-server --help
```

---

## Testing

### Integration Test

```bash
python3 /media/waqasm86/External1/Project-Nvidia/test_llcuda_integration.py
```

Expected output:
```
[1/5] Testing import...
  ✓ llcuda v1.2.2 imported successfully

[2/5] Testing GPU detection...
  Platform:     local
  GPU:          NVIDIA GeForce 940M
  Compute Cap:  5.0
  Compatible:   True

[3/5] Testing llama-server detection...
  ✓ Found: /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/binaries/cuda12/llama-server

[4/5] Testing library detection...
  ✓ Found 15 shared libraries

[5/5] Testing llama-server execution...
  ✓ llama-server executes successfully

Integration Test Complete!
```

### Full Workflow Test

```python
import os
import llcuda

# Auto-configured by __init__.py, but explicit for clarity:
os.environ['LLAMA_SERVER_PATH'] = '/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/binaries/cuda12/llama-server'
os.environ['LD_LIBRARY_PATH'] = '/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/lib'

# Create engine
engine = llcuda.InferenceEngine()

# Load model (downloads on first use)
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    gpu_layers=15,  # Adjust for your GPU
    ctx_size=1024,
    silent=False
)

# Run inference
result = engine.infer("What is artificial intelligence?", max_tokens=100)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")
```

---

## Key Differences: GeForce 940M vs Tesla T4

| Feature | GeForce 940M | Tesla T4 |
|---------|--------------|----------|
| **Compute Capability** | 5.0 | 7.5 |
| **Architecture** | Maxwell | Turing |
| **VRAM** | ~1GB | ~15GB |
| **CMake Arch** | `50` | `75` |
| **FlashAttention** | OFF (not supported) | ON (2x faster) |
| **Force cuBLAS** | ON (better compatibility) | OFF (custom kernels) |
| **Recommended GPU Layers** | 10-15 | 26-35 |
| **Recommended Models** | 1-3B params | 1-13B params |
| **Build Time** | 10-30 min | 5-15 min |

---

## Troubleshooting

### Common Issues

1. **llama-server not found**
   - Check: `ls llcuda/llcuda/binaries/cuda12/llama-server`
   - Fix: Run `BUILD_AND_INTEGRATE.sh` or copy manually

2. **Shared library not found**
   - Check: `echo $LD_LIBRARY_PATH`
   - Fix: `export LD_LIBRARY_PATH="/path/to/llcuda/lib:$LD_LIBRARY_PATH"`

3. **CUDA error: no kernel image available**
   - Cause: Wrong compute capability binary
   - Fix: Rebuild with correct `-DCMAKE_CUDA_ARCHITECTURES`

4. **Server crashes immediately**
   - Debug: Run `./llama-server --help` manually
   - Check: `ldd llama-server | grep "not found"`

See [BUILD_GUIDE.md](BUILD_GUIDE.md) for detailed troubleshooting.

---

## Performance Recommendations

### GeForce 940M (1GB VRAM)

```python
engine.load_model(
    "model.gguf",
    gpu_layers=12,      # Keep low due to VRAM
    ctx_size=512,       # Small context
)
```

Recommended:
- Models: 1-3B parameters
- Quantization: Q4_K_M
- Expected: 10-20 tokens/sec

### Tesla T4 (15GB VRAM)

```python
engine.load_model(
    "model.gguf",
    gpu_layers=30,      # Can use more layers
    ctx_size=4096,      # Larger context
)
```

Recommended:
- Models: 1-13B parameters
- Quantization: Q4_K_M or Q5_K_M
- Expected: 25-60 tokens/sec

---

## File Checksums (After Integration)

Verify your integration:

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda

# Check binary size (should be ~150-200 MB)
ls -lh binaries/cuda12/llama-server

# Count libraries (should have 10-20 .so files)
ls -1 lib/*.so* | wc -l

# Verify executable permissions
file binaries/cuda12/llama-server
# Should output: ELF 64-bit LSB executable
```

---

## Next Steps

### After Successful Build

1. **Local Development:**
   ```bash
   cd /media/waqasm86/External1/Project-Nvidia/llcuda
   pip install -e .
   ```

2. **Test with Model:**
   ```python
   import llcuda
   engine = llcuda.InferenceEngine()
   engine.load_model("gemma-3-1b-Q4_K_M")
   ```

3. **For Google Colab:**
   - Create tar.gz: `tar -czf llcuda-binaries-t4.tar.gz -C llcuda/llcuda binaries lib`
   - Upload to GitHub releases
   - Update bootstrap.py URL if needed

### Publishing to PyPI (Optional)

```bash
cd llcuda
python setup.py sdist bdist_wheel
twine upload dist/*
```

---

## Support & Resources

### Documentation Files
- **BUILD_GUIDE.md** - Comprehensive build guide
- **INTEGRATION_GUIDE.md** - Path detection & integration
- **QUICK_START.md** - Quick reference

### Scripts
- **BUILD_AND_INTEGRATE.sh** - Main integration script
- **test_llcuda_integration.py** - Verification script

### External Links
- llama.cpp: https://github.com/ggml-org/llama.cpp
- llcuda repo: https://github.com/waqasm86/llcuda
- CUDA docs: https://docs.nvidia.com/cuda/

---

## Summary

✅ **Bug Fixed:** stderr.read() AttributeError in Google Colab
✅ **Build Scripts:** Complete CMake configurations for both GPUs
✅ **Integration:** Automated scripts to copy binaries to correct locations
✅ **Documentation:** 1000+ lines of guides, troubleshooting, and examples
✅ **Testing:** Verification scripts to ensure everything works

**You're now ready to build CUDA 12 executables and integrate them into llcuda!**

Run this to get started:
```bash
cd /media/waqasm86/External1/Project-Nvidia
./BUILD_AND_INTEGRATE.sh 940m
```
