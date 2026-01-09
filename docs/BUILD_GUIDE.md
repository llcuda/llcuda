# CUDA 12 Build Guide for llcuda Project

## Overview

This guide provides step-by-step instructions for building CUDA 12 executables for your llcuda project targeting two different GPUs:

1. **NVIDIA GeForce 940M** (Compute Capability 5.0) - Your local Xubuntu 22 system
2. **NVIDIA Tesla T4** (Compute Capability 7.5) - Google Colab

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Bug Fix Applied](#bug-fix-applied)
4. [Build Options](#build-options)
5. [Manual Build Instructions](#manual-build-instructions)
6. [Troubleshooting](#troubleshooting)
7. [Testing](#testing)
8. [Integration with llcuda](#integration-with-llcuda)

---

## Quick Start

### For GeForce 940M (Xubuntu 22):
```bash
cd /media/waqasm86/External1/Project-Nvidia
./build_cuda12_unified.sh 940m
```

### For Tesla T4 (Google Colab):
```bash
cd /content
git clone https://github.com/waqasm86/llcuda
cd llcuda
bash build_cuda12_tesla_t4_colab.sh
```

### Auto-detect and Build:
```bash
./build_cuda12_unified.sh auto
```

---

## Prerequisites

### Xubuntu 22 System (GeForce 940M)

**Required:**
- CUDA Toolkit 12.8 (already installed at `/usr/local/cuda-12.8`)
- CMake >= 3.14
- GCC/G++ compiler
- NVIDIA Driver 570.195.03 (already installed)
- Python 3.11 (already installed)

**Verify:**
```bash
nvcc --version
nvidia-smi
cmake --version
gcc --version
```

### Google Colab (Tesla T4)

**Pre-installed:**
- CUDA 12.4/12.6
- Python 3.12
- CMake
- GCC

**Verify in Colab:**
```python
!nvcc --version
!nvidia-smi
```

---

## Bug Fix Applied

### Issue in Google Colab

**Error:**
```
AttributeError: 'NoneType' object has no attribute 'read'
```

**Root Cause:**
In `llcuda/llcuda/server.py` line 553, the code attempted to read `stderr` when `silent=True` was set, but in silent mode, `stderr` is set to `subprocess.DEVNULL` (which is `None`).

**Fix Applied:**
The code now checks if `stderr` is not `None` before attempting to read it:

```python
# Check if process died
if self.server_process.poll() is not None:
    # Read stderr only if it's not DEVNULL (silent mode)
    if self.server_process.stderr is not None:
        stderr = self.server_process.stderr.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"llama-server process died unexpectedly.\nError output:\n{stderr}")
    else:
        raise RuntimeError(f"llama-server process died unexpectedly. Run with silent=False for error details.")
```

This fix has been applied to `/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/server.py`.

---

## Build Options

### Three Build Scripts Available

#### 1. Unified Build Script (Recommended)
```bash
./build_cuda12_unified.sh [target]
```
- **Targets:** `940m`, `t4`, `both`, `auto`
- **Features:** Auto-detection, profile-based builds, parallel compilation

#### 2. GeForce 940M Specific Script
```bash
./build_cuda12_geforce940m.sh
```
- Optimized for Compute Capability 5.0
- Includes detailed explanations
- Guide-style script

#### 3. Tesla T4 Colab Script
```bash
./build_cuda12_tesla_t4_colab.sh
```
- Fully automated build for Colab
- Creates distribution package
- FlashAttention enabled

---

## Manual Build Instructions

### For GeForce 940M (CC 5.0)

#### Step 1: Navigate to llama.cpp Directory
```bash
cd /media/waqasm86/External1/Project-Nvidia/llama.cpp
```

#### Step 2: Run CMake Configuration
```bash
cmake -B build_cuda12_940m \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="50" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_FORCE_CUBLAS=ON \
    -DGGML_CUDA_FA=OFF \
    -DGGML_CUDA_GRAPHS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_CURL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
```

**CMake Options Explained:**

| Option | Value | Reason |
|--------|-------|--------|
| `CMAKE_CUDA_ARCHITECTURES` | `50` | Target Compute Capability 5.0 (Maxwell) |
| `GGML_NATIVE` | `OFF` | Build for CC 5.0, not the build machine |
| `GGML_CUDA_FORCE_CUBLAS` | `ON` | Use cuBLAS for better compatibility on old GPUs |
| `GGML_CUDA_FA` | `OFF` | FlashAttention requires CC >= 7.0 |
| `GGML_CUDA_GRAPHS` | `ON` | Enable CUDA graphs for optimization |
| `BUILD_SHARED_LIBS` | `ON` | Create .so files for runtime linking |

#### Step 3: Build (10-30 minutes)
```bash
cmake --build build_cuda12_940m --config Release -j$(nproc)
```

**Expected output:**
```
[100%] Built target llama-server
[100%] Built target llama-cli
...
```

#### Step 4: Install Binaries
```bash
# Create directories
mkdir -p ../llcuda/llcuda/binaries/cuda12_940m
mkdir -p ../llcuda/llcuda/lib

# Copy binaries
cp build_cuda12_940m/bin/llama-server ../llcuda/llcuda/binaries/cuda12_940m/
cp build_cuda12_940m/bin/llama-cli ../llcuda/llcuda/binaries/cuda12_940m/
cp build_cuda12_940m/bin/llama-quantize ../llcuda/llcuda/binaries/cuda12_940m/
cp build_cuda12_940m/bin/llama-embedding ../llcuda/llcuda/binaries/cuda12_940m/

# Copy libraries
cp build_cuda12_940m/bin/libllama.so* ../llcuda/llcuda/lib/
cp build_cuda12_940m/bin/libggml*.so* ../llcuda/llcuda/lib/

# Make executable
chmod +x ../llcuda/llcuda/binaries/cuda12_940m/*
```

#### Step 5: Verify Installation
```bash
# Check binaries
ls -lh ../llcuda/llcuda/binaries/cuda12_940m/

# Check CUDA linking
ldd build_cuda12_940m/bin/llama-server | grep cuda

# Test execution
export LD_LIBRARY_PATH=../llcuda/llcuda/lib:$LD_LIBRARY_PATH
build_cuda12_940m/bin/llama-server --help
```

---

### For Tesla T4 (CC 7.5)

#### Step 1: In Google Colab, Clone llama.cpp
```bash
cd /content
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

#### Step 2: Run CMake Configuration
```bash
cmake -B build_cuda12_t4 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_FORCE_CUBLAS=OFF \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_CURL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
```

**T4-Specific Options:**

| Option | Value | Reason |
|--------|-------|--------|
| `CMAKE_CUDA_ARCHITECTURES` | `75` | Target Compute Capability 7.5 (Turing) |
| `GGML_CUDA_FA` | `ON` | Enable FlashAttention (2x faster inference) |
| `GGML_CUDA_FA_ALL_QUANTS` | `ON` | FA for all quantization types |
| `GGML_CUDA_FORCE_CUBLAS` | `OFF` | Use optimized custom kernels |

#### Step 3: Build (5-15 minutes in Colab)
```bash
cmake --build build_cuda12_t4 --config Release -j$(nproc)
```

#### Step 4: Create Distribution Package
```bash
# Create install structure
mkdir -p /content/llama_cuda12_t4/{bin,lib}

# Copy files
cp build_cuda12_t4/bin/llama-server /content/llama_cuda12_t4/bin/
cp build_cuda12_t4/bin/llama-cli /content/llama_cuda12_t4/bin/
cp build_cuda12_t4/bin/libllama.so* /content/llama_cuda12_t4/lib/
cp build_cuda12_t4/bin/libggml*.so* /content/llama_cuda12_t4/lib/

# Create tar.gz for download
cd /content
tar -czf llcuda-binaries-cuda12-t4.tar.gz llama_cuda12_t4/

# Download from Colab
from google.colab import files
files.download('llcuda-binaries-cuda12-t4.tar.gz')
```

---

## Troubleshooting

### Issue: CMake can't find CUDA

**Solution:**
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Issue: Build fails with "compute_50 is not supported"

This means you're using CUDA 13+ which dropped CC 5.0 support.

**Solution:**
Use CUDA 12.x (you have 12.8 installed, which works):
```bash
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc ...
```

### Issue: llama-server crashes immediately

**Possible causes:**
1. Wrong compute capability binary
2. Missing shared libraries
3. Incompatible CUDA runtime

**Debug:**
```bash
# Check dependencies
ldd /path/to/llama-server

# Set library path
export LD_LIBRARY_PATH=/path/to/llcuda/lib:$LD_LIBRARY_PATH

# Run with verbose output
./llama-server --help
```

### Issue: Out of memory on GeForce 940M

The 940M has only ~1GB VRAM.

**Solutions:**
- Use smaller models (1-3B parameters)
- Reduce `gpu_layers` (try 5-15)
- Use 4-bit quantization (Q4_K_M)
- Reduce context size (`ctx_size=512` or `1024`)

---

## Testing

### Test GeForce 940M Build

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda

# Test with Python
python3.11 << 'EOF'
import os
os.environ['LLAMA_SERVER_PATH'] = '/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/binaries/cuda12_940m/llama-server'
os.environ['LD_LIBRARY_PATH'] = '/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/lib'

import llcuda

# Check GPU
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"CC: {compat['compute_capability']}")
print(f"Compatible: {compat['compatible']}")
EOF
```

### Test Tesla T4 Build (in Colab)

```python
import os
os.environ['LLAMA_SERVER_PATH'] = '/content/llama_cuda12_t4/bin/llama-server'
os.environ['LD_LIBRARY_PATH'] = '/content/llama_cuda12_t4/lib'

import llcuda

compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"CC: {compat['compute_capability']}")

# Test with small model
engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    gpu_layers=26,
    ctx_size=2048,
    silent=False  # Use False to see error output if any
)

result = engine.infer("What is 2+2?", max_tokens=50)
print(result.text)
```

---

## Integration with llcuda

### Update llcuda Package

After building, update your llcuda package:

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda

# Reinstall with new binaries
pip uninstall llcuda -y
pip install -e .
```

### Configure for Your System

Create `~/.llcuda_config`:

```ini
[paths]
server_path = /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/binaries/cuda12_940m/llama-server
lib_path = /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/lib

[gpu]
# For GeForce 940M
max_gpu_layers = 15
default_ctx_size = 1024
recommended_quantization = Q4_K_M
```

### Environment Variables

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# llcuda CUDA 12 Configuration
export LLAMA_SERVER_PATH="/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/binaries/cuda12_940m/llama-server"
export LD_LIBRARY_PATH="/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/lib:$LD_LIBRARY_PATH"
```

---

## Performance Recommendations

### GeForce 940M (1GB VRAM)

| Model Size | Quantization | GPU Layers | Context Size | Expected Speed |
|------------|--------------|------------|--------------|----------------|
| 1B params  | Q4_K_M       | 15-20      | 512-1024     | 15-25 tok/s    |
| 3B params  | Q4_K_M       | 8-12       | 512          | 8-15 tok/s     |
| 7B params  | Q4_K_M       | 3-5        | 512          | 2-5 tok/s      |

### Tesla T4 (15GB VRAM)

| Model Size | Quantization | GPU Layers | Context Size | Expected Speed |
|------------|--------------|------------|--------------|----------------|
| 1B params  | Q4_K_M       | 30-35      | 4096         | 50-80 tok/s    |
| 3B params  | Q4_K_M       | 26-30      | 4096         | 40-60 tok/s    |
| 7B params  | Q5_K_M       | 32-35      | 4096         | 25-40 tok/s    |
| 13B params | Q4_K_M       | 28-32      | 2048         | 15-25 tok/s    |

---

## Additional Resources

### CUDA Compute Capabilities

| GPU Model | Architecture | CC | VRAM | Notes |
|-----------|--------------|-----|------|-------|
| GeForce 940M | Maxwell | 5.0 | 1-4GB | Entry-level, no tensor cores |
| Tesla T4 | Turing | 7.5 | 16GB | Tensor cores, INT8 support |
| RTX 3060 | Ampere | 8.6 | 12GB | Modern consumer GPU |
| A100 | Ampere | 8.0 | 40-80GB | High-end datacenter |

### Useful Commands

```bash
# Check CUDA version
nvcc --version

# Check GPU details
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check library dependencies
ldd /path/to/binary | grep -E "(cuda|cublas|cudnn)"

# Test CUDA installation
/usr/local/cuda/samples/bin/x86_64/linux/release/deviceQuery
```

---

## Summary

You now have:

1. ✅ **Bug fix** applied to `llcuda/server.py` for Google Colab compatibility
2. ✅ **Build scripts** for both GeForce 940M and Tesla T4
3. ✅ **Unified build script** for automated builds
4. ✅ **Manual build instructions** with detailed explanations
5. ✅ **Testing procedures** for both platforms
6. ✅ **Performance recommendations** for optimal settings

**Next Steps:**
1. Run the appropriate build script for your target GPU
2. Test the built binaries with llcuda
3. Upload Tesla T4 binaries to your GitHub releases
4. Update llcuda package with new binaries

Good luck with your builds!
