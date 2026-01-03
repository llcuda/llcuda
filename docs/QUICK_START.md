# Quick Start Guide

## Problem Solved

**Google Colab Error:**
```
AttributeError: 'NoneType' object has no attribute 'read'
```

**Solution:** Fixed bug in `llcuda/llcuda/server.py` where code tried to read stderr when it was set to DEVNULL in silent mode.

---

## For Xubuntu 22 (GeForce 940M)

### Option 1: Automated Build (Recommended)
```bash
cd /media/waqasm86/External1/Project-Nvidia
./build_cuda12_unified.sh 940m
```

### Option 2: Manual Build
```bash
cd /media/waqasm86/External1/Project-Nvidia/llama.cpp

# Step 1: Configure (adjust CUDA path if needed)
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

# Step 2: Build (takes 10-30 minutes)
cmake --build build_cuda12_940m --config Release -j$(nproc)

# Step 3: Install
mkdir -p ../llcuda/llcuda/binaries/cuda12_940m
mkdir -p ../llcuda/llcuda/lib
cp build_cuda12_940m/bin/llama-server ../llcuda/llcuda/binaries/cuda12_940m/
cp build_cuda12_940m/bin/libllama.so* ../llcuda/llcuda/lib/
cp build_cuda12_940m/bin/libggml*.so* ../llcuda/llcuda/lib/
chmod +x ../llcuda/llcuda/binaries/cuda12_940m/*
```

---

## For Google Colab (Tesla T4)

### In a Colab Notebook Cell:

```python
# Clone and build
!cd /content && git clone https://github.com/ggml-org/llama.cpp.git
!cd /content/llama.cpp && \
    cmake -B build_t4 \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="75" \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DBUILD_SHARED_LIBS=ON && \
    cmake --build build_t4 --config Release -j$(nproc)

# Create distribution package
!mkdir -p /content/llama_t4/{bin,lib}
!cp /content/llama.cpp/build_t4/bin/llama-server /content/llama_t4/bin/
!cp /content/llama.cpp/build_t4/bin/libllama.so* /content/llama_t4/lib/
!cp /content/llama.cpp/build_t4/bin/libggml*.so* /content/llama_t4/lib/
!tar -czf /content/llcuda-binaries-t4.tar.gz -C /content llama_t4

# Download the package
from google.colab import files
files.download('/content/llcuda-binaries-t4.tar.gz')
```

---

## Testing Your Build

### Test on Xubuntu 22:
```bash
export LD_LIBRARY_PATH=/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/lib:$LD_LIBRARY_PATH
/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/binaries/cuda12_940m/llama-server --help
```

### Test with llcuda:
```python
import os
os.environ['LLAMA_SERVER_PATH'] = '/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/binaries/cuda12_940m/llama-server'
os.environ['LD_LIBRARY_PATH'] = '/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/lib'

import llcuda

# Check GPU compatibility
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"Compatible: {compat['compatible']}")

# Test inference (with fixed server.py)
engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    gpu_layers=15,  # Adjust based on your GPU
    ctx_size=1024,
    silent=False  # Now works without AttributeError!
)
result = engine.infer("What is AI?", max_tokens=50)
print(result.text)
```

---

## Key Files Created

1. **[BUILD_GUIDE.md](BUILD_GUIDE.md)** - Complete documentation with troubleshooting
2. **[build_cuda12_unified.sh](build_cuda12_unified.sh)** - Smart build script with auto-detection
3. **[build_cuda12_geforce940m.sh](build_cuda12_geforce940m.sh)** - Detailed guide for 940M
4. **[build_cuda12_tesla_t4_colab.sh](build_cuda12_tesla_t4_colab.sh)** - Automated Colab build
5. **llcuda/llcuda/server.py** - Bug fix applied

---

## Recommended Settings

### GeForce 940M (1GB VRAM):
- Models: 1-3B parameters
- Quantization: Q4_K_M
- GPU Layers: 10-15
- Context: 512-1024

### Tesla T4 (15GB VRAM):
- Models: 1-13B parameters
- Quantization: Q4_K_M or Q5_K_M
- GPU Layers: 26-35
- Context: 2048-4096

---

## Need Help?

See [BUILD_GUIDE.md](BUILD_GUIDE.md) for:
- Detailed CMake option explanations
- Troubleshooting common errors
- Performance tuning tips
- Integration with llcuda package

---

**Everything is ready! Just run the build script for your target GPU.**
