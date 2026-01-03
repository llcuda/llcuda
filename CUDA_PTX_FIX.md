# CUDA PTX Compatibility Fix for Google Colab

**Problem**: llama-server binaries compiled with CUDA 12.8 produce PTX code incompatible with Google Colab's CUDA driver

**Error**: `CUDA error: the provided PTX was compiled with an unsupported toolchain`

**Root Cause**: Google Colab runs CUDA 12.0-12.2 drivers, but binaries were compiled with CUDA 12.8

---

## Solution: Recompile with Compatible PTX Version

### Option 1: Force SASS Generation (Recommended)

Compile with explicit `-gencode` flags to generate **actual GPU code (SASS/cubin)** instead of relying on JIT compilation from PTX.

**Benefits**:
- No PTX JIT compilation needed
- Works on any CUDA 12.x driver
- Slightly larger binary (~10-20MB more)

**Build Command**:
```bash
cd /media/waqasm86/External1/Project-Nvidia/llama.cpp
rm -rf build
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="50;61;70;75;80;86;89" \
  -DCMAKE_CUDA_FLAGS="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_75,code=compute_75"

cmake --build . --config Release -j$(nproc)
```

**Explanation**:
- `-gencode arch=compute_75,code=sm_75`: Generate native SASS code for SM 7.5 (T4 GPU)
- `-gencode arch=compute_75,code=compute_75`: Also generate PTX for forward compatibility
- This produces **both** native code AND PTX, so older drivers can use the SASS directly

### Option 2: Lower PTX Version Target

Use `-target-dir` to generate PTX compatible with older CUDA versions.

**Build Command**:
```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="50;61;70;75;80;86;89" \
  -DCMAKE_CUDA_FLAGS="--ptxas-options=-v -Xptxas=-O3"
```

### Option 3: Disable PTX, SASS Only (Smallest)

Generate only native code for each architecture - no PTX at all.

**Build Command**:
```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="50;61;70;75;80;86;89" \
  -DCMAKE_CUDA_FLAGS="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89"
```

---

## Recommended Build Script

Create a new build script for Colab-compatible binaries:

```bash
#!/bin/bash
# build_cuda12_colab.sh - Build llama.cpp with Colab-compatible CUDA 12 binaries

set -e

echo "Building llama.cpp with CUDA 12 (Colab-compatible)"

cd /media/waqasm86/External1/Project-Nvidia/llama.cpp

# Clean build
rm -rf build
mkdir -p build
cd build

# Configure with explicit SASS generation for T4 (SM 7.5)
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="50;61;70;75;80;86;89" \
  -DCMAKE_CUDA_FLAGS="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89" \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128

# Build
cmake --build . --config Release -j$(nproc)

# Verify binaries
echo ""
echo "✅ Build complete! Verifying binaries..."
echo ""

./bin/llama-server --version
./bin/llama-cli --version

echo ""
echo "Binaries built in: $(pwd)/bin"
echo ""
echo "Next steps:"
echo "  1. Test on local GPU (GeForce 940M)"
echo "  2. Create binary archive for llcuda"
echo "  3. Test in Google Colab before releasing"
```

---

## Testing Plan

### 1. Local Test (GeForce 940M - SM 5.0)
```bash
cd /media/waqasm86/External1/Project-Nvidia/llama.cpp/build
./bin/llama-server --version
./bin/llama-cli -m /path/to/model.gguf -p "Test" -n 10 -ngl 14
```

Expected: Should work on local GPU

### 2. Create Binary Archive
```bash
cd /media/waqasm86/External1/Project-Nvidia/llama.cpp/build
tar -czf llcuda-binaries-cuda12-colab.tar.gz bin/ lib/
```

### 3. Test in Google Colab
Upload the archive and test:
```python
# In Colab
!wget <your-test-url>/llcuda-binaries-cuda12-colab.tar.gz
!tar -xzf llcuda-binaries-cuda12-colab.tar.gz

# Test llama-server
!./bin/llama-server --version

# Test actual inference
!pip install llcuda -q
import llcuda
engine = llcuda.InferenceEngine()
# Point to new binary
import os
os.environ['LLAMA_SERVER_PATH'] = './bin/llama-server'
engine.load_model("unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf", gpu_layers=26)
result = engine.infer("What is AI?", max_tokens=50)
print(result.text)
```

---

## Alternative: Check Colab CUDA Version First

Before recompiling, run this in Colab to see exact CUDA version:

```python
!nvcc --version
!nvidia-smi | grep "CUDA Version"
```

This will tell us if Colab is using CUDA 12.0, 12.1, 12.2, etc.

---

## Expected Results

After recompilation with SASS generation:
- ✅ Binaries will include native code for SM 7.5 (T4)
- ✅ No PTX JIT compilation needed at runtime
- ✅ Works on Google Colab CUDA 12.0-12.8
- ✅ Works on local systems with CUDA 12.8
- ⚠️  Binary size increases by ~10-20MB (acceptable tradeoff)

---

## Next Steps

1. **Immediate**: Run diagnostic script in Colab to confirm CUDA version
2. **Build**: Recompile llama.cpp with new flags
3. **Test**: Verify on local GPU first
4. **Package**: Create new binary archive
5. **Colab Test**: Upload and test in Colab
6. **Release**: If successful, update v1.1.7 binaries or create v1.1.10

---

**Summary**: The issue is NOT about CUDA 11 vs CUDA 12, it's about PTX generated by CUDA 12.8 being incompatible with older CUDA 12.x drivers in Colab. Solution is to generate native SASS code instead of relying on PTX JIT compilation.
