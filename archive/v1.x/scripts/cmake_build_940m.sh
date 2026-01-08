#!/bin/bash
################################################################################
# Manual CMake Build Script for NVIDIA GeForce 940M (CC 5.0)
# System: Xubuntu 22.04, CUDA 12.8, Python 3.11
#
# This script contains the exact CMake commands you need to run manually.
# It does NOT execute them automatically - you run each command yourself.
################################################################################

cat << 'EOF'
================================================================================
CMake Build Commands for NVIDIA GeForce 940M (Compute Capability 5.0)
================================================================================

Target GPU:    GeForce 940M
Architecture:  Maxwell
Compute Cap:   5.0
CUDA Version:  12.8
System:        Xubuntu 22.04

================================================================================
STEP 1: Navigate to llama.cpp directory
================================================================================

cd /media/waqasm86/External1/Project-Nvidia/llama.cpp

================================================================================
STEP 2: Configure with CMake (run this command)
================================================================================

cmake -B build_cuda12_940m \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="50" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_FORCE_CUBLAS=ON \
    -DGGML_CUDA_FA=OFF \
    -DGGML_CUDA_FA_ALL_QUANTS=OFF \
    -DGGML_CUDA_GRAPHS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_CURL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

================================================================================
CMake Options Explained:
================================================================================

-DCMAKE_BUILD_TYPE=Release
  → Build optimized release version

-DGGML_CUDA=ON
  → Enable CUDA support

-DCMAKE_CUDA_ARCHITECTURES="50"
  → Target Compute Capability 5.0 (Maxwell architecture)
  → CRITICAL for GeForce 940M

-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
  → Use CUDA 12.8 compiler explicitly

-DGGML_NATIVE=OFF
  → Build for CC 5.0, not the build machine's GPU
  → Ensures portability

-DGGML_CUDA_FORCE_CUBLAS=ON
  → Use cuBLAS library for matrix operations
  → More stable for older GPUs like 940M

-DGGML_CUDA_FA=OFF
  → Disable FlashAttention (requires CC >= 7.0)
  → 940M does not support this

-DGGML_CUDA_GRAPHS=ON
  → Enable CUDA graphs for optimized execution

-DLLAMA_BUILD_SERVER=ON
  → Build llama-server executable (required for llcuda)

-DBUILD_SHARED_LIBS=ON
  → Build as shared libraries (.so files)
  → Needed for runtime linking

-DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'
  → Set runtime library search path
  → Helps find .so files at runtime

================================================================================
STEP 3: Build (run this command)
================================================================================

This will take 10-30 minutes depending on your CPU:

cmake --build build_cuda12_940m --config Release -j$(nproc)

Explanation:
  --config Release  : Build in release mode (optimized)
  -j$(nproc)        : Use all CPU cores for parallel compilation

================================================================================
STEP 4: Verify Build Success
================================================================================

After build completes, check these files exist:

ls -lh build_cuda12_940m/bin/llama-server
ls -lh build_cuda12_940m/bin/*.so*

Expected outputs:
  llama-server     : ~150-200 MB
  libllama.so      : ~50-100 MB
  libggml-*.so     : Multiple library files

================================================================================
STEP 5: Test Binary (optional)
================================================================================

export LD_LIBRARY_PATH=$(pwd)/build_cuda12_940m/bin:$LD_LIBRARY_PATH
./build_cuda12_940m/bin/llama-server --help

Should display help text without errors.

================================================================================
NEXT: Create Release Package
================================================================================

After successful build, run:

cd /media/waqasm86/External1/Project-Nvidia
./CREATE_RELEASE_PACKAGE.sh

Select option "1" for GeForce 940M

This will create: llcuda-binaries-cuda12-940m.tar.gz

================================================================================
Notes for GeForce 940M
================================================================================

Hardware Limitations:
  - Only ~1GB VRAM
  - No FlashAttention support
  - Limited to small models (1-3B parameters)

Recommended Settings:
  - gpu_layers: 10-15
  - ctx_size: 512-1024
  - Use Q4_K_M quantization
  - Expected speed: 10-20 tokens/sec

Compatible Models:
  - TinyLlama 1.1B
  - Gemma 2B
  - Phi-2 2.7B
  - Small Qwen models

================================================================================
Troubleshooting
================================================================================

If CMake fails:
  1. Check CUDA installation: nvcc --version
  2. Verify GPU detection: nvidia-smi
  3. Check CUDA path: ls /usr/local/cuda-12.8/bin/nvcc

If build fails:
  1. Check error messages for missing dependencies
  2. Try with single core: -j1 instead of -j$(nproc)
  3. Check disk space: df -h

If llama-server crashes:
  1. Wrong CC: You built for wrong GPU architecture
  2. Missing libraries: Check LD_LIBRARY_PATH
  3. CUDA mismatch: Runtime CUDA != Build CUDA

================================================================================
EOF
