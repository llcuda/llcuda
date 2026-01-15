# Complete Workflow: Build, Package, and Publish llcuda

## Overview

This guide provides the complete end-to-end workflow for building CUDA 12 binaries, creating release packages, and publishing llcuda to both GitHub and PyPI.

---

## Workflow Summary

```
Build Binaries (CMake)
    ↓
Create Release Package (.tar.gz)
    ↓
Upload to GitHub Releases
    ↓
Update llcuda Package (bootstrap.py, version)
    ↓
Upload to PyPI (Python package only, < 100MB)
    ↓
Users: pip install llcuda → Auto-downloads binaries from GitHub
```

---

## Phase 1: Build CUDA 12 Binaries

### For GeForce 940M (Local System)

```bash
cd /media/waqasm86/External1/Project-Nvidia

# 1. Read the build instructions
cat cmake_build_940m.sh

# 2. Navigate to llama.cpp
cd llama.cpp

# 3. Configure with CMake (MANUALLY RUN THIS)
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
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'

# 4. Build (MANUALLY RUN THIS - takes 10-30 min)
cmake --build build_cuda12_940m --config Release -j$(nproc)

# 5. Verify
ls -lh build_cuda12_940m/bin/llama-server
# Should be ~150-200 MB
```

### For Tesla T4 (Google Colab)

```bash
# In Google Colab cell:
!cat cmake_build_t4.sh

# Clone llama.cpp
!git clone https://github.com/ggml-org/llama.cpp
%cd llama.cpp

# Configure (MANUALLY RUN THIS)
!cmake -B build_cuda12_t4 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DBUILD_SHARED_LIBS=ON

# Build (MANUALLY RUN THIS - takes 5-15 min)
!cmake --build build_cuda12_t4 --config Release -j2

# Verify
!ls -lh build_cuda12_t4/bin/llama-server
```

---

## Phase 2: Create Release Packages

### Option A: Both GPUs in One Go

```bash
cd /media/waqasm86/External1/Project-Nvidia

# Run the packaging script
./CREATE_RELEASE_PACKAGE.sh

# Select option 3 (Both)
# This creates:
# - release-packages/llcuda-binaries-cuda12-940m.tar.gz
# - release-packages/llcuda-binaries-cuda12-t4.tar.gz
```

### Option B: One GPU at a Time

```bash
# For 940M only
./CREATE_RELEASE_PACKAGE.sh
# Select option 1

# For T4 only (after building in Colab)
./CREATE_RELEASE_PACKAGE.sh
# Select option 2
```

### Verify Packages

```bash
cd release-packages

# Check sizes
ls -lh *.tar.gz

# Test extraction
tar -xzf llcuda-binaries-cuda12-940m.tar.gz
ls -lh bin/ lib/

# Test execution
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
./bin/llama-server --help

# Should show help without errors
```

---

## Phase 3: Upload to GitHub Releases

### Step 1: Go to Releases Page

Open: https://github.com/waqasm86/llcuda/releases

### Step 2: Create New Release

1. Click **"Draft a new release"**
2. **Tag:** `v1.2.2`
3. **Title:** `llcuda v1.2.2 - CUDA 12 Binaries`
4. **Description:** (see [GITHUB_RELEASE_GUIDE.md](GITHUB_RELEASE_GUIDE.md) for template)

### Step 3: Upload Binaries

Attach files:
- `llcuda-binaries-cuda12-940m.tar.gz` (~120-160 MB)
- `llcuda-binaries-cuda12-t4.tar.gz` (~120-160 MB)

### Step 4: Publish

Click **"Publish release"**

### Step 5: Verify

Test download:
```bash
wget https://github.com/waqasm86/llcuda/releases/download/v1.2.2/llcuda-binaries-cuda12-940m.tar.gz
# Should download successfully
```

---

## Phase 4: Update llcuda Package

### Step 1: Update Bootstrap URL

Edit `llcuda/llcuda/_internal/bootstrap.py`:

```python
# Line 24
GITHUB_RELEASE_URL = "https://github.com/waqasm86/llcuda/releases/download/v1.2.2"
```

### Step 2: Update Version

Edit `llcuda/llcuda/__init__.py`:

```python
__version__ = "1.2.2"
```

Edit `llcuda/setup.py`:

```python
setup(
    name='llcuda',
    version='1.2.2',
    ...
)
```

### Step 3: Verify .gitignore

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda

# Check .gitignore excludes large files
cat .gitignore | grep -E "(binaries|lib|models)"

# Verify no large files staged
git status

# Should NOT show:
# - llcuda/binaries/
# - llcuda/lib/
# - llcuda/models/
# - *.gguf files
```

### Step 4: Commit and Push

```bash
git add llcuda/_internal/bootstrap.py
git add llcuda/__init__.py
git add setup.py
git add .

git commit -m "Release v1.2.2: CUDA 12 support for 940M and T4

- Fixed stderr.read() AttributeError in Google Colab
- Added FlashAttention support for Tesla T4
- Optimized for GeForce 940M (CC 5.0) and Tesla T4 (CC 7.5)
- Updated bootstrap to download v1.2.2 binaries from GitHub Releases
- Package size < 1MB for PyPI compliance
"

git push origin main

# Create tag
git tag v1.2.2
git push origin v1.2.2
```

---

## Phase 5: Upload to PyPI

### Step 1: Clean and Build

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda

# Remove old builds
rm -rf build/ dist/ *.egg-info/

# Build distributions
python setup.py sdist bdist_wheel
```

### Step 2: Verify Package Size

```bash
ls -lh dist/

# Expected:
# llcuda-1.2.2.tar.gz        : < 500 KB (Python code only)
# llcuda-1.2.2-py3-none-any.whl : < 500 KB

# CRITICAL: Must be < 100MB for PyPI
```

### Step 3: Verify Contents

```bash
# Check what's in the package
tar -tzf dist/llcuda-1.2.2.tar.gz | grep -E "(binaries|lib|\.so|\.gguf)"

# Should return NOTHING (no binaries in package)
```

### Step 4: Run Checks

```bash
twine check dist/*

# Should output:
# Checking dist/llcuda-1.2.2.tar.gz: PASSED
# Checking dist/llcuda-1.2.2-py3-none-any.whl: PASSED
```

### Step 5: Test on Test PyPI (Optional but Recommended)

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test install
python3.11 -m venv test_env
source test_env/bin/activate
pip install --index-url https://test.pypi.org/simple/ llcuda
python -c "import llcuda; print(llcuda.__version__)"
deactivate
rm -rf test_env
```

### Step 6: Upload to Production PyPI

```bash
twine upload dist/*

# Enter credentials:
# Username: __token__
# Password: (paste your PyPI API token)
```

### Step 7: Verify

Visit: https://pypi.org/project/llcuda/

Check:
- Version: v1.2.2
- File size: < 1MB
- Description renders correctly

---

## Phase 6: Test Complete Workflow

### Test 1: Fresh Install

```bash
# Clean environment
python3.11 -m venv fresh_env
source fresh_env/bin/activate

# Install from PyPI
pip install llcuda

# This should:
# 1. Download Python package from PyPI (~500KB)
# 2. On first import, download binaries from GitHub (~150MB)
```

### Test 2: Verify Auto-Download

```python
import llcuda

# First import triggers bootstrap
# Should see:
# "Downloading optimized binaries from GitHub..."
# "Extracting llcuda-binaries-cuda12.tar.gz..."
# "✓ Setup Complete!"

# Check version
print(f"Version: {llcuda.__version__}")

# Check GPU
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"Compatible: {compat['compatible']}")
```

### Test 3: Full Inference (Requires Model)

```python
engine = llcuda.InferenceEngine()

# Load model (downloads from HuggingFace first time)
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    gpu_layers=15,
    ctx_size=1024,
    silent=False
)

# Run inference
result = engine.infer("What is 2+2?", max_tokens=50)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")
```

### Test 4: Cleanup

```bash
deactivate
rm -rf fresh_env
```

---

## File Size Summary

### GitHub Main Repository (< 1GB recommended):
- Python source code: ~500KB - 2MB ✅
- Documentation (.md files): ~100-500KB ✅
- Configuration files: ~10-50KB ✅
- **Total:** < 5MB ✅

### GitHub Releases (no limit):
- llcuda-binaries-cuda12-940m.tar.gz: ~120-160MB ✅
- llcuda-binaries-cuda12-t4.tar.gz: ~120-160MB ✅
- **Total:** ~250-320MB ✅

### PyPI Package (< 100MB):
- llcuda-1.2.2.tar.gz: ~300-800KB ✅
- llcuda-1.2.2-py3-none-any.whl: ~300-800KB ✅
- **Total:** < 2MB ✅ (well under limit!)

### NEVER Upload:
- .gguf model files (too large, violates policies) ❌
- User's downloaded models ❌
- Build artifacts (build/, CMakeFiles/) ❌

---

## Troubleshooting Workflow

### Issue: Package too large for PyPI

**Symptom:**
```
HTTPError: 400 Bad Request - File too large
```

**Fix:**
1. Check .gitignore includes binaries/lib/models
2. Verify: `git status` doesn't show large files
3. Clean rebuild: `rm -rf dist/ && python setup.py sdist bdist_wheel`
4. Check size: `ls -lh dist/` (should be < 1MB)

### Issue: Bootstrap can't download binaries

**Symptom:**
Users report "llama-server not found"

**Fix:**
1. Verify GitHub Release exists at correct URL
2. Test download manually:
   ```bash
   wget https://github.com/waqasm86/llcuda/releases/download/v1.2.2/llcuda-binaries-cuda12.tar.gz
   ```
3. Check bootstrap.py URL matches exactly

### Issue: Wrong binaries downloaded

**Symptom:**
GeForce 940M downloads T4 binaries (or vice versa)

**Fix:**
Update bootstrap.py to detect GPU and download appropriate package:
```python
def detect_gpu_compute_capability():
    # Returns ("Tesla T4", "7.5") or ("GeForce 940M", "5.0")

if compute_cap >= 7.0:
    BINARY_BUNDLE_NAME = "llcuda-binaries-cuda12-t4.tar.gz"
else:
    BINARY_BUNDLE_NAME = "llcuda-binaries-cuda12-940m.tar.gz"
```

---

## Checklist for Each Release

### Pre-Release:
- [ ] Build binaries for both GPUs with CMake
- [ ] Test binaries locally
- [ ] Create release packages with CREATE_RELEASE_PACKAGE.sh
- [ ] Test package extraction and execution
- [ ] Write release notes

### GitHub:
- [ ] Upload binaries to GitHub Releases
- [ ] Verify download URLs work
- [ ] Update bootstrap.py with new URL
- [ ] Update version in __init__.py and setup.py
- [ ] Commit and push to main repo
- [ ] Create git tag

### PyPI:
- [ ] Verify .gitignore excludes binaries
- [ ] Clean and rebuild package
- [ ] Verify package size < 1MB
- [ ] Run twine check
- [ ] Test on Test PyPI
- [ ] Upload to production PyPI
- [ ] Verify on pypi.org

### Post-Release:
- [ ] Test fresh install
- [ ] Test bootstrap download
- [ ] Test inference workflow
- [ ] Update documentation
- [ ] Announce release

---

## Quick Reference Commands

```bash
# Build for 940M
cd llama.cpp
cmake -B build_cuda12_940m -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="50" ...
cmake --build build_cuda12_940m --config Release -j$(nproc)

# Create release package
cd /media/waqasm86/External1/Project-Nvidia
./CREATE_RELEASE_PACKAGE.sh

# Upload to GitHub Releases
# (Use web interface)

# Update llcuda and upload to PyPI
cd llcuda
# Update version and bootstrap URL
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*

# Test
pip install llcuda
python -c "import llcuda; print(llcuda.__version__)"
```

---

## Summary

This workflow ensures:

1. ✅ **Binaries built correctly** for both GPUs
2. ✅ **Large files go to GitHub Releases** (not main repo)
3. ✅ **PyPI package stays tiny** (< 1MB)
4. ✅ **Auto-download works** (bootstrap from GitHub)
5. ✅ **No .gguf files uploaded** anywhere
6. ✅ **Clear versioning** across all components

**You're now ready to build, package, and publish llcuda!**

Start with:
```bash
cd /media/waqasm86/External1/Project-Nvidia
cat cmake_build_940m.sh
```
