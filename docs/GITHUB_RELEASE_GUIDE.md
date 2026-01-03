# GitHub Releases Upload Guide for llcuda

## Overview

This guide explains how to upload CUDA 12 binary packages to GitHub Releases page (NOT the main repository) to keep the main repo under 100MB for PyPI compatibility.

---

## Important Rules

### ✅ Upload to GitHub Releases:
- Compiled binaries (`llcuda-binaries-cuda12-*.tar.gz`)
- Shared libraries (included in tar.gz)
- Large build artifacts

### ❌ NEVER Upload Anywhere:
- `.gguf` model files (too large, violates storage policies)
- `.bin`, `.safetensors`, `.pt`, `.pth` model files
- User's downloaded models

### ✅ Upload to Main GitHub Repo:
- Python source code (`.py` files)
- Documentation (`.md` files)
- Configuration files (`setup.py`, `pyproject.toml`)
- Small examples and tests

---

## Step-by-Step: Upload to GitHub Releases

### Step 1: Build the Binaries

**For GeForce 940M (local system):**
```bash
cd /media/waqasm86/External1/Project-Nvidia
./cmake_build_940m.sh  # Read and follow instructions
# Then manually run the cmake commands shown
```

**For Tesla T4 (Google Colab):**
```bash
# In Colab
./cmake_build_t4.sh  # Read and follow instructions
# Then manually run the cmake commands shown
```

### Step 2: Create Release Package

```bash
cd /media/waqasm86/External1/Project-Nvidia
./CREATE_RELEASE_PACKAGE.sh
```

Choose:
- Option 1: GeForce 940M only
- Option 2: Tesla T4 only
- Option 3: Both (recommended for single release)

This creates:
- `release-packages/llcuda-binaries-cuda12-940m.tar.gz` (~120-160 MB)
- `release-packages/llcuda-binaries-cuda12-t4.tar.gz` (~120-160 MB)

### Step 3: Test the Package Locally

```bash
cd release-packages

# Extract
tar -xzf llcuda-binaries-cuda12-940m.tar.gz

# Test
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
./bin/llama-server --help

# Should display help without errors
```

### Step 4: Go to GitHub Releases Page

1. Open browser: https://github.com/waqasm86/llcuda/releases
2. Click **"Draft a new release"** button

### Step 5: Fill Release Information

**Tag version:**
```
v1.2.2
```

**Release title:**
```
llcuda v1.2.2 - CUDA 12 Binaries for GeForce 940M & Tesla T4
```

**Release description (example):**
```markdown
# llcuda v1.2.2 - CUDA 12 Binaries

Official CUDA 12 binary release for llcuda Python package.

## What's Included

Two optimized binary packages:

### 🎮 GeForce 940M (Compute Capability 5.0)
- **File:** `llcuda-binaries-cuda12-940m.tar.gz`
- **Target:** NVIDIA GeForce 940M, Maxwell architecture
- **Optimized for:** Small models (1-3B parameters), limited VRAM
- **Features:** cuBLAS optimization, CUDA graphs

### ☁️ Tesla T4 (Compute Capability 7.5)
- **File:** `llcuda-binaries-cuda12-t4.tar.gz`
- **Target:** NVIDIA Tesla T4, Turing architecture (Google Colab, Kaggle)
- **Optimized for:** Medium models (1-13B parameters)
- **Features:** FlashAttention (2x faster), tensor cores, CUDA graphs

## Installation

### Automatic (via pip)

```bash
pip install llcuda
```

The package will auto-download the appropriate binaries on first import.

### Manual Installation

```bash
# Download the appropriate .tar.gz for your GPU
# Extract and copy to llcuda package

tar -xzf llcuda-binaries-cuda12-t4.tar.gz
cp -r bin/* /path/to/llcuda/llcuda/binaries/cuda12/
cp -r lib/* /path/to/llcuda/llcuda/lib/
chmod +x /path/to/llcuda/llcuda/binaries/cuda12/*
```

## Requirements

- CUDA 12.x runtime
- NVIDIA GPU with Compute Capability 5.0 or higher
- Linux x86_64

## Changes in v1.2.2

- ✅ Fixed stderr.read() AttributeError in Google Colab
- ✅ Optimized for both GeForce 940M and Tesla T4
- ✅ FlashAttention support for T4 (2x faster)
- ✅ Improved library path detection
- ✅ Better error messages

## Performance

### GeForce 940M
- GPU Layers: 10-15
- Context: 512-1024
- Speed: 10-20 tokens/sec
- Models: 1-3B parameters (Q4_K_M)

### Tesla T4
- GPU Layers: 26-35
- Context: 2048-8192
- Speed: 25-60 tokens/sec
- Models: 1-13B parameters (Q4_K_M/Q5_K_M)

## Documentation

- [Quick Start Guide](https://github.com/waqasm86/llcuda/blob/main/QUICK_START.md)
- [Build Guide](https://github.com/waqasm86/llcuda/blob/main/BUILD_GUIDE.md)
- [Integration Guide](https://github.com/waqasm86/llcuda/blob/main/INTEGRATION_GUIDE.md)

## Support

- Issues: https://github.com/waqasm86/llcuda/issues
- Discussions: https://github.com/waqasm86/llcuda/discussions

## License

MIT License - Same as llama.cpp project

---

**Note:** Model files (.gguf) are NOT included. Models are downloaded separately from HuggingFace when needed.
```

### Step 6: Upload Binary Files

In the **"Attach binaries"** section at the bottom:

1. Click **"choose your files"** or drag & drop
2. Upload these files:
   - `llcuda-binaries-cuda12-940m.tar.gz`
   - `llcuda-binaries-cuda12-t4.tar.gz`
   - (Optional) `llcuda-binaries-cuda12-940m.tar.gz.sha256` (checksum)
   - (Optional) `llcuda-binaries-cuda12-t4.tar.gz.sha256` (checksum)

3. Wait for upload to complete (green checkmark appears)

### Step 7: Publish Release

1. Check **"Set as the latest release"** if this is the newest version
2. Click **"Publish release"** button

---

## After Publishing

### Update bootstrap.py URL

Edit `llcuda/llcuda/_internal/bootstrap.py`:

```python
# Line 24-26
GITHUB_RELEASE_URL = "https://github.com/waqasm86/llcuda/releases/download/v1.2.2"
BINARY_BUNDLE_NAME = "llcuda-binaries-cuda12.tar.gz"
```

**Note:** The bootstrap currently uses a generic name. You may need to:
- Upload a combined package as `llcuda-binaries-cuda12.tar.gz`, OR
- Update bootstrap.py to detect GPU and download the appropriate package

### Update Package Version

Edit `llcuda/llcuda/__init__.py`:

```python
__version__ = "1.2.2"
```

### Commit and Push to Main Repo

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda

# Verify .gitignore excludes binaries
git status
# Should NOT show llcuda/binaries/ or llcuda/lib/

# Add changes
git add llcuda/_internal/bootstrap.py
git add llcuda/__init__.py
git add .

# Commit
git commit -m "Release v1.2.2: CUDA 12 support for 940M and T4

- Fixed stderr.read() bug in Google Colab
- Added FlashAttention support for T4
- Optimized for GeForce 940M (CC 5.0) and Tesla T4 (CC 7.5)
- Updated bootstrap to v1.2.2 binaries
"

# Push to GitHub
git push origin main

# Create tag
git tag v1.2.2
git push origin v1.2.2
```

---

## Verify Upload Success

### Check Files Are Available

Visit:
```
https://github.com/waqasm86/llcuda/releases/download/v1.2.2/llcuda-binaries-cuda12-940m.tar.gz
https://github.com/waqasm86/llcuda/releases/download/v1.2.2/llcuda-binaries-cuda12-t4.tar.gz
```

Both should download successfully.

### Test Bootstrap Download

```python
# Test in clean environment
import llcuda

# First import should trigger download from GitHub Releases
# Should see: "Downloading optimized binaries from GitHub..."
```

---

## Create Checksums (Optional but Recommended)

```bash
cd release-packages

# SHA256 checksums
sha256sum llcuda-binaries-cuda12-940m.tar.gz > llcuda-binaries-cuda12-940m.tar.gz.sha256
sha256sum llcuda-binaries-cuda12-t4.tar.gz > llcuda-binaries-cuda12-t4.tar.gz.sha256

# Upload these .sha256 files to the same release
```

Users can verify:
```bash
sha256sum -c llcuda-binaries-cuda12-940m.tar.gz.sha256
```

---

## File Size Guidelines

### GitHub Releases Limits:
- ✅ Individual file: Up to 2GB (you're ~150MB, well within limit)
- ✅ Total release size: Unlimited
- ✅ Number of files: Unlimited

### Main Repository Limits:
- ⚠️ Repository size: Keep under 1GB (recommended)
- ⚠️ Individual file: 100MB hard limit (GitHub will reject larger files)
- ⚠️ PyPI package: Must be under 100MB

### Your Sizes:
- Main repo (Python code only): <10MB ✅
- Each binary package: ~120-160MB ✅ (goes to Releases, not main repo)
- Total release: ~250-320MB ✅

---

## Common Issues

### Issue 1: Upload fails with "file too large"

**Cause:** Trying to upload to main repo instead of Releases

**Fix:**
- Use GitHub Releases page (https://github.com/user/repo/releases)
- NOT the main code repository

### Issue 2: .gitignore not working

**Cause:** Files were committed before adding to .gitignore

**Fix:**
```bash
# Remove from git tracking (keeps local files)
git rm --cached -r llcuda/binaries/
git rm --cached -r llcuda/lib/
git rm --cached -r llcuda/models/
git commit -m "Remove binaries from tracking"
```

### Issue 3: Bootstrap can't download

**Cause:** Wrong URL in bootstrap.py

**Fix:** Check URL matches exactly:
```
https://github.com/waqasm86/llcuda/releases/download/v1.2.2/llcuda-binaries-cuda12.tar.gz
```

---

## Multi-Release Strategy

### Option A: Separate Releases (Recommended for testing)

- Release v1.2.2-940m: GeForce 940M binaries
- Release v1.2.2-t4: Tesla T4 binaries
- Users download appropriate version

### Option B: Single Release with Both Files (Recommended for production)

- Release v1.2.2: Both binaries as separate assets
- Bootstrap detects GPU and downloads appropriate file
- Cleaner versioning

### Option C: Universal Binary (Future)

- Compile with multiple architectures: `50;75`
- Single binary works on both GPUs
- Larger file size (~250MB+)
- Simpler distribution

---

## Automation (Future)

Create GitHub Action to auto-build and release:

`.github/workflows/release.yml`:
```yaml
name: Build and Release CUDA Binaries

on:
  push:
    tags:
      - 'v*'

jobs:
  build-and-release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build CUDA 12 binaries
        run: ./cmake_build_940m.sh
      - name: Create release package
        run: ./CREATE_RELEASE_PACKAGE.sh
      - name: Upload to GitHub Releases
        uses: softprops/action-gh-release@v1
        with:
          files: release-packages/*.tar.gz
```

---

## Summary Checklist

Before publishing:
- [ ] Built binaries with CMake
- [ ] Created release packages
- [ ] Tested binaries locally
- [ ] Written clear release notes
- [ ] Updated version numbers
- [ ] Verified .gitignore excludes binaries
- [ ] Created checksums (optional)

After publishing:
- [ ] Updated bootstrap.py URL
- [ ] Updated __version__ in __init__.py
- [ ] Committed to main repo (without binaries)
- [ ] Tagged release
- [ ] Tested bootstrap download
- [ ] Announced in discussions/README

---

**You're now ready to upload binaries to GitHub Releases!**
