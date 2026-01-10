# Complete Guide: Publishing llcuda to GitHub Releases

**Step-by-step guide for releasing llcuda v2.0.6 exclusively on GitHub**

---

## 📋 Prerequisites

- [ ] Git repository access with push permissions
- [ ] GitHub CLI (`gh`) installed and authenticated
- [ ] Python 3.11+ with `build` package
- [ ] CUDA binaries ready: `llcuda-binaries-cuda12-t4-v2.0.6.tar.gz`

---

## 🔧 Step 1: Prepare the Release

### 1.1 Update Version Numbers

Update version in all files to `2.0.6`:

- [x] `pyproject.toml` - line 11
- [x] `llcuda/__init__.py` - line 179
- [x] `llcuda/_internal/bootstrap.py` - lines 30, 34
- [x] `README.md` - multiple locations

### 1.2 Update Documentation

- [x] Remove all PyPI references
- [x] Update installation instructions to GitHub-only
- [x] Create `GITHUB_INSTALL_GUIDE.md`
- [x] Update `README.md` with new installation methods

---

## 📦 Step 2: Build Release Packages

### 2.1 Clean Previous Builds

```bash
cd /media/waqasm86/External1/Project-Nvidia-Office/llcuda
rm -rf build/ dist/ *.egg-info
```

### 2.2 Build Packages

```bash
# Install build tool if needed
pip install --upgrade build

# Build source and wheel distributions
python3 -m build
```

This creates:
- `dist/llcuda-2.0.6.tar.gz` - Source distribution
- `dist/llcuda-2.0.6-py3-none-any.whl` - Wheel package

### 2.3 Generate Checksums

```bash
cd dist/
sha256sum llcuda-2.0.6.tar.gz > llcuda-2.0.6.tar.gz.sha256
sha256sum llcuda-2.0.6-py3-none-any.whl > llcuda-2.0.6-py3-none-any.whl.sha256
cd ..
```

---

## 🚀 Step 3: Commit and Tag

### 3.1 Commit Changes

```bash
git add -A
git commit -m "Release v2.0.6: GitHub-only distribution

- Remove PyPI dependencies
- Update bootstrap to use GitHub Releases v2.0.6
- Add comprehensive GitHub installation guide
- Update all documentation for GitHub-only install
- Remove PyPI version checking"
```

### 3.2 Create Git Tag

```bash
git tag -a v2.0.6 -m "llcuda v2.0.6 - GitHub Releases Distribution

Key Changes:
- Install from GitHub: pip install git+https://github.com/waqasm86/llcuda.git
- Binaries auto-download from GitHub Releases
- No PyPI dependency
- Tesla T4 optimized (CUDA 12, FlashAttention)
- Complete GitHub installation guide

Assets:
- Python wheel package
- Source distribution
- CUDA 12 binaries (266 MB)
- Checksums for all files"
```

### 3.3 Push to GitHub

```bash
git push origin main
git push origin v2.0.6
```

---

## 📤 Step 4: Create GitHub Release

### 4.1 Create Release

```bash
gh release create v2.0.6 \
  --title "llcuda v2.0.6: GitHub-Only Distribution" \
  --notes "## 🚀 llcuda v2.0.6 - Install from GitHub

**Major Change:** llcuda is now distributed exclusively through GitHub!

### 📦 Installation

\`\`\`bash
pip install git+https://github.com/waqasm86/llcuda.git
\`\`\`

### ✨ What's New

- ✅ **GitHub-only distribution** - No PyPI dependency
- ✅ **Auto-download binaries** - From GitHub Releases on first import
- ✅ **Simpler installation** - One command, everything included
- ✅ **Faster updates** - Direct from source repository

### 📥 Installation Methods

**Method 1: Direct from GitHub (Recommended)**
\`\`\`bash
pip install git+https://github.com/waqasm86/llcuda.git
\`\`\`

**Method 2: From Release Wheel**
\`\`\`bash
pip install https://github.com/waqasm86/llcuda/releases/download/v2.0.6/llcuda-2.0.6-py3-none-any.whl
\`\`\`

**Method 3: From Source**
\`\`\`bash
git clone https://github.com/waqasm86/llcuda.git
cd llcuda
pip install -e .
\`\`\`

### 📖 Full Documentation

See [GITHUB_INSTALL_GUIDE.md](https://github.com/waqasm86/llcuda/blob/main/GITHUB_INSTALL_GUIDE.md) for complete installation instructions.

### 🎯 Features

- **Tesla T4 Optimized** - Exclusive support for Tesla T4 GPUs (SM 7.5)
- **CUDA 12** - Latest CUDA with FlashAttention support
- **Fast Inference** - 45 tok/s on Gemma 3-1B
- **Auto-setup** - Binaries download automatically on first import
- **Google Colab Ready** - Perfect for Colab free tier

### 📦 Release Assets

1. **llcuda-2.0.6-py3-none-any.whl** - Python wheel package (~100 KB)
2. **llcuda-2.0.6.tar.gz** - Source distribution (~100 KB)
3. **llcuda-binaries-cuda12-t4-v2.0.6.tar.gz** - CUDA binaries (266 MB)
4. **\*.sha256** - Checksum files for verification

### 🔧 Technical Details

- **Python:** 3.11+
- **CUDA:** 12.x
- **Target GPU:** Tesla T4 (SM 7.5)
- **Platform:** Google Colab (primary), Linux with T4
- **License:** MIT

### 📝 Changelog

- Removed PyPI distribution and dependencies
- Updated bootstrap to download binaries from GitHub Releases
- Added comprehensive GitHub installation guide
- Updated all documentation for GitHub-only workflow
- Simplified installation process
- Improved version checking (uses GitHub API instead of PyPI)

### ⚠️ Breaking Changes

- **No longer available on PyPI** - Install from GitHub instead
- Installation command changed: Use \`pip install git+https://github.com/waqasm86/llcuda.git\`
- All upgrade commands now use GitHub URLs

### 🔗 Links

- **Repository:** https://github.com/waqasm86/llcuda
- **Installation Guide:** https://github.com/waqasm86/llcuda/blob/main/GITHUB_INSTALL_GUIDE.md
- **Issues:** https://github.com/waqasm86/llcuda/issues

---

**Full Changelog:** [v2.0.3...v2.0.6](https://github.com/waqasm86/llcuda/compare/v2.0.3...v2.0.6)"
```

### 4.2 Upload Release Assets

```bash
# Upload Python packages
gh release upload v2.0.6 \
  dist/llcuda-2.0.6-py3-none-any.whl \
  dist/llcuda-2.0.6-py3-none-any.whl.sha256 \
  dist/llcuda-2.0.6.tar.gz \
  dist/llcuda-2.0.6.tar.gz.sha256

# Upload CUDA binaries (from llcuda-releases directory)
gh release upload v2.0.6 \
  /media/waqasm86/External1/Project-Nvidia-Office/llcuda-releases/llcuda-binaries-cuda12-t4-v2.0.6.tar.gz \
  /media/waqasm86/External1/Project-Nvidia-Office/llcuda-releases/llcuda-binaries-cuda12-t4-v2.0.6.tar.gz.sha256
```

---

## ✅ Step 5: Verify Release

### 5.1 Check Release Page

```bash
gh release view v2.0.6
```

Verify:
- [ ] Release is marked as "Latest"
- [ ] All 6 assets are uploaded
- [ ] Checksums are present
- [ ] Release notes are formatted correctly

### 5.2 Test Installation

**On Google Colab:**

```python
# Fresh installation test
!pip uninstall llcuda -y -q
!pip install -q git+https://github.com/waqasm86/llcuda.git

# Verify
import llcuda
print(f"llcuda version: {llcuda.__version__}")
assert llcuda.__version__ == "2.0.6", "Wrong version!"

# Test basic functionality
from llcuda.core import get_device_count
print(f"CUDA devices: {get_device_count()}")
```

### 5.3 Test Binary Download

```python
# Binaries should auto-download on first import
import llcuda  # This triggers bootstrap

# Check if binaries exist
import os
from pathlib import Path
binaries_dir = Path(llcuda.__file__).parent / "binaries" / "cuda12"
assert (binaries_dir / "llama-server").exists(), "Binaries not downloaded!"
print("✅ Binaries downloaded successfully!")
```

---

## 📊 Step 6: Update Documentation

### 6.1 Update Root PROJECT_OVERVIEW.md

Update the parent directory overview to reflect GitHub-only distribution:

```bash
cd /media/waqasm86/External1/Project-Nvidia-Office/
# Edit PROJECT_OVERVIEW.md to update installation instructions
```

### 6.2 Create Migration Guide (Optional)

If users were previously using PyPI version, create a migration guide:

```markdown
# Migrating from PyPI to GitHub

If you previously installed llcuda from PyPI, follow these steps:

1. Uninstall old version:
   \`\`\`bash
   pip uninstall llcuda -y
   \`\`\`

2. Install from GitHub:
   \`\`\`bash
   pip install git+https://github.com/waqasm86/llcuda.git
   \`\`\`

3. Verify installation:
   \`\`\`python
   import llcuda
   print(llcuda.__version__)  # Should show 2.0.6
   \`\`\`
```

---

## 🎉 Step 7: Announce Release

### 7.1 Update README Badge

The version badge in README.md already shows v2.0.6.

### 7.2 GitHub Discussion (Optional)

Create a discussion thread announcing the move to GitHub-only distribution:

```bash
# Example discussion post
gh api repos/waqasm86/llcuda/discussions \
  -f category="announcements" \
  -f title="llcuda v2.0.6: Now Distributed Exclusively on GitHub!" \
  -f body="We've simplified llcuda distribution! Starting with v2.0.6, llcuda is available exclusively through GitHub. Install with: pip install git+https://github.com/waqasm86/llcuda.git"
```

---

## 📝 Checklist: Complete Release

### Pre-Release
- [x] Updated version numbers in all files
- [x] Updated bootstrap.py to use v2.0.6 binaries
- [x] Removed PyPI references
- [x] Created GITHUB_INSTALL_GUIDE.md
- [x] Updated README.md

### Build
- [ ] Cleaned previous builds
- [ ] Built source distribution
- [ ] Built wheel package
- [ ] Generated checksums

### Git
- [ ] Committed all changes
- [ ] Created git tag v2.0.6
- [ ] Pushed to GitHub

### GitHub Release
- [ ] Created release v2.0.6
- [ ] Uploaded wheel package
- [ ] Uploaded source distribution
- [ ] Uploaded CUDA binaries
- [ ] Uploaded all checksums
- [ ] Marked as latest release

### Testing
- [ ] Tested installation on Google Colab
- [ ] Verified binary auto-download
- [ ] Tested basic functionality
- [ ] Verified version number

### Documentation
- [ ] Updated parent directory docs
- [ ] Verified all links work
- [ ] Checked installation guide

---

## 🔗 Quick Commands Summary

```bash
# Build packages
cd llcuda
python3 -m build
cd dist && sha256sum * > checksums.txt && cd ..

# Git operations
git add -A
git commit -m "Release v2.0.6: GitHub-only distribution"
git tag -a v2.0.6 -m "Release v2.0.6"
git push origin main
git push origin v2.0.6

# Create GitHub release
gh release create v2.0.6 --title "llcuda v2.0.6" --notes-file RELEASE_NOTES.md

# Upload assets
gh release upload v2.0.6 dist/* ../llcuda-releases/*

# Verify
gh release view v2.0.6
```

---

## 📞 Support

If issues arise:
- Check GitHub Issues: https://github.com/waqasm86/llcuda/issues
- Review installation guide: GITHUB_INSTALL_GUIDE.md
- Test on fresh Colab environment

---

**Author:** Waqas Muhammad
**Date:** January 2026
**Version:** 2.0.6
