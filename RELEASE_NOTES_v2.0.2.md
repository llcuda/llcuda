# llcuda v2.0.2 Release Notes

**Release Date:** January 8, 2026

## 🐛 Critical Bug Fixes

This is a critical bug fix release that resolves installation failures on Kaggle and other cloud platforms.

### Fixed Issues

1. **404 Download Error on First Import** ✅
   - **Issue:** Users installing llcuda v2.0.0/v2.0.1 from PyPI encountered HTTP 404 errors when binaries tried to auto-download
   - **Root Cause:** Version mismatch between PyPI package and GitHub release URLs
   - **Fix:** Updated bootstrap to download from v2.0.2 release with corrected tar file structure

2. **Version Number Inconsistency** ✅
   - **Issue:** `__version__` reported "1.2.2" while package was actually v2.0.x
   - **Fix:** Updated `llcuda/__init__.py` to correctly report version "2.0.2"

3. **Tar File Structure Mismatch** ✅
   - **Issue:** Binary tar file had unexpected parent directory (`llcuda-complete-t4/`), causing extraction failures
   - **Expected:** `bin/` and `lib/` at root level
   - **Actual:** `llcuda-complete-t4/bin/` and `llcuda-complete-t4/lib/`
   - **Fix:** Recreated tar file with correct structure for proper extraction

### Technical Details

**Before (Broken):**
```
llcuda-binaries-cuda12-t4.tar.gz
└── llcuda-complete-t4/     ← Extra parent directory
    ├── bin/
    │   └── llama-server
    └── lib/
        └── libggml-cuda.so
```

**After (Fixed):**
```
llcuda-binaries-cuda12-t4.tar.gz
├── bin/                     ← Direct root level
│   └── llama-server
└── lib/
    └── libggml-cuda.so
```

## 📦 Package Improvements

### Enhanced .gitignore
- Strengthened protection against accidentally committing large binary files
- Explicit exclusion of `*.so.*`, `*.a`, and all shared library variants
- Better documentation of what should/shouldn't be committed

### File Information

**Binary Package:**
- Filename: `llcuda-binaries-cuda12-t4-v2.0.2.tar.gz`
- Size: 266 MB
- SHA256: `1dcf78936f3e0340a288950cbbc0e7bf12339d7b9dfbd1fe0344d44b6ead39b5`
- Contents: Tesla T4 optimized binaries with FlashAttention + CUDA Graphs

## 🚀 Upgrade Instructions

### For Existing Users (v2.0.0 or v2.0.1)

If you experienced 404 errors, upgrade immediately:

```bash
pip install --upgrade llcuda
```

Then verify the installation:

```python
import llcuda
print(f"Version: {llcuda.__version__}")  # Should show: 2.0.2

# Test binary download
engine = llcuda.InferenceEngine()
# First import will auto-download fixed binaries
```

### For New Users

```bash
pip install llcuda
```

No special steps required - installation now works correctly out of the box.

## 🔍 Compatibility

- **Python:** 3.11+
- **CUDA:** 12.x
- **GPU:** Tesla T4 (SM 7.5) - primary target
- **Platforms:** Google Colab, Kaggle, local Linux with NVIDIA GPU

## ⚠️ Breaking Changes

**None** - This is a backward-compatible bug fix release.

All v2.0.0 and v2.0.1 code will work without changes after upgrading to v2.0.2.

## 📝 Notes

- v2.0.0 users should upgrade to avoid download failures
- Binary download is one-time only (~266 MB on first import)
- Binaries cached in `~/.cache/llcuda/` for reuse

## 🙏 Acknowledgments

Thanks to users who reported the 404 installation issues on Kaggle and Colab!

---

For full changelog, see [CHANGELOG.md](CHANGELOG.md)
