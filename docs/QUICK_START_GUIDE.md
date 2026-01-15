# llcuda v1.2.2 - Quick Start Guide

## 🚀 Quick Reference

### What Was Fixed
**Problem:** CUDA PTX error on Google Colab T4
```
CUDA error: the provided PTX was compiled with an unsupported toolchain
```

**Solution:** Recompile with CUDA 12.6 in Google Colab (matches Colab's environment)

---

## 📋 3-Step Release Process

### Step 1: Build Binaries (Google Colab)
```
1. Go to: https://colab.research.google.com/
2. Upload: llcuda/build_binaries_colab.ipynb
3. Runtime → Change runtime type → T4 GPU
4. Run all cells (takes ~10 minutes)
5. Download: llcuda-binaries-cuda12-*.tar.gz
6. Rename to: llcuda-binaries-cuda12.tar.gz
```

### Step 2: Upload to GitHub Release
```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda
./upload_to_github_release.sh v1.2.2 llcuda-binaries-cuda12.tar.gz
```

### Step 3: Build & Upload PyPI Package
```bash
# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

---

## ✅ Verification

Test in fresh Google Colab notebook:
```python
!pip install llcuda==1.2.2

import llcuda
engine = llcuda.InferenceEngine()
engine.load_model("unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf", silent=True)
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
```

Expected: No CUDA errors, inference works!

---

## 📁 Key Files

### Build Tools
- `llcuda/build_binaries_colab.ipynb` - **Upload this to Colab**
- `llcuda/build_cuda_binaries_colab.sh` - Alternative bash script
- `llcuda/upload_to_github_release.sh` - GitHub release helper

### Documentation
- `llcuda/RELEASE_INSTRUCTIONS.md` - Detailed step-by-step guide
- `llcuda/FIXES_SUMMARY_v1.2.2.md` - What was fixed
- `llcuda/CHANGELOG.md` - Version history

### Updated Files
- `pyproject.toml` - Version 1.2.2
- `llcuda/_internal/bootstrap.py` - Points to v1.2.2 release
- `.gitignore` - Excludes large files

---

## 🎯 Important Notes

1. **DO NOT compile on your local machine** - Old GPU won't work for Colab
2. **MUST build in Google Colab with T4 GPU** - Ensures compatibility
3. **Binaries go to GitHub Releases** - NOT in PyPI package
4. **PyPI package stays < 100 KB** - Just Python code
5. **Users auto-download binaries** - On first import

---

## 🔧 Troubleshooting

**Q: Build fails in Colab?**
- Make sure you selected T4 GPU runtime
- Check that CUDA is available: `!nvcc --version`

**Q: Package too large for PyPI?**
- Check: `ls -lh dist/*.whl` (should be ~62 KB)
- Binaries should NOT be in wheel
- They download from GitHub releases

**Q: PTX error still occurs?**
- Check bootstrap.py has correct URL: `v1.2.2`
- Verify GitHub release exists with binaries
- Try clearing cache: `rm -rf ~/.cache/llcuda`

---

## 📞 Support

- **GitHub Issues**: https://github.com/waqasm86/llcuda/issues
- **Colab Notebook**: See `p3_llcuda.ipynb` for testing

---

**Ready to release?** Follow [RELEASE_INSTRUCTIONS.md](llcuda/RELEASE_INSTRUCTIONS.md) for detailed steps!
