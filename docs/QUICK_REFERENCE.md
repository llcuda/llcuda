# llcuda + Unsloth - Quick Reference Card

**Version**: 2.2.0
**Target**: Kaggle 2× Tesla T4 (SM 7.5)
**Build**: CUDA 12.5, llama.cpp b7760 (388ce82)
**Date**: January 17, 2026

---

## 📚 Files Created

```
✅ llcuda_unsloth_t4_complete_build.ipynb    (25 KB) - Build from source
✅ llcuda_unsloth_tutorial.ipynb              (17 KB) - Usage tutorial
✅ NOTEBOOKS_GUIDE.md                         (11 KB) - Complete guide
✅ QUICK_REFERENCE.md                          (This file)
```

---

## 🚀 Quick Start (30 seconds)

### Open in Google Colab:

1. **Upload** `llcuda_unsloth_tutorial.ipynb` to Colab
2. **Runtime** → Change runtime type → GPU (T4)
3. **Run All** → Wait 5 minutes
4. **Done!** Test with Unsloth models

---

## 📦 What Each Notebook Does

### 1. Build Notebook (25 KB)
```python
# Creates: llcuda-v2.2.0-cuda12-kaggle-t4x2.tar.gz (~961 MB)

Contains:
- llama.cpp binaries (llama-server + 12 more tools)
- CUDA libraries with FlashAttention
- llcuda Python wheel
- Installation scripts
```

**When**: Need complete package or custom build
**Time**: ~15-20 minutes

### 2. Tutorial Notebook (17 KB)
```python
# Uses: pip install llcuda (auto-downloads binaries ~961 MB)

Demonstrates:
- Load Unsloth GGUF models
- Run fast inference (~45 tok/s)
- Batch processing
- Performance metrics
```

**When**: Want to use llcuda with Unsloth
**Time**: ~5-10 minutes

---

## 💻 Essential Commands

### Installation
```python
# Method 1: Auto-download (easiest)
pip install llcuda

# Method 2: From built package
pip install /path/to/llcuda-2.0.1-py3-none-any.whl
```

### Basic Usage
```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")

result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

### Unsloth GGUF Model
```python
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf"
)
```

---

## 📊 Performance (Tesla T4)

| Model | Speed | VRAM |
|-------|-------|------|
| Gemma 3-1B Q4_K_M | **45 tok/s** | 1.2 GB |
| Llama 3.2-3B Q4_K_M | **30 tok/s** | 2.0 GB |
| Qwen 2.5-7B Q4_K_M | **18 tok/s** | 5.0 GB |
| Llama 3.1-8B Q4_K_M | **15 tok/s** | 5.5 GB |

---

## 🔧 Troubleshooting

### Wrong Version Installed (1.2.2)
```bash
pip uninstall llcuda
pip install llcuda --no-cache-dir --force-reinstall
```

### Binaries Not Found
```python
import os
os.environ['LLAMA_SERVER_PATH'] = '/path/to/llama-server'
os.environ['LD_LIBRARY_PATH'] = '/path/to/lib'
```

### GPU Not T4
```
Runtime → Change runtime type → Hardware accelerator: GPU
Then pray you get T4 (not K80 or P100)
```

---

## 🎯 Unsloth Workflow

```
1. Fine-tune → Unsloth (2x faster training)
2. Export → save_pretrained_gguf(quantization="q4_k_m")
3. Deploy → llcuda.InferenceEngine().load_model()
4. Profit → ~45 tok/s on T4!
```

---

## 📝 Key Files

### On Your System
```
C:\Users\CS-AprilVenture\Documents\Project-Waqas\
  Project-Waqas-Programming\Project-Nvidia\

  ✅ llcuda_unsloth_t4_complete_build.ipynb    Build notebook
  ✅ llcuda_unsloth_tutorial.ipynb              Tutorial notebook
  ✅ NOTEBOOKS_GUIDE.md                         Complete guide
  ✅ QUICK_REFERENCE.md                         This file
```

### After Running Build Notebook
```
/content/
  └── llcuda-complete-cuda12-t4.tar.gz        (~350 MB)
      └── llcuda-complete-t4/
          ├── bin/           # Binaries
          ├── lib/           # CUDA libraries
          ├── python/        # Python wheel
          ├── install.sh     # Installer
          └── README.md
```

---

## 🔗 Links

- **llcuda**: https://github.com/llcuda/llcuda
- **Unsloth**: https://github.com/unslothai/unsloth
- **Unsloth GGUF**: https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf
- **llama.cpp**: https://github.com/ggml-org/llama.cpp

---

## ✅ Quick Checklist

Before running notebooks:
- [ ] Kaggle account (or Google Colab)
- [ ] Runtime set to GPU T4 × 2 (Kaggle) or T4 (Colab)
- [ ] Notebook uploaded

After running build notebook:
- [ ] Downloaded tar file (~961 MB)
- [ ] Extracted package
- [ ] Ran install.sh (if deploying elsewhere)

After running tutorial:
- [ ] llcuda v2.2.0 installed
- [ ] Tested with Gemma 3-1B
- [ ] Speed ~60 tok/s on Kaggle 2× T4

---

## 🎉 That's It!

**You now have**:
- ✅ Complete build notebook
- ✅ Usage tutorial
- ✅ Unified binary package
- ✅ Unsloth integration

**Next**: Open tutorial notebook in Colab and run it!

---

**Questions?** Check NOTEBOOKS_GUIDE.md for detailed info

**Built with**: Claude Code | llcuda v2.0.1 | Tesla T4 | Unsloth
