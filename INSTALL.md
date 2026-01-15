# llcuda v2.1.0 - Installation Guide

**Install directly from GitHub - Fast & Simple!**

---

## 🚀 Quick Install

### One-Line Installation (Recommended)

```bash
pip install git+https://github.com/llcuda/llcuda.git
```

This will:
- ✅ Clone the latest code from GitHub
- ✅ Install the Python package
- ✅ Automatically download CUDA binaries from GitHub Releases on first import

### Alternative Methods

#### Install from Specific Release

```bash
pip install https://github.com/llcuda/llcuda/releases/download/v2.1.0/llcuda-2.1.0-py3-none-any.whl
```

#### Install from Source (Development)

```bash
git clone https://github.com/llcuda/llcuda.git
cd llcuda
pip install -e .
```

---

## 📦 What Gets Installed

### Python Package
- **Source:** GitHub repository (llcuda organization)
- **Size:** ~100 KB (Python code only)
- **Contents:** Pure Python package, no binaries included

### CUDA Binaries (Downloaded Automatically)
- **Source:** GitHub Releases (v2.0.6 binaries - 100% compatible with v2.1.0+)
- **URL:** https://github.com/llcuda/llcuda/releases/download/v2.0.6/llcuda-binaries-cuda12-t4-v2.0.6.tar.gz
- **Size:** 266 MB (downloaded once, cached locally)
- **Downloaded:** On first `import llcuda`
- **Location:** `~/.cache/llcuda/` or `<package>/binaries/`
- **Note:** v2.1.0 uses v2.0.6 binaries (pure Python API layer on top)

---

## 🔄 Upgrading

### Upgrade to Latest Version

```bash
pip install --upgrade --no-cache-dir git+https://github.com/llcuda/llcuda.git
```

### Force Reinstall (if having issues)

```bash
pip uninstall llcuda -y
pip install --no-cache-dir git+https://github.com/llcuda/llcuda.git
```

---

## 🌐 Platform-Specific Instructions

### Google Colab

```python
# Install llcuda
!pip install -q git+https://github.com/llcuda/llcuda.git

# Import (binaries download automatically)
import llcuda
print(f"llcuda version: {llcuda.__version__}")
```

### Kaggle

```python
# In Kaggle notebook
!pip install -q git+https://github.com/llcuda/llcuda.git

import llcuda
```

### Local Ubuntu/Debian

```bash
# Install system dependencies (if needed)
sudo apt-get update
sudo apt-get install -y python3-pip git

# Install llcuda
pip3 install git+https://github.com/llcuda/llcuda.git
```

---

## 🧪 Verify Installation

```python
import llcuda

# Check version
print(f"llcuda version: {llcuda.__version__}")

# Check CUDA availability
from llcuda.core import get_device_count
print(f"CUDA devices: {get_device_count()}")

# Quick test
engine = llcuda.InferenceEngine()
print("✅ llcuda installed successfully!")
```

---

## 📋 Requirements

### System Requirements
- **GPU:** Tesla T4 (SM 7.5) or higher
- **CUDA:** 12.x (pre-installed on Colab/Kaggle)
- **OS:** Linux (Ubuntu/Debian recommended)
- **Python:** 3.11+

### Python Dependencies
All dependencies are automatically installed:
- `numpy>=1.24.0`
- `requests>=2.31.0`
- `huggingface_hub>=0.20.0`
- `tqdm>=4.65.0`

---

## 🔧 Troubleshooting

### Issue: "No module named 'git'"

**Solution:** Install git
```bash
# Ubuntu/Debian
sudo apt-get install -y git

# Or use wheel install method instead
pip install https://github.com/llcuda/llcuda/releases/download/v2.1.0/llcuda-2.1.0-py3-none-any.whl
```

### Issue: Binary download fails

**Solution:** Check internet connection and retry
```bash
pip uninstall llcuda -y
pip install --no-cache-dir git+https://github.com/llcuda/llcuda.git
```

### Issue: "Incompatible GPU"

**Solution:** llcuda v2.0.6+ requires Tesla T4 (SM 7.5) or higher. Use Google Colab for access to T4 GPUs.

---

## 🔗 Links

- **GitHub Repository:** https://github.com/llcuda/llcuda
- **Documentation:** https://llcuda.github.io/
- **Releases:** https://github.com/llcuda/llcuda/releases
- **Issues:** https://github.com/llcuda/llcuda/issues

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

**Author:** Waqas Muhammad (waqasm86@gmail.com)
**Version:** 2.1.0
**Last Updated:** January 2026
