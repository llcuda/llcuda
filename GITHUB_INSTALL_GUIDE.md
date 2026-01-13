# llcuda v2.1.0 - GitHub Installation Guide

**Install directly from GitHub - No PyPI needed!**

---

## 🚀 Quick Install

### Method 1: Direct from GitHub (Recommended)

```bash
pip install git+https://github.com/waqasm86/llcuda.git
```

This will:
- ✅ Clone the latest code from GitHub
- ✅ Install the Python package
- ✅ Automatically download CUDA binaries from GitHub Releases on first import

### Method 2: Install from Specific Release

```bash
pip install https://github.com/waqasm86/llcuda/releases/download/v2.1.0/llcuda-2.1.0-py3-none-any.whl
```

### Method 3: Install from Source

```bash
git clone https://github.com/waqasm86/llcuda.git
cd llcuda
pip install -e .
```

---

## 📦 What Gets Installed

### Python Package
- **Source:** GitHub repository (main branch or release tag)
- **Size:** ~100 KB (Python code only)
- **Contents:** Python package, no binaries included

### CUDA Binaries (Downloaded Automatically)
- **Source:** GitHub Releases (v2.0.6 binaries - 100% compatible with v2.1.0)
- **URL:** https://github.com/waqasm86/llcuda/releases/download/v2.0.6/llcuda-binaries-cuda12-t4-v2.0.6.tar.gz
- **Size:** 266 MB (downloaded once, cached locally)
- **Downloaded:** On first `import llcuda`
- **Location:** `~/.cache/llcuda/` or `<package>/binaries/`
- **Note:** v2.1.0 uses v2.0.6 binaries (pure Python API layer on top)

---

## 🎯 Installation Methods Explained

### 1. Direct GitHub Install (`git+https://...`)

**Best for:** Most users, especially on Google Colab

```bash
pip install git+https://github.com/waqasm86/llcuda.git
```

**Advantages:**
- ✅ Always gets latest code
- ✅ Simple one-line command
- ✅ Works everywhere (Colab, Kaggle, local)
- ✅ Automatic binary download

**How it works:**
1. pip clones the GitHub repository
2. pip installs the Python package
3. On first `import llcuda`, binaries are downloaded from GitHub Releases

### 2. Wheel Install (`.whl` file)

**Best for:** Offline installation or specific version pinning

```bash
# Download the wheel first
wget https://github.com/waqasm86/llcuda/releases/download/v2.1.0/llcuda-2.1.0-py3-none-any.whl

# Install
pip install llcuda-2.1.0-py3-none-any.whl
```

**Advantages:**
- ✅ Specific version control
- ✅ Can be downloaded and saved for offline use
- ✅ Faster installation (no git clone needed)

### 3. Editable Install (Development)

**Best for:** Development and testing

```bash
git clone https://github.com/waqasm86/llcuda.git
cd llcuda
pip install -e .
```

**Advantages:**
- ✅ Live code changes (no reinstall needed)
- ✅ Full access to source code
- ✅ Easy to modify and test

---

## 🔄 Upgrading

### Upgrade to Latest Version

```bash
pip install --upgrade --no-cache-dir git+https://github.com/waqasm86/llcuda.git
```

### Upgrade to Specific Version

```bash
pip install --upgrade https://github.com/waqasm86/llcuda/releases/download/v2.0.6/llcuda-2.0.6-py3-none-any.whl
```

### Force Reinstall (if having issues)

```bash
pip uninstall llcuda -y
pip install --no-cache-dir git+https://github.com/waqasm86/llcuda.git
```

---

## 🌐 Platform-Specific Instructions

### Google Colab

```python
# Install llcuda
!pip install -q git+https://github.com/waqasm86/llcuda.git

# Import (binaries download automatically)
import llcuda
print(f"llcuda version: {llcuda.__version__}")
```

### Kaggle

```python
# In Kaggle notebook
!pip install -q git+https://github.com/waqasm86/llcuda.git

import llcuda
```

### Local Ubuntu/Debian

```bash
# Install system dependencies (if needed)
sudo apt-get update
sudo apt-get install -y python3-pip git

# Install llcuda
pip3 install git+https://github.com/waqasm86/llcuda.git
```

---

## 📥 Manual Binary Download (Optional)

If you prefer to download binaries manually:

### Download Binaries

```bash
# Download the binary package
wget https://github.com/waqasm86/llcuda/releases/download/v2.0.6/llcuda-binaries-cuda12-t4-v2.0.6.tar.gz

# Extract to llcuda package directory
tar -xzf llcuda-binaries-cuda12-t4-v2.0.6.tar.gz -C ~/.cache/llcuda/
```

### Verify Checksum

```bash
# Download checksum
wget https://github.com/waqasm86/llcuda/releases/download/v2.0.6/llcuda-binaries-cuda12-t4-v2.0.6.tar.gz.sha256

# Verify
sha256sum -c llcuda-binaries-cuda12-t4-v2.0.6.tar.gz.sha256
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
- **GPU:** Tesla T4 (SM 7.5)
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
pip install https://github.com/waqasm86/llcuda/releases/download/v2.0.6/llcuda-2.0.6-py3-none-any.whl
```

### Issue: Binary download fails

**Solution 1:** Check internet connection and retry
```bash
pip uninstall llcuda -y
pip install --no-cache-dir git+https://github.com/waqasm86/llcuda.git
```

**Solution 2:** Download binaries manually (see above)

### Issue: "Incompatible GPU"

**Solution:** llcuda v2.0.6 requires Tesla T4 (SM 7.5). Use Google Colab for access to T4 GPUs.

### Issue: Installation succeeds but import fails

**Solution:** Check Python version
```bash
python --version  # Should be 3.11+
```

---

## 📚 Available Release Assets

Each GitHub release includes:

1. **llcuda-{version}-py3-none-any.whl** - Python wheel package
2. **llcuda-{version}.tar.gz** - Source distribution
3. **llcuda-binaries-cuda12-t4-{version}.tar.gz** - CUDA binaries (266 MB)
4. **\*.sha256** - Checksum files for verification

---

## 🔗 Links

- **GitHub Repository:** https://github.com/waqasm86/llcuda
- **Releases:** https://github.com/waqasm86/llcuda/releases
- **Latest Release:** https://github.com/waqasm86/llcuda/releases/latest
- **Issues:** https://github.com/waqasm86/llcuda/issues

---

## ❓ Why GitHub Only?

llcuda v2.0.6 is distributed exclusively through GitHub to:
- ✅ Provide faster updates and releases
- ✅ Reduce dependency on external platforms
- ✅ Give users direct access to source code
- ✅ Simplify the distribution pipeline
- ✅ Keep everything in one place (code + binaries + releases)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

**Author:** Waqas Muhammad (waqasm86@gmail.com)
**Version:** 2.0.6
**Last Updated:** January 2026
