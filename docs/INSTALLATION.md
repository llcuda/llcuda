# llcuda v2.2.0 - Complete Installation Guide

This guide covers all installation methods for llcuda, the CUDA 12 inference backend for Unsloth.

---

## Table of Contents

- [Requirements](#requirements)
- [Quick Install](#quick-install)
- [Installation Methods](#installation-methods)
- [Kaggle Installation](#kaggle-installation)
- [Google Colab Installation](#google-colab-installation)
- [Binary Management](#binary-management)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Requirements

### Hardware
| Component | Requirement |
|-----------|-------------|
| **GPU** | NVIDIA Tesla T4 or compatible (SM 7.5+) |
| **VRAM** | 15GB (single T4) or 30GB (dual T4) |
| **RAM** | 16GB+ recommended |

### Software
| Component | Requirement |
|-----------|-------------|
| **Python** | 3.11 or higher |
| **CUDA** | 12.x runtime |
| **OS** | Linux (Ubuntu 20.04+, Kaggle, Colab) |
| **pip** | 23.0+ |

### Verify Requirements
```bash
# Check Python version
python --version  # Should be 3.11+

# Check CUDA
nvidia-smi  # Should show CUDA 12.x

# Check pip
pip --version
```

---

## Quick Install

### One-Line Install
```bash
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
```

### With Verification
```python
import llcuda
print(f"✅ llcuda {llcuda.__version__} installed")
```

---

## Installation Methods

### Method 1: From GitHub Release (Recommended)

Install the stable v2.2.0 release:

```bash
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
```

**Pros:**
- Stable, tested release
- Reproducible builds
- Best for production

### Method 2: Latest Development

Install the latest development version:

```bash
pip install git+https://github.com/llcuda/llcuda.git
```

**Pros:**
- Newest features
- Bug fixes
- Active development

**Cons:**
- May have breaking changes
- Less tested

### Method 3: From PyPI (When Available)

```bash
pip install llcuda
```

### Method 4: Development Install

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/llcuda/llcuda.git
cd llcuda

# Install in development mode
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

**Pros:**
- Editable installation
- Full source access
- Run tests locally

### Method 5: With Specific Dependencies

```bash
# With Jupyter support
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
pip install -r requirements-jupyter.txt

# With all optional dependencies
pip install "llcuda[full] @ git+https://github.com/llcuda/llcuda.git@v2.2.0"
```

---

## Kaggle Installation

### Standard Kaggle Notebook

```python
# Cell 1: Install llcuda
!pip install -q git+https://github.com/llcuda/llcuda.git@v2.2.0

# Cell 2: Verify installation
import llcuda
print(f"llcuda {llcuda.__version__}")

# Cell 3: Check GPUs
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

### Kaggle with Hugging Face

```python
!pip install -q git+https://github.com/llcuda/llcuda.git@v2.2.0
!pip install -q huggingface_hub

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
    local_dir="/kaggle/working/models"
)
```

### Kaggle Environment Settings

For optimal performance on Kaggle:

| Setting | Value |
|---------|-------|
| GPU | T4 × 2 |
| Internet | Enabled |
| Persistence | Files only |

---

## Google Colab Installation

### Standard Colab Notebook

```python
# Cell 1: Install llcuda
!pip install -q git+https://github.com/llcuda/llcuda.git@v2.2.0

# Cell 2: Check GPU
!nvidia-smi

# Cell 3: Import and verify
import llcuda
print(f"llcuda {llcuda.__version__}")
```

### Colab with Unsloth

```python
# Install Unsloth first
!pip install unsloth

# Then install llcuda
!pip install -q git+https://github.com/llcuda/llcuda.git@v2.2.0
```

---

## Binary Management

llcuda uses pre-compiled CUDA binaries for llama.cpp. These are managed automatically.

### Automatic Download

On first import, llcuda downloads binaries (~961 MB) from GitHub Releases:

```python
import llcuda  # Downloads to ~/.cache/llcuda/
```

### Manual Binary Download

If automatic download fails:

```bash
# Download from GitHub Releases
wget https://github.com/llcuda/llcuda/releases/download/v2.2.0/llcuda-v2.2.0-cuda12-kaggle-t4x2.tar.gz

# Extract to cache directory
mkdir -p ~/.cache/llcuda
tar -xzf llcuda-v2.2.0-cuda12-kaggle-t4x2.tar.gz -C ~/.cache/llcuda/

# Verify
ls ~/.cache/llcuda/bin/
```

### Binary Contents

| Binary | Description |
|--------|-------------|
| `llama-server` | HTTP server with OpenAI API |
| `llama-cli` | Command-line interface |
| `llama-quantize` | GGUF quantization tool |
| `llama-gguf` | GGUF metadata tool |
| `llama-embedding` | Embedding extraction |
| `llama-perplexity` | Perplexity calculation |

### Custom Binary Location

```python
import os
os.environ['LLCUDA_BINARY_PATH'] = '/custom/path/to/binaries'

import llcuda
```

---

## Verification

### Basic Verification

```python
import llcuda

# Check version
print(f"Version: {llcuda.__version__}")

# Check available modules
from llcuda.api import LlamaCppClient
from llcuda.server import ServerManager, ServerConfig
from llcuda.api.gguf import GGUFParser

print("✅ All core modules available")
```

### GPU Verification

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
```

### Multi-GPU Configuration

```python
from llcuda.api import kaggle_t4_dual_config

config = kaggle_t4_dual_config()
print(f"CLI args: {config.to_cli_args()}")
```

### Full System Check

```python
def verify_llcuda():
    """Complete llcuda verification."""
    checks = []
    
    # Check 1: Import
    try:
        import llcuda
        checks.append(("Import", True, llcuda.__version__))
    except ImportError as e:
        checks.append(("Import", False, str(e)))
        return checks
    
    # Check 2: CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        checks.append(("CUDA", cuda_ok, f"{gpu_count} GPU(s)"))
    except Exception as e:
        checks.append(("CUDA", False, str(e)))
    
    # Check 3: Modules
    try:
        from llcuda.api import LlamaCppClient
        from llcuda.server import ServerManager
        checks.append(("Modules", True, "OK"))
    except Exception as e:
        checks.append(("Modules", False, str(e)))
    
    # Check 4: Binary path
    try:
        import os
        binary_path = os.path.expanduser("~/.cache/llcuda/bin")
        exists = os.path.exists(binary_path)
        checks.append(("Binaries", exists, binary_path if exists else "Not found"))
    except Exception as e:
        checks.append(("Binaries", False, str(e)))
    
    return checks

# Run verification
for name, status, info in verify_llcuda():
    icon = "✅" if status else "❌"
    print(f"{icon} {name}: {info}")
```

---

## Troubleshooting

### Common Issues

#### 1. Import Error

**Problem:** `ModuleNotFoundError: No module named 'llcuda'`

**Solution:**
```bash
pip uninstall llcuda
pip cache purge
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
```

#### 2. CUDA Not Found

**Problem:** `CUDA not available` or `nvidia-smi not found`

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# On Kaggle/Colab, ensure GPU is enabled in settings
```

#### 3. Binary Download Failed

**Problem:** Timeout or connection error during binary download

**Solution:**
```bash
# Manual download
wget https://github.com/llcuda/llcuda/releases/download/v2.2.0/llcuda-binaries-cuda12-t4x2-v2.2.0.tar.gz
mkdir -p ~/.cache/llcuda
tar -xzf llcuda-binaries-cuda12-t4x2-v2.2.0.tar.gz -C ~/.cache/llcuda/
```

#### 4. Permission Denied

**Problem:** Cannot execute binaries

**Solution:**
```bash
chmod +x ~/.cache/llcuda/bin/*
```

#### 5. Wrong Python Version

**Problem:** Syntax errors or module issues

**Solution:**
```bash
# Check Python version
python --version

# Use Python 3.11+
python3.11 -m pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
```

#### 6. Conflicting Packages

**Problem:** Version conflicts with other packages

**Solution:**
```bash
# Create fresh environment
python -m venv llcuda-env
source llcuda-env/bin/activate
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
```

### Getting Help

If issues persist:

1. Check [GitHub Issues](https://github.com/llcuda/llcuda/issues)
2. Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. Open a new issue with:
   - Python version
   - CUDA version
   - Error message
   - Platform (Kaggle/Colab/Local)

---

## Next Steps

After installation:

1. **[Quick Start Guide](../QUICK_START.md)** - Get started in 5 minutes
2. **[Configuration Guide](CONFIGURATION.md)** - Server and client options
3. **[Tutorial Notebooks](../notebooks/README.md)** - Step-by-step tutorials
