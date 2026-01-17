# llcuda v2.2.0 - Installation Guide

## Requirements

- **Python:** 3.11+
- **CUDA:** 12.x runtime
- **GPU:** NVIDIA Tesla T4 (single or multi-GPU)
- **Platform:** Linux (Google Colab, Kaggle, or local)

## Installation Methods

### Method 1: From GitHub Release (Recommended)
```bash
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
```

### Method 2: Latest Development
```bash
pip install git+https://github.com/llcuda/llcuda.git
```

### Method 3: Development Install
```bash
git clone https://github.com/llcuda/llcuda.git
cd llcuda
pip install -e .
```

## Binary Download

On first import, llcuda automatically downloads CUDA binaries (~961 MB) from GitHub Releases:

```python
import llcuda  # Downloads binaries to ~/.cache/llcuda/
```

### Manual Binary Download
```bash
wget https://github.com/llcuda/llcuda/releases/download/v2.2.0/llcuda-v2.2.0-cuda12-kaggle-t4x2.tar.gz
mkdir -p ~/.cache/llcuda
tar -xzf llcuda-v2.2.0-cuda12-kaggle-t4x2.tar.gz -C ~/.cache/llcuda/
```

## Verification

```python
import llcuda
print(f"Version: {llcuda.__version__}")  # 2.2.0

from llcuda.api import kaggle_t4_dual_config
config = kaggle_t4_dual_config()
print(f"Multi-GPU config: {config.to_cli_args()}")
```

## Troubleshooting

### GPU Not Detected
```bash
nvidia-smi  # Verify GPU is available
```

### Binary Download Failed
Check internet connection and try manual download above.

### Import Errors
```bash
pip uninstall llcuda
pip cache purge
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
```
