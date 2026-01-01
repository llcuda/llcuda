# 🚀 llcuda v1.1.4 - CUDA-Accelerated LLM Inference for Python

**PyTorch-style self-contained package with CUDA 12.8 binaries and libraries.**  
**No manual setup required - just `pip install llcuda` and use!**

[![PyPI version](https://badge.fury.io/py/llcuda.svg)](https://pypi.org/project/llcuda/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.1+](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **Self-contained**: Includes CUDA 12.8 binaries and libraries - no separate CUDA installation needed
- **Auto-download**: Binaries download automatically on first import (models download on-demand)
- **Multi-GPU Support**: Automatically detects and optimizes for your GPU
- **Colab/Kaggle Ready**: Works out-of-the-box in cloud notebooks
- **GGUF Support**: Load any GGUF model (Llama, Gemma, Phi, Mistral, etc.)
- **Smart Configuration**: Auto-tunes settings for optimal performance
- **Simple API**: PyTorch-style interface for easy adoption

## 📦 Installation

```bash
# Install from PyPI
pip install llcuda

# For development version
pip install git+https://github.com/waqasm86/llcuda.git
