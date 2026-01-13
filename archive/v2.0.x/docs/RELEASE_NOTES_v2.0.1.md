# llcuda v2.0.1 - CUDA 12 Binaries for Tesla T4

**Release Date:** January 7, 2026
**Python Version:** 3.11+
**CUDA Version:** 12.x
**Target GPU:** Tesla T4 (SM 7.5) - Google Colab Optimized

---

## Overview

Official CUDA 12 binary release for llcuda v2.0.1 Python package. This release provides T4-optimized binaries built specifically for Google Colab's Tesla T4 GPU with FlashAttention support.

---

## What's Included

### Tesla T4 Optimized Binary Package
**File:** `llcuda-binaries-cuda12-t4.tar.gz` (140 MB compressed, 371 MB extracted)

#### Binaries (bin/)
- **llama-server** (6.5 MB) - HTTP inference server
- **llama-cli** (4.2 MB) - Command-line interface
- **llama-embedding** (3.3 MB) - Embedding generation tool

#### CUDA Libraries (lib/)
- **libggml-cuda.so** (174 MB) - CUDA kernels with FlashAttention support
- **libggml-base.so** (721 KB) - Base GGML functionality
- **libggml-cpu.so** (1.1 MB) - CPU fallback kernels
- **libllama.so** (2.9 MB) - Llama.cpp core library
- **libmtmd.so** (877 KB) - Multi-threaded operations

### Build Configuration
- **CUDA Architecture:** SM 7.5 (Tesla T4, Turing)
- **CUDA Version:** 12.4/12.6 (compatible with 12.x)
- **FlashAttention:** ✅ Enabled (GGML_CUDA_FA=ON)
- **CUDA Graphs:** ✅ Enabled (GGML_CUDA_GRAPHS=ON)
- **Tensor Cores:** ✅ Optimized for FP16/INT8
- **All Quantization Types:** ✅ Supported (GGML_CUDA_FA_ALL_QUANTS=ON)

---

## Installation

### Automatic (Recommended)

```bash
pip install llcuda
```

The package automatically downloads T4-optimized binaries on first import (one-time, ~140 MB download).

### Verify Installation

```python
import llcuda
from llcuda.core import get_device_properties

# Check GPU
props = get_device_properties(0)
print(f"GPU: {props.name}")
print(f"Compute: SM {props.compute_capability_major}.{props.compute_capability_minor}")
```

---

## Requirements

### Hardware
- **GPU:** NVIDIA Tesla T4 (Compute Capability 7.5)
- **VRAM:** 16 GB
- **Platform:** Google Colab (free tier), Kaggle, or local T4

### Software
- **Python:** 3.11+ (Python 3.12 supported)
- **CUDA Runtime:** 12.x (12.4 or 12.6 recommended)
- **NVIDIA Drivers:** 545+ (included in Google Colab)
- **OS:** Linux x86_64 (Ubuntu 22.04+ recommended)

---

## Quick Start

### Google Colab Example

```python
# Cell 1: Install
!pip install llcuda

# Cell 2: Verify GPU
import llcuda
from llcuda.core import get_device_properties

props = get_device_properties(0)
print(f"✅ GPU: {props.name}")
print(f"   Compute: SM {props.compute_capability_major}.{props.compute_capability_minor}")
print(f"   Memory: {props.total_memory / 1024**3:.1f} GB")

# Cell 3: Run Inference (HTTP API)
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)

result = engine.infer("What is artificial intelligence?", max_tokens=100)
print(result.text)
print(f"\nSpeed: {result.tokens_per_sec:.1f} tokens/sec")
```

---

## Performance Benchmarks (Tesla T4)

| Model | Quantization | VRAM | Speed | Context | Notes |
|-------|--------------|------|-------|---------|-------|
| Gemma 3-1B | Q4_K_M | 1.2 GB | **45 tok/s** | 2048 | Recommended for quick tests |
| Llama 3.2-3B | Q4_K_M | 2.0 GB | **30 tok/s** | 4096 | Good balance |
| Qwen 2.5-7B | Q4_K_M | 5.0 GB | **18 tok/s** | 8192 | High quality |
| Llama 3.1-8B | Q4_K_M | 5.5 GB | **15 tok/s** | 8192 | Best results |

**FlashAttention Impact:** 2-3x faster for contexts > 2048 tokens

---

## Features in v2.0.1

### Core Capabilities
✅ **FlashAttention 2** - 2-3x faster attention for long contexts
✅ **Tensor Core Optimization** - FP16/INT8 acceleration on T4
✅ **CUDA Graphs** - Reduced kernel launch overhead
✅ **All Quantization Types** - Q2_K through Q8_0 supported
✅ **Dynamic GPU Detection** - Auto-configures for T4
✅ **Bootstrap System** - Auto-downloads binaries on first use

### API Options
1. **V1.x HTTP Server API** - OpenAI-compatible, easy to use
2. **V2.0 Native Tensor API** - Custom CUDA operations (planned)

---

## What's New in v2.0.1

### Improvements
- ✅ Updated bootstrap to v2.0.1 release binaries
- ✅ Optimized binary package structure (bin/ + lib/ layout)
- ✅ Reduced download size (140 MB vs previous 264 MB estimate)
- ✅ Improved GPU compatibility checking
- ✅ Better error messages for non-T4 GPUs

### Bug Fixes
- Fixed stderr.read() AttributeError in silent mode (Google Colab)
- Fixed library path detection on Colab/Kaggle
- Improved server startup reliability

---

## Manual Installation

If automatic download fails, manually install binaries:

```bash
# Download release
wget https://github.com/waqasm86/llcuda/releases/download/v2.0.1/llcuda-binaries-cuda12-t4.tar.gz

# Verify checksum (optional)
echo "54bb904f213b38aec4a2c85baa2fab8256200966aa994bb9fa4a77640d699aa4  llcuda-binaries-cuda12-t4.tar.gz" | sha256sum -c

# Extract to cache directory
mkdir -p ~/.cache/llcuda
tar -xzf llcuda-binaries-cuda12-t4.tar.gz -C ~/.cache/llcuda/

# Verify installation
ls -lh ~/.cache/llcuda/bin/llama-server
ls -lh ~/.cache/llcuda/lib/libggml-cuda.so.0
```

---

## GPU Compatibility

### ✅ Supported (Tested)
- **Tesla T4** (Google Colab, Kaggle, Cloud instances)

### ⚠️ May Work (Not Tested)
- RTX 20 series (2060, 2070, 2080) - SM 7.5
- GTX 16 series (1660, 1650) - SM 7.5
- Quadro RTX 4000/5000 - SM 7.5

### ❌ Not Supported
- GPUs with SM < 7.5 (older than Turing architecture)
- For GeForce 940M (SM 5.0), use llcuda v1.2.2

---

## Troubleshooting

### GPU Not Compatible

```
❌ INCOMPATIBLE GPU DETECTED
Your GPU is not Tesla T4
Required: Tesla T4 (SM 7.5)
```

**Solution:** Use Google Colab with Tesla T4, or use llcuda v1.2.2 for older GPUs.

### Binary Download Failed

**Solution:** Manually download binaries (see Manual Installation above).

### Server Startup Issues

```python
# Enable verbose mode to see errors
engine = llcuda.InferenceEngine()
engine.load_model("model-name", silent=False, verbose=True)
```

### Library Loading Errors

```bash
# Check library path
echo $LD_LIBRARY_PATH

# Set manually if needed
export LD_LIBRARY_PATH=~/.cache/llcuda/lib:$LD_LIBRARY_PATH
```

---

## Advanced Usage

### Custom Model Loading

```python
import llcuda

engine = llcuda.InferenceEngine()

# Load from HuggingFace with custom settings
engine.load_model(
    model_name="TheBloke/Llama-2-7B-GGUF",
    filename="llama-2-7b.Q4_K_M.gguf",
    gpu_layers=35,
    context_size=8192,
    silent=True
)

result = engine.infer("Explain quantum computing", max_tokens=200)
print(result.text)
```

### Batch Processing

```python
prompts = [
    "What is machine learning?",
    "Explain neural networks",
    "What is deep learning?"
]

results = engine.batch_infer(prompts, max_tokens=100)
for i, result in enumerate(results):
    print(f"\n{i+1}. {result.text}")
```

---

## Documentation

- **GitHub Repository:** https://github.com/waqasm86/llcuda
- **PyPI Package:** https://pypi.org/project/llcuda/
- **Quick Start Guide:** [README.md](https://github.com/waqasm86/llcuda#readme)
- **Build Guide:** [docs/BUILD_GUIDE.md](https://github.com/waqasm86/llcuda/blob/main/docs/BUILD_GUIDE.md)
- **Integration Guide:** [docs/INTEGRATION_GUIDE.md](https://github.com/waqasm86/llcuda/blob/main/docs/INTEGRATION_GUIDE.md)

---

## Support

- **Issues:** https://github.com/waqasm86/llcuda/issues
- **Discussions:** https://github.com/waqasm86/llcuda/discussions
- **Email:** waqasm86@gmail.com

---

## License

MIT License - see [LICENSE](https://github.com/waqasm86/llcuda/blob/main/LICENSE) file for details.

---

## Acknowledgments

- Built on [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT License)
- FlashAttention from [Dao-AILab](https://github.com/Dao-AILab/flash-attention)
- Designed for [Unsloth](https://github.com/unslothai/unsloth) integration
- Optimized for Google Colab Tesla T4 GPU

---

## Version Information

- **Package Version:** 2.0.1
- **Binary Version:** CUDA 12 (SM 7.5)
- **Build Date:** January 7, 2026
- **llama.cpp Version:** 0.0.7654
- **GGML Version:** 0.9.5

---

## File Checksums

```
SHA256 (llcuda-binaries-cuda12-t4.tar.gz):
54bb904f213b38aec4a2c85baa2fab8256200966aa994bb9fa4a77640d699aa4
```

---

**For older GPUs or different architectures, please see:**
- [llcuda v1.2.2](https://github.com/waqasm86/llcuda/releases/tag/v1.2.2) - GeForce 940M (SM 5.0) support
