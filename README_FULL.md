# llcuda: CUDA-Accelerated LLM Inference

![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Effortless, zero-configuration LLM inference with CUDA acceleration and GPU-specific optimizations.

## 🎉 What's New in v1.2.0

- **GPU-Specific Binaries**: Automatic detection and download of optimized binaries for your GPU
- **FlashAttention Support**: 2x faster inference on modern GPUs (Tesla T4, RTX series)
- **Maxwell GPU Support**: Optimized builds for older GPUs (GeForce 940M, GTX 900 series)
- **Smart Bootstrap**: Auto-selects appropriate binary bundle based on detected GPU
- **Bug Fixes**: Resolved stderr.read() issue in Google Colab

## 🚀 Quick Start

### Installation

```bash
pip install llcuda
```

**Requirements:**
- Python 3.11+
- NVIDIA GPU with Compute Capability 5.0+ (Maxwell or newer)
- CUDA 12.x runtime

### Basic Usage

```python
import llcuda

# Initialize inference engine (auto-downloads optimized binaries on first run)
engine = llcuda.InferenceEngine()

# Load a model (downloads from HuggingFace on first use)
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)

# Run inference
result = engine.infer("What is artificial intelligence?", max_tokens=100)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")
```

## 🎮 GPU Support

### Supported Architectures

| GPU Family | Compute Cap | Example GPUs | Package Size | Features |
|------------|-------------|--------------|--------------|----------|
| **Maxwell** | 5.0-5.2 | GeForce 940M, GTX 950/960 | 26 MB | cuBLAS optimized |
| **Pascal** | 6.0-6.2 | GTX 1060/1070/1080, Tesla P100 | 264 MB | Tensor cores |
| **Volta** | 7.0 | Tesla V100, Titan V | 264 MB | Tensor cores |
| **Turing** | 7.5 | Tesla T4, RTX 2060/2070/2080 | 264 MB | **FlashAttention** |
| **Ampere** | 8.0-8.6 | RTX 3060/3070/3080/3090, A100 | 264 MB | **FlashAttention** |
| **Ada** | 8.9 | RTX 4060/4070/4080/4090 | 264 MB | **FlashAttention** |

### What Gets Downloaded?

On first `import llcuda`, the bootstrap will:
1. Detect your GPU using `nvidia-smi`
2. Select appropriate binary bundle:
   - **Maxwell GPUs (CC 5.x)**: Downloads 26 MB package optimized with forced cuBLAS
   - **Modern GPUs (CC 7.0+)**: Downloads 264 MB package with FlashAttention (2x faster)
3. Extract binaries to `llcuda/binaries/cuda12/`
4. Configure library paths automatically

**This is a one-time download.** Subsequent imports use cached binaries.

## 📊 Performance Benchmarks

### GeForce 940M (Maxwell, CC 5.0)
- **Package**: 26 MB
- **GPU Layers**: 10-15
- **Context**: 512-1024 tokens
- **Models**: 1-3B params (Q4_K_M quantization)
- **Speed**: 10-20 tokens/sec

**Example:**
```
Model: Gemma 3-1B Q4_K_M
Speed: 15 tok/s
Latency: 67ms per token
```

### Tesla T4 (Turing, CC 7.5) with FlashAttention
- **Package**: 264 MB
- **GPU Layers**: 26-35
- **Context**: 2048-8192 tokens
- **Models**: 1-13B params (Q4_K_M/Q5_K_M)
- **Speed**: 25-60 tokens/sec (2x faster than without FlashAttention)

**Example:**
```
Model: Gemma 3-1B Q4_K_M
Speed: 45 tok/s
Latency: 22ms per token
```

### RTX 4090 (Ada, CC 8.9) with FlashAttention
- **Package**: 264 MB
- **GPU Layers**: 35+ (full offload for most models)
- **Context**: 8192+ tokens
- **Models**: 1-70B params
- **Speed**: 120+ tokens/sec for small models, 40-60 tok/s for 13B

**Example:**
```
Model: Gemma 3-1B Q4_K_M
Speed: 125 tok/s
Latency: 8ms per token
```

## 🛠️ Advanced Usage

### Custom Model Loading

```python
import llcuda

engine = llcuda.InferenceEngine()

# Load custom model from HuggingFace
engine.load_model(
    model_name="TheBloke/Llama-2-7B-GGUF",
    filename="llama-2-7b.Q4_K_M.gguf",
    gpu_layers=32,
    context_size=4096,
    silent=True
)

result = engine.infer("Explain quantum computing", max_tokens=200)
```

### Manual Server Management

```python
from llcuda import ServerManager

# Start llama-server manually
server = ServerManager()
server.start(
    model_path="/path/to/model.gguf",
    gpu_layers=30,
    context_size=2048,
    port=8080,
    silent=False  # Show server logs
)

# ... use server ...

server.stop()
```

### Jupyter Notebook Integration

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)

# Interactive chat widget
from llcuda.jupyter import ChatWidget
chat = ChatWidget(engine)
chat.display()

# Streaming inference
for chunk in engine.infer_stream("Write a short story about AI"):
    print(chunk, end='', flush=True)
```

## 📦 Pre-configured Models

The following models are optimized and tested:

| Model | Size | VRAM | Speed (T4) | Best For |
|-------|------|------|------------|----------|
| Gemma 3-1B | 769 MB | ~1 GB | 45 tok/s | Fast inference, chat |
| Gemma 2-2B | 1.6 GB | ~1.5 GB | 35 tok/s | Balanced quality/speed |
| Llama 3.2-3B | 2.0 GB | ~2 GB | 30 tok/s | Higher quality |
| Qwen 2.5-1.5B | 1.0 GB | ~1.2 GB | 40 tok/s | Multilingual |

**Usage:**
```python
engine.load_model("gemma-3-1b-Q4_K_M")
# Shorthand names automatically resolve to HuggingFace repos
```

## 🌐 Platform Support

### Local Systems
- **Ubuntu 22.04+** ✅
- **Windows 11 with WSL2** ✅
- **macOS** (CPU only, no CUDA)

### Cloud Notebooks
- **Google Colab** (Tesla T4/P100/V100/A100) ✅
- **Kaggle Notebooks** ✅
- **Paperspace Gradient** ✅
- **AWS SageMaker** ✅

## 🔧 Troubleshooting

### GPU Not Detected
```python
import llcuda
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"Compute Capability: {compat['compute_capability']}")
```

### Force Specific Binary Bundle
```bash
# Set environment variable before import
export LLCUDA_FORCE_BUNDLE="940m"  # or "t4"
python your_script.py
```

### Check Installed Binaries
```python
from pathlib import Path
import llcuda

binaries_dir = Path(llcuda.__file__).parent / "binaries" / "cuda12"
print(f"Binaries installed: {binaries_dir.exists()}")
print(f"llama-server: {(binaries_dir / 'llama-server').exists()}")
```

### Library Path Issues
If you encounter library loading errors:
```python
import os
import llcuda

# Check library path
lib_path = Path(llcuda.__file__).parent / "lib"
print(f"Library path: {lib_path}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
```

## 📚 Documentation

- **GitHub Repository**: https://github.com/waqasm86/llcuda
- **Build Documentation**: [BUILD_GUIDE.md](BUILD_GUIDE.md)
- **Integration Guide**: [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- **Release Workflow**: [FINAL_WORKFLOW_GUIDE.md](FINAL_WORKFLOW_GUIDE.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## 🐛 Bug Reports

Report issues at: https://github.com/waqasm86/llcuda/issues

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Uses CUDA toolkit from NVIDIA
- FlashAttention implementation from Dao et al.
- Inspired by PyTorch's ease of use

## 🔗 Links

- **PyPI**: https://pypi.org/project/llcuda/
- **GitHub**: https://github.com/waqasm86/llcuda
- **Bug Tracker**: https://github.com/waqasm86/llcuda/issues
- **Changelog**: https://github.com/waqasm86/llcuda/blob/main/CHANGELOG.md

---

**Version**: 1.2.0
**Release Date**: January 4, 2025
**CUDA Version**: 12.x
**Supported GPUs**: Compute Capability 5.0+
**License**: MIT
