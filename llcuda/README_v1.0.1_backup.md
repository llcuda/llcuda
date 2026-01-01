# llcuda v1.0.1 - PyTorch-Style CUDA LLM Inference

**Zero-configuration CUDA-accelerated LLM inference for Python with bundled binaries, smart model loading, and hardware auto-configuration.**

[![PyPI version](https://badge.fury.io/py/llcuda.svg)](https://pypi.org/project/llcuda/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/waqasm86/llcuda)](https://github.com/waqasm86/llcuda/stargazers)

> **Perfect for**: Low-VRAM NVIDIA GPUs (GeForce 900/800 series) • Zero-configuration setup • PyTorch-style API • Production-ready inference

---

## 🎯 What is llcuda v1.0.1?

A **PyTorch-style Python package** that makes LLM inference on low-VRAM NVIDIA GPUs as easy as:

```bash
# Install latest version
pip install --upgrade llcuda

# Or install specific version
pip install llcuda==1.0.1
```

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")  # Auto-downloads from HuggingFace
result = engine.infer("What is AI?")
print(result.text)
```

**That's it.** No manual binary downloads, no LLAMA_SERVER_PATH, no configuration files.

---

## ✨ What's New in v1.0.1

🐛 **Critical Bug Fixes** for low-VRAM GPU compatibility:

- **Fixed `--n-batch` Error** - Corrected parameter mapping for `batch_size` and `ubatch_size`
- **Fixed Library Loading** - Automatic `LD_LIBRARY_PATH` configuration for shared libraries
- **Improved Error Messages** - Better parameter validation and error reporting
- **Tested on GeForce 940M** - Verified working with 1GB VRAM (10-15 tok/s)

### Previous (v1.0.0) Features:

- **Bundled CUDA Binaries** (47 MB wheel) - llama-server + all libraries included
- **Zero Configuration** - Works immediately after `pip install --upgrade llcuda`
- **Smart Model Loading** - 11 curated models with auto-download from HuggingFace
- **Hardware Auto-Config** - Detects your GPU VRAM and optimizes settings automatically
- **Model Registry** - Pre-configured models tested on GeForce 940M
- **Performance Metrics** - Built-in P50/P95/P99 latency tracking

---

## 🚀 Quick Start

### Installation

```bash
# Install or upgrade to latest version
pip install --upgrade llcuda

# Or install specific version
pip install llcuda==1.0.1
```

**That's all you need!** The package includes:
- llama-server executable (CUDA 12.8)
- All required shared libraries
- Auto-configuration on import
- Automatic library path setup (NEW in v1.0.1)

### Basic Usage

```python
import llcuda

# Create inference engine
engine = llcuda.InferenceEngine()

# Load model (auto-downloads with confirmation)
engine.load_model("gemma-3-1b-Q4_K_M")

# Run inference
result = engine.infer("Explain quantum computing in simple terms.")
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

### JupyterLab Usage

```python
import llcuda

engine = llcuda.InferenceEngine()

# Auto-configures for your GPU VRAM
engine.load_model("gemma-3-1b-Q4_K_M")

# Chat with performance tracking
conversation = [
    "What is machine learning?",
    "How does it differ from traditional programming?",
    "Give me a practical example"
]

for message in conversation:
    result = engine.infer(message, max_tokens=100)
    print(f"User: {message}")
    print(f"AI: {result.text}")
    print(f"Speed: {result.tokens_per_sec:.1f} tok/s\n")
```

---

## 📦 Model Registry

llcuda v1.0.0 includes **11 curated models** tested on GeForce 940M (1GB VRAM):

| Model | Size | Min VRAM | Description |
|-------|------|----------|-------------|
| `gemma-3-1b-Q4_K_M` | 700 MB | 0.5 GB | **Recommended for 1GB VRAM** |
| `tinyllama-1.1b-Q5_K_M` | 800 MB | 0.5 GB | Smallest option |
| `gemma-2-2b-Q4_K_M` | 1.5 GB | 1.5 GB | For 2GB+ VRAM |
| `phi-3-mini-Q4_K_M` | 2.2 GB | 2.0 GB | Microsoft Phi-3 |
| `mistral-7b-Q4_K_M` | 4.1 GB | 4.0 GB | For 4GB+ VRAM |
| `llama-3.1-8b-Q4_K_M` | 4.9 GB | 4.5 GB | Meta Llama 3.1 |
| ... and 5 more models | | | |

### List Available Models

```python
from llcuda.models import print_registry_models

# Show all models
print_registry_models()

# Show models compatible with 1GB VRAM
print_registry_models(vram_gb=1.0)
```

### Use Local Models

```python
engine.load_model("/path/to/your/model.gguf")
```

### Use HuggingFace Syntax

```python
engine.load_model("google/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf")
```

---

## 🎯 Features

### Zero Configuration
- **Auto-detects** package location and sets environment variables
- **Bundled binaries** - No need to download llama-server separately
- **Bundled libraries** - All CUDA dependencies included
- **Works immediately** after `pip install`

### Smart Model Loading
- **Registry-based** - 11 pre-configured models
- **Auto-download** - Downloads from HuggingFace with user confirmation
- **Local support** - Use your own GGUF models
- **HuggingFace syntax** - Direct repo:file downloads

### Hardware Auto-Configuration
- **VRAM detection** via nvidia-smi
- **Optimal settings** calculated automatically
- **Model analysis** - Determines best gpu_layers, ctx_size, batch_size
- **Manual override** - Advanced users can specify custom settings

### Performance Tracking
- **P50/P95/P99 latency** tracking
- **Tokens per second** monitoring
- **Request counts** and success rates
- **Built-in metrics** - No external tools needed

### Production Ready
- **Published to PyPI** - Proper versioning and releases
- **47 MB wheel** - Similar to PyTorch CUDA packages
- **Comprehensive docs** - Quick start, API reference, examples
- **Tested extensively** - GeForce 940M to RTX 4090

---

## 📊 Performance

Benchmarks on **GeForce 940M (1GB VRAM, Maxwell architecture)**:

```
Model: gemma-3-1b-Q4_K_M
Hardware: GeForce 940M (1GB VRAM)
Performance: ~15 tokens/second
GPU Layers: 20 (auto-configured)
Context: 512 tokens
Memory Usage: ~800MB VRAM
```

**Auto-Configuration Details:**
- VRAM detected: 1.0 GB
- Optimal settings calculated automatically
- No manual tuning required

Higher-end GPUs will see significantly better performance.

---

## 💡 Advanced Usage

### Manual Configuration

```python
engine = llcuda.InferenceEngine()

# Override auto-configuration
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    gpu_layers=20,
    ctx_size=2048,
    auto_configure=False  # Disable auto-config
)
```

### Performance Metrics

```python
# Run some inferences
for _ in range(10):
    engine.infer("Test prompt", max_tokens=50)

# Get metrics
metrics = engine.get_metrics()
print(f"P50 Latency: {metrics['latency']['p50_ms']:.2f}ms")
print(f"P95 Latency: {metrics['latency']['p95_ms']:.2f}ms")
print(f"P99 Latency: {metrics['latency']['p99_ms']:.2f}ms")
print(f"Throughput: {metrics['throughput']['tokens_per_sec']:.2f} tok/s")
```

### Context Manager

```python
with llcuda.InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M")
    result = engine.infer("Hello!")
    print(result.text)
# Server automatically stopped
```

---

## 🔧 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 5.0+)
- **VRAM**: 1GB+ (depends on model size)
- **RAM**: 4GB+ recommended

### Software
- **Python**: 3.11 or 3.12
- **CUDA**: 12.8 runtime (bundled in package)
- **OS**: Ubuntu 22.04 (tested), likely works on 20.04/24.04

### Tested Hardware
- GeForce 940M (1GB VRAM) ✓
- GeForce GTX 1060 (6GB VRAM) ✓
- RTX 2080 Ti (11GB VRAM) ✓
- RTX 4090 (24GB VRAM) ✓

---

## 📚 Documentation

- **Installation Guide**: See [QUICKSTART.md](QUICKSTART.md)
- **API Reference**: See [docs/](https://waqasm86.github.io/llcuda/)
- **Examples**: See [examples/](examples/)
- **Implementation Details**: See [IMPLEMENTATION_V1.0.md](IMPLEMENTATION_V1.0.md)
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for contribution:**
- Testing on different GPUs and CUDA versions
- Model testing and registry additions
- Documentation improvements
- Windows/macOS support
- Performance optimizations

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **llama.cpp** - GGML/GGUF inference engine by Georgi Gerganov
- **NVIDIA CUDA** - GPU acceleration framework
- **HuggingFace** - Model hosting and distribution
- **PyTorch** - Inspiration for package design

---

## 🔗 Links

- **PyPI**: [pypi.org/project/llcuda](https://pypi.org/project/llcuda/)
- **GitHub**: [github.com/waqasm86/llcuda](https://github.com/waqasm86/llcuda)
- **Documentation**: [waqasm86.github.io](https://waqasm86.github.io/)
- **Issues**: [github.com/waqasm86/llcuda/issues](https://github.com/waqasm86/llcuda/issues)
- **Releases**: [github.com/waqasm86/llcuda/releases](https://github.com/waqasm86/llcuda/releases)

---

## 📈 Project Status

**v1.0.0 - Production/Stable** (December 2025)

- ✅ Zero-configuration installation
- ✅ Bundled CUDA binaries (47 MB)
- ✅ 11 curated models in registry
- ✅ Hardware auto-configuration
- ✅ Performance metrics tracking
- ✅ Comprehensive documentation
- ✅ Published to PyPI

**Future roadmap:**
- Windows support
- More model optimizations
- Advanced batching strategies
- Grafana integration for DevOps monitoring

---

**Built with ❤️ for on-device AI on legacy NVIDIA GPUs**

🤖 *Developed with assistance from Claude Code*
