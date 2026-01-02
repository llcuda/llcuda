# llcuda: CUDA-Accelerated LLM Inference for Python

**Effortless, zero-configuration LLM inference with CUDA acceleration. Compatible with all modern NVIDIA GPUs, Google Colab, Kaggle, and JupyterLab. PyTorch-inspired API for seamless integration.**

> Ideal for:  
> - Google Colab and Kaggle notebooks  
> - Local development on GPUs from GTX 940M to RTX 4090  
> - Production-grade performance without manual setup  
> - Python 3.11+ environments  

---

## What's New in Version 1.1.9

**Fixed llama-server Detection + Silent Mode**

Version 1.1.9 fixes llama-server detection issues and adds silent mode:

- **Fixed Server Detection**: Now finds llama-server in package binaries directory
- **Silent Mode**: Suppress all llama-server warnings with `silent=True`
- **Better Path Detection**: Priority given to package-installed binaries
- **Colab/Kaggle Paths**: Added cloud-specific cache paths

Key Improvements:
- ✅ llama-server properly detected from bootstrap-installed binaries
- ✅ New `silent=True` parameter to suppress server output
- ✅ Works in Google Colab and Kaggle without manual paths
- ✅ Cleaner output for Jupyter notebooks

Example Usage:

```python
# Install the latest v1.1.9 package
pip install llcuda==1.1.9

# Import and load model silently (no llama-server warnings!)
import llcuda
engine = llcuda.InferenceEngine()

# Silent mode - no llama-server output
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
```

---

## Installation

### Quick Install (Recommended)

```bash
pip install llcuda==1.1.9
```

### Python Requirements

- **Python**: 3.11+ (explicitly tested)
- **CUDA**: 11.0+ or 12.0+ recommended
- **GPU**: NVIDIA with compute capability 5.0+ (Maxwell or newer)
- **Memory**: 4GB+ VRAM for small models, 8GB+ recommended

### Verified Platforms

- ✅ **Ubuntu 22.04+** with CUDA 11/12
- ✅ **Google Colab** (T4, P100, V100, A100)
- ✅ **Kaggle Notebooks** (T4)
- ✅ **Windows** with WSL2 + CUDA
- ✅ **macOS** (with cloud GPU access)

---

## Quick Start

### Basic Usage

```python
import llcuda

# Initialize engine
engine = llcuda.InferenceEngine()

# Load a model (auto-downloads from registry)
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

# Run inference
result = engine.infer("What is artificial intelligence?", max_tokens=100)
print(result.text)

# Get performance metrics
metrics = engine.get_metrics()
print(f"Tokens/sec: {metrics['throughput']['tokens_per_sec']:.2f}")
```

### Advanced Usage

```python
import llcuda

# With custom settings
engine = llcuda.InferenceEngine(server_url="http://localhost:8090")

engine.load_model(
    "/path/to/model.gguf",
    gpu_layers=99,        # GPU offload (0=CPU only, -1=maximum)
    ctx_size=2048,        # Context window
    batch_size=512,       # Batch processing size
    temperature=0.7,      # Sampling temperature
    top_p=0.9,          # Nucleus sampling
    auto_configure=True   # Auto-optimize settings
)

# Batch inference
prompts = ["What is AI?", "Explain ML.", "Future of tech?"]
results = engine.batch_infer(prompts, max_tokens=50)

for i, result in enumerate(results):
    print(f"Prompt {i+1}: {result.text}")
```

### Context Manager (Auto-cleanup)

```python
with llcuda.InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M")
    result = engine.infer("Hello, world!")
    print(result.text)
# Server automatically stopped
```

---

## Available Models

### Pre-configured Models (Registry)

These models are tested and optimized for llcuda:

| Model | Size | VRAM | Use Case |
|-------|------|------|----------|
| `gemma-3-1b-Q4_K_M` | 1B | ~1GB | Chat, Q&A |
| `gemma-2-2b-Q4_K_M` | 2B | ~1.5GB | General tasks |
| `llama-3.2-3b-Q4_K_M` | 3B | ~2GB | Advanced reasoning |
| `qwen-2.5-1.5b-Q4_K_M` | 1.5B | ~1.2GB | Multilingual |

### Loading Models

```python
# Registry model (auto-download)
engine.load_model("gemma-3-1b-Q4_K_M")

# Local GGUF file
engine.load_model("/path/to/model.gguf")

# HuggingFace model
engine.load_model("google/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf")
```

---

## Performance

### Benchmarks (RTX 4090, CUDA 12)

| Model | Tokens/sec | Memory | Latency (P50) |
|-------|------------|--------|---------------|
| Gemma 3-1B | 120+ | 1.2GB | 15ms |
| Llama 3.2-3B | 85+ | 2.1GB | 22ms |
| Qwen 2.5-1.5B | 95+ | 1.4GB | 18ms |

### GPU Compatibility

| Architecture | GPUs | Compute Capability | Status |
|--------------|--------|-------------------|---------|
| Maxwell | GTX 900, 940M | 5.0-5.3 | ✅ |
| Pascal | GTX 10xx, P100 | 6.0-6.1 | ✅ |
| Volta | V100 | 7.0 | ✅ |
| Turing | RTX 20xx, T4 | 7.5 | ✅ |
| Ampere | RTX 30xx, A100 | 8.0-8.6 | ✅ |
| Ada Lovelace | RTX 40xx | 8.9 | ✅ |

---

## Advanced Features

### Performance Monitoring

```python
# Get detailed metrics
metrics = engine.get_metrics()

print(f"Mean latency: {metrics['latency']['mean_ms']:.2f}ms")
print(f"P95 latency: {metrics['latency']['p95_ms']:.2f}ms")
print(f"Throughput: {metrics['throughput']['tokens_per_sec']:.2f} tok/s")
```

### Server Management

```python
from llcuda import ServerManager

# Manual server control
server = ServerManager()
server.start_server(
    model_path="model.gguf",
    port=8090,
    gpu_layers=99,
    host="0.0.0.0"  # Allow network access
)

# Check server status
if server.is_running():
    print("Server is running")

# Stop server
server.stop_server()
```

### Jupyter Integration

```python
# For Jupyter notebooks
import llcuda.jupyter as llj

# Interactive chat widget
widget = llj.ChatWidget()
widget.display()

# Streaming output
llj.stream_inference("Explain quantum computing", callback=print_token)
```

---

## Configuration

### Environment Variables

```bash
# Override default paths
export LLAMA_SERVER_PATH="/custom/path/to/llama-server"
export LD_LIBRARY_PATH="/custom/lib/path:$LD_LIBRARY_PATH"

# Cache directory
export LLCUDA_CACHE_DIR="/custom/cache/dir"
```

### Configuration File

```python
import llcuda
from llcuda import create_config_file

# Create default config
config_path = create_config_file()

# Load config
config = llcuda.load_config(config_path)
```

---

## Troubleshooting

### Common Issues

1. **"no kernel image available"**
   - Update to v1.1.6+ with hybrid bootstrap
   - Check GPU compute capability (5.0+ required)

2. **Out of Memory**
   - Reduce `gpu_layers` or `ctx_size`
   - Use smaller model
   - Clear GPU cache between runs

3. **Slow Performance**
   - Increase `gpu_layers` to maximum
   - Check CUDA installation
   - Verify GPU power settings

### Debug Information

```python
import llcuda

# System info
llcuda.print_system_info()

# GPU details
gpu_info = llcuda.get_cuda_device_info()
if gpu_info:
    for gpu in gpu_info['gpus']:
        print(f"GPU: {gpu['name']} ({gpu['memory']}MB)")
```

### Verbose Mode

```python
# Enable verbose output
engine.load_model("model.gguf", verbose=True)

# Check bootstrap logs
import logging
logging.basicConfig(level=logging.INFO)
```

---

## Development

### Installation from Source

```bash
git clone https://github.com/waqasm86/llcuda.git
cd llcuda
pip install -e .
```

### Testing

```bash
# Run tests
pytest tests/

# End-to-end test
python -m llcuda.tests.test_end_to_end
```

### Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Submit pull request

---

## Changelog

### v1.1.9 (2025-01-03)
- **Fixed Server Detection**: llama-server now found in package binaries
- **Silent Mode**: New `silent=True` parameter to suppress output
- **Better Paths**: Priority to package-installed binaries
- **Colab/Kaggle**: Cloud-specific cache paths added

[View full changelog](CHANGELOG.md)

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Links

- **GitHub**: https://github.com/waqasm86/llcuda
- **PyPI**: https://pypi.org/project/llcuda/
- **Documentation**: https://waqasm86.github.io/
- **Issues**: https://github.com/waqasm86/llcuda/issues
- **Releases**: https://github.com/waqasm86/llcuda/releases

---

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Core inference engine
- [Hugging Face](https://huggingface.co) - Model hosting and ecosystem
- CUDA and NVIDIA for GPU acceleration support

---

*CUDA is a trademark of NVIDIA Corporation. This project is not affiliated with NVIDIA.*