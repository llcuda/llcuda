# llcuda - CUDA-Accelerated LLM Inference for Python

High-performance Python package for running LLM inference with CUDA acceleration and **automatic server management**. Designed for ease of use in JupyterLab, notebooks, and production environments. Tested on GeForce 940M (1GB VRAM) to RTX 4090.

[![PyPI version](https://badge.fury.io/py/llcuda.svg)](https://pypi.org/project/llcuda/)
[![Downloads](https://pepy.tech/badge/llcuda)](https://pepy.tech/project/llcuda)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/waqasm86/llcuda)](https://github.com/waqasm86/llcuda/stargazers)

> **Perfect for**: Legacy NVIDIA GPUs • JupyterLab workflows • Local LLM development • No-compilation setup • GeForce 900/800 series
>
> **Keywords**: cuda llm inference python, llama.cpp python wrapper, local llm python, gguf inference, jupyterlab llm, automatic server management, zero configuration

## ✨ What's New in v0.2.0

- 🚀 **Automatic Server Management** - No manual server setup required!
- 🔍 **Auto-Discovery** - Automatically finds llama-server and GGUF models
- 📊 **System Diagnostics** - Built-in tools to check your setup
- 💻 **JupyterLab Ready** - Optimized for notebook workflows
- 🎯 **One-Line Inference** - Get started with minimal code

## Features

- 🚀 **CUDA-Accelerated**: Native CUDA support for maximum performance
- 🤖 **Auto-Start**: Automatically manages llama-server lifecycle
- 🐍 **Pythonic API**: Clean, intuitive interface
- 📊 **Performance Metrics**: Built-in latency and throughput tracking
- 🔄 **Streaming Support**: Real-time token generation
- 📦 **Batch Processing**: Efficient multi-prompt inference
- 🎯 **Smart Discovery**: Finds models and executables automatically
- 💻 **JupyterLab Integration**: Perfect for interactive workflows
- 🛠️ **Context Manager Support**: Automatic resource cleanup

## Installation

### Complete Setup Guide (Tested on Xubuntu 22.04)

Follow these steps exactly as tested on a fresh system:

#### Step 1: Download and Extract llama.cpp Binary

```bash
# Navigate to your Downloads folder (or any location)
cd ~/Downloads

# Download pre-built llama.cpp with CUDA support (290 MB)
wget https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable/releases/download/v0.1.0/llama.cpp-733c851f-bin-ubuntu-cuda-x64.tar.xz

# Extract
tar -xf llama.cpp-733c851f-bin-ubuntu-cuda-x64.tar.xz

# Enter the directory
cd llama-cpp-cuda

# Verify CUDA support
./bin/llama-server --version
```

**Expected output:**
```
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce 940M, compute capability 5.0, VMM: yes
version: 6093 (733c851f)
```

**Save your path** (you'll need it later):
```bash
pwd
# Example output: /home/waqasm86/Downloads/llama-cpp-cuda
```

#### Step 2: Download a GGUF Model

Download a small model for testing and place it in the `bin/` folder:

```bash
# Inside llama-cpp-cuda directory
cd bin

# Download Gemma 3 1B (Q4_K_M quantization, ~700 MB)
wget https://huggingface.co/google/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf

# Return to parent directory
cd ..
```

Or use any GGUF model from HuggingFace.

#### Step 3: Install llcuda

```bash
# Install llcuda from PyPI
pip install llcuda
```

**Requirements**: Python 3.11+, Ubuntu 22.04, NVIDIA GPU with CUDA 11.7+ or 12.x

**Verified working on**: GeForce 940M (1GB VRAM) to RTX 4090

#### Step 4: Launch JupyterLab and Use llcuda

```bash
# Launch JupyterLab
jupyter lab
```

Then in your JupyterLab notebook or Python script:

## Quick Start

### Complete JupyterLab Example (Copy-Paste Ready)

This example matches the exact working setup from Xubuntu 22.04:

```python
import os

# Set the path to llama-server
pwd = '/home/waqasm86/Downloads/llama-cpp-cuda'  # Change to YOUR path from Step 1
os.environ['LLAMA_SERVER_PATH'] = pwd + '/bin/llama-server'

# Import llcuda
import llcuda

# Verify setup
print(f"LLAMA_SERVER_PATH: {os.environ.get('LLAMA_SERVER_PATH')}")
print(f"llcuda version: {llcuda.__version__}")

# Create inference engine
engine = llcuda.InferenceEngine()

# Load model with optimized settings for GeForce 940M (1GB VRAM)
# Adjust gpu_layers, ctx_size for your GPU
engine.load_model(
    pwd + "/bin/gemma-3-1b-it-Q4_K_M.gguf",  # Change to YOUR model path
    auto_start=True,     # Automatically starts llama-server
    gpu_layers=8,        # 8 layers on GPU (adjust for your VRAM)
    ctx_size=512,        # Small context to save memory
    n_parallel=1,        # Single sequence
    verbose=True,
    batch_size=512,      # Reduces from default 2048
    ubatch_size=128,     # CRITICAL - reduces compute buffer
)

print("\n✓ Model loaded successfully!")

# Run inference
result = engine.infer("What is AI?", max_tokens=100)

# Display result
if result.success:
    print("\n" + "="*60)
    print("Generated Text:")
    print("="*60)
    print(result.text)
    print("="*60)
    print(f"\nPerformance:")
    print(f"  Tokens: {result.tokens_generated}")
    print(f"  Speed: {result.tokens_per_sec:.1f} tok/s")
    print(f"  Latency: {result.latency_ms:.0f}ms")
else:
    print(f"Error: {result.error_message}")
```

**Expected output:**
```
LLAMA_SERVER_PATH: /home/waqasm86/Downloads/llama-cpp-cuda/bin/llama-server
llcuda version: 0.2.0

✓ Model loaded and ready for inference
✓ Model loaded successfully!

============================================================
Generated Text:
============================================================
AI, or Artificial Intelligence, is essentially the ability of a
computer or machine to perform tasks that typically require human
intelligence. These tasks include things like learning, problem-
solving, decision-making, and even understanding language...
============================================================

Performance:
  Tokens: 100
  Speed: 12.2 tok/s
  Latency: 8217ms
```

### Simplified Usage (After Setting Environment Variable)

Once `LLAMA_SERVER_PATH` is set, you can use this simpler form:

```python
import llcuda

# Create engine and load model with auto-start
engine = llcuda.InferenceEngine()
engine.load_model(
    "/path/to/model.gguf",
    auto_start=True,  # Automatically starts llama-server
    gpu_layers=20     # Adjust for your GPU VRAM
)

# Run inference
result = engine.infer("What is artificial intelligence?", max_tokens=100)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")
```

### Advanced: Context Manager Usage

```python
import llcuda

# Check system setup
llcuda.print_system_info()

# Find available models
models = llcuda.find_gguf_models()
print(f"Found {len(models)} models")

# Use auto-start with context manager
with llcuda.InferenceEngine() as engine:
    engine.load_model(models[0], auto_start=True, gpu_layers=20)
    result = engine.infer("Explain quantum computing")
    print(result.text)
# Server automatically stopped when exiting context
```

### Traditional Usage (Manual Server)

```bash
# Terminal 1: Start llama-server manually
/path/to/llama-server -m model.gguf --port 8090 -ngl 99 &
```

```python
# Python code
import llcuda

engine = llcuda.InferenceEngine()
result = engine.infer("What is AI?")
print(result.text)
```

## Usage Examples

### System Check

```python
import llcuda

# Comprehensive system information
llcuda.print_system_info()

# Check CUDA availability
if llcuda.check_cuda_available():
    gpu_info = llcuda.get_cuda_device_info()
    print(f"GPUs: {len(gpu_info['gpus'])}")
```

### Basic Inference

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("model.gguf", auto_start=True, gpu_layers=99)

result = engine.infer(
    prompt="What is machine learning?",
    max_tokens=100,
    temperature=0.7,
    top_p=0.9
)

if result.success:
    print(result.text)
    print(f"Latency: {result.latency_ms:.0f}ms")
    print(f"Throughput: {result.tokens_per_sec:.1f} tok/s")
else:
    print(f"Error: {result.error_message}")
```

### Batch Processing

```python
prompts = [
    "What is AI?",
    "What is ML?",
    "What is DL?"
]

results = engine.batch_infer(prompts, max_tokens=50)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result.text}\n")
```

### Streaming Inference

```python
def on_chunk(text):
    print(text, end='', flush=True)

result = engine.infer_stream(
    prompt="Write a story about AI",
    callback=on_chunk,
    max_tokens=200
)
```

### Performance Monitoring

```python
# Run multiple inferences
for _ in range(10):
    engine.infer("Test prompt", max_tokens=50)

# Get detailed metrics
metrics = engine.get_metrics()
print(f"Mean latency: {metrics['latency']['mean_ms']:.2f}ms")
print(f"p95 latency: {metrics['latency']['p95_ms']:.2f}ms")
print(f"Throughput: {metrics['throughput']['tokens_per_sec']:.2f} tok/s")
```

### Advanced: Manual Server Management

```python
from llcuda import ServerManager

# Create and configure server
manager = ServerManager()
manager.start_server(
    model_path="model.gguf",
    port=8090,
    gpu_layers=99,
    ctx_size=4096,
    n_parallel=2
)

# Use the server
engine = llcuda.InferenceEngine()
result = engine.infer("Hello!")

# Stop when done
manager.stop_server()
```

## API Reference

### InferenceEngine

Main interface for LLM inference.

**Methods:**
- `load_model(model_path, gpu_layers=99, auto_start=False, ...)` - Load GGUF model
- `infer(prompt, max_tokens=128, temperature=0.7, ...)` - Single inference
- `infer_stream(prompt, callback, ...)` - Streaming inference
- `batch_infer(prompts, ...)` - Batch inference
- `get_metrics()` - Get performance metrics
- `reset_metrics()` - Reset metrics counters
- `check_server()` - Check if server is running
- `unload_model()` - Stop server and cleanup

**Properties:**
- `is_loaded` - Check if model is loaded

### ServerManager

Low-level server lifecycle management.

**Methods:**
- `start_server(model_path, port=8090, gpu_layers=99, ...)` - Start llama-server
- `stop_server()` - Stop running server
- `restart_server(model_path, ...)` - Restart with new config
- `check_server_health()` - Check server health
- `find_llama_server()` - Find llama-server executable
- `get_server_info()` - Get server status info

### InferResult

Result object from inference.

**Properties:**
- `success` (bool) - Whether inference succeeded
- `text` (str) - Generated text
- `tokens_generated` (int) - Number of tokens generated
- `latency_ms` (float) - Inference latency in milliseconds
- `tokens_per_sec` (float) - Generation throughput
- `error_message` (str) - Error message if failed

### Utility Functions

- `check_cuda_available()` - Check if CUDA is available
- `get_cuda_device_info()` - Get GPU information
- `detect_cuda()` - Detailed CUDA detection
- `find_gguf_models(directory=None)` - Find GGUF models
- `get_llama_cpp_cuda_path()` - Find Ubuntu-Cuda-Llama.cpp-Executable installation
- `print_system_info()` - Print comprehensive system info
- `setup_environment()` - Setup environment variables
- `quick_infer(prompt, model_path=None, ...)` - One-liner inference

## Configuration

### Environment Variables

- `LLAMA_CPP_DIR` - Path to Ubuntu-Cuda-Llama.cpp-Executable installation
- `LLAMA_SERVER_PATH` - Direct path to llama-server executable
- `CUDA_VISIBLE_DEVICES` - Which GPUs to use

Example `.bashrc` / `.profile`:

```bash
export LLAMA_CPP_DIR="/media/waqasm86/External1/Project-Nvidia/Ubuntu-Cuda-Llama.cpp-Executable"
export LD_LIBRARY_PATH="$LLAMA_CPP_DIR/lib:$LD_LIBRARY_PATH"
```

### Config File

llcuda can use a config file at `~/.llcuda/config.json`:

```json
{
  "server": {
    "url": "http://127.0.0.1:8090",
    "port": 8090,
    "auto_start": true
  },
  "model": {
    "gpu_layers": 99,
    "ctx_size": 2048
  },
  "inference": {
    "max_tokens": 128,
    "temperature": 0.7
  }
}
```

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 5.0+)
- **VRAM**: 1GB+ (depends on model size)
- **RAM**: 4GB+ recommended

### Software
- **Python**: 3.11+
- **CUDA**: 11.7+ or 12.0+
- **OS**: Linux (Ubuntu 20.04+), tested on Ubuntu 22.04

### Python Dependencies
- `numpy>=1.20.0`
- `requests>=2.20.0`

## Performance

Benchmarks on NVIDIA GeForce 940M (1GB VRAM):

| Model | Quantization | GPU Layers | Throughput | Latency |
|-------|--------------|------------|------------|---------|
| Gemma 3 1B | Q4_K_M | 20 | ~15 tok/s | ~200ms |
| Gemma 2B | Q4_K_M | 10 | ~12 tok/s | ~250ms |

Higher-end GPUs (T4, P100, V100, A100) will see significantly better performance.

## Troubleshooting

### Server not found

```python
# Check if llama-server can be found
import llcuda
server_path = llcuda.ServerManager().find_llama_server()
print(server_path)  # Should show path to llama-server
```

If None, set `LLAMA_CPP_DIR` or `LLAMA_SERVER_PATH`.

### CUDA out of memory

Reduce GPU layers:

```python
engine.load_model("model.gguf", auto_start=True, gpu_layers=10)
```

Or use smaller context size:

```python
engine.load_model("model.gguf", auto_start=True, ctx_size=1024)
```

### Check System Setup

```python
import llcuda
llcuda.print_system_info()
```

This will show:
- Python version and executable
- CUDA availability and GPU info
- Ubuntu-Cuda-Llama.cpp-Executable installation status
- Available GGUF models

## Examples

See the `examples/` directory:

- `quickstart_jupyterlab.ipynb` - Complete JupyterLab tutorial
- `kaggle_colab_example.ipynb` - Cloud platform example

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Building from Source

```bash
git clone https://github.com/waqasm86/llcuda
cd llcuda
pip install -e .
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{llcuda2024,
  title={llcuda: CUDA-Accelerated LLM Inference for Python},
  author={Muhammad, Waqas},
  year={2024},
  version={0.2.0},
  url={https://github.com/waqasm86/llcuda}
}
```

## Acknowledgments

- **llama.cpp** - GGML/GGUF inference engine
- **NVIDIA CUDA** - GPU acceleration framework
- **Python community** - For amazing tools and libraries

## Links

- **GitHub**: https://github.com/waqasm86/llcuda
- **PyPI**: https://pypi.org/project/llcuda/
- **Issues**: https://github.com/waqasm86/llcuda/issues
- **llama.cpp**: https://github.com/ggerganov/llama.cpp

---

**Built with ❤️ for on-device AI** 🚀
