# llcuda - CUDA-Accelerated LLM Inference for Python

High-performance Python package for running LLM inference with CUDA acceleration. Designed for NVIDIA T4 GPUs in Kaggle and Google Colab environments.

## Features

- 🚀 **CUDA-Accelerated**: Native CUDA kernels for maximum performance
- 🐍 **Pythonic API**: Simple, intuitive interface
- 📊 **Performance Metrics**: Built-in latency and throughput tracking
- 🔄 **Streaming Support**: Real-time token generation
- 📦 **Batch Processing**: Efficient multi-prompt inference
- 🎯 **T4 Optimized**: Tuned for Kaggle/Colab NVIDIA T4 GPUs

## Installation

### From PyPI (when published)

```bash
pip install llcuda
```

### From Source

```bash
# Install dependencies
pip install pybind11 numpy cmake

# Clone repository
git clone https://github.com/waqasm86/local-llama-cuda.git
cd local-llama-cuda

# Set CUDA architecture (75 for T4 GPU)
export CUDA_ARCHITECTURES=75

# Install
pip install -e .
```

### For Kaggle/Colab

```python
# Install in notebook
!pip install pybind11 numpy
!git clone https://github.com/waqasm86/local-llama-cuda.git
%cd local-llama-cuda

import os
os.environ['CUDA_ARCHITECTURES'] = '75'  # T4 GPU
!pip install -e .
```

## Quick Start

```python
import llcuda

# Create engine
engine = llcuda.InferenceEngine()

# Load model
engine.load_model("model.gguf", gpu_layers=8)

# Run inference
result = engine.infer(
    prompt="What is artificial intelligence?",
    max_tokens=100,
    temperature=0.7
)

print(result.text)
print(f"Latency: {result.latency_ms:.2f}ms")
print(f"Throughput: {result.tokens_per_sec:.2f} tok/s")
```

## Usage Examples

### Streaming Inference

```python
def print_token(chunk):
    print(chunk, end='', flush=True)

result = engine.infer_stream(
    prompt="Write a story about AI.",
    callback=print_token,
    max_tokens=200
)
```

### Batch Processing

```python
prompts = [
    "What is ML?",
    "Explain neural networks.",
    "What is deep learning?"
]

results = engine.batch_infer(prompts, max_tokens=50)

for result in results:
    print(result.text)
```

### Performance Benchmarking

```python
# Reset metrics
engine.reset_metrics()

# Run multiple inferences
for _ in range(100):
    result = engine.infer("Hello!", max_tokens=64)

# Get metrics
metrics = engine.get_metrics()
print(f"Mean latency: {metrics['latency']['mean_ms']:.2f}ms")
print(f"p95 latency: {metrics['latency']['p95_ms']:.2f}ms")
print(f"Throughput: {metrics['throughput']['tokens_per_sec']:.2f} tok/s")
```

## Requirements

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (T4, P100, V100, etc.)
- **CUDA**: 11.7+ or 12.0+
- **Python**: 3.11+
- **OS**: Linux (Ubuntu 20.04+, tested on Kaggle/Colab)

### Python Dependencies

- `numpy>=1.20.0`
- `pybind11>=2.10.0` (build-time)
- `cmake>=3.24` (build-time)

### Backend Requirement

This package requires `llama-server` running as a backend. See [Setup Guide](#setup-with-llama-server) below.

## Setup with llama-server

### Install llama.cpp

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j8
```

### Start llama-server

```bash
# Start server (background)
./build/bin/llama-server \
  -m /path/to/model.gguf \
  --port 8090 \
  -ngl 99 \
  -c 4096 &
```

### Use with llcuda

```python
# Point to server
engine = llcuda.InferenceEngine(server_url="http://127.0.0.1:8090")
```

## API Reference

### InferenceEngine

Main interface for LLM inference.

#### Methods

- `load_model(model_path, gpu_layers=0, ...)` - Load a GGUF model
- `infer(prompt, max_tokens=128, ...)` - Single inference
- `infer_stream(prompt, callback, ...)` - Streaming inference
- `batch_infer(prompts, ...)` - Batch inference
- `get_metrics()` - Get performance metrics
- `reset_metrics()` - Reset metrics
- `unload_model()` - Unload current model

#### Properties

- `is_loaded` - Check if model is loaded

### InferResult

Inference result object.

#### Properties

- `success` - Whether inference succeeded
- `text` - Generated text
- `tokens_generated` - Number of tokens generated
- `latency_ms` - Inference latency in milliseconds
- `tokens_per_sec` - Generation throughput
- `error_message` - Error message if failed

### Utility Functions

- `check_cuda_available()` - Check if CUDA is available
- `get_cuda_device_info()` - Get GPU information

## Kaggle/Colab Notebook

See [examples/kaggle_colab_example.ipynb](examples/kaggle_colab_example.ipynb) for a complete notebook example.

## Performance

Tested on NVIDIA T4 GPU (Kaggle/Colab):

| Model | Quantization | Throughput | p95 Latency |
|-------|--------------|------------|-------------|
| Gemma 2B | Q4_K_M | 45 tok/s | 180ms |
| Mistral 7B | Q4_K_M | 28 tok/s | 320ms |
| Llama 2 7B | Q4_K_M | 26 tok/s | 340ms |

*Results with `gpu_layers=99`, `batch_size=512`, `ctx_size=4096`*

## Troubleshooting

### ImportError: No module named '_llcuda'

**Solution**: Reinstall with proper CUDA architecture:
```bash
export CUDA_ARCHITECTURES=75  # For T4 GPU
pip install --force-reinstall -e .
```

### CUDA out of memory

**Solution**: Reduce GPU layers or use smaller model:
```python
engine.load_model("model.gguf", gpu_layers=4)  # Use fewer layers
```

### Server connection failed

**Solution**: Ensure llama-server is running:
```bash
curl http://127.0.0.1:8090/health
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Building Documentation

```bash
pip install -e ".[dev]"
cd docs
make html
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{llcuda2024,
  title={llcuda: CUDA-Accelerated LLM Inference for Python},
  author={Muhammad, Waqas},
  year={2024},
  url={https://github.com/waqasm86/local-llama-cuda}
}
```

## Acknowledgments

- **llama.cpp**: GGML/GGUF inference engine
- **NVIDIA CUDA**: GPU acceleration framework
- **pybind11**: C++/Python bindings

## Links

- **Documentation**: [GitHub Pages](https://waqasm86.github.io/projects/local-llama-cuda/)
- **Source Code**: [GitHub](https://github.com/waqasm86/local-llama-cuda)
- **Issues**: [Bug Tracker](https://github.com/waqasm86/local-llama-cuda/issues)
- **PyPI**: [Package](https://pypi.org/project/llcuda/) (when published)

---

**Built for on-device AI with Python** 🐍🚀
