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
🚀 Quick Start
python
import llcuda

# Create inference engine
engine = llcuda.InferenceEngine()

# Load model (downloads on-demand if not present)
engine.load_model('gemma-3-1b-Q4_K_M')  # Downloads from HuggingFace if needed

# Run inference
result = engine.infer('What is artificial intelligence?', max_tokens=100)
print(result.text)
📖 Detailed Usage
Basic Inference
python
import llcuda

# Initialize engine
engine = llcuda.InferenceEngine()

# Load model with automatic configuration
engine.load_model(
    model_name_or_path='gemma-3-1b-Q4_K_M',  # Or path to local .gguf file
    gpu_layers=99,  # Offload all layers to GPU (auto-detected)
    ctx_size=2048,  # Context window size
    auto_start=True,  # Start server automatically
    verbose=True
)

# Single inference
result = engine.infer(
    prompt='Explain quantum computing in simple terms.',
    max_tokens=150,
    temperature=0.7,
    top_p=0.9,
    top_k=40
)

print(f"Response: {result.text}")
print(f"Tokens generated: {result.tokens_generated}")
print(f"Latency: {result.latency_ms:.2f}ms")
print(f"Throughput: {result.tokens_per_sec:.2f} tokens/sec")

# Batch inference
prompts = [
    'What is machine learning?',
    'Explain neural networks',
    'What are transformers in AI?'
]
results = engine.batch_infer(prompts, max_tokens=100)

# Streaming inference
def stream_callback(chunk):
    print(chunk, end='', flush=True)

engine.infer_stream(
    prompt='Write a short story about AI',
    callback=stream_callback,
    max_tokens=200
)
Model Loading Options
python
# Option 1: Auto-download from registry (recommended)
engine.load_model('gemma-3-1b-Q4_K_M')  # Downloads from waqasm86/llcuda-models

# Option 2: Local GGUF file
engine.load_model('/path/to/your/model.gguf')

# Option 3: HuggingFace download syntax
engine.load_model('microsoft/phi-2-GGUF:phi-2-q4_k_m.gguf')

# Option 4: Manual configuration
engine.load_model(
    model_name_or_path='llama-2-7b-chat.Q4_K_M.gguf',
    gpu_layers=35,  # Manual layer offloading
    ctx_size=4096,  # Custom context size
    batch_size=512,  # Manual batch size
    n_parallel=2,  # Parallel sequences
    n_threads=8,  # CPU threads
    verbose=True
)
Advanced Features
python
# Check CUDA availability
if llcuda.check_cuda_available():
    print("✅ CUDA is available!")
    gpu_info = llcuda.get_cuda_device_info()
    print(f"GPU: {gpu_info['gpus'][0]['name']}")
    print(f"CUDA Version: {gpu_info['cuda_version']}")

# Get performance metrics
metrics = engine.get_metrics()
print(f"Mean latency: {metrics['latency']['mean_ms']:.2f}ms")
print(f"P95 latency: {metrics['latency']['p95_ms']:.2f}ms")
print(f"Throughput: {metrics['throughput']['tokens_per_sec']:.2f} tok/s")

# Reset metrics
engine.reset_metrics()

# Context manager (auto-cleanup)
with llcuda.InferenceEngine() as engine:
    engine.load_model('gemma-3-1b-Q4_K_M', auto_start=True)
    result = engine.infer('Hello world!')
    print(result.text)
Utility Functions
python
import llcuda

# Detect available GGUF models
models = llcuda.find_gguf_models('/path/to/models')
print(f"Found {len(models)} GGUF models")

# Print system info
llcuda.print_system_info()

# Quick inference (one-liner)
response = llcuda.quick_infer(
    prompt='What is Python?',
    model_path='gemma-3-1b-Q4_K_M',
    max_tokens=50
)
print(response)
🛠️ System Requirements
OS: Linux (Ubuntu 20.04+, CentOS 7+, etc.)

Python: 3.8 or higher

GPU: NVIDIA GPU with Compute Capability 5.0+ (7.5+ recommended)

Memory: 8GB+ RAM, 2GB+ VRAM

Disk Space: 2GB+ for binaries + model storage

Supported GPUs
Compute 7.5+: Tesla T4, GeForce RTX 20/30/40 series, RTX A6000, A100

Compute 6.0-7.2: GTX 10 series, Titan V, Tesla P100, V100

Compute 5.0-5.2: GTX 9 series, Titan X, Tesla M40, M60

🔧 Troubleshooting
Common Issues
"CUDA not available"

bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall with verbose output
pip install llcuda -v
"Model download failed"

python
# Try with interactive mode disabled
engine.load_model('gemma-3-1b-Q4_K_M', interactive_download=False)

# Or download manually from HuggingFace:
# https://huggingface.co/waqasm86/llcuda-models
"Server failed to start"

python
# Check port availability
engine = llcuda.InferenceEngine(server_url='http://127.0.0.1:8091')

# Increase timeout
engine.load_model(..., server_timeout=60)
Performance Tips
Use gpu_layers=99 to offload all layers to GPU

Adjust batch_size based on your VRAM (512-2048)

Set n_threads to match your CPU core count

Use Q4_K_M quantization for best speed/quality balance

📊 Benchmarks
Model	Size	GPU	Tokens/sec	VRAM Usage
Gemma 3 1B Q4_K_M	769MB	Tesla T4	~45 tok/s	2.1GB
Llama 2 7B Q4_K_M	3.8GB	RTX 3080	~28 tok/s	5.2GB
Phi-2 Q4_K_M	1.4GB	Tesla P4	~32 tok/s	2.8GB
🚀 What's New in v1.1.4
Removed automatic model download on import (binaries only)

On-demand model downloading via load_model()

Improved bootstrap system with better error handling

Reduced initial download size from ~900MB to ~120MB

Better Colab/Kaggle compatibility

Fixed HuggingFace download warnings

Enhanced GPU detection

Updated CUDA binaries with multi-arch support

📁 Project Structure
text
llcuda/
├── __init__.py              # Main API interface
├── server.py               # Server management
├── engine.py               # Inference engine
├── utils.py                # Utility functions
├── models.py               # Model registry & download
├── _internal/
│   ├── bootstrap.py        # Auto-download binaries
│   └── config.py          # Configuration
├── binaries/              # CUDA binaries (auto-downloaded)
│   └── cuda12/
│       ├── llama-server   # Main inference binary
│       └── ...           # CUDA libraries
└── models/                # Downloaded models storage
    └── ...               # GGUF models
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Built on top of llama.cpp

Uses CUDA for acceleration

Model hosting by Hugging Face

Inspired by llama-cpp-python

📧 Contact
Waqas Mahmood - GitHub - waqas.mahmood@example.com

Project Link: https://github.com/waqasm86/llcuda

⭐ If you find this project useful, please give it a star on GitHub!
