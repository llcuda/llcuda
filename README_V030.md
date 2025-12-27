# llcuda v0.3.0 - CUDA-Accelerated LLM Inference for JupyterLab

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**llcuda** is a Python package that provides a Pythonic, JupyterLab-friendly interface for running large language models with CUDA acceleration via llama.cpp.

## 🆕 What's New in v0.3.0

llcuda v0.3.0 is a major update focused on **JupyterLab integration**:

### ✨ New Features

- 🎯 **Real-time Streaming** - See tokens generated live in Jupyter notebooks
- 💬 **Interactive Chat Widget** - IPython widget with full conversation UI
- 📊 **Rich Metrics Display** - Beautiful tables and visualizations
- 🔗 **Chat Management** - OpenAI-compatible API with conversation history
- 🧠 **Text Embeddings** - Generate embeddings and build semantic search
- 📦 **Model Manager** - Discover, download, and manage GGUF models
- 📈 **Progress Bars** - tqdm integration for batch processing
- 🎨 **Markdown Rendering** - Beautiful formatted output

### 🎥 Quick Demo

```python
from llcuda.jupyter import ChatWidget

chat = ChatWidget(engine)
chat.display()  # Interactive chat in your notebook!
```

![Chat Widget Demo](https://via.placeholder.com/800x400?text=Interactive+Chat+Widget)

## 📦 Installation

### Basic Installation
```bash
pip install llcuda
```

### With JupyterLab Features
```bash
pip install llcuda ipywidgets tqdm matplotlib pandas
```

### Enable Widgets (JupyterLab)
```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## 🚀 Quick Start

### 1. Setup Environment
```python
import os
os.environ['LLAMA_SERVER_PATH'] = '/path/to/llama-server'

import llcuda
engine = llcuda.InferenceEngine()
```

### 2. Load Model
```python
engine.load_model(
    "model.gguf",
    auto_start=True,    # Automatically start llama-server
    gpu_layers=20,      # Offload 20 layers to GPU
    ctx_size=2048,
    verbose=True
)
```

### 3. Generate Text
```python
# Simple inference
result = engine.infer("What is quantum computing?")
print(result.text)

# Streaming (in Jupyter)
from llcuda.jupyter import stream_generate
text = stream_generate(engine, "Explain AI", markdown=True)

# Interactive chat
from llcuda.jupyter import ChatWidget
chat = ChatWidget(engine)
chat.display()
```

## 📚 Key Features

### 🎯 Core Features (v0.2.x)
- ✅ Automatic llama-server management
- ✅ CUDA GPU acceleration
- ✅ Auto-discovery of models and binaries
- ✅ Zero-configuration setup
- ✅ Context manager support
- ✅ Performance metrics tracking

### 🆕 JupyterLab Features (v0.3.0)
- ✅ Real-time streaming with IPython display
- ✅ Interactive chat widget with controls
- ✅ Conversation history management
- ✅ Text embedding generation
- ✅ Semantic search and clustering
- ✅ Model discovery and recommendations
- ✅ Progress bars for batch processing
- ✅ Rich metrics visualization

## 📖 Documentation

- **[Quick Start Guide](QUICK_START_JUPYTER.md)** - Get started in 5 minutes
- **[JupyterLab Features](JUPYTERLAB_FEATURES.md)** - Complete API documentation
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Architecture and design
- **[Tutorial Notebook](../Project-llcuda-jupyterlab/complete-llcuda-tutorial.ipynb)** - Interactive examples

## 🎓 Examples

### Interactive Chat Widget

```python
from llcuda.jupyter import ChatWidget

chat = ChatWidget(
    engine,
    system_prompt="You are a Python expert",
    max_tokens=200,
    temperature=0.7
)
chat.display()
```

### Streaming Generation

```python
from llcuda.jupyter import stream_generate

text = stream_generate(
    engine,
    "Write a short poem about AI",
    max_tokens=150,
    show_timing=True,
    markdown=True
)
```

### Conversation Management

```python
from llcuda.chat import ChatEngine

chat = ChatEngine(engine, system_prompt="You are a helpful assistant")
chat.add_user_message("What is Python?")
response = chat.complete()
print(response)

# Save conversation
chat.save_history("conversation.json")
```

### Text Embeddings & Semantic Search

```python
from llcuda.embeddings import EmbeddingEngine, SemanticSearch

embedder = EmbeddingEngine(engine, normalize=True)
search = SemanticSearch(embedder)

# Index documents
documents = [
    "Python is a programming language",
    "Machine learning uses neural networks",
    "AI is transforming technology"
]
search.add_documents(documents)

# Search
results = search.search("Tell me about AI", top_k=2)
for doc, score, _ in results:
    print(f"[{score:.3f}] {doc}")
```

### Model Management

```python
from llcuda.models import list_models, ModelInfo, print_model_catalog

# List local models
models = list_models()
for m in models:
    print(f"{m['filename']}: {m['file_size_mb']:.1f} MB")

# Get model info and recommendations
info = ModelInfo.from_file("model.gguf")
settings = info.get_recommended_settings(vram_gb=4.0)
print(f"Recommended GPU layers: {settings['gpu_layers']}")

# Browse model catalog
print_model_catalog(vram_gb=8.0)
```

### Performance Metrics

```python
from llcuda.jupyter import display_metrics

# Run some inferences
for prompt in prompts:
    engine.infer(prompt)

# Display formatted metrics
display_metrics(engine)
```

## 🏗️ Architecture

```
llcuda v0.3.0
├── Core (v0.2.x - unchanged)
│   ├── InferenceEngine    - Main inference API
│   ├── ServerManager      - llama-server lifecycle
│   └── Utils              - System detection, model discovery
│
└── JupyterLab (v0.3.0 - new)
    ├── jupyter.py         - Streaming, widgets, visualization
    ├── chat.py            - Chat completion, conversation history
    ├── embeddings.py      - Embedding generation, semantic search
    └── models.py          - Model management, recommendations
```

## 💻 System Requirements

### Minimum
- Ubuntu 22.04 (or similar Linux)
- Python 3.11+
- NVIDIA GPU with CUDA 5.0+ compute capability
- 1GB VRAM (for small models)

### Recommended
- Ubuntu 22.04 LTS
- Python 3.11+
- NVIDIA GPU with 4GB+ VRAM
- CUDA 12.x
- 8GB+ RAM

### Tested On
- GeForce 940M (1GB VRAM) ✅
- GTX 1060 (6GB VRAM) ✅
- RTX 3060 (12GB VRAM) ✅
- RTX 4090 (24GB VRAM) ✅

## 🔧 Installation Options

### Option 1: Use Pre-built Binary (Recommended)

Download Ubuntu-Cuda-Llama.cpp-Executable:

```bash
# Download (290 MB)
wget https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable/releases/download/v0.1.0/llama.cpp-733c851f-bin-ubuntu-cuda-x64.tar.xz

# Extract
tar -xf llama.cpp-733c851f-bin-ubuntu-cuda-x64.tar.xz

# Set environment
export LLAMA_SERVER_PATH="$(pwd)/llama-cpp-cuda/bin/llama-server"
```

Then install llcuda:
```bash
pip install llcuda ipywidgets tqdm matplotlib
```

### Option 2: Build from Source

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with CUDA
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release

# Set environment
export LLAMA_SERVER_PATH="$(pwd)/bin/llama-server"

# Install llcuda
pip install llcuda ipywidgets tqdm matplotlib
```

## 📊 Performance

Benchmarks on **GeForce 940M (1GB VRAM)**:

| Model | Quantization | GPU Layers | Speed | VRAM |
|-------|--------------|------------|-------|------|
| Gemma 3 1B | Q4_K_M | 8 | 12-15 tok/s | ~350 MB |
| Phi-3 Mini | Q4_K_M | 10 | 10-12 tok/s | ~400 MB |
| TinyLlama 1.1B | Q5_K_M | 16 | 15-18 tok/s | ~280 MB |

Benchmarks on **RTX 3060 (12GB VRAM)**:

| Model | Quantization | GPU Layers | Speed | VRAM |
|-------|--------------|------------|-------|------|
| Llama 3.1 8B | Q4_K_M | 99 | 45-50 tok/s | ~5 GB |
| Mistral 7B | Q5_K_M | 99 | 40-45 tok/s | ~5.5 GB |
| Mixtral 8x7B | Q4_K_M | 99 | 25-30 tok/s | ~10 GB |

## 🤝 Integration with llama.cpp

llcuda uses [llama.cpp](https://github.com/ggerganov/llama.cpp) as the inference backend:

- **Communication**: HTTP REST API
- **Format**: GGUF (GPT-Generated Unified Format)
- **Endpoints**: `/completion`, `/v1/chat/completions`, `/embeddings`
- **Server**: llama-server (automatic management)

## 🎯 Use Cases

- **Research** - Experiment with LLMs on consumer hardware
- **Education** - Learn about LLMs without cloud costs
- **Prototyping** - Quick iteration on LLM applications
- **Local AI** - Privacy-focused AI applications
- **JupyterLab Workflows** - Interactive ML notebooks
- **Semantic Search** - Build search engines
- **Chatbots** - Create conversational AI

## 🛠️ Troubleshooting

### Widgets Not Showing
```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Server Won't Start
```python
# Check path
import os
print(os.environ.get('LLAMA_SERVER_PATH'))

# Verify file exists
from pathlib import Path
print(Path(os.environ['LLAMA_SERVER_PATH']).exists())
```

### Out of VRAM
```python
# Reduce GPU layers
engine.load_model("model.gguf", auto_start=True,
                 gpu_layers=8,      # Lower value
                 ctx_size=512,       # Smaller context
                 ubatch_size=128)    # Smaller batch
```

### Embeddings Not Working
Start server with embedding support:
```python
engine.load_model("model.gguf", auto_start=True, embedding=True)
```

## 📜 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Credits

Built on top of:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov
- [GGML](https://github.com/ggerganov/ggml) tensor library
- [Ubuntu-Cuda-Llama.cpp-Executable](https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable) pre-built binaries

## 📞 Support

- **GitHub Issues**: [llcuda/issues](https://github.com/waqasm86/llcuda/issues)
- **Discussions**: [llcuda/discussions](https://github.com/waqasm86/llcuda/discussions)
- **Documentation**: [waqasm86.github.io](https://waqasm86.github.io/)

## 🗺️ Roadmap

### v0.3.1 (Q1 2025)
- [ ] Multimodal support (vision models)
- [ ] Enhanced error messages
- [ ] Model quantization tools

### v0.4.0 (Q2 2025)
- [ ] Function calling / tool use
- [ ] Vision chat widget
- [ ] Audio support

### v1.0.0 (Q3 2025)
- [ ] Full OpenAI API compatibility
- [ ] Production deployment tools
- [ ] Distributed inference

## 🌟 Star History

If you find llcuda useful, please give it a star on GitHub! ⭐

## 📝 Citation

```bibtex
@software{llcuda2025,
  author = {waqasm86},
  title = {llcuda: CUDA-Accelerated LLM Inference for Python},
  year = {2025},
  url = {https://github.com/waqasm86/llcuda}
}
```

---

**Made with ❤️ for the open-source AI community**

**Author**: waqasm86
**Version**: 0.3.0
**Last Updated**: 2025-12-28
