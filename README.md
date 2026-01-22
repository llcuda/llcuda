# llcuda v2.2.0

[![Version](https://img.shields.io/badge/version-2.2.0-blue.svg)](https://github.com/llcuda/llcuda/releases/tag/v2.2.0)
[![Python](https://img.shields.io/badge/python-3.11+-brightgreen.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-2×T4-orange.svg)](https://kaggle.com)
[![Documentation](https://img.shields.io/badge/docs-llcuda.github.io-blue.svg)](https://llcuda.github.io)

**CUDA 12 inference backend for Unsloth** — lightweight Python package with auto-download binaries for Tesla T4 and multi-GPU inference on Kaggle.

🌐 **[Official Documentation](https://llcuda.github.io/)** | 📖 **[Tutorial Notebooks](https://llcuda.github.io/tutorials/index/)** | 🚀 **[Quick Start](https://llcuda.github.io/guides/quickstart/)** | 🔧 **[API Reference](https://llcuda.github.io/api/overview/)**

---

## 📖 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Multi-GPU Inference](#-multi-gpu-kaggle-2-t4)
- [Unsloth Integration](#-unsloth-integration)
- [Split-GPU Architecture](#-split-gpu-architecture)
- [Features](#-features)
- [Performance](#-performance)
- [Tutorial Notebooks](#-tutorial-notebooks)
- [Documentation](#-documentation)
- [Requirements](#-requirements)

---

## 🚀 Installation

### From GitHub (Recommended)
```bash
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
```

### Development Install
```bash
git clone https://github.com/llcuda/llcuda.git
cd llcuda
pip install -e .
```

### Verify Installation
```python
import llcuda
print(f"llcuda {llcuda.__version__}")  # 2.2.0
```

📘 **[Full Installation Guide →](docs/INSTALLATION.md)**

---

## ⚡ Quick Start

### Basic Inference
```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
```

### Using llama-server
```python
from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path="model.gguf",
    n_gpu_layers=99,
    flash_attn=True,
)

server = ServerManager()
server.start_with_config(config)
server.wait_until_ready()

# Now use OpenAI API at http://127.0.0.1:8080
```

📘 **[Quick Start Guide →](QUICK_START.md)**

---

## 🎯 Multi-GPU (Kaggle 2× T4)

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│              KAGGLE DUAL T4 MULTI-GPU INFERENCE                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   GPU 0: Tesla T4 (15GB)    GPU 1: Tesla T4 (15GB)              │
│   ├─ Model Layers 0-39      ├─ Model Layers 40-79               │
│   └─ ~14GB VRAM             └─ ~14GB VRAM                       │
│                                                                 │
│           ← tensor-split 0.5,0.5 (native CUDA) →                │
│                                                                 │
│   Total: 30GB VRAM for models up to 70B (IQ3_XS)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Start Multi-GPU Server
```bash
./bin/llama-server \
    -m model.gguf \
    -ngl 99 \
    --tensor-split 0.5,0.5 \
    --split-mode layer \
    -fa \
    --host 0.0.0.0 \
    --port 8080
```

### Python API
```python
from llcuda.server import ServerManager
from llcuda.api.multigpu import kaggle_t4_dual_config
from llcuda.api.client import LlamaCppClient

# Get optimized configuration for Kaggle dual T4
config = kaggle_t4_dual_config()

# Start server with multi-GPU configuration
server = ServerManager()
tensor_split_str = ",".join(str(x) for x in config.tensor_split)
server.start_server(
    model_path="model.gguf",
    gpu_layers=config.n_gpu_layers,
    tensor_split=tensor_split_str,
    split_mode="layer",
    flash_attn=1 if config.flash_attention else 0,
)

# Use OpenAI-compatible API
client = LlamaCppClient("http://localhost:8080")
response = client.chat.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

> **Note:** llama.cpp uses **native CUDA tensor-split**, NOT NCCL.
> NCCL is available for PyTorch distributed workloads.

📘 **[Kaggle Multi-GPU Guide →](docs/KAGGLE_GUIDE.md)**

---

## 🔗 Unsloth Integration

Complete workflow from fine-tuning to deployment:

```python
# ═══════════════════════════════════════════════════════════════
# STEP 1: Fine-tune with Unsloth
# ═══════════════════════════════════════════════════════════════
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA and train...

# ═══════════════════════════════════════════════════════════════
# STEP 2: Export to GGUF
# ═══════════════════════════════════════════════════════════════
model.save_pretrained_gguf(
    "my_model",
    tokenizer,
    quantization_method="q4_k_m"  # Recommended for T4
)

# ═══════════════════════════════════════════════════════════════
# STEP 3: Deploy with llcuda
# ═══════════════════════════════════════════════════════════════
from llcuda.server import ServerManager, ServerConfig

server = ServerManager()
server.start_with_config(ServerConfig(
    model_path="my_model-Q4_K_M.gguf",
    n_gpu_layers=99,
    tensor_split="0.5,0.5",  # Dual T4
    flash_attn=True,
))
```

📘 **[Unsloth Integration Guide →](notebooks/05-unsloth-integration-llcuda-v2.2.0.ipynb)**

---

## 🔧 Split-GPU Architecture

Run LLM inference on GPU 0 while using GPU 1 for RAPIDS/Graphistry analytics:

```
┌─────────────────┐      ┌─────────────────┐
│   GPU 0 (T4)    │      │   GPU 1 (T4)    │
├─────────────────┤      ├─────────────────┤
│ llama-server    │      │ RAPIDS cuDF     │
│ LLM Inference   │      │ cuGraph         │
│ ~5-12 GB        │      │ Graphistry      │
└─────────────────┘      └─────────────────┘
```

```python
from llcuda import SplitGPUConfig

config = SplitGPUConfig(llm_gpu=0, graph_gpu=1)
# GPU 0: llama-server (LLM inference)
# GPU 1: RAPIDS cuGraph (graph visualization)
```

📘 **[Split-GPU Tutorial →](notebooks/06-split-gpu-graphistry-llcuda-v2.2.0.ipynb)**

---

## 🎨 GGUF Architecture Visualization ⭐ NEW

**Visualize your GGUF models as interactive graphs** with Notebook 11:

```
┌─────────────────────────────────────────────────────────────────┐
│         GGUF NEURAL NETWORK ARCHITECTURE VISUALIZATION          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   📊 929 Nodes: Complete Llama-3.2-3B structure                 │
│   🔗 981 Edges: All connections and data flows                  │
│   🎯 896 Attention Heads: Multi-head attention visualized       │
│   📦 112 Quantization Blocks: Q4_K_M structure revealed         │
│   🌐 Interactive Graphistry Dashboards: Cloud + offline HTML    │
│                                                                 │
│   ✨ First comprehensive GGUF visualization tool                │
│   ✨ GPU-accelerated graph analytics (PageRank, centrality)     │
│   ✨ Dual-GPU architecture (inference + visualization)          │
│   ✨ Multi-scale: From overview to individual attention heads   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**What You Can Visualize:**
- Layer-by-layer transformer structure (35 nodes per layer)
- Attention head importance and connectivity
- Quantization block memory layout
- Information flow through the network
- Critical components via PageRank analysis

📘 **[GGUF Visualization Guide →](docs/GGUF_NEURAL_NETWORK_VISUALIZATION.md)** | 📓 **[Notebook 11 →](notebooks/11-gguf-neural-network-graphistry-visualization.ipynb)**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **FlashAttention** | Enabled for all quantization types |
| **Multi-GPU** | Native `--tensor-split` for Kaggle 2× T4 |
| **Split-GPU** | LLM + RAPIDS/Graphistry workloads |
| **OpenAI API** | Full llama.cpp server compatibility |
| **GGUF Tools** | Parse, quantize, convert models |
| **Auto-download** | 62KB package, binaries fetched on first run |
| **70B Support** | IQ3_XS quantization fits dual T4 |
| **Streaming** | Server-sent events for real-time output |

---

## 📊 Performance

| Platform | GPU | Model | Quantization | Tokens/sec |
|----------|-----|-------|--------------|------------|
| Colab | T4 | Gemma 3-1B | Q4_K_M | ~45 tok/s |
| Kaggle | 2× T4 | Gemma 2-2B | Q4_K_M | ~60 tok/s |
| Kaggle | 2× T4 | Qwen2.5-7B | Q4_K_M | ~35 tok/s |
| Kaggle | 2× T4 | Llama-3.1-70B | IQ3_XS | ~8-12 tok/s |

### VRAM Requirements

| Model Size | Quantization | VRAM | Fits Kaggle? |
|------------|--------------|------|--------------|
| 1-3B | Q4_K_M | 2-3 GB | ✅ Single T4 |
| 7-8B | Q4_K_M | 5-6 GB | ✅ Single T4 |
| 13B | Q4_K_M | 8-9 GB | ✅ Single T4 |
| 32-34B | Q4_K_M | 20-22 GB | ✅ Dual T4 |
| 70B | IQ3_XS | 25-27 GB | ✅ Dual T4 |

---

## 📓 Tutorial Notebooks

Comprehensive Kaggle-ready tutorials in [`notebooks/`](notebooks/):

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Quick Start](notebooks/01-quickstart-llcuda-v2.2.0.ipynb) | 5-minute introduction |
| 02 | [Server Setup](notebooks/02-llama-server-setup-llcuda-v2.2.0.ipynb) | Advanced server configuration |
| 03 | [Multi-GPU](notebooks/03-multi-gpu-inference-llcuda-v2.2.0.ipynb) | Dual T4 tensor-split |
| 04 | [GGUF Quantization](notebooks/04-gguf-quantization-llcuda-v2.2.0.ipynb) | Complete quantization guide |
| 05 | [Unsloth Integration](notebooks/05-unsloth-integration-llcuda-v2.2.0.ipynb) | Train → Export → Deploy |
| 06 | [Split-GPU + Graphistry](notebooks/06-split-gpu-graphistry-llcuda-v2.2.0.ipynb) | LLM + RAPIDS analytics |
| 07 | [OpenAI API Client](notebooks/07-openai-api-client-llcuda-v2.2.0.ipynb) | Full API reference |
| 08 | [NCCL + PyTorch](notebooks/08-nccl-pytorch-llcuda-v2.2.0.ipynb) | Distributed training |
| 09 | [Large Models](notebooks/09-large-models-kaggle-llcuda-v2.2.0.ipynb) | 70B on dual T4 |
| 10 | [Complete Workflow](notebooks/10-complete-workflow-llcuda-v2.2.0.ipynb) | End-to-end tutorial |
| 11 | [**GGUF Visualization**](notebooks/11-gguf-neural-network-graphistry-visualization.ipynb) | ⭐ Interactive architecture graphs |

📘 **[Notebooks Index →](notebooks/README.md)**

---

## 📚 Documentation

### Core Documentation
| Document | Description |
|----------|-------------|
| [QUICK_START.md](QUICK_START.md) | Get started in 5 minutes |
| [INSTALL.md](INSTALL.md) | Detailed installation guide |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

### In-Depth Guides
| Document | Description |
|----------|-------------|
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Complete installation reference |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Server & client configuration |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Python API documentation |
| [docs/KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md) | Kaggle-specific guide |
| [docs/GGUF_GUIDE.md](docs/GGUF_GUIDE.md) | GGUF format & quantization |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues & solutions |

### Contributing
| Document | Description |
|----------|-------------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md) | Building from source |

---

## 📋 Requirements

- **Python:** 3.11+
- **CUDA:** 12.x
- **GPU:** Tesla T4 or compatible (SM 7.5+)
- **Platform:** Linux (Kaggle, Colab, or local)

### Recommended Kaggle Settings
- GPU: T4 × 2
- Internet: Enabled
- Persistence: Enabled (for builds)

---

## 📦 Binary Package

| File | Size | Platform |
|------|------|----------|
| `llcuda-v2.2.0-cuda12-kaggle-t4x2.tar.gz` | 961 MB | Kaggle 2× T4 |

**Build Info:**
- CUDA 12.5, SM 7.5 (Turing)
- llama.cpp b7760 (commit 388ce82)
- Build Date: 2026-01-16
- Contents: 13 binaries (llama-server, llama-cli, llama-quantize, etc.)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/llcuda/llcuda.git
cd llcuda
pip install -e ".[dev]"
pytest tests/
```

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

## 📓 Tutorial Notebooks (10 notebooks)

Complete tutorial series for llcuda v2.2.0 on Kaggle dual T4 GPUs. Click the badges to open directly in Kaggle or view on GitHub.

| # | Notebook | Open in Kaggle | Description |
|---|----------|----------------|-------------|
| 01 | [Quick Start](notebooks/01-quickstart-llcuda-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llcuda/llcuda/blob/main/notebooks/01-quickstart-llcuda-v2.2.0.ipynb) | 5-minute introduction to llcuda |
| 02 | [Llama Server Setup](notebooks/02-llama-server-setup-llcuda-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llcuda/llcuda/blob/main/notebooks/02-llama-server-setup-llcuda-v2.2.0.ipynb) | Server configuration & lifecycle |
| 03 | [Multi-GPU Inference](notebooks/03-multi-gpu-inference-llcuda-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llcuda/llcuda/blob/main/notebooks/03-multi-gpu-inference-llcuda-v2.2.0.ipynb) | Dual T4 tensor-split configuration |
| 04 | [GGUF Quantization](notebooks/04-gguf-quantization-llcuda-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llcuda/llcuda/blob/main/notebooks/04-gguf-quantization-llcuda-v2.2.0.ipynb) | K-quants, I-quants, GGUF parsing |
| 05 | [Unsloth Integration](notebooks/05-unsloth-integration-llcuda-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llcuda/llcuda/blob/main/notebooks/05-unsloth-integration-llcuda-v2.2.0.ipynb) | Fine-tune → GGUF → Deploy |
| 06 | [Split-GPU + Graphistry](notebooks/06-split-gpu-graphistry-llcuda-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llcuda/llcuda/blob/main/notebooks/06-split-gpu-graphistry-llcuda-v2.2.0.ipynb) | LLM on GPU 0 + RAPIDS on GPU 1 |
| 07 | [OpenAI API Client](notebooks/07-openai-api-client-llcuda-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llcuda/llcuda/blob/main/notebooks/07-openai-api-client-llcuda-v2.2.0.ipynb) | Drop-in OpenAI SDK replacement |
| 08 | [NCCL + PyTorch](notebooks/08-nccl-pytorch-llcuda-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llcuda/llcuda/blob/main/notebooks/08-nccl-pytorch-llcuda-v2.2.0.ipynb) | NCCL for distributed PyTorch |
| 09 | [Large Models (70B)](notebooks/09-large-models-kaggle-llcuda-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llcuda/llcuda/blob/main/notebooks/09-large-models-kaggle-llcuda-v2.2.0.ipynb) | 70B models on Kaggle with IQ3_XS |
| 10 | [Complete Workflow](notebooks/10-complete-workflow-llcuda-v2.2.0.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llcuda/llcuda/blob/main/notebooks/10-complete-workflow-llcuda-v2.2.0.ipynb) | End-to-end production workflow |
| 11 | [**GGUF Visualization** ⭐](notebooks/11-gguf-neural-network-graphistry-visualization.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/llcuda/llcuda/blob/main/notebooks/11-gguf-neural-network-graphistry-visualization.ipynb) | **MOST IMPORTANT**: Interactive architecture graphs |

### 🎯 Learning Paths

| Path | Notebooks | Time | Focus |
|------|-----------|------|-------|
| **Quick Start** | 01 → 02 → 03 | 1 hour | Get running fast |
| **Full Course** | 01 → 11 (all) | 4.5 hours | Complete mastery + visualization |
| **Unsloth Focus** | 01 → 04 → 05 → 10 | 2 hours | Fine-tuning workflow |
| **Large Models** | 01 → 03 → 09 | 1.5 hours | 70B on Kaggle |
| **Visualization** ⭐ | 01 → 03 → 04 → 06 → 11 | 2.5 hours | Architecture analysis |

📘 **[Full Notebook Guide →](notebooks/README.md)**
