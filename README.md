# llcuda v2.2.0

[![Version](https://img.shields.io/badge/version-2.2.0-blue.svg)](https://github.com/llcuda/llcuda/releases/tag/v2.2.0)
[![Python](https://img.shields.io/badge/python-3.11+-brightgreen.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**CUDA 12 inference backend for Unsloth** — lightweight Python package with auto-download binaries for Tesla T4 and multi-GPU inference.

## 🚀 Installation

```bash
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
```

## ⚡ Quick Start

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
```

## 🎯 Multi-GPU (Kaggle 2× T4)

```python
from llcuda.api import LlamaCppClient, kaggle_t4_dual_config

# Get optimal config for Kaggle dual T4
config = kaggle_t4_dual_config()
print(config.to_cli_args())

# Connect to server
client = LlamaCppClient("http://localhost:8080")
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

Start the server with:
```bash
./bin/llama-server -m model.gguf -ngl 99 --tensor-split 0.5,0.5 --split-mode layer -fa on --host 0.0.0.0 --port 8080
```

## 🔧 Split-GPU Architecture

For combined LLM + Graph workloads on Kaggle:

```python
from llcuda import SplitGPUConfig

config = SplitGPUConfig(llm_gpu=0, graph_gpu=1)
# GPU 0: llama-server (LLM inference)
# GPU 1: RAPIDS cuGraph (graph visualization)
```

## ✨ Features

| Feature | Description |
|---------|-------------|
| **FlashAttention** | Enabled for all quantization types |
| **Multi-GPU** | Native `--tensor-split` for Kaggle 2× T4 |
| **Split-GPU** | LLM + RAPIDS/Graphistry workloads |
| **OpenAI API** | Full llama.cpp server compatibility |
| **GGUF Tools** | Parse, quantize, convert models |
| **Auto-download** | 62KB package, 961MB binaries fetched on first run |

## 📊 Performance

| Platform | GPU | Model | Tokens/sec |
|----------|-----|-------|------------|
| Colab | T4 | Gemma 3-1B | ~45 tok/s |
| Kaggle | 2× T4 | Gemma 2-2B | ~60 tok/s |
| Kaggle | 2× T4 | Llama 3.1 70B IQ3_XS | ~12 tok/s |

## 📦 Binary Package

| File | Size | Platform |
|------|------|----------|
| `llcuda-v2.2.0-cuda12-kaggle-t4x2.tar.gz` | 961 MB | Kaggle 2× T4 |

**Contents:** 13 binaries (llama-server, llama-cli, llama-quantize, etc.)

## 📋 Requirements

- Python 3.11+
- CUDA 12.x
- Tesla T4 or compatible (SM 7.5+)

## 📚 Documentation

- [Quick Start](QUICK_START.md) · [Installation](INSTALL.md) · [Changelog](CHANGELOG.md)

## 📄 License

MIT — see [LICENSE](LICENSE)
