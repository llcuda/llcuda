# llcuda v2.2.0

[![Version](https://img.shields.io/badge/version-2.2.0-blue.svg)](https://github.com/llcuda/llcuda)
[![Python](https://img.shields.io/badge/python-3.11+-brightgreen.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**CUDA 12 Inference Backend for Unsloth** — 62KB Python package with auto-download binaries for Tesla T4/multi-GPU inference.

## Features

- 🚀 **Ultra-lightweight**: 62KB package with automatic binary download
- 🔥 **FlashAttention**: Enabled for all quantization types
- 🖥️ **Multi-GPU**: Native \`--tensor-split\` support (Kaggle 2× T4)
- 🤖 **OpenAI-compatible**: Full llama.cpp server API coverage
- 🔧 **GGUF Utilities**: Parse, quantize, and convert models
- ⚡ **Unsloth Integration**: Seamless training → GGUF → inference workflow

## Quick Start

### Installation

\`\`\`bash
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
\`\`\`

### Single GPU (Colab T4)

\`\`\`python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
\`\`\`

### Multi-GPU (Kaggle 2× T4)

\`\`\`bash
# Start server with tensor-split
./bin/llama-server \
    -m model.gguf \
    -ngl 99 \
    --tensor-split 0.5,0.5 \
    --split-mode layer \
    -fa \
    --host 0.0.0.0 \
    --port 8080
\`\`\`

\`\`\`python
from llcuda.api import LlamaCppClient

client = LlamaCppClient("http://localhost:8080")
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
\`\`\`

## API Reference

### Core API

\`\`\`python
import llcuda

# InferenceEngine - High-level model loading and inference
engine = llcuda.InferenceEngine()
engine.load_model("model-name")
result = engine.infer("prompt", max_tokens=100)
\`\`\`

### Client API (\`llcuda.api.client\`)

\`\`\`python
from llcuda.api import LlamaCppClient

client = LlamaCppClient("http://localhost:8080")

# Chat completions (OpenAI-compatible)
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
    temperature=0.7
)

# Text completions
response = client.completions.create(
    prompt="Once upon a time",
    max_tokens=100
)

# Embeddings
embeddings = client.embeddings.create(input="Hello world")

# Tokenization
tokens = client.tokenize("Hello world")
text = client.detokenize([1, 2, 3])

# Health check
health = client.health()
\`\`\`

### Multi-GPU Configuration (\`llcuda.api.multigpu\`)

\`\`\`python
from llcuda.api import MultiGPUConfig, kaggle_t4_dual_config

# Pre-configured for Kaggle 2× T4
config = kaggle_t4_dual_config()
print(config.to_cli_args())
# Output: --tensor-split 0.5,0.5 --split-mode layer -ngl 99 -fa

# Custom configuration
config = MultiGPUConfig(
    tensor_split=[0.6, 0.4],  # 60% GPU 0, 40% GPU 1
    split_mode="layer",
    n_gpu_layers=99,
    flash_attention=True
)
\`\`\`

### GGUF Utilities (\`llcuda.api.gguf\`)

\`\`\`python
from llcuda.api.gguf import parse_gguf_header, quantize, convert_hf_to_gguf

# Parse GGUF file header
header = parse_gguf_header("model.gguf")
print(f"Architecture: {header['architecture']}")
print(f"Parameters: {header['n_parameters']}")

# Quantize model
quantize("model-f16.gguf", "model-Q4_K_M.gguf", quant_type="Q4_K_M")

# Convert HuggingFace to GGUF
convert_hf_to_gguf("unsloth/gemma-3-1b", "gemma-3-1b-f16.gguf")
\`\`\`

## Unsloth Workflow

llcuda is designed as the inference backend for Unsloth-trained models:

\`\`\`
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Unsloth       │ ──▶ │   GGUF Export   │ ──▶ │   llcuda        │
│   (Training)    │     │   (Quantize)    │     │   (Inference)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
\`\`\`

### Complete Example

\`\`\`python
# Step 1: Train with Unsloth (in Colab)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/gemma-3-1b")
# ... fine-tune model ...
model.save_pretrained_gguf("gemma-finetuned", tokenizer, quantization="Q4_K_M")

# Step 2: Serve with llcuda
import llcuda
engine = llcuda.InferenceEngine()
engine.load_model("gemma-finetuned-Q4_K_M.gguf")
response = engine.infer("Your prompt here", max_tokens=200)
\`\`\`

## Performance

| Platform | GPU | VRAM | Tokens/sec (Q4_K_M) |
|----------|-----|------|---------------------|
| Colab Free | T4 | 15GB | ~35 tok/s |
| Kaggle | 2× T4 | 30GB | ~60 tok/s |

## Requirements

- Python 3.11+
- CUDA 12.x
- Tesla T4 or compatible GPU (SM 7.5+)

## Documentation

- [Quick Start Guide](QUICK_START.md)
- [Installation Guide](INSTALL.md)
- [Changelog](CHANGELOG.md)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **GitHub**: https://github.com/llcuda/llcuda
- **Issues**: https://github.com/llcuda/llcuda/issues
