# llcuda v2.2.0 - Quick Start Guide

Get started with llcuda in 5 minutes.

## Installation

```bash
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
```

## Single GPU Usage (Colab/Kaggle T4)

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
```

## Multi-GPU Server (Kaggle 2× T4)

### 1. Start Server
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

### 2. Connect with Python
```python
from llcuda.api.client import LlamaCppClient

client = LlamaCppClient("http://localhost:8080")

# Use OpenAI-compatible chat.create() API
response = client.chat.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

## Unsloth Workflow

```python
# 1. Fine-tune with Unsloth
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Llama-3.2-1B-Instruct")

# 2. Export to GGUF
model.save_pretrained_gguf("my_model", tokenizer, quantization_method="q4_k_m")

# 3. Serve with llcuda
# ./bin/llama-server -m my_model-Q4_K_M.gguf -ngl 99 --tensor-split 0.5,0.5
```

## Key APIs

| Module | Purpose |
|--------|---------|
| `llcuda.api.client` | Full llama.cpp server client |
| `llcuda.api.multigpu` | Multi-GPU configuration |
| `llcuda.api.gguf` | GGUF parsing & quantization |

## Links

- [Full Documentation](README.md)
- [Build Notebook](notebooks/build_llcuda_v2_2_0_kaggle_t4x2_complete.ipynb)
- [GitHub Releases](https://github.com/llcuda/llcuda/releases)
