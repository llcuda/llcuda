# llcuda v2.0.2 - Short Descriptions

## PyPI Project Description (Short)
```
CUDA 12 inference backend for Unsloth - Tesla T4 optimized. Python-first API with native tensor operations, custom CUDA kernels, FlashAttention, and NF4 quantization.
```

## GitHub Repository Description (Short)
```
CUDA-accelerated LLM inference for Python. PyTorch-style, zero-configuration, works on Colab Tesla T4 GPU.
```

## PyPI Classifiers Keywords (for pyproject.toml)
```python
keywords = [
    "llm", "cuda", "inference", "unsloth", "t4", "tesla-t4", "tensor-cores",
    "flashattention", "nf4", "quantization", "gguf", "pytorch",
    "deep-learning", "gpu", "colab", "tensor-api", "google-colab"
]
```

## One-Line Summary
```
Production-ready CUDA inference backend for Tesla T4 GPUs with FlashAttention, Tensor Cores, and native Python API.
```

## PyPI Long Description (README Summary - First 3 Paragraphs)

**Python-first CUDA inference backend exposing low-level, quantization-aware GPU execution for Unsloth fine-tuned models.**

llcuda v2.0 is a production-ready inference backend designed **exclusively for Tesla T4 GPU** (Google Colab standard) with:

- **Native Tensor API**: PyTorch-style GPU operations with custom CUDA kernels
- **Tensor Core Optimization**: SM 7.5 targeting for maximum performance
- **FlashAttention Support**: 2-3x faster attention for long contexts
- **CUDA Graphs**: Reduced kernel launch overhead
- **Unsloth Integration**: Direct loading of NF4-quantized fine-tuned models
- **GGUF Support**: Compatible with llama.cpp model format

## GitHub About Section
```
CUDA inference backend for Unsloth - Tesla T4 optimized with FlashAttention, Tensor Cores, and native Python API
```

## Tags for GitHub Topics
```
cuda, llm, inference, tesla-t4, flashattention, tensor-cores, unsloth, gguf, pytorch, google-colab, nf4-quantization, cuda-kernels, deep-learning
```

---

## Version History Clean Description

### Versions to Keep Visible (v2.0.2 and below)
- **v2.0.2** (2026-01-08) - Critical bug fixes: 404 download error, version consistency, tar structure
- **v2.0.1** (2026-01-07) - Cleanup release: removed duplicates, improved .gitignore
- **v2.0.0** (2026-01-06) - Major: Native Tensor API, Tesla T4 exclusive, FlashAttention
- **v1.2.2** (2026-01-04) - Legacy: Multi-GPU support (archived)

### Versions to Archive/Hide (< v1.2.2)
All versions below v1.2.2 should be marked as "legacy" or hidden from main documentation.

---

## Installation Quick Start (for README)

```python
# Install
pip install llcuda

# Verify
import llcuda
print(f"llcuda {llcuda.__version__}")

# Quick test
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)
result = engine.infer("Hello, AI!", max_tokens=50)
print(result.text)
```

---

## Key Message for v2.0.2

**This version fixes critical installation failures on Kaggle/Colab. All v2.0.0/v2.0.1 users should upgrade immediately.**
