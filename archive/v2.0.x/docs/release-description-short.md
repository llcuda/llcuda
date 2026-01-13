## llcuda v2.0.1 - Tesla T4 CUDA 12 Binaries

CUDA 12 inference backend optimized for **Tesla T4 GPU** (Google Colab). Features FlashAttention 2, Tensor Core optimization, and CUDA Graphs for 2-3x faster performance.

### 📦 Package Details

**Binary:** `llcuda-binaries-cuda12-t4.tar.gz` (140 MB)
- llama-server with FlashAttention support
- CUDA 12 libraries optimized for SM 7.5 (Tesla T4)
- Auto-downloaded on first `pip install llcuda`

### 🚀 Installation

```bash
pip install llcuda
```

Binaries download automatically on first import (~140 MB, one-time).

### 📊 Performance (Tesla T4)

| Model | Speed | VRAM |
|-------|-------|------|
| Gemma 3-1B Q4_K_M | 45 tok/s | 1.2 GB |
| Llama 3.2-3B Q4_K_M | 30 tok/s | 2.0 GB |
| Qwen 2.5-7B Q4_K_M | 18 tok/s | 5.0 GB |
| Llama 3.1-8B Q4_K_M | 15 tok/s | 5.5 GB |

### ✨ Features

✅ FlashAttention 2 (2-3x faster for long contexts)
✅ Tensor Core optimization (FP16/INT8)
✅ CUDA Graphs (reduced overhead)
✅ All quantization types (Q2_K - Q8_0)

### 🎯 Requirements

- **GPU:** Tesla T4 (SM 7.5)
- **Python:** 3.11+
- **CUDA:** 12.x runtime
- **Platform:** Google Colab, Kaggle, or local T4

### 🔧 Quick Start (Google Colab)

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

### 🔐 Checksum

```
SHA256: 54bb904f213b38aec4a2c85baa2fab8256200966aa994bb9fa4a77640d699aa4
```

### 🛠️ Build Info

- CUDA: 12.4/12.6
- Compute: SM 7.5 (Turing)
- llama.cpp: 0.0.7654
- GGML: 0.9.5
- Built: January 7, 2026

### 📚 Links

- PyPI: https://pypi.org/project/llcuda/
- Docs: https://github.com/waqasm86/llcuda#readme
- Issues: https://github.com/waqasm86/llcuda/issues

---

**Target:** Google Colab Tesla T4 (free tier) | **License:** MIT
