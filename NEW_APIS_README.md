# llcuda v2.1+ New APIs

Comprehensive new APIs for tight integration with Unsloth and advanced CUDA optimizations on Tesla T4.

## Overview

llcuda v2.1+ introduces four major API modules:

1. **Quantization API** - NF4, GGUF conversion, dynamic quantization
2. **Unsloth Integration API** - Seamless fine-tuning to deployment workflow
3. **CUDA Optimization API** - Tensor Cores, CUDA Graphs, Triton kernels
4. **Advanced Inference API** - FlashAttention, KV-cache, batch optimization

## Quick Start

### Installation

```bash
pip install git+https://github.com/waqasm86/llcuda.git
```

### Complete Workflow

```python
import llcuda
from llcuda.unsloth import export_to_llcuda
from llcuda.cuda import enable_tensor_cores

# 1. Export Unsloth model
export_to_llcuda(model, tokenizer, "model.gguf", quant_type="Q4_K_M")

# 2. Enable optimizations
enable_tensor_cores()

# 3. Deploy
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf")

# 4. Infer
result = engine.infer("What is AI?")
print(f"{result.text} ({result.tokens_per_sec:.1f} tok/s)")
```

## API Modules

### 1. Quantization API

Convert models to GGUF with various quantization schemes.

```python
from llcuda.quantization import (
    quantize_nf4,           # NF4 quantization
    convert_to_gguf,        # GGUF conversion
    DynamicQuantizer,       # Auto-recommendation
)

# NF4 quantization
qweight, state = quantize_nf4(weight, blocksize=64)

# Convert model to GGUF
convert_to_gguf(model, "model.gguf", quant_type="Q4_K_M")

# Dynamic quantization
quantizer = DynamicQuantizer(target_vram_gb=12.0)
config = quantizer.recommend_config(model)
```

**Supported Quantization Types:**

- Q4_K_M (recommended for Tesla T4)
- Q5_K_M (higher quality)
- Q8_0 (near full precision)
- And 20+ more...

### 2. Unsloth Integration API

Seamless integration with Unsloth fine-tuning.

```python
from llcuda.unsloth import (
    load_unsloth_model,     # Load Unsloth models
    export_to_llcuda,       # Export to GGUF
    merge_lora_adapters,    # Merge LoRA
)

# Load Unsloth model
model, tokenizer = load_unsloth_model("model_name")

# Export to llcuda
export_to_llcuda(model, tokenizer, "model.gguf", quant_type="Q4_K_M")

# Merge LoRA adapters
merged = merge_lora_adapters(model)
```

**Complete Workflow:**

```
Unsloth Training → Export to GGUF → llcuda Inference
     (GPU)              (quantize)      (optimized)
```

### 3. CUDA Optimization API

Advanced CUDA features for Tesla T4.

```python
from llcuda.cuda import (
    enable_tensor_cores,    # Tensor Core acceleration
    CUDAGraph,              # CUDA Graphs
    register_kernel,        # Triton kernels
)

# Enable Tensor Cores (2-4x speedup)
enable_tensor_cores(dtype=torch.float16)

# CUDA Graphs (20-40% latency reduction)
graph = CUDAGraph()
with graph.capture():
    output = model(input)
graph.replay()

# Custom Triton kernels
@triton.jit
def my_kernel(...):
    ...
register_kernel("my_kernel", my_kernel)
```

**Features:**

- **Tensor Cores**: 2-4x speedup for FP16 operations
- **CUDA Graphs**: 20-40% latency reduction
- **Triton Kernels**: Custom operations with Python

### 4. Advanced Inference API

Enhanced inference for long contexts and batching.

```python
from llcuda.inference import (
    enable_flash_attention,     # FlashAttention v2
    KVCache,                    # KV-cache management
    batch_inference_optimized,  # Batch optimization
)

# FlashAttention (2-3x faster for long contexts)
model = enable_flash_attention(model)

# KV-cache optimization
cache = KVCache(config)
k_cached, v_cached = cache.update(layer_idx, k, v)

# Optimized batching
results = batch_inference_optimized(prompts, model)
```

**Benefits:**

- **FlashAttention**: 2-3x faster for sequences >1024 tokens
- **KV-Cache**: Efficient sequential generation
- **Batch Optimization**: Maximum throughput

## Performance on Tesla T4

### Benchmarks

| Model | Quant | Speed (tok/s) | VRAM | Context |
|-------|-------|---------------|------|---------|
| Gemma 3-1B | Q4_K_M | 134 | 1.2 GB | 2048 |
| Llama 3.2-3B | Q4_K_M | 85 | 2.5 GB | 4096 |
| Qwen 2.5-7B | Q4_K_M | 45 | 5.0 GB | 4096 |
| Llama 3.1-8B | Q5_K_M | 38 | 6.0 GB | 4096 |

### Optimizations Impact

| Optimization | Benefit |
|--------------|---------|
| Tensor Cores | 2-4x speedup |
| CUDA Graphs | 20-40% latency reduction |
| FlashAttention | 2-3x for long contexts |
| Q4_K_M Quant | 8.5x compression |

## Examples

### Example 1: Quantize and Deploy

```python
from llcuda.quantization import DynamicQuantizer
import llcuda

# Auto-select quantization
quantizer = DynamicQuantizer(target_vram_gb=12.0)
config = quantizer.recommend_config(model_size_gb=3.0)
print(f"Use {config['quant_type']}")

# Deploy
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf")
result = engine.infer("Hello!")
```

### Example 2: Unsloth to llcuda

```python
from unsloth import FastLanguageModel
from llcuda.unsloth import export_to_llcuda
import llcuda

# Train with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained("base")
# ... training ...

# Export
export_to_llcuda(model, tokenizer, "finetuned.gguf")

# Deploy
engine = llcuda.InferenceEngine()
engine.load_model("finetuned.gguf")
```

### Example 3: CUDA Optimizations

```python
from llcuda.cuda import enable_tensor_cores, CUDAGraph
import llcuda

# Enable all optimizations
enable_tensor_cores()

# Use CUDA Graphs for repeated inference
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf")

graph = CUDAGraph()
with graph.capture():
    result = engine.infer("Test")

# Fast replay
for prompt in prompts:
    graph.replay()
```

### Example 4: Long Context with FlashAttention

```python
from llcuda.inference import enable_flash_attention, get_optimal_context_length
import llcuda

# Get optimal context
ctx_len = get_optimal_context_length(
    model_size_b=3.0,
    available_vram_gb=12.0,
    use_flash_attention=True,
)
print(f"Use {ctx_len} tokens")  # ~8192

# Enable FlashAttention
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf", context_size=ctx_len)

# Process long document
result = engine.infer(long_prompt, max_tokens=1000)
```

## Architecture

```
llcuda/
├── quantization/       # Quantization APIs
│   ├── nf4.py         # NF4 quantization
│   ├── gguf.py        # GGUF conversion
│   └── dynamic.py     # Auto-quantization
├── unsloth/           # Unsloth integration
│   ├── loader.py      # Model loading
│   ├── exporter.py    # GGUF export
│   └── adapter.py     # LoRA management
├── cuda/              # CUDA optimizations
│   ├── graphs.py      # CUDA Graphs
│   ├── triton_kernels.py  # Triton kernels
│   └── tensor_core.py # Tensor Core utils
└── inference/         # Advanced inference
    ├── flash_attn.py  # FlashAttention
    ├── kv_cache.py    # KV-cache
    └── batch.py       # Batch optimization
```

## Requirements

- **Python**: 3.11+
- **CUDA**: 12.x
- **GPU**: Tesla T4 (SM 7.5) recommended
- **VRAM**: 8-16GB

### Optional Dependencies

```bash
# For Triton kernels
pip install triton

# For FlashAttention
pip install flash-attn --no-build-isolation

# For Unsloth integration
pip install unsloth
```

## Configuration

### Recommended Settings for Tesla T4

```python
# Quantization
quant_type = "Q4_K_M"  # Best balance

# Context length
context_size = 4096  # With FlashAttention: 8192

# Batch size
batch_size = 4-8  # Optimal throughput

# GPU layers
gpu_layers = 99  # Offload all to GPU

# Tensor Cores
enable_tensor_cores(dtype=torch.float16, allow_tf32=True)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Module not found
   ```bash
   # Reinstall from GitHub
   pip uninstall llcuda -y
   pip install git+https://github.com/waqasm86/llcuda.git
   ```

2. **Triton not available**
   ```bash
   pip install triton
   ```

3. **FlashAttention not available**
   ```bash
   pip install flash-attn --no-build-isolation
   ```

4. **CUDA out of memory**
   - Use more aggressive quantization (Q4_K_M → Q3_K)
   - Reduce context size or batch size
   - Use smaller model

## Migration Guide

### From v2.0 to v2.1+

No breaking changes! All v2.0 APIs still work.

**New capabilities:**

```python
# v2.0 (still works)
import llcuda
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf")

# v2.1+ (new features)
from llcuda.cuda import enable_tensor_cores
enable_tensor_cores()  # 2-4x faster!

from llcuda.unsloth import export_to_llcuda
export_to_llcuda(model, tokenizer, "model.gguf")  # Direct export!
```

## Documentation

- **API Reference**: [API_REFERENCE.md](./API_REFERENCE.md)
- **Examples**: [examples/](./examples/)
- **Website**: https://llcuda.github.io/
- **GitHub**: https://github.com/waqasm86/llcuda

## Support

- **Issues**: https://github.com/waqasm86/llcuda/issues
- **Discussions**: https://github.com/waqasm86/llcuda/discussions
- **Email**: waqasm86@gmail.com

## License

MIT License - see [LICENSE](./LICENSE)

## Citation

If you use llcuda in your research, please cite:

```bibtex
@software{llcuda2025,
  title = {llcuda: CUDA-Accelerated LLM Inference for Tesla T4},
  author = {Muhammad, Waqas},
  year = {2025},
  url = {https://github.com/waqasm86/llcuda}
}
```

---

**Built with ❤️ for the Unsloth and llama.cpp community**

Tesla T4 optimized | CUDA 12 powered | Unsloth integrated
