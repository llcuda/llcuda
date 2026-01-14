# llcuda API Reference v2.1+

Complete API reference for llcuda's new modules introduced in v2.1+.

## Table of Contents

1. [Quantization API](#quantization-api)
2. [Unsloth Integration API](#unsloth-integration-api)
3. [CUDA Optimization API](#cuda-optimization-api)
4. [Advanced Inference API](#advanced-inference-api)

---

## Quantization API

The quantization module provides tools for converting models to GGUF format with various quantization schemes.

### Module: `llcuda.quantization`

#### NF4 Quantization

```python
from llcuda.quantization import quantize_nf4, dequantize_nf4, NF4Quantizer

# Quantize tensor to NF4
weight = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
qweight, state = quantize_nf4(weight, blocksize=64, double_quant=True)

# Dequantize
weight_restored = dequantize_nf4(qweight, state)

# Using NF4Quantizer class
quantizer = NF4Quantizer(blocksize=64, double_quant=True)
qweight, state = quantizer.quantize(weight)
```

**Functions:**

- `quantize_nf4(weight, blocksize=64, double_quant=True)`: Quantize tensor to NF4
  - **Args**: weight (Tensor), blocksize (int), double_quant (bool)
  - **Returns**: (quantized_tensor, quantization_state)

- `dequantize_nf4(quantized, state)`: Dequantize NF4 tensor
  - **Args**: quantized (Tensor), state (dict)
  - **Returns**: Dequantized tensor

#### GGUF Conversion

```python
from llcuda.quantization import convert_to_gguf, GGUFConverter, GGUFQuantType

# Convert model to GGUF
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("model")
convert_to_gguf(model, "model-q4.gguf", quant_type="Q4_K_M")

# Using GGUFConverter class
converter = GGUFConverter(model, tokenizer)
converter.convert("model.gguf", quant_type="Q4_K_M")
```

**Functions:**

- `convert_to_gguf(model, output_path, tokenizer=None, quant_type="Q4_K_M", verbose=True)`: Convert model to GGUF
  - **Args**: model, output_path, tokenizer, quant_type, verbose
  - **Returns**: Path to GGUF file

**Quantization Types:**

- `Q4_K_M`: 4-bit (recommended for Tesla T4)
- `Q5_K_M`: 5-bit (higher quality)
- `Q8_0`: 8-bit (near full precision)
- `F16`: Half precision
- And more...

#### Dynamic Quantization

```python
from llcuda.quantization import DynamicQuantizer, quantize_dynamic

# Recommend quantization based on constraints
quantizer = DynamicQuantizer(target_vram_gb=12.0, strategy="balanced")
config = quantizer.recommend_config(model)

print(f"Recommended: {config['quant_type']}")
print(f"Expected VRAM: {config['expected_vram_gb']:.2f} GB")

# Auto-quantize
quantize_dynamic(model, "model.gguf", target_vram_gb=12.0, strategy="balanced")
```

**Strategies:**

- `aggressive`: Maximum compression
- `balanced`: Quality/speed balance (recommended)
- `quality`: Higher quality
- `minimal`: Minimal compression

---

## Unsloth Integration API

Seamless integration between Unsloth fine-tuning and llcuda inference.

### Module: `llcuda.unsloth`

#### Loading Unsloth Models

```python
from llcuda.unsloth import load_unsloth_model, UnslothModelLoader

# Load model
model, tokenizer = load_unsloth_model(
    "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Using loader class
loader = UnslothModelLoader(max_seq_length=2048)
model, tokenizer = loader.load("model_name")
```

#### Exporting to GGUF

```python
from llcuda.unsloth import export_to_llcuda, UnslothExporter

# Export fine-tuned model
export_to_llcuda(
    model=model,
    tokenizer=tokenizer,
    output_path="model.gguf",
    quant_type="Q4_K_M",
    merge_lora=True,
)

# Using exporter class
exporter = UnslothExporter()
exporter.export(model, tokenizer, "model.gguf")
```

**Functions:**

- `export_to_llcuda(model, tokenizer, output_path, quant_type="Q4_K_M", merge_lora=True, verbose=True)`: Export to GGUF
  - **Args**: model, tokenizer, output_path, quant_type, merge_lora, verbose
  - **Returns**: Path to exported file

#### LoRA Adapter Management

```python
from llcuda.unsloth import merge_lora_adapters, LoRAAdapter

# Merge adapters
merged_model = merge_lora_adapters(model)

# Using adapter class
adapter = LoRAAdapter(model)
if adapter.has_adapters():
    merged = adapter.merge()
    adapter.save_merged(merged, "output_dir")
```

---

## CUDA Optimization API

Advanced CUDA optimizations for Tesla T4.

### Module: `llcuda.cuda`

#### Tensor Core Support

```python
from llcuda.cuda import (
    check_tensor_core_support,
    enable_tensor_cores,
    matmul_tensor_core,
    get_tensor_core_info,
)

# Check support
if check_tensor_core_support():
    print("Tensor Cores available!")

# Enable Tensor Cores
config = enable_tensor_cores(dtype=torch.float16, allow_tf32=True)

# Use Tensor Core matmul
C = matmul_tensor_core(A, B, dtype=torch.float16)

# Get info
info = get_tensor_core_info()
print(f"Speedup: {info['estimated_speedup']}")
```

#### CUDA Graphs

```python
from llcuda.cuda import CUDAGraph, capture_graph, replay_graph

# Capture operations as graph
graph = CUDAGraph()

with graph.capture():
    output = model(input)

# Fast replay
for _ in range(100):
    graph.replay()

# Using functions
def forward():
    return model(input)

graph = capture_graph(forward, warmup_iters=3)
output = replay_graph(graph)
```

**Benefits:**

- 20-40% latency reduction
- Eliminates kernel launch overhead
- Better GPU utilization

#### Triton Kernels

```python
from llcuda.cuda import (
    register_kernel,
    get_kernel,
    list_kernels,
    triton_add,
    triton_layernorm,
)

# List available kernels
kernels = list_kernels()
print(kernels)  # ['add', 'layernorm', 'softmax']

# Use built-in kernels
c = triton_add(a, b)
normalized = triton_layernorm(x, weight, bias)

# Register custom kernel
@triton.jit
def my_kernel(...):
    ...

register_kernel("my_kernel", my_kernel)
```

---

## Advanced Inference API

Enhanced inference capabilities for long contexts and batch optimization.

### Module: `llcuda.inference`

#### FlashAttention

```python
from llcuda.inference import (
    enable_flash_attention,
    flash_attention_forward,
    check_flash_attention_available,
    get_optimal_context_length,
)

# Check availability
if check_flash_attention_available():
    print("FlashAttention available")

# Enable for model
model = enable_flash_attention(model)

# Use FlashAttention
output = flash_attention_forward(query, key, value, causal=True)

# Get optimal context length
ctx_len = get_optimal_context_length(
    model_size_b=3.0,
    available_vram_gb=12.0,
    use_flash_attention=True,
)
```

**Benefits:**

- 2-3x speedup for long sequences (>1024 tokens)
- Reduced memory usage
- Support for 4K-8K contexts on T4

#### KV-Cache Optimization

```python
from llcuda.inference import KVCache, KVCacheConfig, optimize_kv_cache

# Create KV-cache
config = KVCacheConfig(
    max_batch_size=8,
    max_seq_length=4096,
)
cache = KVCache(config)

# Update cache
k_cached, v_cached = cache.update(layer_idx, k_new, v_new)

# Optimize model
model = optimize_kv_cache(model)
```

#### Batch Optimization

```python
from llcuda.inference import BatchInferenceOptimizer, batch_inference_optimized

# Optimize batch inference
optimizer = BatchInferenceOptimizer(max_batch_size=8)
results = optimizer.batch_infer(prompts, inference_fn)

# Using convenience function
results = batch_inference_optimized(
    prompts=prompts,
    model=engine,
    max_batch_size=8,
)
```

---

## Complete Workflow Example

```python
import llcuda
from llcuda.unsloth import export_to_llcuda
from llcuda.cuda import enable_tensor_cores
from unsloth import FastLanguageModel

# 1. Fine-tune with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained("base_model")
# ... training ...

# 2. Export to GGUF with quantization
export_to_llcuda(
    model=model,
    tokenizer=tokenizer,
    output_path="model-q4.gguf",
    quant_type="Q4_K_M",
    merge_lora=True,
)

# 3. Deploy with llcuda
enable_tensor_cores()  # Enable T4 optimizations
engine = llcuda.InferenceEngine()
engine.load_model("model-q4.gguf", silent=True)

# 4. Run inference
result = engine.infer("What is AI?", max_tokens=200)
print(f"{result.text}")
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")

# 5. Batch inference
results = engine.batch_infer(prompts, max_tokens=150)
```

---

## Configuration Reference

### Quantization Configurations

```python
# NF4Config
NF4Config(
    blocksize=64,           # 64, 128, 256, or 512
    double_quant=True,      # Double quantization
    quant_type="nf4",       # Quantization type
    compute_dtype=torch.float16,
)

# AutoQuantConfig
AutoQuantConfig(
    target_vram_gb=12.0,    # Target VRAM
    target_speed_tps=None,  # Target tokens/sec
    strategy="balanced",    # aggressive, balanced, quality, minimal
    preserve_embeddings=True,
    preserve_output=True,
)
```

### CUDA Configurations

```python
# TensorCoreConfig
TensorCoreConfig(
    enabled=True,
    dtype=torch.float16,
    allow_tf32=True,
    allow_fp16=True,
)

# GraphCaptureConfig
GraphCaptureConfig(
    pool=None,
    capture_error_mode="thread_local",
    warmup_iters=3,
)
```

### Inference Configurations

```python
# FlashAttentionConfig
FlashAttentionConfig(
    enabled=True,
    version=2,
    causal=True,
    dropout_p=0.0,
    softmax_scale=None,
    window_size=None,
)

# KVCacheConfig
KVCacheConfig(
    max_batch_size=8,
    max_seq_length=4096,
    num_layers=32,
    num_heads=32,
    head_dim=128,
    dtype=torch.float16,
)
```

---

## Performance Tips

### For Tesla T4 (16GB VRAM)

1. **Quantization**: Use Q4_K_M for best balance
2. **Tensor Cores**: Always enable for FP16 operations
3. **Context Length**: 4K-8K with FlashAttention
4. **Batch Size**: 4-8 for optimal throughput

### Model Size Recommendations

| Model Size | Quant Type | Expected Speed | VRAM Usage |
|------------|------------|----------------|------------|
| 1B params  | Q4_K_M     | ~45 tok/s      | ~1.5 GB    |
| 3B params  | Q4_K_M     | ~30 tok/s      | ~2.5 GB    |
| 7B params  | Q4_K_M     | ~18 tok/s      | ~5.0 GB    |
| 8B params  | Q5_K_M     | ~15 tok/s      | ~6.0 GB    |

---

## Troubleshooting

### Common Issues

1. **Triton not available**: `pip install triton`
2. **FlashAttention not available**: `pip install flash-attn --no-build-isolation`
3. **Unsloth not available**: `pip install unsloth`
4. **CUDA out of memory**: Reduce batch size or use more aggressive quantization

### Getting Help

- Documentation: https://llcuda.github.io/
- GitHub Issues: https://github.com/llcuda/llcuda/issues
- Examples: https://github.com/llcuda/llcuda/tree/main/examples

---

## Version Information

- **API Version**: 2.1+
- **llcuda Version**: 2.1.0+
- **Python**: 3.11+
- **CUDA**: 12.x
- **Target GPU**: Tesla T4 (SM 7.5)
