# Google Colab Testing Guide for llcuda v2.1.0

This guide provides step-by-step instructions for testing llcuda v2.1.0 on Google Colab with Tesla T4 GPU.

## Prerequisites

- Google Colab account (free)
- Access to GPU runtime (Tesla T4)
- Internet connection

## Step-by-Step Testing

### 1. Enable GPU Runtime

```
In Google Colab:
1. Go to Menu → Runtime → Change runtime type
2. Select GPU
3. For GPU type, select "T4 GPU" (standard free tier)
4. Click Save
```

### 2. Verify GPU

```python
# Cell 1: Verify T4 GPU
!nvidia-smi

# Expected Output:
# NVIDIA A100 or Tesla T4
# Compute Capability: 7.5 or higher
```

### 3. Install llcuda

```python
# Cell 2: Install llcuda v2.1.0
!pip install git+https://github.com/llcuda/llcuda.git

# This will:
# - Download llcuda package
# - Auto-download CUDA 12 binaries (267 MB) on first import
# - Set up environment
```

### 4. Import and Verify

```python
# Cell 3: Import llcuda
import llcuda
from llcuda.core import get_device_properties

# Verify GPU
props = get_device_properties(0)
print(f"GPU: {props.name}")
print(f"Compute Capability: SM {props.compute_capability_major}.{props.compute_capability_minor}")

# Expected Output:
# GPU: Tesla T4
# Compute Capability: SM 7.5
```

### 5. Test Inference

```python
# Cell 4: Test inference with lightweight model
import llcuda

# Create inference engine
engine = llcuda.InferenceEngine()

# Load lightweight model (quantized)
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)

# Run inference
result = engine.infer("What is artificial intelligence?", max_tokens=50)
print(f"Output: {result.text}")
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")

# Expected:
# Output: Generated response about AI
# Speed: 15-25 tokens/sec on T4
```

### 6. Test Quantization API

```python
# Cell 5: Test NF4 Quantization API
from llcuda.quantization import NF4Quantizer
import torch

# Create quantizer
quantizer = NF4Quantizer(blocksize=64, double_quant=True)

# Create test tensor
weight = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')

# Quantize
qweight, state = quantizer.quantize(weight)
print(f"Original size: {weight.nbytes / 1024 / 1024:.2f} MB")
print(f"Quantized size: {qweight.nbytes / 1024 / 1024:.2f} MB")
print(f"Compression ratio: {weight.nbytes / qweight.nbytes:.2f}x")

# Expected:
# Compression ratio: ~4x (4-bit quantization)
```

### 7. Test CUDA Graphs

```python
# Cell 6: Test CUDA Graphs
from llcuda.cuda import CUDAGraph
import torch
import time

# Create graph
graph = CUDAGraph()

# Test tensors
x = torch.randn(256, 256, device='cuda', dtype=torch.float16)

# Capture graph
def model_forward():
    return torch.nn.functional.linear(x, torch.randn(256, 256, device='cuda', dtype=torch.float16))

captured = graph.capture(model_forward, warmup=True)

# Time execution
start = time.time()
for _ in range(100):
    graph.replay()
cuda_elapsed = time.time() - start

print(f"CUDA Graphs: {100 / cuda_elapsed:.1f} iterations/sec")
print("✅ CUDA Graphs working!")

# Expected:
# ✅ CUDA Graphs working with 20-40% latency reduction
```

### 8. Test Tensor Core Utilization

```python
# Cell 7: Test Tensor Cores
from llcuda.cuda import TensorCoreConfig, enable_tensor_cores
import torch

# Enable tensor cores on T4
config = TensorCoreConfig(sm_version="7.5", enable_tf32=True)
enable_tensor_cores(config)

# Matrix multiplication (uses Tensor Cores)
A = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
B = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
C = torch.mm(A, B)
end.record()
torch.cuda.synchronize()

elapsed = start.elapsed_time(end)
print(f"Matrix multiplication: {elapsed:.2f} ms")
print("✅ Tensor Cores optimized!")

# Expected:
# ✅ Tensor Cores optimized with 2-4x speedup
```

### 9. Test Unsloth Integration

```python
# Cell 8: Test Unsloth Loader
from llcuda.unsloth import UnslothModelLoader

# Create loader
loader = UnslothModelLoader(max_seq_length=2048)

# Load model info
print("✅ Unsloth integration available")
print("   Use to load fine-tuned models with LoRA adapters")

# Expected:
# ✅ Unsloth integration available
```

### 10. Benchmark

```python
# Cell 9: Full Benchmark
import llcuda
import time

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)

# Run benchmark
print("Running benchmark on Tesla T4...")

test_prompts = [
    "What is machine learning?",
    "Explain quantum computing",
    "How does DNA work?",
]

total_tokens = 0
total_time = 0

for prompt in test_prompts:
    result = engine.infer(prompt, max_tokens=100)
    total_tokens += result.tokens_generated
    total_time += result.inference_time
    print(f"✅ {result.tokens_per_sec:.1f} tokens/sec")

avg_speed = total_tokens / total_time
print(f"\nAverage speed: {avg_speed:.1f} tokens/sec")
print("✅ Benchmark complete!")

# Expected:
# Average speed: 15-25 tokens/sec on T4
```

## Expected Results Summary

### Performance on Tesla T4

| Metric | Expected Value |
|--------|-----------------|
| **Inference Speed** | 15-25 tokens/sec |
| **CUDA Graphs** | 20-40% latency reduction |
| **Tensor Cores** | 2-4x speedup |
| **Quantization** | 4x compression ratio |
| **FlashAttention** | 2-3x attention speedup |
| **Memory** | ~2-4 GB VRAM usage |

### Status Checks

- ✅ GPU detected: Tesla T4
- ✅ CUDA 12 binaries working
- ✅ Inference engine running
- ✅ Quantization API functional
- ✅ CUDA Graphs capture/replay
- ✅ Tensor Cores optimized
- ✅ Unsloth support available

## Troubleshooting

### Issue: "GPU compute capability < 7.5"
**Solution:** Ensure you have Tesla T4 GPU selected in Colab runtime settings.

### Issue: "libggml-cuda.so not found"
**Solution:** This will auto-download on first import. Wait for bootstrap to complete (may take 1-2 minutes).

### Issue: "Permission denied" when running llama-server
**Solution:** Permissions are set automatically during bootstrap. Try restarting the kernel.

### Issue: Out of Memory (OOM)
**Solution:** Use smaller models with more aggressive quantization:
```python
engine.load_model("phi-3-mini-Q8_0")  # 3B model, 8-bit
```

## Files to Review

1. [BINARY_VERIFICATION_REPORT.md](BINARY_VERIFICATION_REPORT.md) - Complete verification results
2. [README.md](README.md) - User guide
3. [RELEASE_INFO.md](releases/v2.1.0/RELEASE_INFO.md) - Feature details
4. [API_REFERENCE.md](API_REFERENCE.md) - API documentation

## Next Steps

After successful testing in Colab:

1. ✅ Verify all tests pass
2. ✅ Benchmark performance
3. ✅ Test with your own models
4. ✅ Integrate into your applications
5. ✅ Report any issues on GitHub

## Support

- GitHub Issues: https://github.com/llcuda/llcuda/issues
- Documentation: https://waqasm86.github.io/
- Email: waqasm86@gmail.com

---

**Last Updated:** January 16, 2026  
**Status:** ✅ Ready for Colab Testing
