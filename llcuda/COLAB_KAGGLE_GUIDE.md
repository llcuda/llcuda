# llcuda on Google Colab and Kaggle

Complete guide for running llcuda on cloud GPU platforms.

## Quick Start

### Google Colab

```python
# Install llcuda
!pip install llcuda

# Import and run
import llcuda

# Check GPU compatibility
compat = llcuda.check_gpu_compatibility()
print(f"Platform: {compat['platform']}")
print(f"GPU: {compat['gpu_name']}")
print(f"Compute Capability: {compat['compute_capability']}")
print(f"Compatible: {compat['compatible']}")

# Create engine and load model
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", gpu_layers=20)

# Run inference
result = engine.infer("What is artificial intelligence?", max_tokens=100)
print(result.text)
```

### Kaggle

```python
# Install llcuda
!pip install llcuda

# Import
import llcuda

# Check GPU (Kaggle typically has 2x Tesla T4)
compat = llcuda.check_gpu_compatibility()
if compat['compatible']:
    print(f"✓ {compat['gpu_name']} is compatible!")
else:
    print(f"✗ {compat['reason']}")

# For T4 GPUs, use conservative settings
engine = llcuda.InferenceEngine()
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    gpu_layers=20,        # Conservative for 15GB VRAM
    ctx_size=2048,        # Reasonable context window
    auto_configure=True   # Let llcuda optimize settings
)

# Run inference
result = engine.infer("Explain machine learning", max_tokens=100)
print(f"Generated: {result.text}")
print(f"Speed: {result.tokens_per_sec:.2f} tokens/sec")
```

---

## Supported GPUs

llcuda v1.1.0+ supports **NVIDIA compute capability 5.0+**:

| Architecture | Compute Cap | Examples | Colab | Kaggle |
|--------------|-------------|----------|-------|--------|
| Maxwell      | 5.0 - 5.3   | GTX 900 series, Tesla M40 | ❌ | ❌ |
| Pascal       | 6.0 - 6.2   | GTX 10xx, Tesla P100 | ✅ P100 | ✅ P100 |
| Volta        | 7.0         | Tesla V100 | ✅ V100 | ❌ |
| Turing       | 7.5         | **Tesla T4**, RTX 20xx | ✅ T4 | ✅ T4 |
| Ampere       | 8.0 - 8.6   | A100, RTX 30xx | ✅ A100 | ❌ |
| Ada Lovelace | 8.9         | RTX 40xx, L40S | ❌ | ❌ |

**Most Common:**
- **Google Colab (Free)**: Tesla T4 (15GB VRAM, compute 7.5)
- **Google Colab (Pro)**: T4, P100, V100, or A100
- **Kaggle**: 2x Tesla T4 (30GB total VRAM, compute 7.5)

---

## Platform-Specific Configuration

### Google Colab

```python
import llcuda

# Detect platform automatically
compat = llcuda.check_gpu_compatibility()
print(f"Platform: {compat['platform']}")  # Outputs: 'colab'

# Auto-configure for Colab GPU
from llcuda.utils import auto_configure_for_model
from pathlib import Path

settings = auto_configure_for_model(
    Path("/path/to/model.gguf"),
    vram_gb=15.0  # T4 has 15GB
)
print(f"Recommended GPU layers: {settings['gpu_layers']}")
print(f"Recommended context: {settings['ctx_size']}")
```

**Recommended Settings for Colab T4:**

| Model Size | GPU Layers | Context | Batch | Performance |
|------------|-----------|---------|-------|-------------|
| 1B (Q4)    | 26 (all)  | 2048    | 512   | ~15 tok/s   |
| 3B (Q4)    | 20-25     | 2048    | 512   | ~10 tok/s   |
| 7B (Q4)    | 10-15     | 1024    | 256   | ~5 tok/s    |
| 13B (Q4)   | 5-10      | 512     | 256   | ~2 tok/s    |

### Kaggle (2x Tesla T4)

```python
import llcuda

# Kaggle detection
compat = llcuda.check_gpu_compatibility()
print(f"Platform: {compat['platform']}")  # Outputs: 'kaggle'

# Kaggle gives you 2 GPUs but llama.cpp uses only GPU 0
# Still, you have 15GB VRAM available

engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    gpu_layers=26,       # All layers for 1B model
    ctx_size=2048,
    batch_size=512,
    ubatch_size=128,
    verbose=True
)
```

**Recommended Settings for Kaggle T4:**

| Model Size | GPU Layers | Context | Batch | Performance |
|------------|-----------|---------|-------|-------------|
| 1B (Q4)    | 26 (all)  | 2048    | 512   | ~15 tok/s   |
| 3B (Q4)    | 25-28     | 2048    | 512   | ~10 tok/s   |
| 7B (Q4)    | 15-20     | 1024    | 512   | ~5 tok/s    |
| 13B (Q4)   | 8-12      | 512     | 256   | ~2 tok/s    |

---

## Complete Examples

### Example 1: Gemma 3 1B on Colab

```python
!pip install llcuda

import llcuda

# Check compatibility
compat = llcuda.check_gpu_compatibility()
if not compat['compatible']:
    raise RuntimeError(f"GPU not compatible: {compat['reason']}")

print(f"✓ Running on {compat['platform']} with {compat['gpu_name']}")

# Load model from registry (auto-downloads)
engine = llcuda.InferenceEngine()
engine.load_model(
    "gemma-3-1b-Q4_K_M",  # Registry name
    gpu_layers=26,        # All 26 layers
    ctx_size=2048,
    auto_start=True,
    verbose=True
)

# Run inference
prompts = [
    "What is machine learning?",
    "Explain neural networks in simple terms",
    "What are transformers in AI?"
]

for prompt in prompts:
    result = engine.infer(prompt, max_tokens=100)
    print(f"\nQ: {prompt}")
    print(f"A: {result.text}")
    print(f"Speed: {result.tokens_per_sec:.2f} tok/s")

# Get metrics
metrics = engine.get_metrics()
print(f"\nPerformance Summary:")
print(f"  Requests: {metrics['throughput']['total_requests']}")
print(f"  Avg Speed: {metrics['throughput']['tokens_per_sec']:.2f} tok/s")
print(f"  Avg Latency: {metrics['latency']['mean_ms']:.0f}ms")
```

### Example 2: Custom Model on Kaggle with HuggingFace

```python
!pip install llcuda huggingface_hub

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
import llcuda

# Authenticate with HuggingFace (if using gated models)
try:
    secret_value = UserSecretsClient().get_secret("HF_TOKEN")
    login(token=secret_value)
    print("✓ Authenticated with HuggingFace")
except Exception as e:
    print(f"Warning: HF authentication failed: {e}")
    print("Continuing without authentication (public models only)")

# Check GPU
compat = llcuda.check_gpu_compatibility()
print(f"\nGPU Check:")
print(f"  Platform: {compat['platform']}")
print(f"  GPU: {compat['gpu_name']}")
print(f"  Compute: {compat['compute_capability']}")
print(f"  Compatible: ✓" if compat['compatible'] else f"  Error: {compat['reason']}")

# Load model (automatically downloads from HuggingFace)
engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    gpu_layers=26,
    ctx_size=2048,
    auto_start=True,
    interactive_download=False  # No confirmation needed on Kaggle
)

# Chat-style inference
conversation = [
    "Hello! Can you help me understand Python?",
    "What's the difference between a list and a tuple?",
    "Can you show me an example of list comprehension?"
]

for user_msg in conversation:
    result = engine.infer(user_msg, max_tokens=150, temperature=0.7)
    print(f"\nUser: {user_msg}")
    print(f"Assistant: {result.text}")
```

### Example 3: Batch Processing on Colab

```python
!pip install llcuda pandas

import llcuda
import pandas as pd

# Load model
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", gpu_layers=26, ctx_size=1024)

# Prepare batch data
prompts = [
    "Summarize: Machine learning is a subset of AI...",
    "Classify sentiment: This movie was terrible!",
    "Extract keywords: Python is a programming language...",
    "Translate to French: Hello, how are you?",
    "Answer: What is 2+2?"
]

# Batch inference
results = engine.batch_infer(prompts, max_tokens=50)

# Create DataFrame
data = {
    'Prompt': prompts,
    'Response': [r.text for r in results],
    'Tokens': [r.tokens_generated for r in results],
    'Speed (tok/s)': [r.tokens_per_sec for r in results]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))

# Save results
df.to_csv('inference_results.csv', index=False)
print("\n✓ Results saved to inference_results.csv")
```

---

##Troubleshooting

### Issue 1: "No kernel image available for execution"

**Cause**: Binary compiled for wrong GPU architecture

**Solution**: Upgrade to llcuda 1.1.0+
```python
!pip install --upgrade llcuda
```

llcuda 1.1.0+ includes binaries for compute capability 5.0-8.9.

### Issue 2: Out of Memory (OOM)

**Symptoms**:
```
CUDA error: out of memory
```

**Solutions**:

1. **Reduce GPU layers**:
```python
engine.load_model("model.gguf", gpu_layers=10)  # Instead of 99
```

2. **Reduce context size**:
```python
engine.load_model("model.gguf", ctx_size=1024)  # Instead of 2048
```

3. **Use smaller model**:
```python
# Use 1B instead of 3B/7B
engine.load_model("gemma-3-1b-Q4_K_M")
```

4. **Check VRAM usage**:
```python
!nvidia-smi
```

### Issue 3: Model Download Fails

**Error**:
```
Repository not found
```

**Solution**: Use correct HuggingFace repo format:
```python
# ✗ Wrong
engine.load_model("google/gemma-3-1b-it-GGUF")

# ✓ Correct
engine.load_model("unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf")
```

### Issue 4: Slow Inference

**Symptoms**: < 5 tokens/second

**Solutions**:

1. **Check GPU is actually being used**:
```python
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")  # Should show GPU name, not "None"
```

2. **Verify GPU layers**:
```python
# Make sure gpu_layers > 0
engine.load_model("model.gguf", gpu_layers=20, verbose=True)
# Should print: "GPU Layers: 20"
```

3. **Use auto-configuration**:
```python
engine.load_model("model.gguf", auto_configure=True)
```

### Issue 5: Server Won't Start

**Error**:
```
RuntimeError: Failed to start llama-server
```

**Solutions**:

1. **Check if port is already in use**:
```python
# Use different port
engine = llcuda.InferenceEngine(server_url="http://127.0.0.1:8091")
```

2. **Skip GPU check if needed**:
```python
# Only use if you know your GPU is compatible
engine.load_model("model.gguf", skip_gpu_check=True)
```

3. **Check server logs**:
```python
import subprocess
result = subprocess.run(['which', 'llama-server'], capture_output=True)
print(result.stdout.decode())
```

---

## Performance Benchmarks

### Tesla T4 (15GB VRAM) - Colab/Kaggle

| Model | Quantization | GPU Layers | tok/s | VRAM Usage |
|-------|--------------|-----------|-------|------------|
| Gemma 3 1B | Q4_K_M | 26 (all) | ~15 | ~1.2 GB |
| Gemma 3 3B | Q4_K_M | 28 (all) | ~10 | ~3.5 GB |
| Llama 3.2 3B | Q4_K_M | 28 (all) | ~10 | ~3.5 GB |
| Llama 3.1 7B | Q4_K_M | 20 | ~5 | ~8 GB |
| Llama 3.1 7B | Q4_K_M | 32 (all) | ~8 | ~12 GB |

### Tesla V100 (16GB VRAM) - Colab Pro

| Model | Quantization | GPU Layers | tok/s | VRAM Usage |
|-------|--------------|-----------|-------|------------|
| Gemma 3 1B | Q4_K_M | 26 (all) | ~20 | ~1.2 GB |
| Llama 3.1 7B | Q4_K_M | 32 (all) | ~12 | ~12 GB |
| Llama 3.1 13B | Q4_K_M | 20 | ~5 | ~14 GB |

---

## Best Practices

### 1. Always Check GPU Compatibility First

```python
import llcuda

compat = llcuda.check_gpu_compatibility()
if not compat['compatible']:
    print(f"Error: {compat['reason']}")
    # Fall back to CPU or different model
else:
    print(f"✓ Compatible: {compat['gpu_name']}")
```

### 2. Use Auto-Configuration

```python
# Let llcuda optimize settings for your hardware
engine.load_model("model.gguf", auto_configure=True)
```

### 3. Monitor VRAM Usage

```python
# Check VRAM before loading large models
!nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
```

### 4. Use Context Managers for Cleanup

```python
with llcuda.InferenceEngine() as engine:
    engine.load_model("model.gguf", auto_start=True)
    result = engine.infer("Hello!")
    print(result.text)
# Server automatically stopped
```

### 5. Start with Small Models

```python
# Test with 1B model first
engine.load_model("gemma-3-1b-Q4_K_M")

# Then try larger models
# engine.load_model("llama-3.1-8b-Q4_K_M")
```

---

## Additional Resources

- **llcuda GitHub**: https://github.com/waqasm86/llcuda
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **GGUF Models**: https://huggingface.co/models?library=gguf
- **Documentation**: https://waqasm86.github.io/

---

## Version Information

- **llcuda**: 1.1.0+
- **Required**: Python 3.11+
- **Supported Compute Capability**: 5.0 - 8.9
- **Platforms**: JupyterLab, Google Colab, Kaggle, Local

## License

MIT License - Free for commercial and personal use.
