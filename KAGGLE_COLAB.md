# llcuda for Kaggle & Google Colab

Complete guide for using llcuda on NVIDIA T4 GPUs in Kaggle and Google Colab.

## Quick Start (Copy-Paste Ready)

### Kaggle Notebook Setup

```python
# ============================================
# KAGGLE SETUP - Copy this entire cell
# ============================================

# 1. Check GPU
!nvidia-smi

# 2. Install system dependencies
!apt-get update -qq
!apt-get install -y cmake build-essential

# 3. Install llcuda from PyPI
!pip install -q pybind11 numpy

import os
os.environ['CUDA_ARCHITECTURES'] = '75'  # T4 GPU

print("🔨 Installing llcuda from PyPI... (takes 2-3 minutes)")
!pip install llcuda --no-binary llcuda

print("✅ llcuda installed!")

# 3. Install llama.cpp
!git clone https://github.com/ggerganov/llama.cpp.git /kaggle/working/llama_cpp
%cd /kaggle/working/llama_cpp
!mkdir build && cd build && cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build . --config Release -j4

# 4. Download model (Gemma 2B - fast and small)
!pip install -q huggingface_hub
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="google/gemma-2b-it-GGUF",
    filename="gemma-2b-it-q4_k_m.gguf",
    local_dir="/kaggle/working/models"
)
print(f"✓ Model: {model_path}")

# 5. Start llama-server (background)
import subprocess
import time
import requests

server = subprocess.Popen([
    '/kaggle/working/llama_cpp/build/bin/llama-server',
    '-m', model_path,
    '--port', '8090',
    '-ngl', '99',
    '-c', '4096'
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for server
print("Starting server...")
for i in range(30):
    try:
        r = requests.get('http://127.0.0.1:8090/health', timeout=1)
        if r.status_code == 200:
            print("✓ Server ready!")
            break
    except:
        time.sleep(1)

# 6. Test llcuda
import llcuda

engine = llcuda.InferenceEngine()
os.makedirs('/tmp', exist_ok=True)
open('/tmp/model.gguf', 'a').close()
engine.load_model('/tmp/model.gguf')

result = engine.infer("What is AI?", max_tokens=50)
print(f"\n✓ llcuda working! Throughput: {result.tokens_per_sec:.1f} tok/s")

print("\n" + "="*60)
print("✓ SETUP COMPLETE! Ready to use llcuda.")
print("="*60)
```

### Google Colab Setup

```python
# ============================================
# COLAB SETUP - Copy this entire cell
# ============================================

# 1. Check GPU
!nvidia-smi

# 2. Install llcuda
!pip install -q pybind11 numpy cmake
!git clone https://github.com/waqasm86/local-llama-cuda.git
%cd local-llama-cuda

import os
os.environ['CUDA_ARCHITECTURES'] = '75'
!pip install -q -e .

# 3. Install llama.cpp
!git clone https://github.com/ggerganov/llama.cpp.git /content/llama_cpp
%cd /content/llama_cpp
!mkdir build && cd build && cmake .. -DGGML_CUDA=ON && cmake --build . --config Release -j$(nproc)

# 4. Download model
!pip install -q huggingface_hub
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="google/gemma-2b-it-GGUF",
    filename="gemma-2b-it-q4_k_m.gguf",
    local_dir="/content/models"
)

# 5. Start server
import subprocess, time, requests

server = subprocess.Popen([
    '/content/llama_cpp/build/bin/llama-server',
    '-m', model_path,
    '--port', '8090',
    '-ngl', '99'
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

for i in range(30):
    try:
        if requests.get('http://127.0.0.1:8090/health', timeout=1).status_code == 200:
            print("✓ Ready!")
            break
    except:
        time.sleep(1)

# 6. Test
import llcuda
engine = llcuda.InferenceEngine()
open('/tmp/model.gguf', 'a').close()
engine.load_model('/tmp/model.gguf')

print(engine.infer("Hello!", max_tokens=20).text)
print("✓ Setup complete!")
```

## Usage Examples

### Example 1: Basic Inference

```python
import llcuda

# Create engine
engine = llcuda.InferenceEngine()

# Load model (dummy file - using llama-server backend)
open('/tmp/model.gguf', 'a').close()
engine.load_model('/tmp/model.gguf', gpu_layers=99)

# Run inference
result = engine.infer(
    prompt="Explain machine learning in one sentence.",
    max_tokens=100,
    temperature=0.7
)

print(result.text)
print(f"Latency: {result.latency_ms:.0f}ms")
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

**Expected Output:**
```
Machine learning is a subset of artificial intelligence that enables...
Latency: 1234ms
Speed: 45.2 tok/s
```

### Example 2: Streaming

```python
print("Streaming: ", end='')

def callback(chunk):
    print(chunk, end='', flush=True)

result = engine.infer_stream(
    prompt="Write a haiku about coding.",
    callback=callback,
    max_tokens=50
)

print(f"\n\nDone! ({result.tokens_per_sec:.1f} tok/s)")
```

### Example 3: Batch Processing

```python
prompts = [
    "What is Python?",
    "What is CUDA?",
    "What is deep learning?"
]

results = engine.batch_infer(prompts, max_tokens=30)

for i, result in enumerate(results):
    print(f"\nQ{i+1}: {prompts[i]}")
    print(f"A{i+1}: {result.text}")
```

### Example 4: Benchmarking

```python
import numpy as np

# Benchmark
latencies = []
for i in range(50):
    result = engine.infer("Test", max_tokens=64)
    latencies.append(result.latency_ms)
    print(f"{i+1}/50", end='\r')

print("\n" + "="*40)
print(f"Mean: {np.mean(latencies):.0f}ms")
print(f"p50:  {np.percentile(latencies, 50):.0f}ms")
print(f"p95:  {np.percentile(latencies, 95):.0f}ms")
print(f"p99:  {np.percentile(latencies, 99):.0f}ms")
print("="*40)
```

## Recommended Models for T4 GPU

### Small Models (< 3B parameters)

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| Gemma 2B Q4_K_M | 1.6GB | ~45 tok/s | General, fast |
| Phi-2 2.7B Q4_K_M | 1.8GB | ~35 tok/s | Reasoning |
| TinyLlama 1.1B Q4_K_M | 0.6GB | ~80 tok/s | Ultra-fast |

### Medium Models (3-7B parameters)

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| Mistral 7B Q4_K_M | 4.4GB | ~28 tok/s | Best quality |
| Llama 2 7B Q4_K_M | 4.1GB | ~26 tok/s | General |
| CodeLlama 7B Q4_K_M | 4.2GB | ~25 tok/s | Coding |

### Download Examples

```python
from huggingface_hub import hf_hub_download

# Gemma 2B (recommended for T4)
model = hf_hub_download(
    repo_id="google/gemma-2b-it-GGUF",
    filename="gemma-2b-it-q4_k_m.gguf",
    local_dir="/kaggle/working/models"
)

# Mistral 7B (best quality)
model = hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    local_dir="/kaggle/working/models"
)
```

## Performance Tuning

### GPU Layer Offloading

```python
# All layers on GPU (fastest, most VRAM)
engine.load_model(model_path, gpu_layers=99)

# Hybrid CPU/GPU (less VRAM)
engine.load_model(model_path, gpu_layers=20)

# CPU only (slowest, no VRAM)
engine.load_model(model_path, gpu_layers=0)
```

### Batch Size Tuning

```python
# Large batch (faster, more VRAM)
engine.load_model(model_path, gpu_layers=99, batch_size=512)

# Small batch (slower, less VRAM)
engine.load_model(model_path, gpu_layers=99, batch_size=128)
```

### Context Size

```python
# Large context (more VRAM)
engine.load_model(model_path, ctx_size=4096)

# Small context (less VRAM)
engine.load_model(model_path, ctx_size=2048)
```

## Resource Limits

### Kaggle

- **GPU Time**: 30 hours/week (T4 x2)
- **VRAM**: 16GB per GPU
- **RAM**: 30GB
- **Storage**: 20GB persistent
- **Session**: 9 hours max

### Google Colab Free

- **GPU Time**: ~12 hours/day (T4)
- **VRAM**: 15GB
- **RAM**: 12GB
- **Storage**: 108GB (temporary)
- **Session**: 12 hours max

### Google Colab Pro

- **GPU**: T4, P100, V100, or A100
- **VRAM**: Up to 40GB (A100)
- **Session**: 24 hours
- **Priority**: GPU access

## Troubleshooting

### Server won't start

```python
# Check if port is already in use
!lsof -i :8090

# Kill existing server
!pkill -f llama-server

# Restart
# ... (run server startup code again)
```

### Out of memory

```python
# Reduce GPU layers
engine.load_model(model_path, gpu_layers=10)

# Use smaller model
# Download TinyLlama instead of Mistral 7B

# Reduce context size
engine.load_model(model_path, ctx_size=2048)
```

### Slow inference

```python
# Check GPU is being used
!nvidia-smi

# Verify GPU layers
# Make sure gpu_layers > 0

# Check server logs
!tail -f /tmp/llama-server.log
```

## Complete Working Example

See [kaggle_colab_example.ipynb](examples/kaggle_colab_example.ipynb) for a complete working notebook.

## Tips & Tricks

### 1. Persistent Storage (Kaggle)

```python
# Save to Kaggle dataset for reuse
!mkdir -p /kaggle/working/llcuda_cache
# ... download models to this directory
# Then: Create dataset from output
```

### 2. Google Drive (Colab)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Use Drive for models
model_path = '/content/drive/MyDrive/models/model.gguf'
```

### 3. Multiple Models

```python
# Switch models
engine.unload_model()
engine.load_model('model2.gguf')
```

### 4. Error Handling

```python
result = engine.infer("Test", max_tokens=100)
if result.success:
    print(result.text)
else:
    print(f"Error: {result.error_message}")
```

## Getting Help

- **Notebook Example**: [kaggle_colab_example.ipynb](examples/kaggle_colab_example.ipynb)
- **Documentation**: [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/waqasm86/local-llama-cuda/issues)

---

**Happy inferencing on T4 GPUs!** 🚀
