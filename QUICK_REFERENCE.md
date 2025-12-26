# llcuda v0.2.0 - Quick Reference Card

## 🚀 Installation (One-Time Setup)

```bash
# 1. Set environment variables (add to ~/.bashrc)
export LLAMA_CPP_DIR="/media/waqasm86/External1/Project-Nvidia/llama-cpp-cuda"
export LD_LIBRARY_PATH="$LLAMA_CPP_DIR/lib:${LD_LIBRARY_PATH}"

# 2. Install llcuda
cd /media/waqasm86/External1/Project-Nvidia/llcuda
/usr/local/bin/python3.11 -m pip install -e .

# 3. Verify installation
/usr/local/bin/python3.11 test_setup.py
```

---

## 💻 Basic Usage

### Simplest Possible Code

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model(
    "/media/waqasm86/External1/Project-Nvidia/llama-cpp-cuda/bin/gemma-3-1b-it-Q4_K_M.gguf",
    auto_start=True,
    gpu_layers=20
)

result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
```

### With Context Manager (Recommended)

```python
import llcuda

with llcuda.InferenceEngine() as engine:
    engine.load_model("model.gguf", auto_start=True, gpu_layers=20)
    result = engine.infer("Hello!", max_tokens=100)
    print(result.text)
# Auto-cleanup happens here
```

---

## 🎯 Common Tasks

### Check System

```python
import llcuda
llcuda.print_system_info()
```

### Find Models

```python
import llcuda
models = llcuda.find_gguf_models()
print([m.name for m in models])
```

### Single Inference

```python
result = engine.infer(
    prompt="Your prompt here",
    max_tokens=100,
    temperature=0.7
)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

### Batch Inference

```python
prompts = ["Q1?", "Q2?", "Q3?"]
results = engine.batch_infer(prompts, max_tokens=50)
for r in results:
    print(r.text)
```

### Get Metrics

```python
metrics = engine.get_metrics()
print(f"Mean latency: {metrics['latency']['mean_ms']:.0f}ms")
print(f"Throughput: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")
```

---

## ⚙️ GPU Settings (GeForce 940M - 1GB VRAM)

```python
# Conservative (safest)
engine.load_model(path, auto_start=True, gpu_layers=10, ctx_size=1024)

# Balanced (recommended)
engine.load_model(path, auto_start=True, gpu_layers=20, ctx_size=2048)

# Aggressive (may fail)
engine.load_model(path, auto_start=True, gpu_layers=99, ctx_size=4096)
```

---

## 🛠️ Manual Server Control

```python
from llcuda import ServerManager

manager = ServerManager()

# Start
manager.start_server("model.gguf", gpu_layers=20, verbose=True)

# Check status
info = manager.get_server_info()
print(f"Running: {info['running']}, PID: {info['process_id']}")

# Stop
manager.stop_server()
```

---

## 📊 JupyterLab Workflow

```python
import llcuda

# 1. Check system
llcuda.print_system_info()

# 2. Find models
models = llcuda.find_gguf_models()

# 3. Create engine
with llcuda.InferenceEngine() as engine:
    # 4. Load model
    engine.load_model(str(models[0]), auto_start=True, gpu_layers=20)

    # 5. Run inference
    result = engine.infer("Your prompt", max_tokens=100)
    print(result.text)

    # 6. Get metrics
    metrics = engine.get_metrics()
    print(metrics)
```

---

## 🔍 Troubleshooting

### Check if CUDA is available
```python
import llcuda
if llcuda.check_cuda_available():
    print("✓ CUDA available")
    print(llcuda.get_cuda_device_info())
```

### Find llama-server
```python
from llcuda import ServerManager
path = ServerManager().find_llama_server()
print(f"llama-server: {path}")
```

### Check server health
```python
engine = llcuda.InferenceEngine()
if engine.check_server():
    print("✓ Server running")
else:
    print("✗ Server not running")
```

---

## 📝 Key Files

| File | Purpose |
|------|---------|
| `test_setup.py` | Verify installation |
| `examples/quickstart_jupyterlab.ipynb` | Complete tutorial |
| `README_NEW.md` | Full documentation |
| `SETUP_GUIDE_V2.md` | Setup instructions |
| `RESTRUCTURE_SUMMARY.md` | What changed in v0.2.0 |

---

## 🚀 Launch JupyterLab

```bash
/usr/local/bin/python3.11 -m jupyter lab
# Then open: examples/quickstart_jupyterlab.ipynb
```

---

## 🆘 Quick Help

```python
# Import
import llcuda

# Check what's available
dir(llcuda)

# Get help on a function
help(llcuda.InferenceEngine)
help(llcuda.ServerManager)
```

---

**Version**: 0.2.0
**Python**: 3.11.11
**CUDA**: 12.8
**GPU**: NVIDIA GeForce 940M (1GB)
