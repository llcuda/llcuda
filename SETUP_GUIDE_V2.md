# llcuda v0.2.0 - Complete Setup Guide for Ubuntu 22.04

This guide will help you set up llcuda with Python 3.11.11 and JupyterLab on your Ubuntu 22.04 system with NVIDIA CUDA support.

## ✅ What You Have

Your system is already well-configured with:

- ✅ Ubuntu 22.04 (Xubuntu)
- ✅ Python 3.11.11 at `/usr/local/bin/python3.11`
- ✅ NVIDIA GeForce 940M GPU (1GB VRAM)
- ✅ CUDA 12.8 with driver 570.195.03
- ✅ JupyterLab installed
- ✅ llama-cpp-cuda at `/media/waqasm86/External1/Project-Nvidia/llama-cpp-cuda/`
- ✅ Gemma 3 1B model (Q4_K_M) ready to use

## 🚀 Quick Setup (5 minutes)

### Step 1: Set Environment Variables

Add these to your `~/.bashrc`:

```bash
# llcuda environment
export LLAMA_CPP_DIR="/media/waqasm86/External1/Project-Nvidia/llama-cpp-cuda"
export LD_LIBRARY_PATH="$LLAMA_CPP_DIR/lib:${LD_LIBRARY_PATH}"
export PATH="/usr/local/bin:$PATH"  # Ensure Python 3.11 is first
```

Apply changes:

```bash
source ~/.bashrc
```

### Step 2: Install/Update llcuda

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda

# Install in development mode (recommended for local development)
/usr/local/bin/python3.11 -m pip install -e .

# Or install from PyPI (when published)
# pip install llcuda
```

### Step 3: Verify Installation

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda
/usr/local/bin/python3.11 test_setup.py
```

You should see all checks passing with ✓ symbols.

### Step 4: Launch JupyterLab

```bash
# Make sure JupyterLab uses Python 3.11
/usr/local/bin/python3.11 -m jupyter lab
```

Then open: `examples/quickstart_jupyterlab.ipynb`

## 📖 Detailed Usage

### Ultra-Simple Example

Create a new Python file or Jupyter cell:

```python
import llcuda

# Auto-start mode - simplest way to use llcuda
engine = llcuda.InferenceEngine()

# This automatically finds llama-server and starts it
engine.load_model(
    "/media/waqasm86/External1/Project-Nvidia/llama-cpp-cuda/bin/gemma-3-1b-it-Q4_K_M.gguf",
    auto_start=True,
    gpu_layers=20,  # Use 20 layers on GPU (safe for 1GB VRAM)
    ctx_size=2048
)

# Run inference
result = engine.infer("What is artificial intelligence?", max_tokens=100)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")

# Cleanup (optional - automatic on exit)
engine.unload_model()
```

### JupyterLab Usage

```python
import llcuda

# Check system
llcuda.print_system_info()

# Create engine with context manager (auto-cleanup)
with llcuda.InferenceEngine() as engine:
    # Find models automatically
    models = llcuda.find_gguf_models()
    print(f"Found {len(models)} models: {[m.name for m in models]}")

    # Load first model with auto-start
    engine.load_model(str(models[0]), auto_start=True, gpu_layers=20)

    # Run inference
    result = engine.infer("Explain quantum computing in simple terms", max_tokens=100)
    print(result.text)

# Server automatically stopped here
```

## 🎯 Recommended GPU Settings for GeForce 940M (1GB VRAM)

Your GPU has limited VRAM, so adjust these parameters:

```python
# Conservative settings (safest)
engine.load_model(
    model_path,
    auto_start=True,
    gpu_layers=10,   # Fewer layers
    ctx_size=1024,   # Smaller context
    n_parallel=1     # Single sequence
)

# Balanced settings (recommended)
engine.load_model(
    model_path,
    auto_start=True,
    gpu_layers=20,   # More layers
    ctx_size=2048,   # Standard context
    n_parallel=1
)

# Aggressive settings (may run out of memory)
engine.load_model(
    model_path,
    auto_start=True,
    gpu_layers=99,   # All layers
    ctx_size=4096,   # Large context
    n_parallel=1
)
```

**Tip**: Start with conservative settings and increase gradually.

## 📁 Project Structure

After setup, your llcuda directory looks like:

```
llcuda/
├── llcuda/
│   ├── __init__.py       # Main API with InferenceEngine
│   ├── server.py         # ServerManager for llama-server control
│   └── utils.py          # Utility functions (detect_cuda, find_models, etc.)
├── examples/
│   ├── quickstart_jupyterlab.ipynb  # Complete JupyterLab tutorial
│   └── kaggle_colab_example.ipynb   # Cloud platform example
├── tests/
│   └── test_llcuda.py    # Unit tests
├── test_setup.py         # Verify installation
├── setup.py              # Package setup
├── pyproject.toml        # Modern Python packaging
├── README_NEW.md         # Complete documentation
└── SETUP_GUIDE_V2.md     # This file
```

## 🛠️ Advanced Configuration

### Custom Server Location

If llama-server is in a different location:

```bash
export LLAMA_SERVER_PATH="/path/to/custom/llama-server"
```

Or in Python:

```python
import os
os.environ['LLAMA_SERVER_PATH'] = '/path/to/llama-server'
```

### Multiple GPU Support

If you have multiple GPUs:

```bash
# Use first GPU only
export CUDA_VISIBLE_DEVICES=0

# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1
```

### Manual Server Control

For maximum control:

```python
from llcuda import ServerManager

# Start server manually
manager = ServerManager()
manager.start_server(
    model_path="/path/to/model.gguf",
    port=8090,
    gpu_layers=20,
    ctx_size=2048,
    verbose=True
)

# Check status
info = manager.get_server_info()
print(f"Server running: {info['running']}")
print(f"PID: {info['process_id']}")

# Use the server
from llcuda import InferenceEngine
engine = InferenceEngine()
result = engine.infer("Test prompt")

# Stop when done
manager.stop_server()
```

## 🔍 Troubleshooting

### Issue: "llama-server not found"

**Solution 1**: Set environment variable
```bash
export LLAMA_CPP_DIR="/media/waqasm86/External1/Project-Nvidia/llama-cpp-cuda"
```

**Solution 2**: Check path
```python
import llcuda
print(llcuda.ServerManager().find_llama_server())
```

### Issue: "CUDA out of memory"

**Solution**: Reduce GPU layers
```python
engine.load_model(model_path, auto_start=True, gpu_layers=10)
```

Or reduce context size:
```python
engine.load_model(model_path, auto_start=True, ctx_size=1024)
```

### Issue: "Server failed to start"

**Solution**: Check server manually
```bash
/media/waqasm86/External1/Project-Nvidia/llama-cpp-cuda/bin/llama-server \
  -m /media/waqasm86/External1/Project-Nvidia/llama-cpp-cuda/bin/gemma-3-1b-it-Q4_K_M.gguf \
  --port 8090 \
  -ngl 20 \
  -c 2048
```

If this works, the issue is with auto-start. Use manual server management instead.

### Issue: Wrong Python version in JupyterLab

**Solution**: Install JupyterLab kernel for Python 3.11
```bash
/usr/local/bin/python3.11 -m pip install ipykernel
/usr/local/bin/python3.11 -m ipykernel install --user --name python311 --display-name "Python 3.11"
```

Then select "Python 3.11" kernel in JupyterLab.

## 📊 Performance Tips

### 1. Optimize GPU Layers

Run this to find optimal settings:

```python
import llcuda

engine = llcuda.InferenceEngine()
model_path = "/path/to/model.gguf"

# Test different GPU layer counts
for gpu_layers in [10, 20, 30, 40, 99]:
    try:
        engine.load_model(model_path, auto_start=True, gpu_layers=gpu_layers)

        # Run test inference
        result = engine.infer("Test prompt", max_tokens=50)

        print(f"GPU Layers: {gpu_layers}")
        print(f"  Speed: {result.tokens_per_sec:.1f} tok/s")
        print(f"  Latency: {result.latency_ms:.0f}ms")

        engine.unload_model()
    except Exception as e:
        print(f"GPU Layers: {gpu_layers} - Failed: {e}")
```

### 2. Batch Processing

Process multiple prompts efficiently:

```python
prompts = [f"Question {i}?" for i in range(10)]
results = engine.batch_infer(prompts, max_tokens=50)
```

### 3. Monitor Performance

```python
# Reset metrics
engine.reset_metrics()

# Run inferences
for _ in range(100):
    engine.infer("Test", max_tokens=50)

# Get statistics
metrics = engine.get_metrics()
print(f"Mean latency: {metrics['latency']['mean_ms']:.0f}ms")
print(f"p95 latency: {metrics['latency']['p95_ms']:.0f}ms")
print(f"Throughput: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")
```

## 🎓 Next Steps

1. **Explore Examples**
   - Open `examples/quickstart_jupyterlab.ipynb`
   - Run all cells to see llcuda in action

2. **Try Different Models**
   - Download more GGUF models from HuggingFace
   - Place them in `llama-cpp-cuda/bin/`
   - Use `llcuda.find_gguf_models()` to discover them

3. **Build Applications**
   - Create chatbots
   - Build Q&A systems
   - Develop text generation tools

4. **Optimize Performance**
   - Experiment with different quantizations (Q2, Q3, Q4, Q5, Q8)
   - Tune GPU layers and context size
   - Profile with different batch sizes

## 📚 Resources

- **llcuda GitHub**: https://github.com/waqasm86/llcuda
- **llcuda PyPI**: https://pypi.org/project/llcuda/
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **GGUF Models**: https://huggingface.co/models?search=gguf

## 💬 Support

- Issues: https://github.com/waqasm86/llcuda/issues
- Discussions: https://github.com/waqasm86/llcuda/discussions

---

**Happy Inferencing! 🚀**

Built with ❤️ for on-device AI
