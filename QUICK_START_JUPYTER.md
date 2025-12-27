# llcuda Quick Start for JupyterLab

5-minute guide to get started with llcuda v0.3.0 in JupyterLab.

## 1. Installation

```bash
# Install llcuda with JupyterLab features
pip install llcuda ipywidgets tqdm matplotlib

# Enable ipywidgets in JupyterLab (if needed)
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## 2. Setup (Run in first cell)

```python
import os

# Point to your llama-server binary
os.environ['LLAMA_SERVER_PATH'] = '/path/to/llama-server'

import llcuda
```

## 3. Load Model (Run in second cell)

```python
engine = llcuda.InferenceEngine()

engine.load_model(
    "/path/to/model.gguf",
    auto_start=True,
    gpu_layers=20,     # Adjust for your GPU
    ctx_size=2048,
    verbose=True
)
```

## 4. Choose Your Style

### Option A: Interactive Chat Widget (Easiest)

```python
from llcuda.jupyter import ChatWidget

chat = ChatWidget(engine)
chat.display()
```

Now just type and click Send!

### Option B: Streaming Generation

```python
from llcuda.jupyter import stream_generate

text = stream_generate(
    engine,
    "Explain machine learning",
    max_tokens=256,
    markdown=True
)
```

### Option C: Programmatic Chat

```python
from llcuda.chat import ChatEngine

chat = ChatEngine(engine, system_prompt="You are a helpful assistant")
chat.add_user_message("What is Python?")
response = chat.complete()
print(response)
```

### Option D: Simple Inference

```python
result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
```

## 5. Advanced Features

### Semantic Search

```python
from llcuda.embeddings import EmbeddingEngine, SemanticSearch

embedder = EmbeddingEngine(engine)
search = SemanticSearch(embedder)

search.add_documents([
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Neural networks mimic the brain"
])

results = search.search("Tell me about AI", top_k=2)
for doc, score, _ in results:
    print(f"[{score:.3f}] {doc}")
```

### Model Discovery

```python
from llcuda.models import list_models, print_model_catalog

# List local models
models = list_models()
for m in models:
    print(f"{m['filename']}: {m['file_size_mb']:.1f} MB")

# Get recommendations for your GPU
print_model_catalog(vram_gb=4.0)
```

### Performance Metrics

```python
from llcuda.jupyter import display_metrics

# Run some inferences first
for prompt in ["Q1", "Q2", "Q3"]:
    engine.infer(prompt)

# Display metrics table
display_metrics(engine)
```

### Temperature Comparison

```python
from llcuda.jupyter import compare_temperatures

compare_temperatures(
    engine,
    "Write a creative story opening",
    temperatures=[0.3, 0.7, 1.0, 1.5]
)
```

## 6. Cleanup (Run in last cell)

```python
engine.unload_model()
print("✓ Server stopped")
```

## Common Issues

### Issue: ipywidgets not showing
**Fix**: Run in terminal:
```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Issue: Model not found
**Fix**: Check the path:
```python
from llcuda.models import list_models
models = list_models()
for m in models:
    print(m['filepath'])
```

### Issue: Server won't start
**Fix**: Check environment variable:
```python
import os
print(os.environ.get('LLAMA_SERVER_PATH'))

# Verify the file exists
from pathlib import Path
print(Path(os.environ['LLAMA_SERVER_PATH']).exists())
```

### Issue: Out of VRAM
**Fix**: Reduce GPU layers:
```python
# For 1GB VRAM
engine.load_model("model.gguf", auto_start=True,
                 gpu_layers=8, ctx_size=512, ubatch_size=128)

# For 4GB VRAM
engine.load_model("model.gguf", auto_start=True,
                 gpu_layers=20, ctx_size=2048)

# For 8GB+ VRAM
engine.load_model("model.gguf", auto_start=True,
                 gpu_layers=99, ctx_size=4096)
```

## System Info

Check your setup:
```python
llcuda.print_system_info()
```

This shows:
- Python version
- CUDA availability
- GPU information
- Available models
- llama-server location

## Complete Example

Copy-paste this into a notebook:

```python
# Cell 1: Setup
import os
os.environ['LLAMA_SERVER_PATH'] = '/home/user/llama-cpp-cuda/bin/llama-server'
import llcuda

# Cell 2: Load model
engine = llcuda.InferenceEngine()
engine.load_model(
    "/home/user/models/gemma-3-1b-it-Q4_K_M.gguf",
    auto_start=True,
    gpu_layers=20,
    verbose=True
)

# Cell 3: Interactive chat
from llcuda.jupyter import ChatWidget
chat = ChatWidget(engine)
chat.display()

# Cell 4: Cleanup (when done)
engine.unload_model()
```

## Next Steps

- See `complete-llcuda-tutorial.ipynb` for full examples
- Read `JUPYTERLAB_FEATURES.md` for detailed API docs
- Check `IMPLEMENTATION_SUMMARY.md` for architecture details

## Links

- GitHub: https://github.com/waqasm86/llcuda
- Binary: https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable
- Docs: https://waqasm86.github.io/

---

**Happy coding!** 🚀
