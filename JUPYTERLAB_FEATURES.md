# llcuda v0.3.0 - JupyterLab Features

This document describes the new JupyterLab-optimized features added in llcuda v0.3.0.

## What's New in v0.3.0

llcuda v0.3.0 introduces comprehensive JupyterLab integration with four new modules designed specifically for interactive notebook environments:

### 1. `llcuda.jupyter` - JupyterLab Enhancements

Interactive features optimized for Jupyter notebooks:

**Real-time Streaming**
```python
from llcuda.jupyter import stream_generate

text = stream_generate(
    engine,
    "Explain quantum computing",
    max_tokens=256,
    show_timing=True,
    markdown=True  # Render as markdown
)
```

**Interactive Chat Widget**
```python
from llcuda.jupyter import ChatWidget

chat = ChatWidget(engine, system_prompt="You are a helpful assistant")
chat.display()  # Shows interactive chat interface with controls
```

**Performance Metrics Display**
```python
from llcuda.jupyter import display_metrics

display_metrics(engine)  # Shows formatted metrics table
```

**Temperature Comparison**
```python
from llcuda.jupyter import compare_temperatures

results = compare_temperatures(
    engine,
    prompt="Write a creative story opening",
    temperatures=[0.3, 0.7, 1.0, 1.5]
)
```

**Batch Processing with Progress**
```python
from llcuda.jupyter import progress_generate

results = progress_generate(engine, prompts_list)  # Shows progress bar
```

### 2. `llcuda.chat` - Conversation Management

OpenAI-compatible chat completion with history management:

**Basic Chat**
```python
from llcuda.chat import ChatEngine

chat = ChatEngine(engine, system_prompt="You are a coding assistant")
chat.add_user_message("How do I write a Python decorator?")
response = chat.complete()
```

**Streaming Chat**
```python
for chunk in chat.complete_stream():
    print(chunk, end='', flush=True)
```

**Conversation Persistence**
```python
# Save conversation
chat.save_history("conversation.json")

# Load conversation
chat2 = ChatEngine(engine)
chat2.load_history("conversation.json")
```

**Multi-Session Management**
```python
from llcuda.chat import ConversationManager

manager = ConversationManager(engine)
manager.create_conversation("coding", "You are a coding assistant")
manager.create_conversation("writing", "You are a writing coach")
manager.switch_to("coding")
response = manager.chat("Explain list comprehensions")
```

### 3. `llcuda.embeddings` - Text Embeddings

Generate and use text embeddings for semantic tasks:

**Basic Embedding**
```python
from llcuda.embeddings import EmbeddingEngine

embedder = EmbeddingEngine(engine, normalize=True)
vector = embedder.embed("Machine learning is amazing")
```

**Batch Embedding**
```python
texts = ["text1", "text2", "text3"]
vectors = embedder.embed_batch(texts, show_progress=True)
```

**Similarity Comparison**
```python
from llcuda.embeddings import cosine_similarity

sim = cosine_similarity(vector1, vector2)
print(f"Similarity: {sim:.3f}")
```

**Semantic Search**
```python
from llcuda.embeddings import SemanticSearch

search = SemanticSearch(embedder)
search.add_documents(document_list)
results = search.search("query text", top_k=5)

for doc, score, metadata in results:
    print(f"[{score:.3f}] {doc}")
```

**Text Clustering**
```python
from llcuda.embeddings import TextClustering

clustering = TextClustering(embedder, n_clusters=3)
labels = clustering.fit(texts)
clusters = clustering.get_clusters(texts, labels)
```

### 4. `llcuda.models` - Model Management

Discover, analyze, and manage GGUF models:

**List Local Models**
```python
from llcuda.models import list_models

models = list_models()
for model in models:
    print(f"{model['filename']}: {model['file_size_mb']:.1f} MB")
```

**Model Information**
```python
from llcuda.models import ModelInfo

info = ModelInfo.from_file("model.gguf")
print(f"Architecture: {info.architecture}")
print(f"Context length: {info.context_length}")

settings = info.get_recommended_settings(vram_gb=4.0)
print(f"Recommended GPU layers: {settings['gpu_layers']}")
```

**Download from HuggingFace**
```python
from llcuda.models import download_model

path = download_model(
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)
```

**Model Recommendations**
```python
from llcuda.models import print_model_catalog

print_model_catalog(vram_gb=8.0)  # Shows recommended models for your VRAM
```

**Model Manager**
```python
from llcuda.models import ModelManager

manager = ModelManager(["/path/to/models"])
small_models = manager.find_by_size(max_mb=2000)
best_model = manager.get_best_for_vram(vram_gb=4.0)
```

## Installation

### Basic Installation
```bash
pip install llcuda
```

### With JupyterLab Features
```bash
pip install llcuda ipywidgets tqdm
```

### With All Features
```bash
pip install llcuda ipywidgets tqdm matplotlib pandas scikit-learn huggingface_hub
```

## Dependencies

### Core Dependencies (Required)
- `numpy >= 1.20.0`
- `requests >= 2.20.0`

### JupyterLab Features (Recommended)
- `ipywidgets >= 7.6.0` - Interactive widgets
- `tqdm >= 4.60.0` - Progress bars
- `IPython >= 7.0.0` - Rich display

### Optional Features
- `matplotlib >= 3.5.0` - Visualization
- `pandas >= 1.3.0` - Metrics display
- `scikit-learn >= 1.0.0` - Text clustering
- `huggingface_hub >= 0.10.0` - Model downloads
- `gguf >= 0.1.0` - GGUF metadata parsing

## Quick Start

### 1. Setup Environment
```python
import os
os.environ['LLAMA_SERVER_PATH'] = '/path/to/llama-server'

import llcuda
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf", auto_start=True, gpu_layers=20)
```

### 2. Interactive Chat
```python
from llcuda.jupyter import ChatWidget

chat = ChatWidget(engine)
chat.display()
```

### 3. Streaming Generation
```python
from llcuda.jupyter import stream_generate

text = stream_generate(engine, "Explain AI", markdown=True)
```

### 4. Semantic Search
```python
from llcuda.embeddings import EmbeddingEngine, SemanticSearch

embedder = EmbeddingEngine(engine)
search = SemanticSearch(embedder)
search.add_documents(["doc1", "doc2", "doc3"])
results = search.search("query")
```

## Examples

See the complete tutorial notebook:
- `examples/complete-llcuda-tutorial.ipynb`

## API Changes from v0.2.1

### Backward Compatible
All v0.2.1 features remain unchanged and fully compatible.

### New Modules
- `llcuda.jupyter` - JupyterLab features
- `llcuda.chat` - Chat management
- `llcuda.embeddings` - Embedding generation
- `llcuda.models` - Model utilities

### Version Bump
- v0.2.1 → v0.3.0

## Feature Comparison

| Feature | v0.2.1 | v0.3.0 |
|---------|--------|--------|
| Basic Inference | ✅ | ✅ |
| Auto Server Start | ✅ | ✅ |
| Streaming Support | ❌ | ✅ |
| Chat Widget | ❌ | ✅ |
| Conversation History | ❌ | ✅ |
| Embeddings | ❌ | ✅ |
| Semantic Search | ❌ | ✅ |
| Model Discovery | Basic | Advanced |
| Progress Bars | ❌ | ✅ |
| Metrics Display | Basic | Rich |
| Model Download | ❌ | ✅ |

## Performance

All new features are designed for efficiency:

- **Streaming**: Real-time display with minimal overhead
- **Caching**: Embedding cache with configurable size
- **Lazy Loading**: Modules loaded only when imported
- **Progress Tracking**: Optional, zero overhead when disabled

## Platform Support

### Tested On
- Ubuntu 22.04 with CUDA 12.x
- GeForce 940M (1GB VRAM) to RTX 4090
- JupyterLab 3.x and 4.x
- Python 3.11+

### Should Work On
- Any Linux with CUDA support
- macOS (CPU or Metal)
- Windows with WSL2 + CUDA

## Troubleshooting

### Widgets Not Displaying
```bash
# Install and enable ipywidgets
pip install ipywidgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Embeddings Not Working
Start llama-server with `--embedding` flag:
```python
engine.load_model(
    "model.gguf",
    auto_start=True,
    embedding=True  # Add this parameter
)
```

### Import Errors
Install optional dependencies:
```bash
pip install llcuda[jupyter]  # JupyterLab features
pip install llcuda[all]      # All features
```

## Contributing

Found a bug or have a feature request? Open an issue at:
https://github.com/waqasm86/llcuda/issues

## License

MIT License - same as llcuda core

## Credits

Built on top of:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGML/GGUF inference engine
- [Ubuntu-Cuda-Llama.cpp-Executable](https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable) - Pre-built binaries

Developed by: waqasm86
