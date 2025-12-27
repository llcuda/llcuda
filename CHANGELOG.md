# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-28

### 🎉 Major Release - JupyterLab Integration

This release adds comprehensive JupyterLab support with four new modules for interactive notebook workflows, chat management, embeddings, and model discovery.

### Added

#### New Modules (4 modules, ~1,780 lines of code)

**llcuda.jupyter** - JupyterLab-Specific Features (497 lines)
- `stream_generate()` - Real-time streaming with IPython display and markdown rendering
- `ChatWidget` - Interactive chat interface with ipywidgets (text input, controls, history display)
- `display_metrics()` - Rich metrics visualization with pandas DataFrames and HTML tables
- `compare_temperatures()` - Side-by-side comparison of different temperature settings
- `progress_generate()` - Batch processing with tqdm progress bars
- `visualize_tokens()` - Token boundary visualization for debugging

**llcuda.chat** - Chat Completion and Conversation Management (419 lines)
- `Message` class - Individual message representation with role, content, timestamp
- `ChatEngine` - OpenAI-compatible chat completion API with history management
- `ConversationManager` - Multi-session conversation handler for managing multiple topics
- Streaming chat completion support via `/v1/chat/completions`
- Conversation persistence to/from JSON files
- Automatic token counting via tokenize endpoint
- Context window management with intelligent message trimming

**llcuda.embeddings** - Text Embeddings and Semantic Search (405 lines)
- `EmbeddingEngine` - Generate and cache text embeddings with configurable pooling
- `SemanticSearch` - Vector similarity search with document indexing and metadata
- `TextClustering` - K-means clustering for text grouping
- Similarity functions: `cosine_similarity()`, `dot_product_similarity()`, `euclidean_distance()`
- LRU embedding cache with configurable size (default 1000 entries)
- Batch embedding generation with optional progress tracking
- Index and cache persistence to disk (JSON format)

**llcuda.models** - Model Discovery and Management (459 lines)
- `ModelInfo` - Extract metadata from GGUF files (architecture, size, context length)
- `ModelManager` - Manage collections of models with filtering and search
- `list_models()` - Discover all local GGUF models with metadata
- `download_model()` - Download models from HuggingFace Hub with progress tracking
- `get_model_recommendations()` - VRAM-based model suggestions (1GB to 24GB+)
- `print_model_catalog()` - Display formatted model recommendations
- Automatic GGUF metadata parsing using gguf library
- Hardware-specific configuration recommendations (GPU layers, context size, batch sizes)

#### Documentation (5 new files)
- `JUPYTERLAB_FEATURES.md` - Complete API documentation for all new modules and functions
- `IMPLEMENTATION_SUMMARY.md` - Architecture, design decisions, and technical details
- `QUICK_START_JUPYTER.md` - 5-minute quick start guide for JupyterLab users
- `README_V030.md` - Updated comprehensive README for v0.3.0
- `requirements-jupyter.txt` - JupyterLab feature dependencies specification
- `complete-llcuda-tutorial.ipynb` - Comprehensive 12-section tutorial notebook covering all features

#### Dependencies
New optional dependency groups for modular installation:
- `jupyter` - JupyterLab integration (ipywidgets, tqdm, IPython, matplotlib, pandas)
- `embeddings` - Text clustering support (scikit-learn)
- `models` - Model management (huggingface_hub, gguf)
- `all` - All optional features combined

### Changed
- **Version**: 0.2.1 → 0.3.0
- **Description**: Updated to "CUDA-accelerated LLM inference for Python with JupyterLab integration and automatic server management"
- **Keywords**: Added jupyter, jupyterlab, chat, embeddings, semantic-search, gguf
- **Classifiers**: Added Framework :: Jupyter and Framework :: Jupyter :: JupyterLab
- **Documentation URLs**: Updated to point to waqasm86.github.io
- **`__init__.py`**: Added new modules to `__all__` for proper exports

### Improved
- **User Experience**: Rich interactive features for notebook-based development
- **Visualization**: Beautiful formatted outputs with markdown, HTML, and matplotlib
- **Productivity**: Progress tracking, metrics display, and interactive widgets
- **Model Discovery**: Automatic scanning and metadata extraction from GGUF files
- **Search Capabilities**: Full semantic search with embeddings and vector similarity

### Fixed
- None (this is a feature-only release with no bug fixes)

### Deprecated
- None (all v0.2.1 APIs remain fully supported)

### Security
- None

### Breaking Changes
**None** - v0.3.0 is 100% backward compatible with v0.2.1

All existing code continues to work without modification:
```python
# This v0.2.1 code works identically in v0.3.0
import llcuda
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf", auto_start=True, gpu_layers=20)
result = engine.infer("What is AI?")
print(result.text)
```

### Migration Guide

#### No Changes Required
Existing v0.2.1 code works without modification. Simply upgrade:
```bash
pip install --upgrade llcuda
```

#### Using New Features

**Interactive Chat Widget:**
```python
from llcuda.jupyter import ChatWidget
chat = ChatWidget(engine, system_prompt="You are a helpful assistant")
chat.display()  # Shows interactive UI in Jupyter
```

**Streaming Generation:**
```python
from llcuda.jupyter import stream_generate
text = stream_generate(engine, "Explain quantum computing", markdown=True)
```

**Chat Management:**
```python
from llcuda.chat import ChatEngine
chat = ChatEngine(engine)
chat.add_user_message("What is Python?")
response = chat.complete()
chat.save_history("conversation.json")
```

**Semantic Search:**
```python
from llcuda.embeddings import EmbeddingEngine, SemanticSearch
embedder = EmbeddingEngine(engine)
search = SemanticSearch(embedder)
search.add_documents(["doc1", "doc2", "doc3"])
results = search.search("query", top_k=3)
```

**Model Discovery:**
```python
from llcuda.models import list_models, print_model_catalog
models = list_models()  # Find all local models
print_model_catalog(vram_gb=4.0)  # Get recommendations
```

### Installation

**Basic (same as before):**
```bash
pip install llcuda
```

**With JupyterLab features (recommended):**
```bash
pip install llcuda[jupyter]
```

**With all features:**
```bash
pip install llcuda[all]
```

**Enable widgets in JupyterLab:**
```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Performance
All new features are optimized for efficiency:
- Streaming: Real-time display with minimal overhead
- Caching: Embedding cache with LRU eviction
- Lazy Loading: Modules imported only when needed
- Progress Tracking: Optional, zero overhead when disabled

### Testing
Tested on:
- Ubuntu 22.04 with CUDA 12.8
- Python 3.11
- GeForce 940M (1GB VRAM) to RTX 4090
- JupyterLab 3.x and 4.x
- llama.cpp commit 733c851f

### Technical Details
- Total new code: ~1,780 lines across 4 modules
- Total documentation: ~3,500 lines across 5 markdown files
- Tutorial notebook: 12 interactive sections
- New dependencies: 8 optional packages
- API endpoints used: `/completion`, `/v1/chat/completions`, `/v1/embeddings`, `/tokenize`

### Known Issues
- Embeddings require llama-server started with `--embedding` flag
- ChatWidget requires ipywidgets extension in JupyterLab
- Model downloads require huggingface_hub package
- Text clustering requires scikit-learn package

### What's Next (v0.3.1)
- Multimodal support (vision models)
- Enhanced error messages and debugging tools
- Model quantization utilities
- Streaming embeddings support

---

## [0.2.1] - 2025-12-27

### Documentation
- **Added**: Pre-built binary as recommended installation option in README.md
- **Added**: Pre-built binary section to SETUP_GUIDE_V2.md
- **Updated**: Installation guide now references Ubuntu-Cuda-Llama.cpp-Executable repository
- **Improved**: Setup instructions now highlight fastest installation method (pre-built binary)

### Links
- Pre-built binary: [Ubuntu-Cuda-Llama.cpp-Executable](https://github.com/waqasm86/Ubuntu-Cuda-Llama.cpp-Executable)
- Release v0.1.0: Contains llama.cpp commit 733c851f with CUDA 12.x support

## [0.2.0] - 2025-12-26

### 🚀 Major Release - Automatic Server Management

This release transforms llcuda into a production-ready package with automatic server management, zero-configuration setup, and comprehensive JupyterLab integration.

### Added
- **Automatic Server Management**: New `ServerManager` class (`llcuda/server.py`) for automatic llama-server lifecycle management
- **Auto-Start Capability**: `InferenceEngine.load_model()` now accepts `auto_start=True` to automatically start llama-server
- **Auto-Discovery System**:
  - Automatically finds llama-server executable in common locations
  - Auto-discovers GGUF models via `find_gguf_models()`
  - Locates llama-cpp-cuda installation automatically
- **System Diagnostics**: New `print_system_info()` for comprehensive system checks (Python, CUDA, GPU, models)
- **Context Manager Support**: Use `with InferenceEngine() as engine:` for automatic cleanup
- **Utility Module** (`llcuda/utils.py`):
  - `detect_cuda()` - Full CUDA detection with GPU details
  - `find_gguf_models()` - Auto-discover GGUF models
  - `get_llama_cpp_cuda_path()` - Find llama-cpp-cuda installation
  - `setup_environment()` - Auto-configure environment variables
  - `get_recommended_gpu_layers()` - Smart GPU layer recommendations based on VRAM
  - `validate_model_path()` - Model file validation
  - `load_config()` / `create_config_file()` - Configuration file support
- **JupyterLab Integration**:
  - Complete tutorial notebook (`examples/quickstart_jupyterlab.ipynb`) with 13 interactive sections
  - Optimized for notebook workflows
  - Performance visualization examples
- **Installation Verification**: New `test_setup.py` script to verify complete installation
- **Comprehensive Documentation**:
  - `README.md` - Complete rewrite with full API reference and examples
  - `SETUP_GUIDE_V2.md` - Step-by-step setup for Ubuntu 22.04
  - `QUICK_REFERENCE.md` - Quick command lookup card
  - `RESTRUCTURE_SUMMARY.md` - Detailed documentation of all changes

### Changed
- **InferenceEngine.load_model()**: New parameters:
  - `auto_start` (bool) - Automatically start server if not running
  - `n_parallel` (int) - Number of parallel sequences
  - `verbose` (bool) - Print status messages
- **Package Version**: 0.1.2 → 0.2.0
- **Package Description**: Updated to "CUDA-accelerated LLM inference for Python with automatic server management"
- **Error Messages**: Significantly improved with actionable suggestions
- **Resource Management**: Automatic cleanup of server processes on exit via `__del__` and context managers

### Improved
- **User Experience**: Zero-configuration setup - works out of the box with auto-discovery
- **Documentation**: 10x increase in documentation coverage (4 new guides + tutorial)
- **Examples**: Added comprehensive 13-section JupyterLab tutorial
- **Error Handling**: Better error messages with troubleshooting guidance
- **Performance**: Smart GPU layer recommendations for low-VRAM GPUs

### Fixed
- Automatic cleanup of server processes when `InferenceEngine` is destroyed
- Proper handling of library paths for llama-cpp-cuda via `LD_LIBRARY_PATH`
- Environment variable setup for optimal CUDA performance
- Server health checking before making requests

### Breaking Changes
**None** - v0.2.0 is fully backward compatible with v0.1.2

### Migration Guide

**Old way** (still works):
```python
# Terminal 1: Start llama-server manually
# $ llama-server -m model.gguf --port 8090 -ngl 99

# Python code:
import llcuda
engine = llcuda.InferenceEngine()
result = engine.infer("Hello")
```

**New way** (recommended):
```python
import llcuda
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf", auto_start=True, gpu_layers=99)
result = engine.infer("Hello")
# Server automatically stopped when done
```

### Technical Details
- Added `ServerManager` class for low-level server control
- Integrated server management into `InferenceEngine`
- Smart path discovery algorithm for llama-server and models
- Automatic environment variable configuration
- Support for `~/.llcuda/config.json` configuration file

### Performance
Tested on NVIDIA GeForce 940M (1GB VRAM):
- Gemma 3 1B Q4_K_M: ~15 tok/s with 20 GPU layers
- Auto-start overhead: <5 seconds
- Context manager cleanup: Instant

### Requirements
- Python 3.11+
- CUDA 11.7+ or 12.0+
- NVIDIA GPU with CUDA support
- llama-server executable (from llama.cpp)

## [0.1.2] - 2024-12-26

### Changed
- **Converted to pure Python package** - No longer requires C++ compilation!
- Removed C++ extension dependencies (CMake, pybind11, CUDA headers)
- Now installs instantly with `pip install llcuda` on all platforms
- Uses HTTP client to communicate with llama-server backend via requests library

### Added
- Added `requests>=2.20.0` as a dependency

### Fixed
- **Fixed PyPI installation failure** - Package now installs without compilation errors
- Works on Kaggle, Colab, Windows, Linux, macOS without build tools
- No more "Failed building wheel" errors

### Removed
- C++ extension build system
- CMake requirement
- pybind11 requirement
- CUDA development headers requirement

## [0.1.0] - 2024-12-26

### Added
- Initial release of llcuda
- CUDA-accelerated LLM inference engine
- Python API with Pythonic interface
- Support for GGUF model format
- Streaming inference support
- Batch processing capabilities
- Performance metrics tracking (latency, throughput, GPU stats)
- CMake-based build system with pybind11
- PyPI package configuration
- Comprehensive documentation:
  - Installation guide (INSTALL.md)
  - Quick start guide (QUICKSTART.md)
  - Kaggle/Colab guide (KAGGLE_COLAB.md)
  - PyPI publishing guide (PYPI_PUBLISHING_GUIDE.md)
- Example Jupyter notebook for Kaggle/Colab
- Unit tests
- MIT License

### Requirements
- Python 3.11+
- CUDA 11.7+ or 12.0+
- NVIDIA GPU (T4, P100, V100, etc.)
- CMake 3.24+
- pybind11 2.10.0+

### Technical Details
- Optimized for NVIDIA T4 GPUs (Kaggle/Colab)
- Supports GPU layer offloading
- Configurable context size and batch size
- Temperature, top-p, top-k sampling
- Custom stop sequences

### Known Limitations
- Requires llama-server backend
- Source distribution only (no pre-built wheels)
- Linux-focused (tested on Ubuntu 20.04+)

[0.1.0]: https://github.com/waqasm86/llcuda/releases/tag/v0.1.0
