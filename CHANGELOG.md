# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
