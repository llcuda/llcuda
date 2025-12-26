# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
