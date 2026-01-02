# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.7] - 2025-01-03

### 🚀 CUDA 12.8 Support and Enhanced Binary Distribution

This release brings full CUDA 12.8 compatibility with optimized binaries for both modern and legacy GPUs.

### Major Changes
- **CUDA 12.8 Support**: Binaries compiled with CUDA Toolkit 12.8 for latest GPU drivers
- **Optimized Binaries**: Reduced binary distribution from 551MB to 161MB (70% reduction)
- **Enhanced Compatibility**: Improved support for Google Colab T4, Kaggle notebooks, and local systems
- **Python 3.11 Focus**: Continued testing and optimization for Python 3.11+
- **Package Size**: Maintained ultra-lightweight 62KB wheel, 61KB source distribution

### Improvements
- **Binary Distribution**: Streamlined archive includes only essential llama.cpp executables and libraries
- **Download Speed**: Faster binary downloads for Colab and Kaggle users
- **CUDA Runtime**: Full compatibility with CUDA 12.8 runtime and latest NVIDIA drivers
- **GPU Support**: Tested on Maxwell (GTX 940M) through Ada Lovelace (RTX 4090) architectures
- **Documentation**: Updated all docs with CUDA 12.8 compatibility information

### Performance
- **Binary Size**: Reduced from 551MB to 161MB (70% smaller)
- **Installation**: Faster package installation and first-run bootstrap
- **Memory**: Same efficient memory usage as v1.1.6
- **Throughput**: Maintained performance across all supported GPUs

### Maintained Features
- ✅ Hybrid bootstrap architecture (auto-download binaries/models)
- ✅ Universal GPU support (SM 5.0-8.9: Maxwell to Ada Lovelace)
- ✅ All existing APIs and functionality from v1.1.6
- ✅ Colab/Kaggle compatibility with T4 GPUs
- ✅ Python 3.11+ support
- ✅ CUDA 11/12 compatibility

### Package Info
- **Wheel Size**: 62 KB
- **Source Size**: 61 KB
- **Binary Archive**: 161 MB (llcuda-binaries-cuda12.tar.gz)
- **Dependencies**: Unchanged (numpy, requests, huggingface_hub, tqdm)
- **Python Support**: 3.11+ (explicitly tested)
- **CUDA Support**: 11.0+ and 12.0+ (12.8 recommended)

---

## [1.1.6] - 2025-01-03

### 🧹 Project Cleanup and Structure Optimization

This release focuses on cleaning up the project structure while maintaining all functionality for Python 3.11+.

### Major Changes
- **Repository Size**: Reduced from 14GB+ to <100MB for faster cloning
- **Project Cleanup**: Removed unnecessary binaries, old scripts, and deprecated files
- **Python 3.11 Focus**: Explicit testing and optimization for Python 3.11+
- **Package Size**: Ultra-lightweight 62KB wheel, 61KB source distribution
- **Git Performance**: Significantly improved clone and operation times

### Removed Files
- `bundles/` directory (large binary bundles)
- `check_binaries/` directory (testing binaries)
- `releases/` directory (old release assets)
- Old version scripts: `build_wheel.sh`, `create_release.sh`, `finalize_release.sh`
- Test scripts: `test_bootstrap.py`, `test_installation.py`, `verify_versions.sh`
- Upload scripts: `upload_to_huggingface.py`, `upload_to_pypi.sh`
- Large source archives: `llcuda-v1.1.*-source.*`

### Improvements
- **.gitignore**: Enhanced with comprehensive exclusions
- **Documentation**: Updated all README, examples, and docstrings
- **Development Environment**: Cleaner, more maintainable codebase
- **CI/CD Ready**: Better structure for automated workflows
- **GitHub/PyPI Integration**: Streamlined for deployment

### Maintained Features
- ✅ Hybrid bootstrap architecture (auto-download binaries/models)
- ✅ Universal GPU support (SM 5.0-8.9)
- ✅ All existing APIs and functionality
- ✅ Colab/Kaggle compatibility
- ✅ Python 3.11+ support
- ✅ CUDA 11/12 compatibility

### Performance
- **Clone Time**: Reduced from minutes to seconds
- **Disk Usage**: 99%+ reduction in local storage
- **Development**: Faster build and test cycles
- **Upload**: Quicker PyPI/GitHub releases

### Package Info
- **Wheel Size**: 62.2 KB (vs previous 51KB with additional docs)
- **Source Size**: 60.6 KB (vs previous 49KB)
- **Dependencies**: Unchanged (numpy, requests, huggingface_hub, tqdm)
- **Python Support**: 3.11+ (explicitly tested)

---

## [1.1.5] - 2026-01-02

### 🔧 Version Skip - PyPI Filename Resolution

This release skips to version 1.1.5 to resolve PyPI filename conflicts from previous upload attempts.

### No Functional Changes
- Contains all fixes from v1.1.2 and v1.1.3
- Binary extraction fixes for Google Colab
- Updated download URLs
- Enhanced library path detection
- PyPI upload compatibility fix

---

## Version 1.1.5 (2026-01-02)

### New Features
- Enhanced compatibility with older NVIDIA GPUs (SM 5.0+)
- Improved auto-download system for binaries and models
- Better error handling for cloud environments (Colab/Kaggle)

### Bug Fixes
- Fixed binary path resolution in hybrid bootstrap
- Improved GPU detection for legacy hardware
- Resolved PyPI filename conflicts from previous versions

### Performance
- Optimized memory usage for GPUs with limited VRAM
- Faster model loading on first import
- Reduced package size for PyPI distribution

## Version 1.1.5
- Updated binary compatibility for broader GPU support
- Fixed PyPI filename conflicts
- Improved auto-download system

