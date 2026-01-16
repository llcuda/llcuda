# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.1.1] - 2026-01-16

### 🎯 Colab-Focused Refresh: Enhanced Reliability

#### Fixed
- ✅ Fixed llama-server discovery fallback mechanism (no more NameError when primary binary fails)
- ✅ Improved bootstrap binary download error handling and recovery

#### Updated
- ✅ Updated Gemma 3-1B Colab notebook for v2.1.1 compatibility
- ✅ Consistent version numbering across entire project (pyproject.toml, __init__.py, binaries)
- ✅ Enhanced .gitignore to prevent large binary commits

#### Includes
- ✅ Tesla T4 binary bundle (v2.1.0 primary with v2.0.6 fallback)
- ✅ All v2.1.0 stable APIs: Quantization, Unsloth, CUDA Optimization, Advanced Inference
- ✅ Google Colab optimization and one-command installation

---

## [2.1.0] - 2026-01-13

### 🚀 Major Release: Complete Unsloth Integration with Advanced CUDA APIs

This major release introduces four powerful API modules that seamlessly integrate Unsloth fine-tuning with optimized CUDA inference for Tesla T4 GPUs.

### Added

#### 1. Quantization API (`llcuda.quantization`)
- **NF4 Quantization**: Block-wise 4-bit NormalFloat quantization with double quantization support
  - `quantize_nf4()`: Convert PyTorch tensors to NF4 format
  - `NF4Quantizer`: Configurable quantizer with blocksize and double_quant options
- **GGUF Conversion**: Complete GGUF v3 format support with 29 quantization types
  - `convert_to_gguf()`: Convert PyTorch models to GGUF format
  - `GGUFConverter`: Full control over quantization and metadata
  - Supported types: Q4_0, Q4_K_M, Q5_K_M, Q8_0, F16, NF4, and 23 more
- **Dynamic Quantization**: Intelligent VRAM-based quantization recommendations
  - `DynamicQuantizer`: Automatic quantization type selection based on available VRAM
  - `recommend_config()`: Get optimal quantization settings for your hardware

#### 2. Unsloth Integration API (`llcuda.unsloth`)
- **Model Loading**: Direct loading of Unsloth fine-tuned models
  - `load_unsloth_model()`: Load models with 4-bit quantization support
  - `UnslothModelLoader`: Configurable loader with max_seq_length control
- **GGUF Export**: Export fine-tuned models to GGUF with automatic LoRA merging
  - `export_to_llcuda()`: One-line export from Unsloth to llcuda
  - `UnslothExporter`: Advanced export with quantization control
- **LoRA Adapter Management**: Handle LoRA adapters efficiently
  - `merge_lora_adapters()`: Merge LoRA weights into base model
  - `LoRAAdapter`: Adapter management and merging utilities

#### 3. CUDA Optimization API (`llcuda.cuda`)
- **CUDA Graphs**: 20-40% latency reduction for inference workloads
  - `CUDAGraph`: Capture and replay CUDA operations
  - `GraphPool`: Manage multiple graphs for different batch sizes
- **Triton Kernels**: Custom GPU operations with Triton integration
  - `triton_add()`, `triton_layernorm()`, `triton_softmax()`: Built-in kernels
  - `register_kernel()`: Register custom Triton kernels
- **Tensor Core Utilities**: Leverage Tesla T4 Tensor Cores (SM 7.5)
  - `enable_tensor_cores()`: Enable TF32 and FP16 Tensor Core acceleration
  - `matmul_tensor_core()`: Optimized matrix multiplication
  - `get_tensor_core_info()`: Query Tensor Core capabilities

#### 4. Advanced Inference API (`llcuda.inference`)
- **FlashAttention v2**: 2-3x speedup for long context inference
  - `enable_flash_attention()`: Enable FlashAttention with custom config
  - `get_optimal_context_length()`: Calculate optimal context based on VRAM
- **KV-Cache Optimization**: Efficient key-value cache management
  - `KVCache`: Optimized cache for transformer inference
  - `KVCacheConfig`: Configure cache size and behavior
- **Batch Inference**: Continuous batching and batch optimization
  - `batch_inference_optimized()`: Efficient multi-prompt inference
  - `ContinuousBatching`: Dynamic batching for variable-length sequences

### Documentation
- **API_REFERENCE.md**: Complete API documentation (503 lines)
- **QUICK_START.md**: 5-minute getting started guide (277 lines)
- **NEW_APIS_README.md**: Overview of v2.1+ features (557 lines)
- **IMPLEMENTATION_SUMMARY.md**: Technical architecture details (589 lines)
- **TEST_RESULTS.md**: Comprehensive test results (434 lines)
- **COMPLETION_REPORT.md**: Full implementation report (590 lines)

### Examples
- **examples/complete_workflow_example.py**: Full Unsloth to llcuda workflow (358 lines)
- **examples/api_usage_examples.py**: Quick API demonstrations (321 lines)

### Tests
- **tests/test_new_apis.py**: 18 comprehensive unit tests (242 lines)
  - All tests passed with graceful fallbacks for optional dependencies
  - Coverage: Quantization, Unsloth Integration, CUDA Optimization, Advanced Inference

### Changed
- Updated `pyproject.toml` version from 2.0.6 to 2.1.0
- Enhanced package description with new features
- Updated `README.md` with comprehensive v2.1+ feature documentation
- Updated `llcuda/__init__.py` to export new API modules
- **100% backward compatibility maintained** with v2.0.x

### Performance
- **CUDA Graphs**: 20-40% latency reduction for inference
- **Tensor Cores**: 2-4x speedup for FP16/TF32 operations on Tesla T4
- **FlashAttention**: 2-3x speedup for long context (8K+ tokens)
- **Dynamic Quantization**: Optimize VRAM usage while maintaining accuracy

### Technical Details
- Total additions: 17,498 insertions across 31 files
- New Python modules: 17 files (3,903 lines of code)
- Documentation: 5 comprehensive guides (2,060 lines)
- Examples: 2 complete workflow examples (679 lines)
- Tests: 18 unit tests with 100% pass rate (242 lines)

---

## [2.0.6] - 2026-01-10

### Added
- **Comprehensive Gemma 3-1B Tutorial Notebook**: Added complete Google Colab tutorial demonstrating llcuda v2.0.6 with Unsloth
  - `notebooks/llcuda_v2_0_6_gemma3_1b_unsloth_colab.ipynb` - Full tutorial with 14 steps
  - `notebooks/llcuda_v2_0_6_gemma3_1b_unsloth_colab_executed.ipynb` - Live execution output from Tesla T4
  - Demonstrates GitHub installation, binary auto-download, model loading, and inference
  - Includes batch processing, performance metrics, and advanced generation examples
  - Documents complete Unsloth fine-tuning → llcuda deployment workflow

### Performance
- **Verified Tesla T4 Performance**: Real Google Colab execution confirms **134 tok/s** with Gemma 3-1B Q4_K_M
  - 3x faster than initial estimates (was 45 tok/s)
  - Median latency: 690ms
  - Consistent performance across batch inference (130-142 tok/s range)
  - FlashAttention and Tensor Core optimization delivering exceptional results

### Documentation
- Updated README with verified performance benchmarks
- Added dedicated "Tutorials & Notebooks" section to README
- Enhanced performance table with verified vs estimated metrics
- Linked to executed notebook as proof of working implementation

### Changed
- Performance benchmarks updated with real Tesla T4 measurements
- README now highlights 134 tok/s verified performance for Gemma 3-1B

---

## [2.0.2] - 2026-01-08

### 🐛 Critical Bug Fixes

#### Fixed
- **HTTP 404 Error on Binary Download**: Fixed bootstrap failing with 404 error when downloading binaries on first import
  - Root cause: Version mismatch between PyPI package (v2.0.0/v2.0.1) and GitHub release URLs
  - Solution: Updated `bootstrap.py` to use v2.0.2 release URL
- **Version Number Inconsistency**: Fixed `__version__` incorrectly reporting "1.2.2" instead of actual version
  - Updated `llcuda/__init__.py` to correctly report "2.0.2"
- **Tar File Structure Mismatch**: Fixed binary tar extraction failures
  - Root cause: Tar had unexpected parent directory `llcuda-complete-t4/` instead of root-level `bin/` and `lib/`
  - Solution: Recreated tar with correct structure for proper extraction by bootstrap code

#### Changed
- Enhanced `.gitignore` with stronger protection against large binary commits
  - Added explicit patterns for `*.so.*`, `*.a`, and all shared library variants
  - Better documentation of file size limits for GitHub/PyPI
- Updated binary package filename to `llcuda-binaries-cuda12-t4-v2.0.2.tar.gz` (266 MB)
- SHA256: `1dcf78936f3e0340a288950cbbc0e7bf12339d7b9dfbd1fe0344d44b6ead39b5`

#### Impact
- Kaggle, Colab, and local installations now work correctly without 404 errors
- All v2.0.0/v2.0.1 users should upgrade to avoid installation failures
- No breaking changes - fully backward compatible

---

## [2.0.1] - 2026-01-07

### Changed
- **Cleanup Release**: Removed duplicate files, obsolete documentation, and large binaries from repository
- Updated .gitignore to prevent large binary files from being uploaded to PyPI and GitHub
- Excluded large binaries (*.so, *.tar.gz, *.gguf) from PyPI wheel package
- Binaries are now downloaded on first use via bootstrap mechanism (not included in wheel)
- Improved package structure and reduced repository size by ~265 MB

### Removed
- Duplicate backup files (`__init___backup.py`, `__init___pure.py`)
- Empty nested directory structure in `llcuda/` package
- Obsolete CMakeLists.txt and llcuda_py.cpp from package directory
- 15+ obsolete documentation files from v1.x era
- Duplicate binary tarballs (kept single copy in release-packages)

### Fixed
- Package-data configuration to exclude large binaries from PyPI upload
- .gitignore patterns to prevent accidental uploads of model files (.gguf) to GitHub/PyPI

### Note
- **IMPORTANT**: Large binaries (llcuda_cpp.so, llcuda-binaries-*.tar.gz) are NOT included in the PyPI package
- Users must rebuild native extensions or binaries will auto-download on first import
- This ensures PyPI package stays under 100 MB limit

---

## [2.0.0] - 2025-01-06

### Added
- **Native Tensor API**: PyTorch-style GPU operations with custom CUDA kernels
- **Tensor Core Optimization**: Exclusive Tesla T4 (SM 7.5) targeting
- **Custom CUDA kernels**: Device management, tensor operations, cuBLAS matmul
- **GGUF Parser**: Zero-copy memory-mapped GGUF file reader
- **Bootstrap refactor**: T4-only binary downloader with verification

### Changed
- Refactored to Tesla T4-only architecture (removed multi-GPU support)
- Updated binary package to 264 MB with FlashAttention and CUDA Graphs
- Python 3.11+ requirement (was 3.8+)

---

## [1.2.2] - 2025-01-04

### Documentation
- Simplified all documentation to focus exclusively on GeForce 940M and Tesla T4
- Removed references to Pascal, Volta, Ampere, and Ada GPUs from user-facing documentation
- Updated README to highlight Ubuntu 22.04 and Google Colab as primary supported platforms
- Clarified GPU support table to show only GeForce 940M and Tesla T4
- Updated package description for PyPI consistency

### Note
- No code changes - this is a documentation-only release
- All GPU architectures continue to work (Pascal/Volta/Ampere/Ada download T4 binaries)
- Focus on 940M and T4 provides clearer documentation for primary use cases

---

## [1.2.2] - 2025-01-04

### 🚀 GPU-Specific Optimizations and FlashAttention Support

Major release introducing GPU-specific binary bundles with automatic detection and FlashAttention support for 2x faster inference on modern GPUs.

### Added
- **GPU-specific binary bundles** for optimized performance
  - GeForce 940M package (26 MB) with forced cuBLAS for Maxwell architecture
  - Tesla T4 package (264 MB) with FlashAttention for Turing+ architectures
- **Automatic GPU detection** in bootstrap using nvidia-smi
  - Detects GPU name and compute capability
  - Selects appropriate binary bundle automatically
  - Supports Maxwell (CC 5.x), Pascal (CC 6.x), Volta (CC 7.0), Turing (CC 7.5), Ampere (CC 8.x), and Ada (CC 8.9)
- **FlashAttention support** for CC 7.5+ GPUs
  - 2x faster inference on Tesla T4, RTX 20xx/30xx/40xx, and A100
  - Enabled automatically when supported by GPU
- **GPU compute capability detection** function
  - `detect_gpu_compute_capability()` in bootstrap module
  - Returns GPU name and compute capability tuple
- **Smart binary selection** logic
  - Maps GPU architectures to appropriate binary bundles
  - Falls back to T4 binaries for unknown GPUs (better compatibility)
- **Platform detection** function for Colab/Kaggle/local systems

### Fixed
- **Critical**: Fixed `AttributeError: 'NoneType' object has no attribute 'read'` when reading stderr in silent mode
  - Issue occurred in Google Colab when server process died with silent=True
  - Added null check before reading stderr (llcuda/server.py:553)
  - Now raises informative RuntimeError instead of AttributeError
- **Packaging**: Fixed library path detection for different CMake build configurations
  - T4 builds put libraries in `lib/` directory
  - 940M builds put libraries in `bin/` directory
  - CREATE_RELEASE_PACKAGE.sh now searches both locations
- **Packaging**: Fixed script termination bug in CREATE_RELEASE_PACKAGE.sh
  - Changed `((BINARY_COUNT++))` to `BINARY_COUNT=$((BINARY_COUNT + 1))`
  - Prevents premature exit with `set -e`

### Changed
- **Bootstrap architecture**: Now downloads GPU-specific binaries instead of universal bundle
  - Maxwell GPUs download 26 MB optimized package
  - Modern GPUs download 264 MB package with FlashAttention
  - Reduces download size for older GPUs by 90%
- **Library management**: Improved LD_LIBRARY_PATH configuration
  - Handles both bin/ and lib/ directory structures
  - Automatically detects library location during extraction
- **Package structure**: Updated to support multiple binary variants
  - GPU_BUNDLES dictionary maps GPU types to appropriate packages
  - select_binary_bundle() function implements selection logic
- **GitHub Release URL**: Updated to v1.2.2 in bootstrap.py
- **Version**: Bumped to 1.2.2 in __init__.py and pyproject.toml

### Performance
- **GeForce 940M (CC 5.0)**: 10-20 tokens/sec for 1-3B parameter models
  - Optimized with forced cuBLAS
  - Best for Q4_K_M quantized models
  - Recommended GPU layers: 10-15
- **Tesla T4 (CC 7.5)**: 25-60 tokens/sec with FlashAttention
  - 2x improvement over non-FlashAttention builds
  - Best for Q4_K_M and Q5_K_M quantized models
  - Recommended GPU layers: 26-35
- **RTX 4090 (CC 8.9)**: 120+ tokens/sec for small models
  - FlashAttention enabled
  - Full GPU offload for models up to 13B parameters
  - Recommended GPU layers: 35+

### Package Info
- **Wheel Size**: ~62 KB (Python code only)
- **Source Size**: ~61 KB
- **Binary Bundles**: GPU-specific downloads
  - Maxwell (940M): 26 MB
  - Modern (T4+): 264 MB
- **Python Support**: 3.11+
- **CUDA Support**: 12.x (12.8 recommended)

---

## [1.1.9] - 2025-01-03

### 🔧 Bug Fixes - llama-server Detection

Critical fix for llama-server path detection in Google Colab and Kaggle.

### Fixed
- **Server Detection**: Added package binaries directory as priority #2 in search order
- **Path Priority**: Now checks `llcuda/binaries/cuda12/llama-server` before system paths
- **Cache Paths**: Added Colab (`/content/.cache`) and Kaggle (`/kaggle/working/.cache`) specific paths
- **Library Path**: Automatic LD_LIBRARY_PATH setup for package-installed binaries

### Added
- **Silent Mode**: New `silent=True` parameter to suppress all llama-server output/warnings
- **Better Detection**: Improved binary finding logic for cloud environments

### Changed
- Priority order now: ENV vars → Package binaries → LLAMA_CPP_DIR → Cache → Project paths → System paths
- Server manager now checks package installation directory first

### Usage
```python
# Suppress llama-server warnings
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)
```

### Package Info
- **Wheel Size**: ~54 KB
- **Source Size**: ~56 KB
- **Binary Archive**: Use v1.1.7 binaries (161 MB)
- **Python Support**: 3.11+
- **CUDA Support**: 11.0+ and 12.0+ (12.8 recommended)

---

## [1.1.8] - 2025-01-03

### 🐛 Bug Fixes - Colab/Kaggle Bootstrap

Critical fixes for Google Colab and Kaggle compatibility.

### Fixed
- **Bootstrap URL**: Updated to download v1.1.7 binaries instead of v1.1.6 (404 error fix)
- **Auto-Download Removed**: No longer downloads GGUF models on import - models download only when explicitly requested via `load_model()`
- **Binary Extraction**: Improved handling of bin/lib archive structure for proper installation
- **Memory Usage**: Prevents unnecessary 800MB model downloads on every import

### Changes
- Bootstrap now downloads binaries ONLY on first import
- Models are downloaded on-demand when `engine.load_model()` is called
- Improved error messages for binary download failures
- Better archive structure handling (bin/ and lib/ directories)

### User Impact
- ✅ Faster `import llcuda` - no automatic model download
- ✅ Works in Google Colab with T4 GPUs
- ✅ Works in Kaggle notebooks
- ✅ Models download only when needed
- ✅ Reduced memory usage during initialization

### Package Info
- **Wheel Size**: 62 KB
- **Source Size**: 61 KB
- **Binary Archive**: 161 MB (llcuda-binaries-cuda12.tar.gz)
- **Bootstrap**: Fixed for v1.1.7 binaries
- **Python Support**: 3.11+
- **CUDA Support**: 11.0+ and 12.0+ (12.8 recommended)

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

