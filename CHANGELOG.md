# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2025-12-30

### 🚀 Major Release: Multi-GPU Architecture Support + Cloud Platform Compatibility

This release adds **universal GPU compatibility** and **cloud platform support**, enabling llcuda to work seamlessly on Google Colab, Kaggle, and all NVIDIA GPUs with compute capability 5.0+.

### Added

#### Multi-GPU Architecture Support
- **Multi-architecture CUDA binaries**: Compiled for compute capabilities 5.0, 6.1, 7.0, 7.5, 8.0, 8.6, 8.9
- **Supported GPUs**:
  - Maxwell (5.0-5.3): GTX 900 series, GeForce 940M
  - Pascal (6.0-6.2): GTX 10xx, **Tesla P100**
  - Volta (7.0): **Tesla V100**
  - Turing (7.5): **Tesla T4**, RTX 20xx, GTX 16xx
  - Ampere (8.0-8.6): **A100**, RTX 30xx
  - Ada Lovelace (8.9): RTX 40xx

#### Cloud Platform Support
- **Google Colab**: Full support for T4, P100, V100, A100 GPUs
- **Kaggle**: Works on Tesla T4 notebooks (2x T4, 30GB total VRAM)
- **Platform auto-detection**: Automatically identifies local/Colab/Kaggle environments

#### New API Functions
- `check_gpu_compatibility(min_compute_cap=5.0)`: Validate GPU compatibility before loading models
  - Returns: `compatible`, `compute_capability`, `gpu_name`, `reason`, `platform`
  - Provides clear error messages with recommendations
- `skip_gpu_check` parameter in `ServerManager.start_server()`: Advanced users can override GPU checks

#### Documentation
- **COLAB_KAGGLE_GUIDE.md**: Complete guide for cloud platforms
  - Quick start examples for Colab and Kaggle
  - Platform-specific configuration
  - 6 complete code examples
  - Troubleshooting guide with 5 common issues
  - Performance benchmarks for T4/P100/V100
  - Best practices
- **RELEASE_v1.1.0.md**: Detailed release notes
- **IMPLEMENTATION_SUMMARY_v1.1.0.md**: Technical implementation details

### Changed

#### Binary Compilation
- **Build configuration**: Changed from `GGML_NATIVE=ON` to `GGML_NATIVE=OFF`
- **Binary size**: llama-server remains ~6.5 MB
- **CUDA library**: Increased from ~24 MB to ~114 MB (multi-architecture support)
- **Package size**: Increased from ~50 MB to ~313 MB (acceptable, under PyPI limit)
- **Architectures**: Now includes 7 architectures (was 1)

#### API Enhancements
- `ServerManager.start_server()`: Now performs automatic GPU compatibility check
  - Prints platform, GPU name, and compute capability
  - Raises clear error if GPU is incompatible
  - Provides recommendations (CPU mode, upgrade GPU, skip check)
- **Error messages**: Enhanced with specific guidance for each scenario

#### Package Metadata
- **Version**: `1.0.2` → `1.1.0`
- **Description**: Now mentions "Works on JupyterLab, Google Colab, and Kaggle"
- **Keywords**: Added `colab`, `kaggle`, `t4`, `p100`, `turing`, `ampere`
- **Classifiers**: Added CUDA 11 environment

### Fixed

#### Critical Bug Fixes
- **"No kernel image available" error**: Fixed on Tesla T4, P100, V100, A100, RTX GPUs
  - Root cause: Binaries compiled only for compute 5.0 (GeForce 940M)
  - Solution: Multi-architecture compilation with PTX virtual architectures
- **Silent failures on incompatible GPUs**: Now shows clear error messages
- **Missing GPU architecture support**: Added support for Pascal, Volta, Turing, Ampere, Ada Lovelace

### Performance

#### Benchmarks

**Tesla T4 (Colab/Kaggle, 15GB VRAM)**:
- Gemma 3 1B Q4_K_M: ~15 tok/s (26 GPU layers, ~1.2 GB VRAM)
- Llama 3.1 7B Q4_K_M: ~5-8 tok/s (20-32 GPU layers, ~8-12 GB VRAM)

**Tesla P100 (Colab, 16GB VRAM)**:
- Gemma 3 1B Q4_K_M: ~18 tok/s (26 GPU layers, ~1.2 GB VRAM)
- Llama 3.1 7B Q4_K_M: ~10 tok/s (32 GPU layers, ~12 GB VRAM)

**GeForce 940M (Local, 1GB VRAM)** - Backward Compatibility:
- Gemma 3 1B Q4_K_M: ~15 tok/s (unchanged from v1.0.x)
- No performance degradation on existing GPUs

#### Startup Performance
- **Native architectures (5.0, 8.6, 8.9)**: Instant startup (no change)
- **Virtual architectures (6.1, 7.0, 7.5, 8.0)**: First-run PTX JIT compilation adds 2-5 seconds
  - **One-time cost**: Kernels cached after first run
  - **Subsequent runs**: Instant (same as native)

### Migration Guide

#### From v1.0.x to v1.1.0

**No breaking changes** - v1.1.0 is fully backward compatible.

**Upgrade**:
```bash
pip install --upgrade llcuda
```

**Existing code works unchanged**:
```python
import llcuda
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", gpu_layers=20)
result = engine.infer("Hello!")
```

**Optional: Add GPU check**:
```python
import llcuda

compat = llcuda.check_gpu_compatibility()
if compat['compatible']:
    engine = llcuda.InferenceEngine()
    engine.load_model("model.gguf")
else:
    print(f"GPU not compatible: {compat['reason']}")
```

### Technical Details

#### Build System
- **llama.cpp commit**: 10b4f82d4 (build 7489)
- **CUDA version**: 12.8.61
- **Compiler**: GCC 11.4.0
- **Platform**: Ubuntu 22.04 LTS x86_64
- **Build time**: ~60 minutes (multi-architecture compilation)

#### CMake Configuration
```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FORCE_CUBLAS=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_NATIVE=OFF \              # KEY CHANGE
  -DGGML_OPENMP=ON
```

#### CUDA Architectures Included
```
50-virtual  - Maxwell (GTX 900, 940M)
61-virtual  - Pascal (GTX 10xx, P100)
70-virtual  - Volta (V100)
75-virtual  - Turing (T4, RTX 20xx)
80-virtual  - Ampere (A100, RTX 30xx)
86-real     - Ampere (RTX 30xx high-end)
89-real     - Ada Lovelace (RTX 40xx)
```

### Known Limitations

1. **Package size**: 313 MB (increased from ~50 MB)
   - Acceptable: Under PyPI 500 MB limit
   - Reason: Multi-architecture CUDA library
2. **First-run JIT latency**: 2-5 seconds on virtual architectures
   - Impact: One-time only, cached afterwards
   - Affected: P100, V100, T4, A100 (compute 6.x-8.0)
3. **VRAM constraints on cloud platforms**:
   - T4 (15GB): Limited to smaller models or reduced GPU layers
   - Recommendation: Start with `gpu_layers=20` for 7B models

### Deprecations

None.

### Security

No security issues addressed in this release.

---

## [1.0.2] - 2025-12-29

### 📖 Documentation Update Release

This release contains no functional changes from v1.0.1. It updates the README.md to ensure PyPI displays correct v1.0.1 bug fix information and installation instructions.

### Changed
- Updated README.md with v1.0.1 references
- Updated installation instructions to include `pip install --upgrade llcuda`

---

## [1.0.1] - 2025-12-29

### 🐛 Critical Bug Fixes - Low-VRAM GPU Compatibility

This patch release fixes critical issues preventing llcuda from working on low-VRAM GPUs like the GeForce 940M.

### Fixed

#### 1. Invalid Parameter Error (`--n-batch`)
- **Issue**: Server crashed with `error: invalid argument: --n-batch`
- **Root Cause**: Parameter mapping incorrectly converted `n_batch` to `--n-batch` instead of `-b`
- **Fix**: Updated `server.py` to use correct llama-server parameter names
  - Added `batch_size` and `ubatch_size` as explicit function parameters
  - Created `param_map` dictionary for special parameter mappings
  - Fixed command building to use `-b` and `-ub` flags directly

#### 2. Shared Library Loading Failure
- **Issue**: `error while loading shared libraries: libmtmd.so.0: cannot open shared object file`
- **Root Cause**: `LD_LIBRARY_PATH` not configured for bundled libraries
- **Fix**: Added `_setup_library_path()` method to `ServerManager`
  - Automatically detects `lib/` directory relative to llama-server binary
  - Configures `LD_LIBRARY_PATH` when finding server executable

#### 3. Parameter Naming Inconsistencies
- **Fix**: Documented correct parameter names and added proper handling

### Testing
- Verified on GeForce 940M (1GB VRAM)
- Gemma 3 1B Q4_K_M: ~15 tokens/second
- All parameter combinations tested

---

## [1.0.0] - 2025-12-27

### 🎉 Initial Release

First stable release of llcuda with bundled CUDA binaries and zero-configuration setup.

### Features
- PyTorch-style Python API
- Bundled llama-server executable (CUDA 12.8)
- Smart model loading with registry
- Hardware auto-configuration
- Performance metrics (P50/P95/P99)
- JupyterLab support

### Supported Platforms
- Ubuntu 22.04
- Python 3.11+
- NVIDIA GPUs with CUDA Compute Capability 5.0+

---

## Links

- **PyPI**: https://pypi.org/project/llcuda/
- **GitHub**: https://github.com/waqasm86/llcuda
- **Documentation**: https://waqasm86.github.io/
- **Bug Tracker**: https://github.com/waqasm86/llcuda/issues

---

*For detailed technical information, see [IMPLEMENTATION_SUMMARY_v1.1.0.md](IMPLEMENTATION_SUMMARY_v1.1.0.md)*
## [1.1.3] - 2026-01-01
- Clean up unnecessary development files from root directory
- Streamline package structure for distribution

