

# llcuda Integration Guide: Complete Path Detection and Execution Flow

## Overview

This document explains how llcuda locates, detects, and executes the llama-server binary after you build it with CMake. Understanding this flow is crucial for successful integration.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Detection Flow](#detection-flow)
4. [Integration Steps](#integration-steps)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### llcuda's Hybrid Bootstrap Design

llcuda uses a **hybrid bootstrap architecture**:

1. **Minimal Package (62KB)**: The pip package is tiny, containing only Python code
2. **Auto-Download**: On first import, downloads CUDA binaries from GitHub releases
3. **Platform Detection**: Automatically detects GPU (GeForce 940M, Tesla T4, etc.)
4. **Path Auto-Configuration**: Sets up `LD_LIBRARY_PATH` and `LLAMA_SERVER_PATH`

### How It Works

```
User: pip install llcuda
       ↓
User: import llcuda
       ↓
llcuda/__init__.py:
  1. Checks if binaries exist at llcuda/binaries/cuda12/llama-server
  2. If NOT found → calls bootstrap() from _internal/bootstrap.py
  3. bootstrap() downloads llcuda-binaries-cuda12.tar.gz from GitHub
  4. Extracts to llcuda/binaries/cuda12/ and llcuda/lib/
  5. Sets LD_LIBRARY_PATH and LLAMA_SERVER_PATH environment variables
       ↓
User: engine = llcuda.InferenceEngine()
User: engine.load_model(...)
       ↓
llcuda/server.py:
  1. ServerManager.find_llama_server() searches for executable
  2. Uses the detection order (see below)
  3. Starts llama-server as subprocess
  4. Waits for server health check
       ↓
User: result = engine.infer("prompt")
       ↓
Makes HTTP request to llama-server at http://127.0.0.1:8090
```

---

## Directory Structure

### After CMake Build (llama.cpp)

```
/media/waqasm86/External1/Project-Nvidia/llama.cpp/
├── build_cuda12_940m/               # Build directory for GeForce 940M
│   ├── bin/
│   │   ├── llama-server             # Main executable ⭐
│   │   ├── llama-cli
│   │   ├── llama-quantize
│   │   ├── llama-embedding
│   │   ├── libllama.so              # Shared library
│   │   ├── libggml-base.so
│   │   └── libggml-cuda.so          # CUDA backend
│   └── ggml/src/
│       └── libggml*.so*             # Additional GGML libraries
└── build_cuda12_t4/                 # Build directory for Tesla T4
    └── ... (same structure)
```

### After Integration (llcuda package)

```
/media/waqasm86/External1/Project-Nvidia/llcuda/
├── llcuda/                          # Python package root
│   ├── __init__.py                  # Main entry point with auto-config
│   ├── server.py                    # ServerManager with find_llama_server()
│   ├── models.py
│   ├── utils.py
│   ├── binaries/                    # ⭐ Binaries directory
│   │   └── cuda12/                  # CUDA 12 binaries
│   │       ├── llama-server         # ← Copied from build/bin/
│   │       ├── llama-cli
│   │       ├── llama-quantize
│   │       └── llama-embedding
│   ├── lib/                         # ⭐ Shared libraries directory
│   │   ├── libllama.so              # ← Copied from build/bin/
│   │   ├── libggml-base.so
│   │   ├── libggml-cuda.so
│   │   └── ... (other .so files)
│   ├── models/                      # Model cache (created automatically)
│   └── _internal/
│       ├── bootstrap.py             # Auto-download logic
│       └── registry.py              # Model registry
├── setup.py
└── README.md
```

---

## Detection Flow

### llcuda/__init__.py (Lines 45-163)

**On Import:**

```python
import llcuda  # Triggers __init__.py

# Step 1: Set up paths
_LLCUDA_DIR = Path(__file__).parent  # .../llcuda/llcuda/
_BIN_DIR = _LLCUDA_DIR / "binaries" / "cuda12"
_LIB_DIR = _LLCUDA_DIR / "lib"

# Step 2: Auto-configure LD_LIBRARY_PATH
if _LIB_DIR.exists():
    os.environ["LD_LIBRARY_PATH"] = f"{_LIB_DIR}:{existing_path}"

# Step 3: Auto-configure LLAMA_SERVER_PATH
_LLAMA_SERVER = _BIN_DIR / "llama-server"
if _LLAMA_SERVER.exists():
    os.environ["LLAMA_SERVER_PATH"] = str(_LLAMA_SERVER.absolute())
else:
    # Step 4: Bootstrap (download binaries)
    from ._internal.bootstrap import bootstrap
    bootstrap()  # Downloads from GitHub, extracts to binaries/cuda12/ and lib/
```

### llcuda/server.py - find_llama_server() (Lines 72-166)

**Search Order (Priority 1-6):**

```python
def find_llama_server(self) -> Optional[Path]:
    # Priority 1: LLAMA_SERVER_PATH environment variable
    env_path = os.getenv("LLAMA_SERVER_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Priority 2: Package's installed binaries ⭐ MAIN PATH
    package_dir = Path(llcuda.__file__).parent
    paths = [
        package_dir / "binaries" / "cuda12" / "llama-server",  # After integration
        package_dir / "binaries" / "llama-server",
    ]
    for path in paths:
        if path.exists():
            return path

    # Priority 3: LLAMA_CPP_DIR environment variable
    llama_cpp_dir = os.getenv("LLAMA_CPP_DIR")
    if llama_cpp_dir:
        path = Path(llama_cpp_dir) / "bin" / "llama-server"
        if path.exists():
            return path

    # Priority 4: Cache directory (bootstrap download location)
    cache_paths = [
        Path.home() / ".cache" / "llcuda" / "bin" / "llama-server",
        Path("/content/.cache/llcuda/llama-server"),  # Google Colab
    ]
    for path in cache_paths:
        if path.exists():
            return path

    # Priority 5: Project-specific locations
    project_paths = [
        Path("/media/waqasm86/External1/Project-Nvidia/llama.cpp/build/bin/llama-server"),
        # ... more paths
    ]

    # Priority 6: System locations
    system_paths = [
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
    ]

    return None  # Not found
```

### llcuda/_internal/bootstrap.py (Lines 144-263)

**Auto-Download Logic:**

```python
def download_binaries():
    llama_server = BINARIES_DIR / "cuda12" / "llama-server"
    if llama_server.exists():
        return  # Already downloaded

    # Download from GitHub releases
    bundle_url = f"{GITHUB_RELEASE_URL}/{BINARY_BUNDLE_NAME}"
    # GITHUB_RELEASE_URL = "https://github.com/waqasm86/llcuda/releases/download/v1.2.2"
    # BINARY_BUNDLE_NAME = "llcuda-binaries-cuda12.tar.gz"

    cache_tarball = Path.home() / ".cache" / "llcuda" / BINARY_BUNDLE_NAME
    download_file(bundle_url, cache_tarball)

    # Extract
    temp_extract_dir = Path.home() / ".cache" / "llcuda" / "extract"
    extract_tarball(cache_tarball, temp_extract_dir)

    # Install to package directory
    bin_dir = temp_extract_dir / "bin"
    lib_dir = temp_extract_dir / "lib"

    # Copy binaries to llcuda/binaries/cuda12/
    for item in bin_dir.iterdir():
        shutil.copy2(item, BINARIES_DIR / "cuda12" / item.name)
        chmod(0o755)  # Make executable

    # Copy libraries to llcuda/lib/
    for item in lib_dir.iterdir():
        shutil.copy2(item, LIB_DIR / item.name)
```

---

## Integration Steps

### Manual Integration (After Building with CMake)

#### Step 1: Build with CMake

```bash
cd /media/waqasm86/External1/Project-Nvidia/llama.cpp

# Configure
cmake -B build_cuda12_940m \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="50" \
    -DGGML_CUDA_FORCE_CUBLAS=ON \
    -DGGML_CUDA_FA=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DBUILD_SHARED_LIBS=ON

# Build
cmake --build build_cuda12_940m --config Release -j$(nproc)
```

#### Step 2: Create Directory Structure

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda

# Create directories
mkdir -p binaries/cuda12
mkdir -p lib
mkdir -p models
```

#### Step 3: Copy Binaries

```bash
BUILD_DIR="../../../llama.cpp/build_cuda12_940m"

# Copy executables
cp $BUILD_DIR/bin/llama-server binaries/cuda12/
cp $BUILD_DIR/bin/llama-cli binaries/cuda12/
cp $BUILD_DIR/bin/llama-quantize binaries/cuda12/
cp $BUILD_DIR/bin/llama-embedding binaries/cuda12/

# Make executable
chmod +x binaries/cuda12/*
```

#### Step 4: Copy Libraries

```bash
# Copy all .so files
cp $BUILD_DIR/bin/*.so* lib/

# Also check ggml/src
cp $BUILD_DIR/ggml/src/*.so* lib/ 2>/dev/null || true
```

#### Step 5: Verify

```bash
# Check structure
ls -lh binaries/cuda12/
ls -lh lib/

# Test execution
export LD_LIBRARY_PATH="$(pwd)/lib:$LD_LIBRARY_PATH"
./binaries/cuda12/llama-server --help
```

### Automated Integration

**Use the provided script:**

```bash
cd /media/waqasm86/External1/Project-Nvidia
./BUILD_AND_INTEGRATE.sh 940m
```

This script:
1. Guides you through CMake configuration
2. Prompts you to run the build
3. Automatically copies binaries and libraries to correct locations
4. Sets up proper permissions
5. Verifies the integration
6. Creates a test script

---

## Verification

### Test 1: Directory Structure

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda

# Check binaries exist
ls -lh binaries/cuda12/llama-server
# Expected: ~150-200 MB file

# Check libraries exist
ls -lh lib/*.so*
# Expected: Multiple .so files (libllama.so, libggml*.so, etc.)
```

### Test 2: Python Import

```python
import os
import sys

# Set up paths
os.environ['LLAMA_SERVER_PATH'] = '/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/binaries/cuda12/llama-server'
os.environ['LD_LIBRARY_PATH'] = '/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/lib'

# Import llcuda
import llcuda

# Check version
print(f"llcuda version: {llcuda.__version__}")

# Test server detection
from llcuda.server import ServerManager
manager = ServerManager()
server_path = manager.find_llama_server()
print(f"Server found at: {server_path}")
```

### Test 3: GPU Compatibility

```python
import llcuda

compat = llcuda.check_gpu_compatibility()
print(f"Platform: {compat['platform']}")
print(f"GPU: {compat['gpu_name']}")
print(f"Compute Capability: {compat['compute_capability']}")
print(f"Compatible: {compat['compatible']}")
```

### Test 4: Full Workflow (Requires Model)

```python
import llcuda

engine = llcuda.InferenceEngine()

# This will:
# 1. Find llama-server using detection flow
# 2. Download model from HuggingFace (first time)
# 3. Start llama-server subprocess
# 4. Wait for health check
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    gpu_layers=15,
    ctx_size=1024,
    silent=False  # Show output for debugging
)

# Run inference
result = engine.infer("What is 2+2?", max_tokens=50)
print(result.text)
```

---

## Troubleshooting

### Issue 1: llama-server not found

**Symptom:**
```
FileNotFoundError: llama-server not found
```

**Check:**
```bash
# 1. Verify binaries directory
ls -lh /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/binaries/cuda12/

# 2. Check if file exists
file /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/binaries/cuda12/llama-server

# 3. Check permissions
ls -l /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/binaries/cuda12/llama-server
# Should show: -rwxr-xr-x (executable)
```

**Fix:**
```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda
chmod +x binaries/cuda12/llama-server
```

### Issue 2: Shared library not found

**Symptom:**
```
error while loading shared libraries: libllama.so: cannot open shared object file
```

**Check:**
```bash
# 1. Verify libraries exist
ls -lh /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/lib/

# 2. Check LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# 3. Check what libraries llama-server needs
ldd /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/binaries/cuda12/llama-server
```

**Fix:**
```bash
export LD_LIBRARY_PATH="/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/lib:$LD_LIBRARY_PATH"
```

### Issue 3: Server starts but crashes immediately

**Symptom:**
```
RuntimeError: llama-server process died unexpectedly
```

**Debug:**
```bash
# Run llama-server manually to see errors
cd /media/waqasm86/External1/Project-Nvidia/llcuda/llcuda
export LD_LIBRARY_PATH="$(pwd)/lib:$LD_LIBRARY_PATH"
./binaries/cuda12/llama-server --help

# If it crashes, check:
# 1. CUDA compatibility
nvidia-smi

# 2. Library dependencies
ldd binaries/cuda12/llama-server | grep "not found"

# 3. Run with verbose output in Python
engine.load_model(..., silent=False, verbose=True)
```

### Issue 4: Wrong compute capability

**Symptom:**
```
CUDA error: no kernel image is available for execution on the device
```

**Cause:** Binary was compiled for wrong compute capability (e.g., CC 7.5 on CC 5.0 GPU)

**Fix:**
```bash
# Rebuild with correct architecture
cmake -DCMAKE_CUDA_ARCHITECTURES="50" ...  # For your GPU
```

### Issue 5: Google Colab specific

**For Colab, llcuda will:**
1. Detect platform as "colab"
2. Try to download binaries from GitHub releases
3. Extract to package directory

**Manual intervention if auto-download fails:**
```python
# In Colab
!pip install llcuda

# If bootstrap fails, manually download
!wget https://github.com/waqasm86/llcuda/releases/download/v1.2.2/llcuda-binaries-cuda12.tar.gz
!mkdir -p /usr/local/lib/python3.12/dist-packages/llcuda/binaries/cuda12
!mkdir -p /usr/local/lib/python3.12/dist-packages/llcuda/lib
!tar -xzf llcuda-binaries-cuda12.tar.gz
!cp -r bin/* /usr/local/lib/python3.12/dist-packages/llcuda/binaries/cuda12/
!cp -r lib/* /usr/local/lib/python3.12/dist-packages/llcuda/lib/
!chmod +x /usr/local/lib/python3.12/dist-packages/llcuda/binaries/cuda12/*
```

---

## Summary

### Key Points

1. **llcuda uses auto-detection**: On import, it automatically finds and configures llama-server
2. **Priority search order**: Environment vars → Package binaries → Cache → System paths
3. **Integration is simple**: Just copy binaries to `llcuda/binaries/cuda12/` and libs to `llcuda/lib/`
4. **Bootstrap handles Colab**: Auto-downloads binaries in Google Colab
5. **Bug fixed**: The `stderr.read()` AttributeError is now fixed in `server.py:553`

### Integration Checklist

- [ ] Build llama.cpp with correct CUDA architecture
- [ ] Create `llcuda/binaries/cuda12/` directory
- [ ] Create `llcuda/lib/` directory
- [ ] Copy `llama-server` to `binaries/cuda12/`
- [ ] Copy all `.so*` files to `lib/`
- [ ] Make binaries executable (`chmod +x`)
- [ ] Test with Python import
- [ ] Verify server detection
- [ ] Test full inference workflow

### Quick Test

```bash
# Run integration script
./BUILD_AND_INTEGRATE.sh 940m

# Or run test directly
python3 test_llcuda_integration.py
```

---

**You're now ready to build and integrate CUDA 12 binaries into llcuda!**
