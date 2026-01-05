# llcuda v2.0 - T4-Only Refactoring Plan
## CUDA 12 Inference Backend for Unsloth on Google Colab

**Date**: January 6, 2026
**Target**: Google Colab Tesla T4 GPU (SM 7.5)
**Purpose**: CUDA-first backend inference tool integrated with Unsloth

---

## Executive Summary

Based on comprehensive analysis of:
1. Entire llcuda project (195 files)
2. T4 binary build process from Colab notebook
3. llama.cpp server architecture
4. GGUF format specification
5. Unsloth integration requirements

**Decision**: Refactor llcuda v2.0 to be T4-exclusive, removing all 940M support, and focusing on tensor core optimization (SM 7.5+).

---

## Key Insights from T4 Build Notebook

### T4 Binary Build Configuration
```bash
cmake -B build_cuda12_t4 -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75" \        # T4 only
    -DGGML_CUDA_FA=ON \                       # FlashAttention enabled
    -DGGML_CUDA_FA_ALL_QUANTS=ON \           # FA for all quant types
    -DGGML_CUDA_GRAPHS=ON \                   # CUDA Graphs
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \    # Multi-GPU batch
    -DLLAMA_BUILD_SERVER=ON \
    -DBUILD_SHARED_LIBS=ON
```

### T4 Binary Package Contents
```
package_t4/
├── bin/
│   ├── llama-server (6.5 MB)
│   ├── llama-cli
│   ├── llama-quantize
│   └── llama-embedding
└── lib/
    ├── libggml-base.so.0.9.5 (721 KB)
    ├── libggml-cpu.so.0.9.5 (949 KB)
    ├── libggml-cuda.so.0.9.5 (219 MB) ← Main CUDA kernels
    ├── libggml.so.0.9.5 (54 KB)
    ├── libllama.so.0.0.7621 (2.8 MB)
    └── libmtmd.so.0.0.7621 (868 KB)

Total: 264 MB
```

**Key Finding**: The CUDA kernels are already compiled with FlashAttention support for T4!

---

## Phase 1: Clean Up 940M-Specific Files

### Files to Remove (18 files)
1. `scripts/build_cuda12_geforce940m.sh`
2. `scripts/cmake_build_940m.sh`
3. `release-packages/llcuda-binaries-cuda12-940m.tar.gz` (26 MB)
4. `docs/Xubuntu22-Nvidia-Details.txt`
5. `README_FULL.md`
6. `README_SIMPLIFIED.md`
7. `ORGANIZATION_SUMMARY.md`
8. `ORGANIZATION_COMPLETE.md`
9. `WORK_COMPLETED_SUMMARY.md`
10. `V1.1.9_RELEASE_SUMMARY.md`
11. `examples/colab_test_v1.1.9.ipynb`
12. `examples/colab_test_v1.1.9.py`

### Files to Archive (Move to `archive/v1.x/`)
- Entire `release-info/` directory
- Old documentation variants

---

## Phase 2: Update Build System for T4-Only

### 2.1 Update CMakeLists.txt

**Current**:
```cmake
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
```

**New (T4-only)**:
```cmake
# Force Tesla T4 architecture (SM 7.5) - Google Colab standard
set(CMAKE_CUDA_ARCHITECTURES "75")
message(STATUS "Building for Tesla T4 (SM 7.5) - FlashAttention enabled")
```

### 2.2 Enable FlashAttention and Tensor Cores

Add to CMakeLists.txt:
```cmake
# Enable FlashAttention (requires SM 7.0+)
add_compile_definitions(GGML_CUDA_FA=ON)

# Enable CUDA Graphs for optimization
add_compile_definitions(GGML_CUDA_GRAPHS=ON)

# Tensor Core support
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
```

### 2.3 Simplify Build Scripts

**New**: `scripts/build_t4_native.sh`
```bash
#!/bin/bash
# Build llcuda v2.0 native extension for Tesla T4 (SM 7.5)

set -e

echo "=== Building llcuda v2.0 for Tesla T4 (SM 7.5) ==="

BUILD_DIR="build/native_t4"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake ../.. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DPython3_EXECUTABLE=$(which python3.11) \
    -DCMAKE_CUDA_FLAGS="-use_fast_math"

make -j$(nproc)

# Copy extension to package
cp llcuda_cpp*.so ../../

echo "✓ Build complete for Tesla T4"
echo "✓ FlashAttention enabled"
echo "✓ Tensor Core optimizations enabled"
```

---

## Phase 3: GGUF Format Integration

### 3.1 Create GGUF Parser

**New**: `llcuda/gguf_parser.py`
```python
"""
GGUF file format parser for llcuda v2.0
Based on: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import struct
from typing import Dict, List, Tuple, Any
from pathlib import Path

GGUF_MAGIC = 0x47474655  # "GGUF"
GGUF_VERSION = 3

class GGUFMetadataType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

class GGMLType:
    """Quantization types"""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    # K-quants (higher quality)
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    # ... more types

class GGUFReader:
    def __init__(self, path: str):
        self.path = Path(path)
        self.file = None
        self.metadata = {}
        self.tensors = {}

    def __enter__(self):
        self.file = open(self.path, 'rb')
        self._parse_header()
        self._parse_metadata()
        self._parse_tensor_info()
        return self

    def __exit__(self, *args):
        if self.file:
            self.file.close()

    def _parse_header(self):
        """Parse GGUF header"""
        magic = struct.unpack('<I', self.file.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {hex(magic)}")

        self.version = struct.unpack('<I', self.file.read(4))[0]
        if self.version != GGUF_VERSION:
            raise ValueError(f"Unsupported GGUF version: {self.version}")

        self.tensor_count = struct.unpack('<Q', self.file.read(8))[0]
        self.metadata_count = struct.unpack('<Q', self.file.read(8))[0]

    def _read_string(self) -> str:
        """Read GGUF string (length + data)"""
        length = struct.unpack('<Q', self.file.read(8))[0]
        return self.file.read(length).decode('utf-8')

    def _read_value(self, value_type: int) -> Any:
        """Read typed value"""
        if value_type == GGUFMetadataType.UINT8:
            return struct.unpack('<B', self.file.read(1))[0]
        elif value_type == GGUFMetadataType.INT8:
            return struct.unpack('<b', self.file.read(1))[0]
        elif value_type == GGUFMetadataType.UINT32:
            return struct.unpack('<I', self.file.read(4))[0]
        elif value_type == GGUFMetadataType.INT32:
            return struct.unpack('<i', self.file.read(4))[0]
        elif value_type == GGUFMetadataType.FLOAT32:
            return struct.unpack('<f', self.file.read(4))[0]
        elif value_type == GGUFMetadataType.UINT64:
            return struct.unpack('<Q', self.file.read(8))[0]
        elif value_type == GGUFMetadataType.INT64:
            return struct.unpack('<q', self.file.read(8))[0]
        elif value_type == GGUFMetadataType.FLOAT64:
            return struct.unpack('<d', self.file.read(8))[0]
        elif value_type == GGUFMetadataType.BOOL:
            return struct.unpack('<?', self.file.read(1))[0]
        elif value_type == GGUFMetadataType.STRING:
            return self._read_string()
        elif value_type == GGUFMetadataType.ARRAY:
            array_type = struct.unpack('<I', self.file.read(4))[0]
            array_len = struct.unpack('<Q', self.file.read(8))[0]
            return [self._read_value(array_type) for _ in range(array_len)]
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _parse_metadata(self):
        """Parse all metadata key-value pairs"""
        for _ in range(self.metadata_count):
            key = self._read_string()
            value_type = struct.unpack('<I', self.file.read(4))[0]
            value = self._read_value(value_type)
            self.metadata[key] = value

    def _parse_tensor_info(self):
        """Parse tensor metadata"""
        for _ in range(self.tensor_count):
            name = self._read_string()

            # Dimensions
            n_dims = struct.unpack('<I', self.file.read(4))[0]
            shape = [struct.unpack('<Q', self.file.read(8))[0] for _ in range(n_dims)]

            # Type and offset
            ggml_type = struct.unpack('<I', self.file.read(4))[0]
            offset = struct.unpack('<Q', self.file.read(8))[0]

            self.tensors[name] = {
                'shape': shape,
                'type': ggml_type,
                'offset': offset
            }

    def get_tensor_data(self, name: str) -> bytes:
        """Get raw tensor data"""
        if name not in self.tensors:
            raise KeyError(f"Tensor {name} not found")

        tensor_info = self.tensors[name]

        # Calculate tensor size based on type
        # ... implementation
```

---

## Phase 4: Update Bootstrap for T4-Only

### 4.1 Refactor `llcuda/_internal/bootstrap.py`

**Current**: Downloads binaries based on GPU detection
**New**: Always download T4 binaries, verify SM 7.5+

```python
def verify_gpu_compatibility():
    """Verify GPU is SM 7.5+ (Turing or newer)"""
    import subprocess

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        compute_cap = result.stdout.strip()
        major, minor = map(int, compute_cap.split('.'))

        if major < 7 or (major == 7 and minor < 5):
            raise RuntimeError(
                f"llcuda v2.0 requires GPU with SM 7.5+ (Turing or newer). "
                f"Your GPU: SM {major}.{minor}\\n"
                f"Recommended: Tesla T4, RTX 20xx, RTX 30xx, RTX 40xx, A100, H100"
            )

        return f"{major}.{minor}"

    except Exception as e:
        print(f"Warning: Could not verify GPU compatibility: {e}")
        return None

def download_t4_binaries():
    """Download T4 CUDA 12 binaries"""
    import requests
    import tarfile
    from pathlib import Path

    binary_url = "https://github.com/waqasm86/llcuda/releases/download/v2.0.0/llcuda-binaries-cuda12-t4.tar.gz"
    binary_dir = Path(__file__).parent.parent / "binaries" / "cuda12"

    if (binary_dir / "bin" / "llama-server").exists():
        return  # Already downloaded

    print("Downloading T4 binaries (264 MB)...")
    response = requests.get(binary_url, stream=True)

    tar_path = binary_dir / "llcuda-binaries-cuda12-t4.tar.gz"
    binary_dir.mkdir(parents=True, exist_ok=True)

    with open(tar_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting binaries...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(binary_dir)

    tar_path.unlink()  # Clean up tar file
    print("✓ T4 binaries ready")
```

---

## Phase 5: Integrate llama.cpp Server Patterns

### 5.1 Continuous Batching Support

**New**: `llcuda/server/batching.py`
```python
"""
Continuous batching implementation inspired by llama.cpp server
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class SlotState(Enum):
    IDLE = 0
    PROCESSING_PROMPT = 1
    GENERATING = 2

@dataclass
class RequestSlot:
    """Slot for request handling (like llama.cpp)"""
    id: int
    state: SlotState = SlotState.IDLE
    prompt_tokens: List[int] = None
    generated_tokens: List[int] = None
    n_decoded: int = 0
    max_tokens: int = 512

    def is_active(self) -> bool:
        return self.state != SlotState.IDLE

    def is_finished(self) -> bool:
        return self.n_decoded >= self.max_tokens

class ContinuousBatchScheduler:
    """
    Continuous batching scheduler
    Based on llama.cpp server architecture
    """
    def __init__(self, model, max_batch_size: int = 32):
        self.model = model
        self.max_batch_size = max_batch_size
        self.slots = [RequestSlot(id=i) for i in range(max_batch_size)]

    def step(self):
        """Single inference step with continuous batching"""
        # 1. Collect active slots
        active_slots = [s for s in self.slots if s.is_active()]

        if not active_slots:
            return

        # 2. Build batch
        batch_input_ids = []
        batch_positions = []

        for slot in active_slots:
            if slot.state == SlotState.PROCESSING_PROMPT:
                # Add prompt tokens
                tokens = slot.prompt_tokens[slot.n_decoded:slot.n_decoded + 32]
                batch_input_ids.extend(tokens)
                slot.n_decoded += len(tokens)
            elif slot.state == SlotState.GENERATING:
                # Add last generated token
                batch_input_ids.append(slot.generated_tokens[-1])
                slot.n_decoded += 1

        # 3. Run inference
        logits = self.model.forward(batch_input_ids)

        # 4. Sample and update slots
        # ... implementation
```

### 5.2 CUDA Graphs Support

**New**: `llcuda/cuda_graphs.py`
```python
"""
CUDA Graphs optimization for repeated inference
Inspired by llama.cpp GGML_CUDA_GRAPHS
"""

import torch

class CUDAGraphWrapper:
    """Capture and replay CUDA graphs for inference"""
    def __init__(self, model):
        self.model = model
        self.graph = None
        self.static_input = None
        self.static_output = None

    def capture(self, input_shape):
        """Capture CUDA graph for fixed input shape"""
        # Warm up
        for _ in range(3):
            _ = self.model.forward(torch.zeros(input_shape).cuda())

        # Capture graph
        self.static_input = torch.zeros(input_shape).cuda()
        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph):
            self.static_output = self.model.forward(self.static_input)

    def replay(self, input_tensor):
        """Replay graph with new input"""
        self.static_input.copy_(input_tensor)
        self.graph.replay()
        return self.static_output.clone()
```

---

## Phase 6: Unsloth Integration Points

### 6.1 Model Export from Unsloth

**Add to Unsloth** (via PR):
```python
# unsloth/models/llama.py
def save_pretrained_llcuda(
    model,
    save_directory: str,
    tokenizer=None,
    quantization: str = "nf4",
    **kwargs
):
    """
    Export Unsloth model to llcuda format

    Args:
        quantization: "nf4" (BitsAndBytes), "gguf:Q4_K_M", or "fp16"
    """
    from safetensors.torch import save_file
    import json

    # Merge LoRA adapters
    if hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
    else:
        merged = model

    # Save config
    config = merged.config.to_dict()
    config["_llcuda_quantization"] = quantization
    config["_llcuda_version"] = "2.0.0"

    with open(f"{save_directory}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save weights
    state_dict = merged.state_dict()

    if quantization == "nf4":
        # Use llcuda's NF4 quantizer
        from llcuda.quantization import quantize_nf4
        for name, param in state_dict.items():
            if "weight" in name and param.dim() >= 2:
                state_dict[name] = quantize_nf4(param)

    save_file(state_dict, f"{save_directory}/model.safetensors")

    # Save tokenizer
    if tokenizer:
        tokenizer.save_pretrained(save_directory)

    print(f"✓ Saved to {save_directory} in llcuda format ({quantization})")
```

### 6.2 Direct Loading in llcuda

**New**: `llcuda/models/unsloth_loader.py`
```python
"""
Load Unsloth-exported models directly
"""

from pathlib import Path
import json
from safetensors import safe_open

def load_unsloth_model(model_path: str):
    """
    Load Unsloth model exported via save_pretrained_llcuda()

    Returns:
        model: llcuda.Module instance
        config: Model configuration
    """
    model_path = Path(model_path)

    # Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)

    quantization = config.get("_llcuda_quantization", "fp16")
    model_type = config["model_type"]

    # Create model architecture
    if model_type == "llama":
        from llcuda.models.llama import LlamaModel, LlamaConfig
        model_config = LlamaConfig(**config)
        model = LlamaModel(model_config)
    elif model_type == "gemma":
        from llcuda.models.gemma import GemmaModel, GemmaConfig
        model_config = GemmaConfig(**config)
        model = GemmaModel(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load weights
    with safe_open(model_path / "model.safetensors", framework="pt") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            # Set parameter in model
            # ... implementation

    return model, config
```

---

## Phase 7: Documentation Update

### 7.1 Update README.md

```markdown
# llcuda v2.0 - CUDA 12 Inference Backend for Unsloth

**Tesla T4 optimized | FlashAttention enabled | Tensor Core accelerated**

llcuda v2.0 is a Python-first CUDA inference backend designed for Google Colab and Unsloth integration. It exposes low-level, quantization-aware GPU execution for maximum control.

## Why llcuda v2.0?

- **Tensor Core Optimized**: Built specifically for SM 7.5+ GPUs (T4, RTX 20xx+, A100, H100)
- **FlashAttention Native**: 2-3x faster long-context inference
- **NF4 Quantization**: Native support for Unsloth's quantization format
- **PyTorch-Style API**: Familiar tensor operations with direct CUDA control
- **Production Ready**: Continuous batching, CUDA graphs, multi-GPU support

## Quick Start (Google Colab)

```python
# Install
!pip install llcuda

# Import v2.0 native API
from llcuda.core import Tensor, matmul, get_device_properties

# Check GPU (should be T4)
props = get_device_properties(0)
print(f"{props.name} - SM {props.compute_capability_major}.{props.compute_capability_minor}")

# Create tensors
A = Tensor.zeros([1024, 1024], device=0)
B = Tensor.zeros([1024, 1024], device=0)
C = A @ B  # cuBLAS + Tensor Cores

# Load Unsloth model
from llcuda.models import load_unsloth_model
model, config = load_unsloth_model("username/my-unsloth-model")
```

## System Requirements

- **GPU**: Tesla T4, RTX 20xx series or newer (SM 7.5+)
- **CUDA**: 12.x
- **Python**: 3.11+
- **Platform**: Google Colab, Kaggle, local Linux

**Note**: llcuda v2.0 is optimized for T4. For older GPUs (Maxwell/Pascal), use llcuda v1.x.

## Architecture

llcuda v2.0 combines two modes:

1. **Native Tensor API** (NEW): Direct CUDA operations with custom kernels
2. **HTTP Server Mode** (v1.x compat): llama.cpp backend for GGUF models

...
```

---

## Implementation Checklist

### Week 1: Core Refactoring
- [ ] Remove all 940M-specific files
- [ ] Archive v1.x release history
- [ ] Update CMakeLists.txt for T4-only (SM 7.5)
- [ ] Add FlashAttention and CUDA Graphs flags
- [ ] Update bootstrap.py for T4-only downloads
- [ ] Test build on local system

### Week 2: GGUF Integration
- [ ] Implement GGUF parser
- [ ] Add quantization block structures
- [ ] Test GGUF loading with T4 binaries
- [ ] Integrate with existing model registry

### Week 3: Server Patterns
- [ ] Implement continuous batching
- [ ] Add CUDA Graphs wrapper
- [ ] Create slot-based scheduler
- [ ] Test with multiple concurrent requests

### Week 4: Unsloth Integration
- [ ] Create Unsloth export method
- [ ] Implement Unsloth model loader
- [ ] Add NF4 quantization support
- [ ] Test end-to-end workflow

### Week 5: Documentation & Release
- [ ] Update all documentation
- [ ] Create Colab demo notebook
- [ ] Write migration guide (v1.x → v2.0)
- [ ] Prepare PyPI package
- [ ] Create GitHub release

---

## Success Metrics

1. **Build Size**: ≤ 300 MB (T4 binaries + extension)
2. **Performance**: ≥ 50 tok/s on T4 for 3B models
3. **FlashAttention**: 2x speedup vs standard attention
4. **Compatibility**: Load Unsloth models directly
5. **Ease of Use**: `pip install llcuda` → working in Colab

---

## Next Steps

Execute Phase 1 immediately:
1. Create `archive/v1.x/` directory
2. Move 940M files to archive
3. Delete 940M binary package
4. Update CMakeLists.txt
5. Test clean build

**Status**: Ready to execute ✅
