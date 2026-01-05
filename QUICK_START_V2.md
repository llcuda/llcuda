# llcuda v2.0 Quick Start Guide

## What is llcuda v2.0?

llcuda v2.0 transforms llcuda from a lightweight HTTP wrapper into a **Python-first CUDA inference backend** with native tensor operations and custom kernels. It's designed for tight Unsloth integration and production LLM serving.

**Key Philosophy**: Expose low-level, quantization-aware GPU execution that production servers like TGI intentionally hide.

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- CUDA 12.x
- CMake 3.24+
- GPU with Compute Capability 5.0+ (Maxwell or newer)

### Build from Source

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda

# Build native extension
./build_native.sh

# Verify installation
python3.11 -c "import llcuda_cpp; print('Success!')"
```

---

## Quick Examples

### 1. Device Management

```python
from llcuda.core import get_device_count, get_device_properties

# Check available GPUs
num_gpus = get_device_count()
print(f"Found {num_gpus} GPU(s)")

# Get device info
props = get_device_properties(0)
print(f"Device: {props.name}")
print(f"Compute: SM {props.compute_capability_major}.{props.compute_capability_minor}")
print(f"Memory: {props.total_memory / 1024**3:.2f} GB")
```

**Output**:
```
Found 1 GPU(s)
Device: NVIDIA GeForce 940M
Compute: SM 5.2
Memory: 2.00 GB
```

---

### 2. Tensor Creation

```python
from llcuda.core import Tensor, DType

# Create tensor on GPU 0
t = Tensor([1024, 1024], dtype=DType.Float32, device=0)

print(f"Shape: {t.shape}")
print(f"Elements: {t.numel()}")
print(f"Memory: {t.nbytes() / 1024**2:.2f} MB")

# Create zero-filled tensor
zeros = Tensor.zeros([512, 512], dtype=DType.Float16, device=0)
```

**Output**:
```
Shape: [1024, 1024]
Elements: 1048576
Memory: 4.00 MB
```

---

### 3. Matrix Multiplication

```python
from llcuda.core import Tensor, DType, matmul
import time

# Create tensors
A = Tensor.zeros([2048, 2048], dtype=DType.Float32, device=0)
B = Tensor.zeros([2048, 2048], dtype=DType.Float32, device=0)

# Matrix multiplication (cuBLAS)
start = time.time()
C = matmul(A, B)
elapsed = time.time() - start

print(f"Result shape: {C.shape}")
print(f"Time: {elapsed*1000:.2f} ms")

# Using @ operator
D = A @ B  # Same as matmul(A, B)
```

---

### 4. Multi-GPU Operations

```python
from llcuda.core import Tensor, get_device_count

# Check for multiple GPUs
if get_device_count() >= 2:
    # Create tensor on GPU 0
    t0 = Tensor.zeros([100, 100], device=0)

    # Transfer to GPU 1
    t1 = t0.to(1)

    print(f"Original device: {t0.device}")
    print(f"New device: {t1.device}")
```

---

## Current Capabilities (Phase 1)

### ✅ Implemented
- [x] CUDA device management
- [x] Tensor creation and memory management
- [x] cuBLAS matrix multiplication (FP32, FP16)
- [x] Multi-GPU tensor transfer
- [x] PyTorch-style Python API
- [x] Automatic GPU architecture detection

### 🚧 In Progress
- [ ] NumPy integration (`from_numpy()`, `to_numpy()`)
- [ ] NF4 quantization kernels (Phase 2)
- [ ] Flash Attention 2 (Phase 3)
- [ ] Model architecture support (Phase 4)

### 📅 Planned
- [ ] HuggingFace model loader (Phase 5)
- [ ] Multi-GPU parallelism (Phase 6)
- [ ] Inference server (Phase 7)
- [ ] Benchmarks (Phase 8)
- [ ] Unsloth integration (Phase 9)

---

## API Reference

### Device Class

```python
from llcuda.core import Device

# Static methods
Device.get_device_count() -> int
Device.get_device_properties(device_id: int) -> DeviceProperties
Device.set_device(device_id: int)
Device.get_device() -> int
Device.synchronize(device_id: int = -1)
Device.get_free_memory(device_id: int = -1) -> int
Device.get_total_memory(device_id: int = -1) -> int
```

### Tensor Class

```python
from llcuda.core import Tensor, DType

# Constructor
Tensor(shape: List[int], dtype: DType = DType.Float32, device: int = 0)

# Factory methods
Tensor.zeros(shape, dtype=DType.Float32, device=0)
Tensor.ones(shape, dtype=DType.Float32, device=0)  # Coming soon
Tensor.from_numpy(arr, device=0)  # Coming soon

# Properties
tensor.shape -> List[int]
tensor.dtype -> DType
tensor.device -> int
tensor.ndim -> int
tensor.numel() -> int
tensor.element_size() -> int
tensor.nbytes() -> int

# Methods
tensor.to(device_id: int) -> Tensor
tensor.is_contiguous() -> bool
tensor.contiguous() -> Tensor

# Operators
tensor1 @ tensor2  # Matrix multiplication
```

### DType Enum

```python
from llcuda.core import DType

DType.Float32   # 32-bit float
DType.Float16   # 16-bit float (half precision)
DType.BFloat16  # 16-bit bfloat16 (Google Brain format)
DType.Int32     # 32-bit integer
DType.Int64     # 64-bit integer
DType.UInt8     # 8-bit unsigned integer
```

### Operations

```python
from llcuda.core import matmul

# Matrix multiplication
result = matmul(A, B)  # A: [M, K], B: [K, N] -> result: [M, N]

# Batched matrix multiplication (coming soon)
# result = batched_matmul(A, B)  # A: [B, M, K], B: [B, K, N]
```

---

## Testing

### Run All Tests
```bash
python3.11 -m pytest tests/test_tensor_api.py -v
```

### Run Specific Test Class
```bash
python3.11 -m pytest tests/test_tensor_api.py::TestDevice -v
python3.11 -m pytest tests/test_tensor_api.py::TestTensor -v
python3.11 -m pytest tests/test_tensor_api.py::TestMatmul -v
```

### Run with Output
```bash
python3.11 -m pytest tests/test_tensor_api.py -v -s
```

---

## Troubleshooting

### Issue: `undefined symbol: fatbinData`

**Cause**: CUDA device code linking issue.

**Solution**:
```bash
# Use shared CUDA libraries instead of static
sed -i 's/cudart_static/cudart/g' CMakeLists.txt
sed -i 's/cublas_static/cublas/g' CMakeLists.txt
sed -i 's/cublasLt_static/cublasLt/g' CMakeLists.txt

# Rebuild
rm -rf build/native/*
./build_native.sh
```

### Issue: `ImportError: No module named llcuda_cpp`

**Cause**: Extension not built or not in Python path.

**Solution**:
```bash
# Check if .so file exists
ls -lh llcuda_cpp*.so

# If missing, rebuild
./build_native.sh

# Verify Python can find it
python3.11 -c "import sys; print(sys.path)"
```

### Issue: `CUDA_ERROR_OUT_OF_MEMORY`

**Cause**: Trying to allocate more memory than available on GPU.

**Solution**:
```python
from llcuda.core import Device

# Check available memory
free = Device.get_free_memory(0)
print(f"Free: {free / 1024**3:.2f} GB")

# Reduce tensor size or use quantization
```

---

## Performance Tips

### 1. Use Appropriate Data Types
```python
# FP16 is 2x faster on modern GPUs (SM 7.0+)
t_fp16 = Tensor([1024, 1024], dtype=DType.Float16)

# FP32 for compatibility
t_fp32 = Tensor([1024, 1024], dtype=DType.Float32)
```

### 2. Minimize Device Transfers
```python
# BAD: Multiple transfers
for i in range(100):
    t = Tensor.zeros([10, 10], device=0)
    t_cpu = t.cpu()  # Slow!

# GOOD: Batch operations on GPU
t = Tensor.zeros([100, 10, 10], device=0)
# Process on GPU
# Transfer once at the end
```

### 3. Leverage cuBLAS
```python
# Matrix multiplication is optimized with cuBLAS
# Use large batch sizes for best performance
A = Tensor.zeros([4096, 4096], dtype=DType.Float16)
B = Tensor.zeros([4096, 4096], dtype=DType.Float16)
C = A @ B  # Very fast!
```

---

## Roadmap to Unsloth Integration

### Current Status: Phase 1 Complete ✅

### Next Phases:

**Phase 2 (Weeks 4-6)**: NF4 Quantization
- Custom NF4 dequantization kernels
- Quantized matrix multiplication
- Compatibility with Unsloth's quantization format

**Phase 3 (Weeks 7-9)**: Flash Attention 2
- Tiled attention algorithm
- Long context support (up to 128K tokens)
- 2-3x speedup over standard attention

**Phase 4 (Weeks 10-12)**: Model Architectures
- Llama (1, 2, 3, 3.1, 3.2, 3.3)
- Gemma (2, 3)
- Mistral/Mixtral (MoE support)

**Phase 5 (Weeks 13-14)**: HuggingFace Integration
- Direct model loading from HuggingFace Hub
- Automatic architecture detection
- Tokenizer integration

**Phase 6 (Weeks 15-17)**: Multi-GPU
- Tensor parallelism (split model across GPUs)
- Pipeline parallelism (layer-wise distribution)
- Hybrid parallelism

**Phase 7 (Weeks 18-20)**: Inference Server
- Continuous batching
- OpenAI-compatible API
- Multi-user serving

**Phase 8 (Weeks 21-22)**: Benchmarking
- vs vLLM, TGI, llama.cpp
- Throughput, latency, memory benchmarks

**Phase 9 (Weeks 23-24)**: Unsloth Partnership
- `model.save_pretrained_llcuda()` export method
- Official integration
- Documentation and examples

---

## Contributing

llcuda v2.0 is under active development. Areas where contributions are welcome:

1. **Kernel Optimization**: Improve CUDA kernel performance
2. **Model Support**: Add new architectures
3. **Testing**: Expand test coverage
4. **Documentation**: Improve guides and examples
5. **Benchmarking**: Performance comparisons

---

## Resources

- [Full Integration Plan](LLCUDA_UNSLOTH_INTEGRATION_PLAN.md)
- [Phase 1 Summary](PHASE1_COMPLETE.md)
- [GitHub Repository](https://github.com/waqasm86/llcuda)
- [Unsloth Repository](https://github.com/unslothai/unsloth)

---

## License

MIT License - See LICENSE file for details

---

**Last Updated**: January 5, 2026
**Version**: 2.0.0-alpha
**Status**: Phase 1 Complete
