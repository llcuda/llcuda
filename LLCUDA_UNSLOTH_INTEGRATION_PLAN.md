# llcuda-Unsloth Integration Plan
## Python-first CUDA Inference Backend for Unsloth Models

**Version:** 2.0.0
**Target:** Production-grade inference server with tight Unsloth integration
**Philosophy:** Expose low-level, quantization-aware GPU execution that production servers like TGI intentionally hide

---

## Executive Summary

llcuda will become the official inference backend for Unsloth-trained models, bridging the gap between research (Unsloth fine-tuning) and production deployment (high-throughput serving). Unlike existing solutions:

- **vLLM/TGI**: Hide low-level details, focus on ease of use
- **llama.cpp**: C++ first, Python bindings are secondary
- **llcuda**: Python-first with direct CUDA control, quantization-aware from ground up

### Key Differentiators

1. **PyTorch-style tensor API** - Natural for ML practitioners
2. **Custom NF4 kernels** - Native support for Unsloth's quantization format
3. **Flash Attention 2** - State-of-the-art long-context optimization
4. **Multi-GPU without paywall** - Open-source tensor/pipeline parallelism
5. **Educational transparency** - Well-documented kernels for learning

---

## Architecture Overview

### Current State (v1.2.2)
```
llcuda Python Package (54KB)
    ↓ (HTTP API)
llama-server binary
    ↓ (links to)
ggml-cuda kernels (libggml-cuda.so)
    ↓ (uses)
CUDA 12.x runtime
```

### Target State (v2.0.0)
```
llcuda Python Package
    ├── HTTP Server Mode (maintained for compatibility)
    │   └── llama-server binary (GGUF models)
    └── Native Tensor API (NEW)
        ├── llcuda.Tensor (PyTorch-style)
        ├── llcuda.Module (model abstraction)
        ├── Custom CUDA Kernels
        │   ├── NF4 quantization
        │   ├── Flash Attention 2
        │   ├── RoPE embeddings
        │   ├── MLP/SwiGLU
        │   └── Multi-GPU primitives
        └── Model Loaders
            ├── HuggingFace (Unsloth models)
            ├── GGUF (llama.cpp compatibility)
            └── Native llcuda format
```

---

## Phase 1: Core Tensor API (Weeks 1-3)

### 1.1 Memory Management
**Goal**: PyTorch-style CUDA memory abstraction

**Components**:
```python
# llcuda/core/tensor.py
class Tensor:
    """CUDA tensor with automatic memory management"""
    def __init__(self, data, dtype, device):
        self._data_ptr = None      # CUDA device pointer
        self._shape = None
        self._strides = None
        self._dtype = dtype
        self._device = device

    @property
    def device(self) -> Device:
        """Returns device (cuda:0, cuda:1, etc.)"""

    def to(self, device) -> 'Tensor':
        """Move tensor to different device"""

    def cpu(self) -> np.ndarray:
        """Copy to CPU as NumPy array"""

    def cuda(self) -> 'Tensor':
        """Move to CUDA device"""

# llcuda/core/device.py
class Device:
    """CUDA device abstraction"""
    @staticmethod
    def get_device_count() -> int:
        """Number of available GPUs"""

    @staticmethod
    def get_device_properties(device_id: int) -> DeviceProperties:
        """Compute capability, memory, name"""

    @staticmethod
    def set_device(device_id: int):
        """Set active CUDA device"""
```

**CUDA Implementation** (`llcuda/csrc/memory.cu`):
```cuda
// Memory pool implementation
class CUDAMemoryPool {
public:
    void* allocate(size_t size, int device_id);
    void deallocate(void* ptr);
    void reset();  // Clear pool

private:
    std::unordered_map<void*, size_t> allocations_;
    std::vector<std::pair<size_t, void*>> free_blocks_;
};

// Pinned memory for fast CPU-GPU transfers
void* allocate_pinned(size_t size);
void free_pinned(void* ptr);
```

**Testing**:
- Allocate/deallocate tensors (memory leak detection)
- Multi-device tensor creation
- CPU-GPU data transfers (correctness + bandwidth)
- Memory pool efficiency (fragmentation testing)

---

### 1.2 Basic Operations
**Goal**: Foundational tensor operations

**Operations**:
```python
# llcuda/ops/__init__.py
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication (uses cuBLAS)"""

def add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition"""

def mul(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication"""

def copy(src: Tensor, dst: Tensor):
    """Memory copy (same or cross-device)"""
```

**CUDA Implementation** (`llcuda/csrc/ops.cu`):
```cuda
// cuBLAS wrapper
void cublas_gemm(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    bool trans_a, bool trans_b
);

// Element-wise kernels
template<typename T>
__global__ void elementwise_add_kernel(
    const T* a, const T* b, T* out, int64_t size
);
```

**Testing**:
- Matrix multiplication correctness (vs NumPy)
- Large matrix performance (compare to cuBLAS)
- Mixed precision (fp16, bf16, fp32)

---

### 1.3 Pybind11 Bindings
**Goal**: Expose C++/CUDA code to Python

**Implementation** (`llcuda/csrc/bindings.cpp`):
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(llcuda_cpp, m) {
    m.doc() = "llcuda native CUDA operations";

    // Tensor class
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<py::array, std::string, int>())
        .def("to_numpy", &Tensor::to_numpy)
        .def("shape", &Tensor::shape)
        .def("dtype", &Tensor::dtype);

    // Device management
    m.def("get_device_count", &get_device_count);
    m.def("get_device_properties", &get_device_properties);

    // Operations
    m.def("matmul", &matmul, "Matrix multiplication");
    m.def("add", &elementwise_add, "Element-wise addition");
}
```

**Build System** (update `CMakeLists.txt`):
```cmake
cmake_minimum_required(VERSION 3.24)
project(llcuda CUDA CXX)

# CUDA setup
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Source files
file(GLOB_RECURSE CUDA_SOURCES "llcuda/csrc/*.cu")
file(GLOB_RECURSE CPP_SOURCES "llcuda/csrc/*.cpp")

# Build Python module
pybind11_add_module(llcuda_cpp
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

target_link_libraries(llcuda_cpp PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
)

# Compute architectures (Pascal to Ada Lovelace)
set_target_properties(llcuda_cpp PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
)
```

---

## Phase 2: Quantization Kernels (Weeks 4-6)

### 2.1 NF4 Quantization Format
**Goal**: Support Unsloth's primary quantization format

**Theory**: NormalFloat 4-bit quantization
- 16 quantization bins: [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 0.0911, 0.1848, 0.2844, 0.3949, 0.5251, 0.6962, 1.0, NaN]
- Information-theoretically optimal for neural network weights (assuming normal distribution)
- Per-block scaling factors (typically 64-element blocks)

**Block Structure** (`llcuda/csrc/quantization/nf4.h`):
```cpp
// NF4 quantization block (matches bitsandbytes)
struct NF4Block {
    half scale;              // FP16 scale factor
    uint8_t qvals[32];       // 64 4-bit values packed into 32 bytes

    static constexpr int BLOCK_SIZE = 64;
};

// Double quantization (quantize the scales)
struct NF4BlockDQ {
    half scale;              // Scale for the scales
    uint8_t block_scales[4]; // 8 quantized scales per block
    uint8_t qvals[32];       // 64 4-bit values
};
```

**Dequantization Kernel** (`llcuda/csrc/quantization/nf4_dequant.cu`):
```cuda
// NF4 lookup table (16 values)
__device__ const half NF4_QUANT_TABLE[16] = {
    -1.0f, -0.6962f, -0.5251f, -0.3949f,
    -0.2844f, -0.1848f, -0.0911f, 0.0f,
    0.0911f, 0.1848f, 0.2844f, 0.3949f,
    0.5251f, 0.6962f, 1.0f, 0.0f  // Last is NaN, treat as 0
};

__global__ void nf4_dequantize_kernel(
    const NF4Block* blocks,
    half* output,
    int n_blocks
) {
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    if (block_idx >= n_blocks) return;

    const NF4Block& block = blocks[block_idx];
    half scale = block.scale;

    // Each thread processes 2 values (packed in 1 byte)
    int byte_idx = thread_idx;
    if (byte_idx < 32) {
        uint8_t packed = block.qvals[byte_idx];

        uint8_t val0 = packed & 0x0F;
        uint8_t val1 = (packed >> 4) & 0x0F;

        int out_idx = block_idx * 64 + byte_idx * 2;
        output[out_idx] = __hmul(NF4_QUANT_TABLE[val0], scale);
        output[out_idx + 1] = __hmul(NF4_QUANT_TABLE[val1], scale);
    }
}
```

---

### 2.2 NF4 Matrix Multiplication
**Goal**: Efficient GEMM with quantized weights

**Approach**: Dequantize-on-the-fly during GEMM
- Block-wise dequantization in shared memory
- Use Tensor Cores (WMMA/MMA) for fp16 computation
- Minimize global memory bandwidth

**Kernel** (`llcuda/csrc/quantization/nf4_matmul.cu`):
```cuda
// Matrix multiplication: Y = X @ W^T
// X: [M, K] fp16 (activations)
// W: [N, K] nf4 (weights, stored as NF4Block[N, K/64])
// Y: [M, N] fp16 (output)

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void nf4_matmul_kernel(
    const half* X,           // [M, K]
    const NF4Block* W,       // [N, K/64]
    half* Y,                 // [M, N]
    int M, int N, int K
) {
    __shared__ half smem_X[BLOCK_M][BLOCK_K];
    __shared__ half smem_W[BLOCK_N][BLOCK_K];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Output tile coordinates
    int out_row = by * BLOCK_M + ty;
    int out_col = bx * BLOCK_N + tx;

    half acc = 0.0f;

    // Loop over K dimension in tiles
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // Load X tile to shared memory (coalesced)
        if (out_row < M && k_tile + tx < K) {
            smem_X[ty][tx] = X[out_row * K + k_tile + tx];
        }

        // Load and dequantize W tile
        if (out_col < N && k_tile + ty < K) {
            int block_idx = out_col * (K / 64) + (k_tile + ty) / 64;
            int in_block_idx = (k_tile + ty) % 64;

            const NF4Block& block = W[block_idx];
            uint8_t packed = block.qvals[in_block_idx / 2];
            uint8_t qval = (in_block_idx % 2 == 0)
                ? (packed & 0x0F)
                : ((packed >> 4) & 0x0F);

            smem_W[tx][ty] = __hmul(NF4_QUANT_TABLE[qval], block.scale);
        }

        __syncthreads();

        // Compute partial dot product using Tensor Cores
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            acc = __hfma(smem_X[ty][k], smem_W[tx][k], acc);
        }

        __syncthreads();
    }

    // Write output
    if (out_row < M && out_col < N) {
        Y[out_row * N + out_col] = acc;
    }
}
```

**Optimizations**:
- Use WMMA (Warp Matrix Multiply-Accumulate) for Tensor Core acceleration
- Tune BLOCK_M, BLOCK_N, BLOCK_K for different GPUs (Pascal: 16x16, Ampere: 32x32)
- Implement heuristics to choose kernel based on matrix dimensions

---

### 2.3 GGUF Quantization (Compatibility)
**Goal**: Maintain support for existing GGUF models

**Formats to Support**:
- Q4_0, Q4_1 (4-bit with/without bias)
- Q5_0, Q5_1 (5-bit)
- Q8_0 (8-bit)
- Q4_K_M, Q5_K_M (K-quants - higher quality)

**Implementation Strategy**:
- Reuse ggml's block structures (already in your llama.cpp build)
- Port dequantization kernels from `ggml-cuda/dequantize.cuh`
- Optimize matmul kernels (ggml's `mmq.cu` is reference)

---

## Phase 3: Flash Attention 2 (Weeks 7-9)

### 3.1 Flash Attention Theory
**Goal**: Understand and implement FA2 algorithm

**Key Concepts**:
- **Tiling**: Break Q, K, V into blocks that fit in SRAM
- **Online softmax**: Compute softmax incrementally without materializing full attention matrix
- **Recomputation**: Recompute attention in backward pass (memory-efficient)

**Algorithm** (simplified):
```
Input: Q [N, d], K [N, d], V [N, d]
Output: O [N, d]

1. Divide Q, K, V into blocks: Qi, Kj, Vj
2. For each Qi:
   a. Initialize Oi = 0, li = 0, mi = -∞
   b. For each Kj, Vj:
      - Load Qi, Kj, Vj to SRAM
      - Compute Sij = Qi @ Kj^T / √d
      - Compute local mi_new = max(mi, rowmax(Sij))
      - Compute Pij = exp(Sij - mi_new)
      - Update li = li * exp(mi - mi_new) + rowsum(Pij)
      - Update Oi = Oi * exp(mi - mi_new) + Pij @ Vj
      - Update mi = mi_new
   c. Output Oi = Oi / li
```

**Reference**: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

---

### 3.2 Forward Pass Implementation
**Goal**: Efficient attention with long context support

**Kernel** (`llcuda/csrc/attention/flash_attention.cu`):
```cuda
// Flash Attention forward pass
// Q, K, V: [batch, num_heads, seq_len, head_dim]
// Output: [batch, num_heads, seq_len, head_dim]

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__global__ void flash_attention_fwd_kernel(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    float* L,     // row sums (for backward pass)
    float* M,     // row maxes (for backward pass)
    int batch_size,
    int num_heads,
    int seq_len,
    float scale
) {
    extern __shared__ half smem[];

    // Partition shared memory
    half* smem_Q = smem;
    half* smem_K = smem + BLOCK_M * HEAD_DIM;
    half* smem_V = smem_K + BLOCK_N * HEAD_DIM;
    half* smem_S = smem_V + BLOCK_N * HEAD_DIM;  // Attention scores

    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block_idx = blockIdx.x;

    // Load Q block to shared memory
    int q_start = q_block_idx * BLOCK_M;
    load_tile_Q(smem_Q, Q, batch_idx, head_idx, q_start, seq_len, HEAD_DIM);

    // Initialize output accumulators
    float acc_O[BLOCK_M][HEAD_DIM] = {0};
    float max_scores[BLOCK_M];
    float sum_exp[BLOCK_M];

    #pragma unroll 1
    for (int i = 0; i < BLOCK_M; i++) {
        max_scores[i] = -INFINITY;
        sum_exp[i] = 0.0f;
    }

    // Loop over K, V blocks
    for (int kv_block = 0; kv_block < (seq_len + BLOCK_N - 1) / BLOCK_N; kv_block++) {
        int kv_start = kv_block * BLOCK_N;

        // Load K, V blocks
        load_tile_K(smem_K, K, batch_idx, head_idx, kv_start, seq_len, HEAD_DIM);
        load_tile_V(smem_V, V, batch_idx, head_idx, kv_start, seq_len, HEAD_DIM);
        __syncthreads();

        // Compute S = Q @ K^T
        matmul_qk(smem_S, smem_Q, smem_K, BLOCK_M, BLOCK_N, HEAD_DIM, scale);
        __syncthreads();

        // Apply causal mask if needed
        apply_causal_mask(smem_S, q_start, kv_start, BLOCK_M, BLOCK_N);

        // Online softmax update
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i++) {
            // Find max in this row
            float row_max = -INFINITY;
            for (int j = 0; j < BLOCK_N; j++) {
                row_max = fmaxf(row_max, __half2float(smem_S[i * BLOCK_N + j]));
            }

            // Update global max
            float old_max = max_scores[i];
            float new_max = fmaxf(old_max, row_max);

            // Rescale previous sum
            float scale_factor = expf(old_max - new_max);
            sum_exp[i] *= scale_factor;

            // Rescale previous output
            for (int d = 0; d < HEAD_DIM; d++) {
                acc_O[i][d] *= scale_factor;
            }

            // Compute P = exp(S - max)
            half smem_P[BLOCK_N];
            for (int j = 0; j < BLOCK_N; j++) {
                float exp_val = expf(__half2float(smem_S[i * BLOCK_N + j]) - new_max);
                smem_P[j] = __float2half(exp_val);
                sum_exp[i] += exp_val;
            }

            // Update output: O += P @ V
            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int j = 0; j < BLOCK_N; j++) {
                    sum += __half2float(smem_P[j]) * __half2float(smem_V[j * HEAD_DIM + d]);
                }
                acc_O[i][d] += sum;
            }

            max_scores[i] = new_max;
        }

        __syncthreads();
    }

    // Normalize output by sum_exp
    for (int i = 0; i < BLOCK_M; i++) {
        int q_idx = q_start + i;
        if (q_idx < seq_len) {
            for (int d = 0; d < HEAD_DIM; d++) {
                O[((batch_idx * num_heads + head_idx) * seq_len + q_idx) * HEAD_DIM + d]
                    = __float2half(acc_O[i][d] / sum_exp[i]);
            }
            L[q_idx] = sum_exp[i];
            M[q_idx] = max_scores[i];
        }
    }
}
```

**Launch Configuration**:
```python
# Optimal block sizes vary by GPU architecture
CONFIGS = {
    "sm_60": {"BLOCK_M": 16, "BLOCK_N": 16, "HEAD_DIM": 64},  # Pascal
    "sm_70": {"BLOCK_M": 32, "BLOCK_N": 32, "HEAD_DIM": 64},  # Volta
    "sm_75": {"BLOCK_M": 32, "BLOCK_N": 32, "HEAD_DIM": 64},  # Turing
    "sm_80": {"BLOCK_M": 64, "BLOCK_N": 64, "HEAD_DIM": 64},  # Ampere
    "sm_86": {"BLOCK_M": 64, "BLOCK_N": 64, "HEAD_DIM": 64},  # Ampere (RTX)
    "sm_89": {"BLOCK_M": 64, "BLOCK_N": 64, "HEAD_DIM": 128}, # Ada Lovelace
}
```

---

### 3.3 Backward Pass (Future)
**Note**: For inference-only v2.0, backward pass is not needed. Document for future training support.

---

## Phase 4: Model Architecture Support (Weeks 10-12)

### 4.1 Llama Architecture
**Goal**: Support Llama 1/2/3/3.1/3.2/3.3

**Model Structure**:
```python
# llcuda/models/llama.py
class LlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # GQA support
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None  # For long context

class LlamaModel(llcuda.Module):
    def __init__(self, config: LlamaConfig):
        self.embed_tokens = llcuda.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = llcuda.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = llcuda.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states, past_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

class LlamaDecoderLayer(llcuda.Module):
    def __init__(self, config: LlamaConfig):
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = llcuda.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = llcuda.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None):
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_kv

class LlamaAttention(llcuda.Module):
    def __init__(self, config: LlamaConfig):
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Grouped Query Attention (GQA) support
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = llcuda.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = llcuda.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = llcuda.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = llcuda.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rope = llcuda.RotaryEmbedding(self.head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None):
        bsz, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        Q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        Q, K = self.rope(Q, K, seq_len)

        # Repeat K, V for GQA
        if self.num_kv_groups > 1:
            K = K.repeat_interleave(self.num_kv_groups, dim=1)
            V = V.repeat_interleave(self.num_kv_groups, dim=1)

        # Flash Attention
        attn_output = llcuda.flash_attention(Q, K, V, causal=True)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, (K, V)

class LlamaMLP(llcuda.Module):
    def __init__(self, config: LlamaConfig):
        self.gate_proj = llcuda.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = llcuda.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = llcuda.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        # SwiGLU activation
        return self.down_proj(llcuda.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
```

**CUDA Kernels Needed**:
- RMSNorm: `llcuda/csrc/ops/rmsnorm.cu`
- RoPE: `llcuda/csrc/ops/rope.cu`
- SwiGLU (fused): `llcuda/csrc/ops/swiglu.cu`

---

### 4.2 Gemma Architecture
**Goal**: Support Gemma 2 and 3

**Key Differences from Llama**:
- GeGLU instead of SwiGLU (Gated GELU)
- Different RoPE scaling
- Gemma 2: Sliding window attention + global attention (interleaved layers)
- Gemma 3: Multi-head latent attention (MHLA) - query pooling

**Implementation**:
```python
# llcuda/models/gemma.py
class GemmaAttention(llcuda.Module):
    """Gemma 3 uses MHLA - query pooling for efficiency"""
    def __init__(self, config):
        # Query pooling parameters
        self.query_pooling_layer = config.query_pooling_layer if hasattr(config, 'query_pooling_layer') else None

        if self.query_pooling_layer is not None:
            # Gemma 3: Query pooling
            self.q_proj = llcuda.Linear(config.hidden_size, config.query_num_heads * config.head_dim, bias=False)
        else:
            # Gemma 2: Standard
            self.q_proj = llcuda.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)

    # ... rest similar to Llama with attention modifications

class GemmaMLP(llcuda.Module):
    """GeGLU activation"""
    def forward(self, hidden_states):
        # GeGLU instead of SwiGLU
        return self.down_proj(llcuda.gelu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
```

---

### 4.3 Mistral/Mixtral (MoE)
**Goal**: Support Mixture of Experts

**MoE Challenges**:
- Expert routing (top-k selection per token)
- Load balancing (auxiliary loss not needed for inference)
- Efficient expert execution (batch tokens by expert)

**Implementation**:
```python
# llcuda/models/mixtral.py
class MixtralSparseMoeBlock(llcuda.Module):
    def __init__(self, config):
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        self.gate = llcuda.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = [MixtralMLP(config) for _ in range(self.num_experts)]

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Flatten for routing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router logits
        router_logits = self.gate(hidden_states_flat)

        # Top-k selection
        routing_weights, selected_experts = llcuda.topk(router_logits, self.top_k, dim=-1)
        routing_weights = llcuda.softmax(routing_weights, dim=-1)

        # Prepare output
        output = llcuda.zeros_like(hidden_states_flat)

        # Process each expert (can be parallelized)
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            token_indices = (selected_experts == expert_idx).any(dim=-1).nonzero()

            if token_indices.numel() > 0:
                # Get tokens for this expert
                expert_input = hidden_states_flat[token_indices]

                # Expert computation
                expert_output = self.experts[expert_idx](expert_input)

                # Weight by routing weights
                expert_weights = routing_weights[token_indices][selected_experts[token_indices] == expert_idx]
                expert_output = expert_output * expert_weights.unsqueeze(-1)

                # Accumulate to output
                output[token_indices] += expert_output

        return output.view(batch_size, seq_len, hidden_dim)
```

**MoE Optimization**:
- Batch tokens by expert (reduce kernel launches)
- Optimize top-k selection (custom CUDA kernel)
- Parallelize expert computation across GPUs

---

## Phase 5: HuggingFace Integration (Weeks 13-14)

### 5.1 Model Loader
**Goal**: Load Unsloth models directly from HuggingFace

**Implementation**:
```python
# llcuda/models/loader.py
class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        quantization: Optional[str] = None,  # "nf4", "gguf:q4_k_m", etc.
        device: str = "cuda:0",
        trust_remote_code: bool = False,
        **kwargs
    ) -> llcuda.Module:
        """
        Load model from HuggingFace Hub or local path

        Examples:
            # Load Unsloth fine-tuned model
            model = llcuda.AutoModelForCausalLM.from_pretrained(
                "username/my-unsloth-model",
                quantization="nf4"
            )

            # Load from GGUF
            model = llcuda.AutoModelForCausalLM.from_pretrained(
                "TheBloke/Llama-2-7B-GGUF",
                quantization="gguf:Q4_K_M"
            )
        """
        # 1. Download/locate model files
        if os.path.isdir(model_name_or_path):
            model_path = model_name_or_path
        else:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_name_or_path)

        # 2. Load config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        # 3. Determine architecture
        model_type = config_dict["model_type"]

        if model_type == "llama":
            from .llama import LlamaConfig, LlamaModel
            config = LlamaConfig(**config_dict)
            model = LlamaModel(config)
        elif model_type == "gemma":
            from .gemma import GemmaConfig, GemmaModel
            config = GemmaConfig(**config_dict)
            model = GemmaModel(config)
        elif model_type == "mistral":
            from .mistral import MistralConfig, MistralModel
            config = MistralConfig(**config_dict)
            model = MistralModel(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 4. Load weights
        if quantization == "nf4":
            load_nf4_weights(model, model_path)
        elif quantization and quantization.startswith("gguf:"):
            quant_type = quantization.split(":")[1]
            load_gguf_weights(model, model_path, quant_type)
        else:
            load_fp16_weights(model, model_path)

        # 5. Move to device
        model.to(device)
        model.eval()

        return model

def load_nf4_weights(model: llcuda.Module, model_path: str):
    """Load NF4 quantized weights (from bitsandbytes or Unsloth)"""
    # Check for adapter_model.safetensors (LoRA adapters)
    adapter_path = os.path.join(model_path, "adapter_model.safetensors")

    if os.path.exists(adapter_path):
        # Unsloth model with LoRA adapters
        # Need to load base model + merge adapters
        load_base_and_merge_lora(model, model_path)
    else:
        # Fully merged NF4 model
        from safetensors import safe_open

        weights_path = os.path.join(model_path, "model.safetensors")
        with safe_open(weights_path, framework="pt") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)

                # Convert to NF4 format if needed
                if "weight" in name:
                    # Check if already quantized
                    if tensor.dtype == torch.uint8:
                        # Already NF4 quantized
                        set_parameter(model, name, tensor)
                    else:
                        # Quantize on-the-fly
                        quantized = quantize_nf4(tensor)
                        set_parameter(model, name, quantized)

def load_gguf_weights(model: llcuda.Module, model_path: str, quant_type: str):
    """Load GGUF quantized weights"""
    gguf_path = find_gguf_file(model_path, quant_type)

    # Parse GGUF file
    from .gguf_parser import GGUFReader
    reader = GGUFReader(gguf_path)

    # Map GGUF tensor names to model parameter names
    name_mapping = create_name_mapping(model)

    for tensor_name, tensor_data in reader.tensors.items():
        param_name = name_mapping.get(tensor_name)
        if param_name:
            set_parameter(model, param_name, tensor_data)
```

---

### 5.2 Tokenizer Integration
**Goal**: Use HuggingFace tokenizers seamlessly

```python
# llcuda/tokenization.py
from transformers import AutoTokenizer

class LLCUDATokenizer:
    """Wrapper around HuggingFace tokenizer"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(tokenizer)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """Apply chat template (critical for Unsloth compatibility)"""
        return self.tokenizer.apply_chat_template(messages, **kwargs)
```

---

### 5.3 Unsloth Export Method
**Goal**: Add `save_pretrained_llcuda()` to Unsloth

**Proposed Addition to Unsloth** (submit as PR):
```python
# In unsloth/models/llama.py (or wherever FastLanguageModel is defined)
def save_pretrained_llcuda(
    model,
    save_directory: str,
    tokenizer = None,
    quantization: str = "nf4",
    **kwargs
):
    """
    Export Unsloth model to llcuda format

    Args:
        save_directory: Output directory
        tokenizer: Tokenizer to save alongside model
        quantization: "nf4" or "fp16"
    """
    import os
    from safetensors.torch import save_file

    # 1. Merge LoRA adapters if present
    if hasattr(model, "merge_and_unload"):
        merged_model = model.merge_and_unload()
    else:
        merged_model = model

    # 2. Prepare state dict
    state_dict = merged_model.state_dict()

    # 3. Quantize if requested
    if quantization == "nf4":
        from llcuda.quantization import quantize_nf4
        for name, param in state_dict.items():
            if "weight" in name and param.dim() >= 2:
                state_dict[name] = quantize_nf4(param)

    # 4. Save model
    os.makedirs(save_directory, exist_ok=True)
    save_file(state_dict, os.path.join(save_directory, "model.safetensors"))

    # 5. Save config
    config = merged_model.config.to_dict()
    config["_llcuda_quantization"] = quantization
    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 6. Save tokenizer
    if tokenizer:
        tokenizer.save_pretrained(save_directory)

    print(f"Model saved to {save_directory} in llcuda format ({quantization})")
```

**Usage Example**:
```python
from unsloth import FastLanguageModel

# Fine-tune with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(...)
# ... training ...

# Export to llcuda
model.save_pretrained_llcuda("./my_llcuda_model", tokenizer, quantization="nf4")

# Load in llcuda
import llcuda
model = llcuda.AutoModelForCausalLM.from_pretrained("./my_llcuda_model")
```

---

## Phase 6: Multi-GPU Support (Weeks 15-17)

### 6.1 Tensor Parallelism
**Goal**: Split model across GPUs (vertical parallelism)

**Strategy**: Column/Row parallel for linear layers
- Q, K, V projections: column parallel (split output dim)
- O projection: row parallel (split input dim)
- MLP gate/up: column parallel
- MLP down: row parallel

**Implementation**:
```python
# llcuda/distributed/tensor_parallel.py
class ColumnParallelLinear(llcuda.Module):
    """Linear layer with column parallelism"""
    def __init__(self, in_features, out_features, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        self.out_features_per_partition = out_features // world_size

        # Each GPU holds a slice of the weight matrix
        self.weight = llcuda.Parameter(
            torch.empty(self.out_features_per_partition, in_features),
            device=f"cuda:{rank}"
        )

    def forward(self, x):
        # Input is replicated across all GPUs
        # Output is split along feature dimension
        output_parallel = llcuda.matmul(x, self.weight.t())
        return output_parallel

class RowParallelLinear(llcuda.Module):
    """Linear layer with row parallelism"""
    def __init__(self, in_features, out_features, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        self.in_features_per_partition = in_features // world_size

        self.weight = llcuda.Parameter(
            torch.empty(out_features, self.in_features_per_partition),
            device=f"cuda:{rank}"
        )

    def forward(self, x):
        # Input is split along feature dimension
        # Output needs all-reduce across GPUs
        output_parallel = llcuda.matmul(x, self.weight.t())

        # All-reduce to sum partial results
        output = all_reduce(output_parallel)
        return output
```

**Communication** (NCCL backend):
```cpp
// llcuda/csrc/distributed/nccl_ops.cu
#include <nccl.h>

class NCCLCommunicator {
public:
    void all_reduce(
        void* send_buf,
        void* recv_buf,
        size_t count,
        ncclDataType_t dtype,
        ncclRedOp_t op,
        cudaStream_t stream
    ) {
        ncclAllReduce(send_buf, recv_buf, count, dtype, op, comm_, stream);
    }

    void all_gather(
        void* send_buf,
        void* recv_buf,
        size_t count,
        ncclDataType_t dtype,
        cudaStream_t stream
    ) {
        ncclAllGather(send_buf, recv_buf, count, dtype, comm_, stream);
    }

private:
    ncclComm_t comm_;
};
```

---

### 6.2 Pipeline Parallelism
**Goal**: Split model by layers (horizontal parallelism)

**Strategy**: Assign layer ranges to different GPUs
- GPU 0: Layers 0-7
- GPU 1: Layers 8-15
- GPU 2: Layers 16-23
- GPU 3: Layers 24-31

**Micro-batching**: To improve GPU utilization, split batch into micro-batches

**Implementation**:
```python
# llcuda/distributed/pipeline_parallel.py
class PipelineParallelModel(llcuda.Module):
    def __init__(self, model, num_stages):
        self.num_stages = num_stages
        self.rank = get_rank()

        # Assign layers to this stage
        layers_per_stage = len(model.layers) // num_stages
        start_layer = self.rank * layers_per_stage
        end_layer = (self.rank + 1) * layers_per_stage

        if self.rank == 0:
            self.embedding = model.embed_tokens

        self.layers = model.layers[start_layer:end_layer]

        if self.rank == num_stages - 1:
            self.norm = model.norm
            self.lm_head = model.lm_head

    def forward(self, input_ids):
        # Stage 0: Embedding
        if self.rank == 0:
            hidden_states = self.embedding(input_ids)
        else:
            # Receive from previous stage
            hidden_states = recv_from_prev_stage()

        # All stages: Process assigned layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Send to next stage or return output
        if self.rank < self.num_stages - 1:
            send_to_next_stage(hidden_states)
            return None
        else:
            # Final stage: Output
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits
```

**Micro-batching for efficiency**:
```python
def pipeline_parallel_forward_with_microbatches(
    model, input_ids, num_microbatches
):
    # Split input into microbatches
    microbatches = torch.chunk(input_ids, num_microbatches, dim=0)

    outputs = []
    for microbatch in microbatches:
        output = model(microbatch)
        if output is not None:  # Only last stage returns output
            outputs.append(output)

    if outputs:
        return torch.cat(outputs, dim=0)
    return None
```

---

### 6.3 Hybrid Parallelism
**Goal**: Combine tensor and pipeline parallelism

**Example**: 8 GPUs with 2-way TP and 4-way PP
- TP group 0: [GPU 0, GPU 1] - Layers 0-7
- TP group 1: [GPU 2, GPU 3] - Layers 8-15
- TP group 2: [GPU 4, GPU 5] - Layers 16-23
- TP group 3: [GPU 6, GPU 7] - Layers 24-31

**Implementation**:
```python
# llcuda/distributed/hybrid_parallel.py
def setup_hybrid_parallelism(world_size, tp_size, pp_size):
    """
    Setup process groups for hybrid parallelism

    Args:
        world_size: Total number of GPUs
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
    """
    assert world_size == tp_size * pp_size

    rank = get_rank()

    # TP group: GPUs in same pipeline stage
    tp_rank = rank % tp_size
    pp_rank = rank // tp_size

    tp_group_ranks = [pp_rank * tp_size + i for i in range(tp_size)]
    tp_group = torch.distributed.new_group(tp_group_ranks)

    # PP group: GPUs across pipeline stages (same TP rank)
    pp_group_ranks = [i * tp_size + tp_rank for i in range(pp_size)]
    pp_group = torch.distributed.new_group(pp_group_ranks)

    return {
        "tp_group": tp_group,
        "pp_group": pp_group,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "tp_size": tp_size,
        "pp_size": pp_size,
    }
```

---

## Phase 7: Inference Server (Weeks 18-20)

### 7.1 Continuous Batching
**Goal**: High-throughput serving with dynamic batching

**Implementation** (similar to llama.cpp server):
```python
# llcuda/server/scheduler.py
class ContinuousBatchScheduler:
    def __init__(self, model, max_batch_size=32, max_seq_len=4096):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Slot management
        self.slots = [RequestSlot(i) for i in range(max_batch_size)]
        self.request_queue = queue.Queue()

    def add_request(self, request: InferenceRequest):
        self.request_queue.put(request)

    def step(self):
        # 1. Assign new requests to available slots
        while not self.request_queue.empty():
            slot = self._get_available_slot()
            if slot is None:
                break

            request = self.request_queue.get()
            slot.assign(request)

        # 2. Build batch with active slots
        batch_input_ids = []
        batch_positions = []

        for slot in self.slots:
            if slot.is_active():
                if slot.state == SlotState.PROCESSING_PROMPT:
                    # Add prompt tokens (up to max batch size)
                    tokens = slot.get_next_prompt_tokens(self.max_batch_size)
                    batch_input_ids.extend(tokens)
                    batch_positions.extend(range(slot.n_decoded, slot.n_decoded + len(tokens)))
                    slot.n_decoded += len(tokens)
                elif slot.state == SlotState.GENERATING:
                    # Add last generated token
                    batch_input_ids.append(slot.last_token)
                    batch_positions.append(slot.n_decoded)
                    slot.n_decoded += 1

        if not batch_input_ids:
            return

        # 3. Run inference
        input_ids = torch.tensor([batch_input_ids]).to("cuda")
        with torch.no_grad():
            logits = self.model(input_ids)

        # 4. Sample and update slots
        idx = 0
        for slot in self.slots:
            if slot.is_active():
                slot_logits = logits[0, idx]
                token_id = self._sample(slot_logits, slot.sampling_params)

                slot.add_token(token_id)

                if slot.is_finished():
                    slot.complete()

                idx += 1

    def _sample(self, logits, params):
        # Top-k, top-p, temperature sampling
        if params.temperature > 0:
            logits = logits / params.temperature

        if params.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, params.top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits[top_k_indices] = top_k_logits

        if params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > params.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1)

        return token_id.item()
```

---

### 7.2 OpenAI-Compatible API
**Goal**: Drop-in replacement for OpenAI API

```python
# llcuda/server/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="llcuda Inference Server")

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 512
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in request.messages],
        tokenize=False,
        add_generation_prompt=True
    )

    # Create inference request
    inference_request = InferenceRequest(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=request.stream
    )

    # Submit to scheduler
    scheduler.add_request(inference_request)

    # Wait for completion
    if request.stream:
        return StreamingResponse(
            stream_tokens(inference_request),
            media_type="text/event-stream"
        )
    else:
        output = await inference_request.wait_for_completion()

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output.text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": output.n_prompt_tokens,
                "completion_tokens": output.n_generated_tokens,
                "total_tokens": output.n_prompt_tokens + output.n_generated_tokens
            }
        )

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model", "owned_by": "llcuda"}]
    }
```

---

## Phase 8: Benchmarking & Optimization (Weeks 21-22)

### 8.1 Benchmark Suite
**Goal**: Measure performance against competitors

```python
# benchmarks/throughput_benchmark.py
import llcuda
import time
import numpy as np

def benchmark_throughput(
    model_name: str,
    batch_sizes: List[int],
    seq_lengths: List[int],
    num_iterations: int = 100
):
    # Load model
    model = llcuda.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = llcuda.LLCUDATokenizer.from_pretrained(model_name)

    results = []

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            # Prepare dummy input
            input_ids = torch.randint(0, 32000, (batch_size, seq_len)).cuda()

            # Warmup
            for _ in range(10):
                _ = model(input_ids)

            # Benchmark
            torch.cuda.synchronize()
            start = time.time()

            for _ in range(num_iterations):
                _ = model(input_ids)

            torch.cuda.synchronize()
            elapsed = time.time() - start

            throughput = (batch_size * num_iterations) / elapsed

            results.append({
                "batch_size": batch_size,
                "seq_len": seq_len,
                "throughput_samples_per_sec": throughput,
                "latency_ms": (elapsed / num_iterations) * 1000
            })

            print(f"Batch={batch_size}, SeqLen={seq_len}: "
                  f"{throughput:.2f} samples/s, {results[-1]['latency_ms']:.2f} ms/sample")

    return results

# benchmarks/latency_benchmark.py
def benchmark_latency(model_name: str, prompts: List[str]):
    model = llcuda.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = llcuda.LLCUDATokenizer.from_pretrained(model_name)

    ttfts = []  # Time to first token
    itls = []   # Inter-token latency

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)

        # Time to first token
        start = time.time()
        first_token = model.generate(input_ids, max_new_tokens=1)
        ttft = time.time() - start
        ttfts.append(ttft)

        # Inter-token latency (generate 50 tokens)
        start = time.time()
        output = model.generate(input_ids, max_new_tokens=50)
        total_time = time.time() - start
        itl = (total_time - ttft) / 49
        itls.append(itl)

    print(f"TTFT: {np.mean(ttfts)*1000:.2f} ms (±{np.std(ttfts)*1000:.2f})")
    print(f"ITL: {np.mean(itls)*1000:.2f} ms (±{np.std(itls)*1000:.2f})")

# benchmarks/memory_benchmark.py
def benchmark_memory(model_name: str, batch_sizes: List[int]):
    for batch_size in batch_sizes:
        torch.cuda.reset_peak_memory_stats()

        model = llcuda.AutoModelForCausalLM.from_pretrained(model_name)
        input_ids = torch.randint(0, 32000, (batch_size, 512)).cuda()

        _ = model(input_ids)

        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"Batch={batch_size}: Peak memory = {peak_memory_mb:.2f} MB")

# benchmarks/multi_gpu_benchmark.py
def benchmark_multi_gpu_scaling(
    model_name: str,
    num_gpus_list: List[int],
    batch_size: int = 8
):
    single_gpu_throughput = None

    for num_gpus in num_gpus_list:
        # Initialize distributed
        llcuda.distributed.init_process_group(world_size=num_gpus)

        # Load model with parallelism
        model = llcuda.AutoModelForCausalLM.from_pretrained(
            model_name,
            tensor_parallel_size=num_gpus
        )

        # Benchmark
        throughput = run_throughput_benchmark(model, batch_size)

        if single_gpu_throughput is None:
            single_gpu_throughput = throughput

        speedup = throughput / single_gpu_throughput
        efficiency = speedup / num_gpus

        print(f"GPUs={num_gpus}: {throughput:.2f} tok/s, "
              f"Speedup={speedup:.2f}x, Efficiency={efficiency*100:.1f}%")
```

---

### 8.2 Comparison Targets
**Benchmark against**:
1. **vLLM** (PagedAttention, SOTA for serving)
2. **TGI** (HuggingFace Text Generation Inference)
3. **llama.cpp server** (Your current backend)
4. **PyTorch native** (Baseline)

**Metrics to Report**:
- Throughput: tokens/sec at various batch sizes
- Latency: TTFT and ITL (P50, P95, P99)
- Memory: VRAM usage vs batch size
- Multi-GPU: Scaling efficiency (1, 2, 4, 8 GPUs)

---

## Phase 9: Documentation & Community (Weeks 23-24)

### 9.1 Documentation Structure
```
docs/
├── index.md                          # Landing page
├── getting-started/
│   ├── installation.md
│   ├── quick-start.md
│   └── unsloth-integration.md        # NEW: Unsloth-specific guide
├── guides/
│   ├── model-loading.md
│   ├── quantization.md
│   ├── multi-gpu.md
│   └── serving.md
├── api-reference/
│   ├── tensor.md
│   ├── models.md
│   ├── distributed.md
│   └── server.md
├── kernel-documentation/              # NEW: Educational content
│   ├── nf4-quantization.md
│   ├── flash-attention.md
│   ├── rope-embeddings.md
│   └── tensor-parallelism.md
├── benchmarks/
│   ├── throughput.md
│   ├── latency.md
│   └── comparisons.md
└── contributing.md
```

---

### 9.2 Example Notebooks
**Create Jupyter notebooks**:
1. `01_basic_inference.ipynb` - Load model, run inference
2. `02_unsloth_export.ipynb` - Fine-tune with Unsloth, export to llcuda
3. `03_quantization.ipynb` - Compare NF4 vs GGUF quantization
4. `04_multi_gpu.ipynb` - Scale to multiple GPUs
5. `05_custom_kernels.ipynb` - Write custom CUDA kernels

---

### 9.3 Unsloth Partnership
**Outreach to Unsloth team**:
1. Present llcuda capabilities (share benchmarks)
2. Propose integration (`save_pretrained_llcuda()` method)
3. Offer to maintain integration code
4. Request inclusion in Unsloth docs

**Draft email**:
```
Subject: llcuda - Open-Source Inference Backend for Unsloth Models

Hi Unsloth Team,

I've developed llcuda, a Python-first CUDA inference backend specifically designed
for deploying Unsloth-trained models to production. Key features:

- Native NF4 quantization support (custom CUDA kernels)
- Flash Attention 2 for long context
- Multi-GPU tensor/pipeline parallelism (no paywall)
- OpenAI-compatible API for serving
- PyTorch-style API for researchers

I'd love to contribute a save_pretrained_llcuda() export method to Unsloth,
making llcuda the official inference backend for production deployment.

Benchmarks show competitive performance with vLLM/TGI while exposing low-level
details that production servers typically hide - perfect for researchers
experimenting with quantization-aware inference.

Would you be open to discussing integration?

Best regards,
[Your Name]
```

---

## Summary: Implementation Timeline

| Phase | Weeks | Deliverable | Status |
|-------|-------|-------------|--------|
| 1. Core Tensor API | 1-3 | llcuda.Tensor, memory management, cuBLAS ops | Pending |
| 2. Quantization | 4-6 | NF4 kernels, GGUF support | Pending |
| 3. Flash Attention | 7-9 | FA2 forward pass | Pending |
| 4. Model Architectures | 10-12 | Llama, Gemma, Mistral/Mixtral | Pending |
| 5. HuggingFace Integration | 13-14 | Model loader, tokenizer | Pending |
| 6. Multi-GPU | 15-17 | Tensor + pipeline parallelism | Pending |
| 7. Inference Server | 18-20 | Continuous batching, OpenAI API | Pending |
| 8. Benchmarking | 21-22 | Throughput, latency, memory tests | Pending |
| 9. Documentation | 23-24 | Docs, notebooks, Unsloth partnership | Pending |

**Total Duration**: ~6 months for complete v2.0 implementation

---

## Success Criteria

### Technical Metrics
- [ ] NF4 matmul within 5% of bitsandbytes performance
- [ ] Flash Attention 2 within 10% of official implementation
- [ ] Throughput competitive with vLLM (within 20%)
- [ ] Multi-GPU scaling efficiency >80% (up to 4 GPUs)
- [ ] Memory usage <10% higher than competitors

### Adoption Metrics
- [ ] Unsloth integration merged (PR accepted)
- [ ] 100+ GitHub stars within 3 months
- [ ] 10+ community contributions
- [ ] Featured in Unsloth documentation
- [ ] 5+ research papers cite llcuda

### Educational Impact
- [ ] 1000+ views on kernel documentation
- [ ] 10+ blog posts/tutorials from community
- [ ] Used in university courses

---

## Risk Mitigation

### Technical Risks
1. **NF4 kernel performance**: If custom kernels underperform, can fall back to bitsandbytes
2. **Flash Attention complexity**: Reference implementations available, can port from Triton
3. **Multi-GPU bugs**: Extensive testing with synthetic data before production

### Partnership Risks
1. **Unsloth team unresponsive**: Proceed independently, create converter tool
2. **API incompatibility**: Maintain compatibility layer
3. **Breaking changes in Unsloth**: Version pinning, adapters

### Community Risks
1. **Low adoption**: Focus on quality benchmarks, documentation
2. **Contributor burnout**: Start with solo development, grow organically
3. **Competition**: Differentiate through education + Unsloth integration

---

## Next Steps

1. **Immediate** (This week):
   - Set up development environment
   - Implement basic Tensor class
   - Write first CUDA kernel (matmul)

2. **Short-term** (Next month):
   - Complete Phase 1 (Tensor API)
   - Begin NF4 quantization kernels
   - Draft documentation structure

3. **Mid-term** (3 months):
   - Complete quantization + Flash Attention
   - Support Llama models
   - Initial benchmarks

4. **Long-term** (6 months):
   - Full v2.0 release
   - Unsloth partnership established
   - Production deployments

---

**This plan positions llcuda as the bridge between research (Unsloth fine-tuning) and production (high-throughput inference), with a focus on transparency, education, and Python-first design.**
