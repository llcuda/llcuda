# llcuda v2.2.0 - Tutorial Notebooks

Complete tutorial notebook series for llcuda on Kaggle with dual Tesla T4 GPUs.

---

## Overview

This directory contains 10 comprehensive tutorial notebooks covering all aspects of llcuda v2.2.0:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLCUDA TUTORIAL PATH                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   FUNDAMENTALS          INTERMEDIATE          ADVANCED          │
│   ────────────          ────────────          ────────          │
│   01 Quick Start        04 GGUF               07 OpenAI API     │
│   02 Server Setup       05 Unsloth            08 NCCL/PyTorch   │
│   03 Multi-GPU          06 Split-GPU          09 Large Models   │
│                                               10 Complete       │
│                                                                 │
│   ──────────────────────────────────────────────────────────▶   │
│          Beginner                              Expert           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Notebook Index

### Beginner Level

| # | Notebook | Description | Time |
|---|----------|-------------|------|
| 01 | [Quick Start](01-quickstart-llcuda-v2.2.0.ipynb) | Get started in 5 minutes with basic setup and first query | 5 min |
| 02 | [Llama Server Setup](02-llama-server-setup-llcuda-v2.2.0.ipynb) | Deep dive into server configuration and lifecycle management | 15 min |
| 03 | [Multi-GPU Inference](03-multi-gpu-inference-llcuda-v2.2.0.ipynb) | Use both T4 GPUs with tensor-split for larger models | 20 min |

### Intermediate Level

| # | Notebook | Description | Time |
|---|----------|-------------|------|
| 04 | [GGUF Quantization](04-gguf-quantization-llcuda-v2.2.0.ipynb) | Understanding GGUF format, K-quants, I-quants, and parsing | 20 min |
| 05 | [Unsloth Integration](05-unsloth-integration-llcuda-v2.2.0.ipynb) | Fine-tune with Unsloth → export GGUF → deploy with llcuda | 30 min |
| 06 | [Split-GPU Graphistry](06-split-gpu-graphistry-llcuda-v2.2.0.ipynb) | LLM on GPU 0 + RAPIDS/Graphistry on GPU 1 | 30 min |

### Advanced Level

| # | Notebook | Description | Time |
|---|----------|-------------|------|
| 07 | [OpenAI API Client](07-openai-api-client-llcuda-v2.2.0.ipynb) | Use OpenAI SDK with llama-server for drop-in replacement | 15 min |
| 08 | [NCCL PyTorch](08-nccl-pytorch-llcuda-v2.2.0.ipynb) | NCCL for distributed PyTorch workloads alongside llcuda | 25 min |
| 09 | [Large Models](09-large-models-kaggle-llcuda-v2.2.0.ipynb) | Run 70B models on Kaggle dual T4 with I-quants | 30 min |
| 10 | [Complete Workflow](10-complete-workflow-llcuda-v2.2.0.ipynb) | End-to-end: Unsloth → GGUF → Multi-GPU → Production | 45 min |

### Advanced Visualization

| # | Notebook | Description | Time |
|---|----------|-------------|------|
| 11 | [GGUF Neural Network Visualization](11-gguf-neural-network-graphistry-visualization.ipynb) | **MOST IMPORTANT**: Visualize GGUF model architecture as interactive Graphistry graphs | 60 min |

---

## Detailed Descriptions

### 01 - Quick Start

**File:** `01-quickstart-llcuda-v2.2.0.ipynb`

Get started with llcuda in just 5 minutes. This notebook covers:

- Installing llcuda
- Downloading a GGUF model
- Starting the llama-server
- Making your first chat completion
- Cleaning up resources

**Prerequisites:** None  
**VRAM Required:** 3-5 GB (single T4)

```python
# Sample from notebook
from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(model_path="model.gguf")
server = ServerManager()
server.start_with_config(config)
```

---

### 02 - Llama Server Setup

**File:** `02-llama-server-setup-llcuda-v2.2.0.ipynb`

Deep dive into server configuration and management:

- ServerConfig parameter reference
- Server lifecycle (start → ready → stop)
- Health checking and monitoring
- Log access and debugging
- Multiple server configurations

**Prerequisites:** Complete notebook 01  
**VRAM Required:** 5-8 GB (single T4)

---

### 03 - Multi-GPU Inference

**File:** `03-multi-gpu-inference-llcuda-v2.2.0.ipynb`

Harness both Kaggle T4 GPUs for larger models:

- GPU detection and VRAM monitoring
- tensor-split configuration
- split-mode options (layer vs row)
- Performance optimization
- Memory management

**Prerequisites:** Complete notebooks 01-02  
**VRAM Required:** 15-25 GB (dual T4)

**Key Concept:**
```python
# Split model across both GPUs
config = ServerConfig(
    model_path="model.gguf",
    tensor_split="0.5,0.5",  # 50% each GPU
    split_mode="layer",
)
```

---

### 04 - GGUF Quantization

**File:** `04-gguf-quantization-llcuda-v2.2.0.ipynb`

Master the GGUF format and quantization:

- GGUF file structure
- K-quants (Q4_K_M, Q5_K_M, Q6_K)
- I-quants (IQ3_XS, IQ2_XXS)
- VRAM estimation
- Quality vs size trade-offs
- Using GGUFParser

**Prerequisites:** Complete notebooks 01-03  
**VRAM Required:** Varies by model

---

### 05 - Unsloth Integration

**File:** `05-unsloth-integration-llcuda-v2.2.0.ipynb`

Complete Unsloth fine-tuning → llcuda deployment workflow:

- Loading models with Unsloth
- LoRA fine-tuning basics
- Exporting to GGUF format
- Deploying with llcuda
- Performance comparison

**Prerequisites:** Complete notebooks 01-04  
**VRAM Required:** 10-15 GB for training, 5-8 GB for inference

**Workflow:**
```
Unsloth (Train) → GGUF (Export) → llcuda (Deploy)
```

---

### 06 - Split-GPU Graphistry

**File:** `06-split-gpu-graphistry-llcuda-v2.2.0.ipynb`

Advanced architecture: LLM + RAPIDS on separate GPUs:

- Split-GPU architecture design
- LLM on GPU 0 (llama-server)
- RAPIDS/Graphistry on GPU 1
- Inter-GPU coordination
- LLM-powered graph analytics

**Prerequisites:** Complete notebooks 01-05  
**VRAM Required:** GPU 0: 5-10 GB, GPU 1: 2-8 GB

**Architecture:**
```
GPU 0: llama-server (LLM inference)
GPU 1: RAPIDS cuDF, cuGraph, Graphistry
```

---

### 07 - OpenAI API Client

**File:** `07-openai-api-client-llcuda-v2.2.0.ipynb`

Use OpenAI SDK with llama-server:

- OpenAI SDK compatibility
- Drop-in replacement setup
- Chat completions
- Streaming responses
- Function calling (tools)
- Embeddings (if supported)

**Prerequisites:** Complete notebooks 01-03  
**VRAM Required:** 5-10 GB

**Key Feature:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

### 08 - NCCL PyTorch

**File:** `08-nccl-pytorch-llcuda-v2.2.0.ipynb`

NCCL for distributed PyTorch alongside llcuda:

- NCCL vs tensor-split (key differences)
- When to use each approach
- NCCL for PyTorch DDP
- Combining inference + training
- Multi-GPU memory management

**Prerequisites:** Complete notebooks 01-06  
**VRAM Required:** 15-25 GB

**Important Note:**
```
llama-server: Uses native CUDA tensor-split (NOT NCCL)
PyTorch DDP:  Uses NCCL for distributed training
```

---

### 09 - Large Models on Kaggle

**File:** `09-large-models-kaggle-llcuda-v2.2.0.ipynb`

Run 70B models on Kaggle's dual T4 setup:

- I-quant selection for 70B
- Memory-optimized configuration
- Context size management
- Performance expectations
- Quality vs feasibility trade-offs

**Prerequisites:** Complete notebooks 01-04  
**VRAM Required:** 25-30 GB (dual T4)

**Key Configuration:**
```python
# 70B model on 30GB VRAM
config = ServerConfig(
    model_path="llama-70b-IQ3_XS.gguf",
    tensor_split="0.48,0.48",
    context_size=2048,  # Smaller context
    n_batch=128,        # Smaller batch
)
```

---

### 10 - Complete Workflow

**File:** `10-complete-workflow-llcuda-v2.2.0.ipynb`

End-to-end production workflow:

1. Environment setup
2. Model selection and download
3. Unsloth fine-tuning
4. GGUF export and quantization
5. Multi-GPU deployment
6. OpenAI API client usage
7. Performance monitoring
8. Production best practices

**Prerequisites:** Complete notebooks 01-09  
**VRAM Required:** Varies (single or dual T4)

---

## Running on Kaggle

### Setup Steps

1. **Create New Notebook**
   - Go to [kaggle.com/code](https://kaggle.com/code)
   - Click "New Notebook"

2. **Configure GPU**
   - Settings → Accelerator → GPU T4 × 2
   - Settings → Internet → On

3. **Upload or Copy Notebook**
   - Upload `.ipynb` file, or
   - Copy cells into new notebook

4. **Run All Cells**
   - Kernel → Run All

### Kaggle-Specific Notes

- **Session limit:** 12 hours maximum
- **Disk space:** 73 GB available
- **Internet:** Required for package installation
- **Persistence:** Only `/kaggle/working` persists

---

## Learning Path

### Path 1: Quick Start (1 hour)
```
01 → 02 → 03
Quick Start → Server Setup → Multi-GPU
```

### Path 2: Full Course (4 hours)
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 10 → 11
All fundamentals through visualization
```

### Path 3: Advanced Topics (2 hours)
```
01 → 03 → 08 → 09
Focus on multi-GPU and large models
```

### Path 4: Unsloth Focus (2 hours)
```
01 → 04 → 05 → 10
Fine-tuning and deployment
```

### Path 5: Visualization & Analysis (2.5 hours) ⭐ RECOMMENDED
```
01 → 03 → 04 → 06 → 11
Quick start → Multi-GPU → GGUF → Split-GPU → Architecture Visualization
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| GPU not detected | Check Settings → Accelerator → GPU T4 × 2 |
| Out of memory | Reduce context_size, use smaller model |
| Server won't start | Check logs with `server.get_logs()` |
| Slow inference | Enable flash_attn=True |
| Import errors | Restart kernel after pip install |

### Getting Help

- **Documentation:** See [`../docs/`](../docs/) for detailed guides
- **API Reference:** See [`API_REFERENCE.md`](../docs/API_REFERENCE.md)
- **Troubleshooting:** See [`TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md)

---

## Contributing

Want to improve these notebooks? See the [Contributing Guide](../CONTRIBUTING.md).

---

### 11 - GGUF Neural Network Visualization ⭐ MOST IMPORTANT

**File:** `11-gguf-neural-network-graphistry-visualization.ipynb`

**The most comprehensive GGUF visualization tool available** - visualizes internal model architecture as interactive graphs:

- **Complete architecture visualization** with 929 nodes, 981 edges
- **Layer-by-layer breakdown** showing 5 transformer layers (35 nodes, 34 edges each)
- **896 attention heads** visualized across 28 layers
- **112 quantization blocks** showing Q4_K_M structure
- **Interactive Graphistry dashboards** with cloud URLs
- **GPU-accelerated analytics** using RAPIDS cuGraph (PageRank, centrality)
- **Dual-GPU architecture** (LLM on GPU 0, visualization on GPU 1)

**Prerequisites:** Complete notebooks 01-06
**VRAM Required:** GPU 0: 10GB (LLM), GPU 1: 0.5GB (analytics)

**What You'll Learn:**
```
1. Extract architecture from running GGUF models
2. Build graph representations of neural networks
3. Apply PageRank to identify important components
4. Create interactive visualizations with Graphistry
5. Understand quantization block structure
6. Analyze information flow through transformer layers
```

**Key Visualizations:**
- **Main Architecture** (929 nodes): Complete Llama-3.2-3B structure
- **Layers 1-5** (35 nodes each): Transformer block internals
- **Attention Heads** (896 nodes): Multi-head attention focus
- **Quantization Blocks** (112 nodes): Q4_K_M memory layout
- **Complete Dashboard**: All-in-one HTML with statistics

**Novel Features:**
- First tool to visualize GGUF quantization as graphs
- Runtime model introspection (no binary file parsing)
- Graph theory metrics applied to neural architectures
- Dual-GPU split for concurrent inference + visualization

**Outputs:**
- 8 interactive Graphistry cloud URLs
- `/kaggle/working/complete_dashboard.html` (downloadable)
- `/kaggle/working/attention_dashboard.html`

**Research Applications:**
- Compare different quantization methods (Q4_K_M vs IQ3_XS)
- Identify pruning opportunities (low-importance heads)
- Understand information flow and bottlenecks
- Validate GGUF conversions vs original models

**Technical Stack:**
- llcuda v2.2.0 (ServerManager, LlamaCppClient)
- RAPIDS cuGraph (GPU-accelerated PageRank, centrality)
- Graphistry (interactive cloud visualization)
- Pandas, cuDF, PyArrow

📘 **[Detailed Documentation →](../docs/GGUF_NEURAL_NETWORK_VISUALIZATION.md)**

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.2.0 | 2026-01-22 | Initial 10-notebook series + notebook 11 (GGUF visualization) |

---

## License

MIT License - See [LICENSE](../LICENSE)
