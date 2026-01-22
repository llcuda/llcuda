# llcuda v2.2.0 - Kaggle Tutorial Notebooks

Complete tutorial series demonstrating **split-GPU LLM inference and visualization** on Kaggle's dual Tesla T4 GPUs (30GB total VRAM).

## Overview

This collection contains **11 comprehensive Jupyter notebooks** that progressively teach modern GPU-accelerated LLM deployment, from basic inference to advanced graph analytics and visualization. All notebooks are optimized for Kaggle's dual T4 environment with full support for split-GPU architecture.

### Key Innovation: Split-GPU Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 SPLIT-GPU ARCHITECTURE                       │
├──────────────────────────────────────────────────────────────┤
│  GPU 0 (15GB)              │  GPU 1 (15GB)                   │
│  ┌──────────────────┐      │  ┌──────────────────────────┐  │
│  │  llama-server    │      │  │  RAPIDS cuGraph/cuDF    │  │
│  │  (LLM Inference) │      │  │  Graphistry             │  │
│  │  tensor_split    │      │  │  (Graph Analytics)      │  │
│  │  1.0, 0.0        │      │  │  CUDA_VISIBLE_DEVICES=1 │  │
│  └──────────────────┘      │  └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Tutorial Progression

```
Phase 1: Foundation (01-04)
├─ 01: Quick start and basic inference
├─ 02: Server configuration mastery
├─ 03: Multi-GPU tensor parallelism
└─ 04: Quantization understanding

Phase 2: Integration (05-06)
├─ 05: Fine-tuning with Unsloth
└─ 06: Split-GPU workflows

Phase 3: Advanced Applications (07-08)
├─ 07: Knowledge graph extraction
└─ 08: Document network analysis

Phase 4: Optimization & Production (09-11)
├─ 09: Large model deployment (70B)
├─ 10: Complete production workflow
└─ 11: Neural architecture visualization
```

---

## Notebook Index

### Phase 1: Foundation (Beginner Level)

| # | Notebook | Goal | Technologies | Time |
|---|----------|------|--------------|------|
| **01** | [Quick Start](01-quickstart-llcuda-v2.2.0.ipynb) | Get started with llcuda setup, model download, and first inference | llama-server, GGUF | 10 min |
| **02** | [Server Setup](02-llama-server-setup-llcuda-v2.2.0.ipynb) | Master server configuration, optimization, and performance tuning | llama-server, Health monitoring | 15 min |
| **03** | [Multi-GPU Inference](03-multi-gpu-inference-llcuda-v2.2.0.ipynb) | Learn tensor parallelism and multi-GPU configurations | Tensor split, CUDA | 20 min |
| **04** | [GGUF Quantization](04-gguf-quantization-llcuda-v2.2.0.ipynb) | Understand quantization types, size estimation, and quality tradeoffs | K-Quants, I-Quants, GGUF | 20 min |

### Phase 2: Integration (Intermediate Level)

| # | Notebook | Goal | Technologies | Time |
|---|----------|------|--------------|------|
| **05** | [Unsloth Integration](05-unsloth-integration-llcuda-v2.2.0.ipynb) | Fine-tune models with Unsloth and deploy with llcuda | Unsloth, LoRA, GGUF export | 30 min |
| **06** | [Split-GPU Graphistry](06-split-gpu-graphistry-llcuda-v2-2-0.ipynb) | Run LLM on GPU 0 and RAPIDS/Graphistry on GPU 1 simultaneously | RAPIDS cuGraph, Graphistry, Split-GPU | 30 min |

### Phase 3: Advanced Applications

| # | Notebook | Goal | Technologies | Time |
|---|----------|------|--------------|------|
| **07** | [Knowledge Graph Extraction](07-knowledge-graph-extraction-graphistry-v2.2.0.ipynb) | Extract knowledge graphs from text using LLM and visualize with Graphistry | LLM extraction, cuGraph, GFQL, Hypergraphs | 35 min |
| **08** | [Document Network Analysis](08-document-network-analysis-graphistry-llcuda-v2-2-0.ipynb) | Analyze document collections through similarity networks and clustering | TF-IDF, Louvain, Community detection | 30 min |

### Phase 4: Optimization & Production

| # | Notebook | Goal | Technologies | Time |
|---|----------|------|--------------|------|
| **09** | [Large Models on Kaggle](09-large-models-kaggle-llcuda-v2-2-0.ipynb) | Deploy 70B models on dual T4 with performance visualization | IQ3_XS, Performance benchmarking, Graphistry dashboard | 35 min |
| **10** | [Complete Workflow](10-complete-workflow-llcuda-v2-2-0.ipynb) | End-to-end production pipeline integrating all components | Full stack, OpenAI API, Production deployment | 45 min |
| **11** | [Neural Network Visualization](11-gguf-neural-network-graphistry-visualization.ipynb) | Visualize GGUF model architecture and internal structure | Model inspection, Architecture graphs, Parameter analysis | 40 min |

---

## Detailed Descriptions

### 01 - Quick Start (Foundation)

**File:** `01-quickstart-llcuda-v2.2.0.ipynb`
**VRAM Required:** 3-5 GB (single T4)
**Prerequisites:** None

**What You'll Learn:**
- Install llcuda v2.2.0 on Kaggle
- Download GGUF models from HuggingFace
- Start llama-server with basic configuration
- Run first chat completion
- Monitor GPU memory usage
- Clean up resources properly

**Key Concepts:**
- ServerManager API
- GGUF model format
- OpenAI-compatible endpoints
- GPU memory management

**Sample Code:**
```python
from llcuda.server import ServerManager

server = ServerManager()
server.start_server(
    model_path="model.gguf",
    host="127.0.0.1",
    port=8090,
    gpu_layers=99  # All layers on GPU
)
```

---

### 02 - Server Setup (Foundation)

**File:** `02-llama-server-setup-llcuda-v2.2.0.ipynb`
**VRAM Required:** 5-8 GB (single T4)
**Prerequisites:** Complete notebook 01

**What You'll Learn:**
- Complete server configuration reference
- Kaggle T4 optimization presets
- High-performance tuning (batch size, parallelism)
- Health monitoring and debugging
- Command-line deployment options

**Key Concepts:**
- Server configuration parameters
- Performance optimization
- Health check endpoints
- Log management

**Configuration Options:**
```python
# High-performance configuration
server.start_server(
    model_path="model.gguf",
    gpu_layers=99,
    ctx_size=8192,        # Large context
    batch_size=1024,      # Large batch
    n_parallel=4,         # Parallel slots
    flash_attn=True       # Flash attention
)
```

---

### 03 - Multi-GPU Inference (Foundation)

**File:** `03-multi-gpu-inference-llcuda-v2.2.0.ipynb`
**VRAM Required:** 15-25 GB (dual T4)
**Prerequisites:** Complete notebooks 01-02

**What You'll Learn:**
- Tensor split configurations (50/50, 70/30, 100/0)
- Split-mode options (layer, row, none)
- Memory distribution verification
- Split-GPU mode for LLM + other tasks
- Performance comparison across configurations

**Key Concepts:**
- Tensor parallelism
- Multi-GPU VRAM distribution
- Layer-based splitting
- Reserved GPU for analytics

**Tensor Split Examples:**
```python
# Equal split (50/50) - Max model size
tensor_split="0.5,0.5"

# LLM + Light task (70/30)
tensor_split="0.7,0.3"

# Split-GPU mode (100/0) - GPU 1 free
tensor_split="1.0,0.0"
```

---

### 04 - GGUF Quantization (Foundation)

**File:** `04-gguf-quantization-llcuda-v2.2.0.ipynb`
**VRAM Required:** Varies by model
**Prerequisites:** Complete notebooks 01-03

**What You'll Learn:**
- Quantization families (Legacy, K-Quants, I-Quants)
- Bits per weight and quality scores
- VRAM estimation for any model size
- Kaggle T4 recommendations
- Quality vs size tradeoffs
- Quantization benchmarking

**Key Concepts:**
- Q4_K_M (recommended default)
- Q5_K_M (higher quality)
- IQ3_XS (enables 70B on dual T4)
- Size estimation calculator

**Quantization Reference:**
```
Model Size | Quant    | VRAM   | Fits Dual T4?
─────────────────────────────────────────────
1-3B       | Q4_K_M   | ~2 GB  | ✅ Single GPU
7-8B       | Q4_K_M   | ~5 GB  | ✅ Single GPU
13B        | Q4_K_M   | ~8 GB  | ✅ Single GPU
32B        | Q4_K_M   | ~20 GB | ✅ Dual GPU
70B        | IQ3_XS   | ~25 GB | ✅ Dual GPU
```

---

### 05 - Unsloth Integration (Intermediate)

**File:** `05-unsloth-integration-llcuda-v2.2.0.ipynb`
**VRAM Required:** 10-15 GB (training), 5-8 GB (inference)
**Prerequisites:** Complete notebooks 01-04

**What You'll Learn:**
- Load models with Unsloth (4-bit quantization)
- Add LoRA adapters (parameter-efficient)
- Fine-tune with custom datasets
- Export to GGUF format
- Deploy fine-tuned models with llcuda
- Compare performance

**Key Concepts:**
- FastLanguageModel
- LoRA fine-tuning
- GGUF export
- GPU memory cleanup

**Workflow:**
```
Unsloth (Train) → GGUF (Export) → llcuda (Deploy)
     GPU 0           CPU           GPU 0
```

---

### 06 - Split-GPU Graphistry (Intermediate)

**File:** `06-split-gpu-graphistry-llcuda-v2-2-0.ipynb`
**VRAM Required:** GPU 0: 5-10 GB, GPU 1: 2-8 GB
**Prerequisites:** Complete notebooks 01-05

**What You'll Learn:**
- Split-GPU architecture design
- LLM inference on GPU 0
- RAPIDS cuGraph on GPU 1 (parallel)
- Louvain community detection
- Interactive Graphistry dashboards
- Multi-channel encodings (color, size, icons, badges)

**Key Concepts:**
- Simultaneous GPU operations
- Social network analytics
- PageRank and centrality metrics
- Interactive visualization

**Architecture:**
```python
# GPU 0: LLM
server.start_server(tensor_split="1.0,0.0")

# GPU 1: RAPIDS
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cudf, cugraph
```

---

### 07 - Knowledge Graph Extraction (Advanced)

**File:** `07-knowledge-graph-extraction-graphistry-v2.2.0.ipynb`
**VRAM Required:** GPU 0: 5-8 GB, GPU 1: 3-8 GB
**Prerequisites:** Complete notebooks 01-06

**What You'll Learn:**
- LLM-powered entity and relationship extraction
- Knowledge graph construction with cuGraph
- GPU-accelerated graph algorithms
- Advanced Graphistry layouts (Force Atlas 2, UMAP, Ring)
- GFQL pattern mining (hubs, 2-hop paths)
- Hypergraph visualization

**Key Concepts:**
- Structured entity extraction
- Knowledge graph analytics
- GFQL query language
- Multiple visualization layouts

**Workflow:**
```
Text → LLM Extract → Knowledge Graph → cuGraph Analytics → Graphistry Viz
```

---

### 08 - Document Network Analysis (Advanced)

**File:** `08-document-network-analysis-graphistry-llcuda-v2-2-0.ipynb`
**VRAM Required:** GPU 0: 3-5 GB, GPU 1: 2-5 GB
**Prerequisites:** Complete notebooks 01-07

**What You'll Learn:**
- LLM document summarization
- TF-IDF similarity calculation
- Louvain community detection
- Document clustering visualization
- Community theme extraction
- Intra-community analysis

**Key Concepts:**
- Document similarity networks
- Community-based clustering
- Theme discovery with LLM
- Interactive cluster exploration

**Pipeline:**
```
Documents → LLM Summarize → TF-IDF → Similarity Network → Louvain → Viz
```

---

### 09 - Large Models on Kaggle (Advanced)

**File:** `09-large-models-kaggle-llcuda-v2-2-0.ipynb`
**VRAM Required:** 25-30 GB (dual T4)
**Prerequisites:** Complete notebooks 01-04

**What You'll Learn:**
- I-quant selection for 70B models
- Memory-optimized configurations
- Performance benchmarking
- Model selection with Graphistry dashboard
- Streaming vs non-streaming comparison
- Context size management

**Key Concepts:**
- IQ3_XS quantization
- 70B deployment on 30GB
- Performance landscape visualization
- Memory optimization tips

**70B Configuration:**
```python
server.start_server(
    model_path="llama-70b-IQ3_XS.gguf",
    tensor_split="0.48,0.48",
    ctx_size=2048,      # Smaller context
    batch_size=128,     # Smaller batch
    flash_attn=True
)
```

**Graphistry Dashboard:**
- 12 model configurations visualized
- Quantization tradeoff comparison
- GPU compatibility indicators
- Upgrade path suggestions

---

### 10 - Complete Workflow (Production)

**File:** `10-complete-workflow-llcuda-v2-2-0.ipynb`
**VRAM Required:** Varies (single or dual T4)
**Prerequisites:** Complete notebooks 01-09

**What You'll Learn:**
- Complete environment setup and verification
- Model download with validation
- Fine-tuning with Unsloth
- GGUF export and quantization
- Multi-GPU deployment
- OpenAI SDK integration
- Combined LLM + RAPIDS pipelines
- Production best practices
- Workflow visualization with Graphistry

**Key Concepts:**
- End-to-end workflow
- Production deployment
- API integration
- Workflow pipeline visualization

**11-Stage Workflow:**
```
1. Environment Setup
2. Model Download
3. Unsloth Fine-tuning (GPU 0)
4. GGUF Export (CPU)
5. Model Validation
6. llcuda Deployment (GPU 0)
7. LLM Inference (GPU 0)
8. RAPIDS Analytics (GPU 1, parallel)
9. Combined Results
10. Graphistry Visualization
11. Production API
```

---

### 11 - Neural Network Visualization (Deep Dive)

**File:** `11-gguf-neural-network-graphistry-visualization.ipynb`
**VRAM Required:** GPU 0: 3-5 GB, GPU 1: 2-5 GB
**Prerequisites:** Complete notebooks 01-08

**What You'll Learn:**
- GGUF model metadata extraction
- Neural network architecture parsing
- Transformer layer visualization
- Attention head network mapping (896 heads)
- Parameter distribution analysis
- Quantization impact visualization
- Interactive architecture exploration

**Key Concepts:**
- Model internal structure
- Architecture component graphs
- Attention mechanism visualization
- Parameter allocation

**Architecture Breakdown (Llama-3.2-3B):**
```
Total: 28 transformer layers, 896 attention heads

Parameter Distribution:
├─ Embedding:      10.0%
├─ Attention:      26.7%
├─ Feed-forward:   53.4%
└─ Output:         10.0%

Visualization: 929 nodes, 981 edges
```

---

## Running on Kaggle

### Initial Setup

1. **Create New Notebook**
   - Visit [kaggle.com/code](https://kaggle.com/code)
   - Click "New Notebook"

2. **Configure GPU**
   - Settings → Accelerator → **GPU T4 × 2**
   - Settings → Internet → **On** (required for pip install)

3. **Upload Notebook**
   - Upload `.ipynb` file, or
   - Copy-paste cells into Kaggle notebook

4. **Run All Cells**
   - Kernel → Restart & Run All

### Kaggle Environment Details

| Resource | Specification |
|----------|---------------|
| **GPUs** | 2× Tesla T4 (15GB each, 30GB total) |
| **VRAM** | 30GB total GPU memory |
| **RAM** | 30GB system RAM |
| **Disk** | 73GB temporary storage |
| **Session** | 12 hours maximum |
| **Internet** | Required for package installation |
| **Persistence** | Only `/kaggle/working` persists |

### Environment Variables

```python
# GPU configuration
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'      # Use only GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '1'      # Use only GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'    # Use both GPUs (default)
```

---

## Learning Paths

### Path 1: Quick Start (1 hour)
**Goal:** Get up and running with basic inference

```
01 → 02 → 03
Quick Start → Server Setup → Multi-GPU
```

**Outcome:** Deploy and run LLM inference on Kaggle T4

---

### Path 2: Full Foundation (3 hours)
**Goal:** Complete understanding of llcuda ecosystem

```
01 → 02 → 03 → 04 → 05 → 06 → 10
All core concepts through production workflow
```

**Outcome:** Deploy production-ready LLM systems

---

### Path 3: Graph Analytics Focus (2.5 hours)
**Goal:** Master LLM + graph visualization

```
01 → 03 → 06 → 07 → 08 → 11
Split-GPU architecture with Graphistry
```

**Outcome:** Build LLM-powered graph analytics applications

---

### Path 4: Large Model Specialist (2 hours)
**Goal:** Deploy 70B models efficiently

```
01 → 03 → 04 → 09
Multi-GPU and quantization focus
```

**Outcome:** Run 70B models on Kaggle dual T4

---

### Path 5: Fine-tuning Expert (2.5 hours)
**Goal:** Custom model training and deployment

```
01 → 04 → 05 → 10
Fine-tuning workflow mastery
```

**Outcome:** Train and deploy custom fine-tuned models

---

## Core Technologies

### llcuda v2.2.0
- **Size:** 62KB Python package + 961MB CUDA binaries
- **Backend:** CUDA 12 inference engine
- **API:** OpenAI-compatible endpoints
- **Multi-GPU:** Native tensor-split support

### RAPIDS 25.6.0
- **cuDF:** GPU DataFrames (pandas-like API)
- **cuGraph:** Graph algorithms (PageRank, Louvain, Betweenness)
- **cuML:** Machine learning on GPU
- **Performance:** 50-100× faster than CPU

### Graphistry 0.50.4
- **Visualization:** Interactive graph dashboards
- **Encodings:** Color, size, icons, badges
- **Layouts:** Force Atlas 2, UMAP, Ring, Geospatial
- **GFQL:** Graph query language

### Unsloth
- **Training:** 2× faster than standard fine-tuning
- **Memory:** 4-bit quantization for efficiency
- **Export:** Direct GGUF export

### NCCL
- **Communication:** Multi-GPU coordination
- **Use Case:** PyTorch DDP alongside llcuda

---

## Hardware Optimization Guide

### Kaggle Dual Tesla T4 (30GB Total VRAM)

| Model Size | Quantization | VRAM Usage | Speed (tok/s) | Configuration |
|------------|--------------|------------|---------------|---------------|
| **1-3B** | Q4_K_M | ~2-3 GB | 80+ | Single T4 |
| **7-8B** | Q4_K_M | ~5-6 GB | 50-60 | Single T4 |
| **13-14B** | Q4_K_M | ~8-9 GB | 30-35 | Single T4 or Dual (split) |
| **32-34B** | Q4_K_M | ~20-22 GB | 20-25 | Dual T4 (tensor_split="0.5,0.5") |
| **70B** | IQ3_XS | ~25-27 GB | 8-10 | Dual T4 (tensor_split="0.48,0.48") |

### Recommended Configurations

**Single T4 (15GB):**
```python
# Best for: 1-13B models
config = {
    "gpu_layers": 99,
    "ctx_size": 4096,
    "flash_attn": True
}
```

**Dual T4 Equal Split (30GB):**
```python
# Best for: 13-70B models
config = {
    "gpu_layers": 99,
    "tensor_split": "0.5,0.5",
    "ctx_size": 4096,
    "flash_attn": True
}
```

**Split-GPU Mode (LLM + RAPIDS):**
```python
# GPU 0: LLM
config = {
    "gpu_layers": 99,
    "tensor_split": "1.0,0.0",
    "ctx_size": 4096
}

# GPU 1: RAPIDS
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cudf, cugraph, graphistry
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **GPU not detected** | Settings → Accelerator → GPU T4 × 2 |
| **Out of memory (OOM)** | Reduce `ctx_size` (4096 → 2048), use smaller model or lower quantization |
| **Server won't start** | Check logs with `server.get_logs()`, verify model path |
| **Slow inference** | Enable `flash_attn=True`, increase `batch_size` |
| **Import errors** | Restart kernel after `pip install` |
| **RAPIDS import fails** | Avoid `--force-reinstall` which breaks cupy C extensions |
| **Graphistry registration fails** | Add Kaggle secrets: `Graphistry_Personal_Key_ID`, `Graphistry_Personal_Secret_Key` |
| **NCCL errors** | Ensure both GPUs are available, check `CUDA_VISIBLE_DEVICES` |

### Performance Optimization

**Memory:**
- Use smaller `ctx_size` for 70B models (2048 vs 4096)
- Reduce `batch_size` if OOM (512 → 256 → 128)
- Enable `flash_attn=True` for attention efficiency
- Use appropriate quantization (Q4_K_M default, IQ3_XS for 70B)

**Speed:**
- Increase `batch_size` for throughput (128 → 256 → 512)
- Enable `n_parallel=4` for concurrent requests
- Use `flash_attn=True` for faster attention
- Consider dual GPU split for larger models

**Quality:**
- Use higher quantization (Q5_K_M, Q6_K) when VRAM allows
- Larger `ctx_size` for longer context understanding
- Temperature tuning (0.7 default, 0.3 for factual, 0.9 for creative)

---

## API Reference

### ServerManager

```python
from llcuda.server import ServerManager

server = ServerManager()

# Start server
server.start_server(
    model_path="model.gguf",
    host="127.0.0.1",
    port=8090,
    gpu_layers=99,
    tensor_split="0.5,0.5",
    ctx_size=4096,
    batch_size=512,
    n_parallel=4,
    flash_attn=True,
    timeout=120
)

# Health check
if server.check_server_health():
    print("Server ready")

# Get logs
logs = server.get_logs()

# Stop server
server.stop_server()
```

### LlamaCppClient

```python
from llcuda.api.client import LlamaCppClient

client = LlamaCppClient(base_url="http://127.0.0.1:8090")

# Chat completion
response = client.chat.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
    temperature=0.7
)

# Streaming
for chunk in client.chat.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
    stream=True
):
    print(chunk.choices[0].delta.content, end="")
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8090/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
```

---

## File Structure

```
llcuda-other-notebooks-practice/
├── README.md (this file)
│
├── Phase 1: Foundation
│   ├── 01-quickstart-llcuda-v2.2.0.ipynb
│   ├── 02-llama-server-setup-llcuda-v2.2.0.ipynb
│   ├── 03-multi-gpu-inference-llcuda-v2.2.0.ipynb
│   └── 04-gguf-quantization-llcuda-v2.2.0.ipynb
│
├── Phase 2: Integration
│   ├── 05-unsloth-integration-llcuda-v2.2.0.ipynb
│   └── 06-split-gpu-graphistry-llcuda-v2-2-0.ipynb
│
├── Phase 3: Advanced Applications
│   ├── 07-knowledge-graph-extraction-graphistry-v2.2.0.ipynb
│   └── 08-document-network-analysis-graphistry-llcuda-v2-2-0.ipynb
│
└── Phase 4: Optimization & Production
    ├── 09-large-models-kaggle-llcuda-v2-2-0.ipynb
    ├── 10-complete-workflow-llcuda-v2-2-0.ipynb
    └── 11-gguf-neural-network-graphistry-visualization.ipynb
```

---

## Prerequisites

### Software
- Python 3.10+
- CUDA 12.x
- Kaggle account (for notebook execution)

### Python Packages
All packages are installed within notebooks:
- `llcuda>=2.2.0`
- `unsloth` (for notebook 05)
- `graphistry[ai]>=0.50.4` (for notebooks 06-11)
- `cugraph-cu12>=25.6.0` (for notebooks 06-11)
- `cudf-cu12>=25.6.0` (for notebooks 06-11)

---

## Contributing

Contributions are welcome! To improve these notebooks:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Test on Kaggle T4 × 2
5. Submit a pull request

---

## Citation

If you use these notebooks in your research or projects, please cite:

```bibtex
@misc{llcuda-kaggle-tutorials,
  title={llcuda v2.2.0 Kaggle Tutorial Notebooks},
  author={llcuda Contributors},
  year={2026},
  howpublished={\url{https://github.com/llcuda/llcuda}},
  note={11-notebook tutorial series for split-GPU LLM inference and visualization}
}
```

---

## License

MIT License - See [LICENSE](../LICENSE)

---

## Support and Resources

- **Documentation:** [llcuda.github.io](https://llcuda.github.io)
- **GitHub:** [github.com/llcuda/llcuda](https://github.com/llcuda/llcuda)
- **Issues:** [github.com/llcuda/llcuda/issues](https://github.com/llcuda/llcuda/issues)
- **Discussions:** [github.com/llcuda/llcuda/discussions](https://github.com/llcuda/llcuda/discussions)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **v2.2.0** | January 2026 | Initial 11-notebook series with split-GPU architecture |
| | | - Added notebooks 06-08: Graphistry visualizations |
| | | - Added notebook 09: Large model deployment with dashboard |
| | | - Added notebook 10: Complete production workflow |
| | | - Added notebook 11: Neural architecture visualization |
| | | - Full descriptions for all 183 code cells |

---

**Last Updated:** January 22, 2026
**Tutorial Series Version:** 2.2.0
**Target Platform:** Kaggle Dual Tesla T4 (30GB VRAM)
