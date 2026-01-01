# llcuda: CUDA-Accelerated LLM Inference for Python

**Effortless, zero-configuration LLM inference with CUDA acceleration. Compatible with all modern NVIDIA GPUs, Google Colab, Kaggle, and JupyterLab. PyTorch-inspired API for seamless integration.**

> Ideal for:  
> - Google Colab and Kaggle notebooks  
> - Local development on GPUs from GTX 940M to RTX 4090  
> - Production-grade performance without manual setup  

---

## What's New in Version 1.1.0

**Universal GPU Compatibility and Cloud-Optimized Design**

In previous versions (1.0.x), compatibility was limited, often leading to errors like "no kernel image available" on older architectures such as Tesla T4 in Colab/Kaggle.

With v1.1.0, we've introduced a **hybrid bootstrap architecture** for broader support:

- **Ultra-Light PyPI Package**: Only 51 KB (down from 327 MB) – pure Python code.  
- **Auto-Download System**: Binaries (~700 MB) and models (~770 MB) fetch automatically on first import, based on your GPU.  
- **Expanded GPU Support**: All NVIDIA architectures with compute capability 5.0+ (Maxwell to Ada Lovelace).  
- **Full Cloud Integration**: Seamless on Google Colab (T4, P100, V100, A100) and Kaggle (T4).  
- **No Breaking Changes**: Backward-compatible with v1.0.x APIs.  

Example Upgrade:

```python
# Previously (v1.0.x) – Error on T4/P100
!pip install llcuda==1.0.0
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")  # Fails on incompatible GPUs

# Now (v1.1.0) – Works everywhere
!pip install llcuda
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")  # Auto-configures and runs
