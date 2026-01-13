# llcuda Examples

This directory contains example scripts and notebooks for using llcuda.

## Google Colab / Kaggle Examples

### Gemma 3-1B + Unsloth Tutorial
Complete tutorial for using llcuda v2.1.0 with Gemma 3-1B model on Google Colab.

**Features demonstrated:**
- Binary auto-download on first import
- Loading GGUF models from HuggingFace
- Quantization API usage
- Unsloth integration workflow
- Inference and batch inference
- Performance metrics and optimization
- Verified 134 tok/s on Tesla T4

**Notebooks:**
1. [llcuda_v2_1_0_gemma3_1b_unsloth_colab.ipynb](../notebooks/llcuda_v2_1_0_gemma3_1b_unsloth_colab.ipynb) - Tutorial notebook
2. [llcuda_v2_1_0_gemma3_1b_unsloth_colab_executed.ipynb](../notebooks/llcuda_v2_1_0_gemma3_1b_unsloth_colab_executed.ipynb) - Executed with results

**Recommended Runtime:**
- Python 3.11+
- Tesla T4 GPU (Google Colab)

**Usage:**
1. Open in Google Colab: [Open Tutorial](https://colab.research.google.com/github/waqasm86/llcuda/blob/main/notebooks/llcuda_v2_1_0_gemma3_1b_unsloth_colab.ipynb)
2. Select Runtime → Change runtime type → T4 GPU
3. Run all cells

### Legacy Examples (v1.x)

For legacy v1.x examples, see the [archive/v1.x/examples](../archive/v1.x/examples) directory.

## Local Examples

More examples coming soon for local development and production deployments.

## Contributing

Have an example you'd like to share? Submit a pull request!

---

**Links:**
- [llcuda on PyPI](https://pypi.org/project/llcuda/)
- [llcuda on GitHub](https://github.com/waqasm86/llcuda)
- [Documentation](https://waqasm86.github.io/)
