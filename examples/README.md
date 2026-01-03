# llcuda Examples

This directory contains example scripts and notebooks for using llcuda.

## Google Colab / Kaggle Examples

### colab_test_v1.2.2.ipynb
Interactive Jupyter notebook for testing llcuda v1.2.2 in Google Colab or Kaggle.

**Features tested:**
- Binary auto-download on first import
- llama-server detection from package binaries
- Silent mode to suppress llama-server warnings
- Model download only when explicitly called
- Inference and batch inference
- Performance metrics

**Recommended Runtime:**
- Python 3.11+
- T4 GPU (or any NVIDIA GPU with compute capability 5.0+)

**Usage:**
1. Open in Google Colab: [Open in Colab](https://colab.research.google.com/github/waqasm86/llcuda/blob/main/examples/colab_test_v1.2.2.ipynb)
2. Select Runtime → Change runtime type → T4 GPU
3. Run all cells

### colab_test_v1.2.2.py
Python script version of the Colab test. Can be run directly in Colab or Kaggle.

**Usage in Colab:**
```bash
!pip install llcuda==1.2.2
!wget https://raw.githubusercontent.com/waqasm86/llcuda/main/examples/colab_test_v1.2.2.py
!python3 colab_test_v1.2.2.py
```

**Usage in Kaggle:**
```bash
!pip install llcuda==1.2.2
!python3 /kaggle/input/llcuda-examples/colab_test_v1.2.2.py
```

## Local Examples

More examples coming soon for local development and production deployments.

## Contributing

Have an example you'd like to share? Submit a pull request!

---

**Links:**
- [llcuda on PyPI](https://pypi.org/project/llcuda/)
- [llcuda on GitHub](https://github.com/waqasm86/llcuda)
- [Documentation](https://waqasm86.github.io/)
