#!/usr/bin/env python3
"""
Upload llcuda models to Hugging Face Hub
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, login

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set via environment variable
REPO_ID = "waqasm86/llcuda-models"
MODEL_FILE = "/media/waqasm86/External1/Project-Nvidia/llcuda/llcuda/models/google_gemma-3-1b-it-Q4_K_M.gguf"

def main():
    print("=" * 60)
    print("llcuda Model Upload to Hugging Face")
    print("=" * 60)
    print()

    # Login to Hugging Face
    print("🔐 Logging in to Hugging Face...")
    try:
        login(token=HF_TOKEN)
        print("✅ Login successful!")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        sys.exit(1)

    print()

    # Initialize API
    api = HfApi()

    # Create repository if it doesn't exist
    print(f"📦 Creating repository: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print("✅ Repository created/verified!")
    except Exception as e:
        print(f"⚠️  Repository may already exist: {e}")

    print()

    # Check if model file exists
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: Model file not found: {MODEL_FILE}")
        sys.exit(1)

    model_size = os.path.getsize(MODEL_FILE) / (1024 * 1024)  # MB
    print(f"📁 Model file: {MODEL_FILE}")
    print(f"📊 File size: {model_size:.1f} MB")
    print()

    # Upload model file
    print(f"📤 Uploading model to Hugging Face...")
    print(f"   Repository: {REPO_ID}")
    print(f"   File: {os.path.basename(MODEL_FILE)}")
    print()
    print("   This may take several minutes (769 MB upload)...")
    print()

    try:
        upload_file(
            path_or_fileobj=MODEL_FILE,
            path_in_repo=os.path.basename(MODEL_FILE),
            repo_id=REPO_ID,
            repo_type="model",
            token=HF_TOKEN
        )
        print("✅ Model uploaded successfully!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)

    print()

    # Create README
    readme_content = """---
license: apache-2.0
tags:
  - llama.cpp
  - gguf
  - gemma
  - quantized
  - cuda
language:
  - en
pipeline_tag: text-generation
---

# llcuda Models

Optimized GGUF models for llcuda - Zero-config CUDA-accelerated LLM inference.

## Models

### google_gemma-3-1b-it-Q4_K_M.gguf

- **Model**: Google Gemma 3 1B Instruct
- **Quantization**: Q4_K_M (4-bit)
- **Size**: 769 MB
- **Use case**: General-purpose chat, Q&A, code assistance
- **Recommended for**: 1GB+ VRAM GPUs

**Performance:**
- Tesla T4 (Colab/Kaggle): ~15 tok/s
- Tesla P100 (Colab): ~18 tok/s
- GeForce 940M (1GB): ~15 tok/s
- RTX 30xx/40xx: ~25+ tok/s

## Usage

### With llcuda (Recommended)

```python
pip install llcuda

import llcuda
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")
result = engine.infer("What is AI?")
print(result.text)
```

### With llama.cpp

```bash
# Download model
huggingface-cli download waqasm86/llcuda-models google_gemma-3-1b-it-Q4_K_M.gguf --local-dir ./models

# Run with llama.cpp
./llama-server -m ./models/google_gemma-3-1b-it-Q4_K_M.gguf -ngl 26
```

## Supported Platforms

- ✅ Google Colab (T4, P100, V100, A100)
- ✅ Kaggle (Tesla T4)
- ✅ Local GPUs (GeForce, RTX, Tesla)
- ✅ All NVIDIA GPUs with compute capability 5.0+

## Links

- **PyPI**: [pypi.org/project/llcuda](https://pypi.org/project/llcuda/)
- **GitHub**: [github.com/waqasm86/llcuda](https://github.com/waqasm86/llcuda)
- **Documentation**: [waqasm86.github.io](https://waqasm86.github.io/)

## License

Apache 2.0 - Models are provided as-is for educational and research purposes.

## Credits

- Model: Google Gemma 3 1B
- Quantization: llama.cpp GGUF format
- Package: llcuda by Waqas Muhammad
"""

    print("📝 Creating README...")
    try:
        readme_path = "/tmp/llcuda_hf_README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            token=HF_TOKEN
        )
        print("✅ README uploaded!")
    except Exception as e:
        print(f"❌ README upload failed: {e}")

    print()
    print("=" * 60)
    print("✅ Upload Complete!")
    print("=" * 60)
    print()
    print(f"🌐 Repository URL: https://huggingface.co/{REPO_ID}")
    print()
    print("Next steps:")
    print("  1. Visit the repository to verify upload")
    print("  2. Update llcuda Python code to use this model")
    print("  3. Test download in Colab/Kaggle")
    print()

if __name__ == "__main__":
    main()
