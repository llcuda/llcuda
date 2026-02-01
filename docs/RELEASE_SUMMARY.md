# llcuda v2.2.0 Release Summary

**Date:** 2026-02-01
**Status:** Ready for GitHub Release
**Target:** Kaggle dual Tesla T4 (SM 7.5)

---

## What This Release Is

- CUDA 12-first inference stack optimized for Kaggle dual T4.
- Split-GPU workflow: GPU 0 for llama.cpp inference, GPU 1 for Graphistry/RAPIDS.
- Intended model range: 1B-5B GGUF (Q4_K_M recommended).
- Distribution: GitHub Releases (primary) + optional HuggingFace mirror (no PyPI).

---

## Release Artifact (Binary Bundle)

- **Filename:** `llcuda-v2.2.0-cuda12-kaggle-t4x2.tar.gz`
- **Size:** ~961 MB
- **Contents:** `bin/` (llama-server + tools), `lib/` (CUDA libraries)
- **Build Info:** CUDA 12.5, SM 7.5 (Turing), llama.cpp b7760 (388ce82)

---

## Packaging/Bootstrap Notes

- `llcuda/_internal/bootstrap.py` points to the v2.2.0 bundle and validates SHA256.
- First import auto-downloads binaries to `~/.cache/llcuda`.
- GitHub is the primary distribution channel (no PyPI for v2.2.0).

---

## Release Steps (High-Level)

1. **Create GitHub Release**
   - Tag: `v2.2.0`
   - Title: `llcuda v2.2.0 - Kaggle Dual T4 CUDA 12 Binaries`
   - Upload: `llcuda-v2.2.0-cuda12-kaggle-t4x2.tar.gz`

2. **Verify Installation (Kaggle)**
   ```python
   !pip install -q --no-cache-dir --force-reinstall git+https://github.com/llcuda/llcuda.git@v2.2.0
   import llcuda
   print(llcuda.__version__)
   ```

3. **Smoke Test**
   - Download a 1B-5B GGUF model
   - Start llama-server with `tensor_split="1.0,0.0"`
   - Run a simple completion

---

## Related Docs

- `docs/INSTALLATION.md`
- `docs/KAGGLE_GUIDE.md`
- `docs/GITHUB_RELEASE_GUIDE.md`
