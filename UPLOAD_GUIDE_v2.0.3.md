# llcuda v2.0.3 Upload Guide

## ⚠️ SECURITY FIRST

**CRITICAL**: The PyPI token you shared earlier is now **publicly exposed**. You MUST:

1. **Immediately revoke it**: https://pypi.org/manage/account/token/
2. **Create a new token** when ready to upload
3. **NEVER share tokens** - use them only in your local terminal

---

## 📋 Pre-Upload Checklist

### ✅ Version Updates (Already Done)
- [x] `pyproject.toml` → v2.0.3
- [x] `llcuda/__init__.py` → v2.0.3
- [x] `README.md` → v2.0.3
- [x] `.gitignore` → Updated for bundled binaries
- [x] Release notes created

### ✅ Binaries Prepared
- [x] Binaries extracted to `llcuda/binaries/cuda12/`
- [x] Libraries extracted to `llcuda/lib/`
- [x] Verified with `prepare_binaries.py` ✅

---

## 🔨 Step 1: Build the Package

```bash
cd C:\Users\CS-AprilVenture\Documents\Project-Waqas\Project-Waqas-Programming\Project-Nvidia-Office\Project-Nvidia-Office\llcuda

# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build wheel and source distribution
python -m build
```

**Expected output:**
```
Successfully built llcuda-2.0.3-py3-none-any.whl (270 MB)
Successfully built llcuda-2.0.3.tar.gz (268 MB)
```

### Verify Build

```bash
ls -lh dist/

# Should show:
# llcuda-2.0.3-py3-none-any.whl  (~270 MB)
# llcuda-2.0.3.tar.gz            (~268 MB)
```

---

## 🧪 Step 2: Test Locally

```bash
# Install from local wheel
pip install dist/llcuda-2.0.3-py3-none-any.whl --force-reinstall

# Test import (should be instant, no downloads)
python -c "import llcuda; print(f'Version: {llcuda.__version__}')"

# Expected: Version: 2.0.3
```

### Test in Clean Environment (Recommended)

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate

# Install and test
pip install dist/llcuda-2.0.3-py3-none-any.whl
python -c "import llcuda; print(llcuda.__version__)"

# Deactivate when done
deactivate
```

---

## 📤 Step 3: Upload to PyPI

### Create New API Token

1. Go to: https://pypi.org/manage/account/token/
2. Click "Add API token"
3. **Name**: `llcuda-v2.0.3-upload`
4. **Scope**: "Project: llcuda"
5. Copy the token (starts with `pypi-...`)

### Upload with Twine

```bash
# Upload to PyPI
python -m twine upload dist/llcuda-2.0.3*

# You'll be prompted:
# Username: __token__
# Password: <paste your NEW token here>
```

**Note**: Use `__token__` (with two underscores) as username, not your PyPI username.

---

## 🐙 Step 4: Update GitHub

### Commit Changes

```bash
cd C:\Users\CS-AprilVenture\Documents\Project-Waqas\Project-Waqas-Programming\Project-Nvidia-Office\Project-Nvidia-Office\llcuda

# Check status (binaries should NOT be staged - they're in .gitignore)
git status

# Stage changes
git add .
git commit -m "Release v2.0.3: Bundled binaries in PyPI package

- Bundle CUDA binaries in wheel (~270 MB)
- No runtime downloads, instant import
- PyTorch-style distribution
- Updated documentation
- Fixed 404 download errors
"

# Push to GitHub
git push origin main
```

### Create GitHub Release

1. Go to: https://github.com/waqasm86/llcuda/releases/new
2. **Tag**: `v2.0.3`
3. **Title**: `v2.0.3 - Bundled Binaries`
4. **Description**: Copy from `RELEASE_NOTES_v2.0.3.md`
5. **Attach binary** (optional): `llcuda-binaries-cuda12-t4-v2.0.2.tar.gz` from `build-artifacts/`
6. Click "Publish release"

---

## ✅ Step 5: Verify Deployment

### Test PyPI Installation

```bash
# Wait ~5 minutes for PyPI CDN propagation

# Test in fresh environment
python -m venv verify_env
source verify_env/bin/activate  # Windows: verify_env\Scripts\activate

# Install from PyPI
pip install llcuda==2.0.3

# Verify
python -c "
import llcuda
print(f'Version: {llcuda.__version__}')
print(f'Import successful: {llcuda.__file__}')
"

# Should show:
# Version: 2.0.3
# Import successful: .../site-packages/llcuda/__init__.py

deactivate
```

### Test on Google Colab

Create new Colab notebook:

```python
# Cell 1: Install
!pip install llcuda==2.0.3

# Cell 2: Verify
import llcuda
print(f"Version: {llcuda.__version__}")

# Cell 3: Quick test
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")
result = engine.infer("Hello!", max_tokens=20)
print(result.text)
```

**Expected:** Instant import, no binary downloads! ✅

---

## 📝 Post-Upload Tasks

### Update Links

1. **PyPI page**: Should automatically update to v2.0.3
2. **GitHub README**: Verify badges show v2.0.3
3. **Documentation**: Update any version-specific links

### Announce Release

Consider posting on:
- GitHub Discussions
- Twitter/X
- Reddit (r/MachineLearning, r/LocalLLaMA)
- Discord communities

### Clean Up

```bash
# Remove old cached downloads (users upgrading from v2.0.2)
rm -rf ~/.cache/llcuda/llcuda-binaries-cuda12-t4-v2.0.2.tar.gz

# Remove test environments
rm -rf test_env/ verify_env/
```

---

## 🐛 Troubleshooting

### "File too large" error on PyPI

PyPI has 100 MB default limit, but allows larger files for packages with binaries.

If you get this error:
1. Request limit increase: https://pypi.org/help/#file-size-limit
2. Reference PyTorch (2.5 GB wheels are accepted)

### Import still downloads binaries

This means old version is installed:
```bash
pip uninstall llcuda
pip install --no-cache-dir llcuda==2.0.3
```

### Binaries not found after install

Package integrity issue:
```bash
pip install --force-reinstall --no-cache-dir llcuda==2.0.3
```

---

## 📊 Success Metrics

After upload, verify:

- ✅ PyPI shows v2.0.3
- ✅ `pip install llcuda` downloads ~270 MB
- ✅ `import llcuda` is instant (no downloads)
- ✅ Works on Google Colab with T4 GPU
- ✅ GitHub release published with notes

---

## 🎉 You're Done!

llcuda v2.0.3 is now live with bundled binaries!

Users can now:
```bash
pip install llcuda  # One command, everything included!
```

No more two-stage installation confusion! 🚀
