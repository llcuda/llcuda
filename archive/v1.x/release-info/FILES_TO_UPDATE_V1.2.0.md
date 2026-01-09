# Files to Update for v1.2.0 Release

## ✅ Already Completed

These files have already been updated to v1.2.0:

1. **`llcuda/llcuda/__init__.py`** - Version 1.2.0 ✅
2. **`llcuda/pyproject.toml`** - Version 1.2.0 ✅
3. **`llcuda/llcuda/_internal/bootstrap.py`** - Updated with GPU detection and v1.2.0 URL ✅
4. **`llcuda/llcuda/server.py:553`** - Fixed stderr.read() bug ✅
5. **`llcuda/CHANGELOG.md`** - Added v1.2.0 entry ✅
6. **`CREATE_RELEASE_PACKAGE.sh`** - Fixed both packaging bugs ✅

## 📝 Files to Update on GitHub

### 1. README.md (Critical)

**Current Location:** `llcuda/README.md`

**New Content:** See `llcuda/README_V1.2.0.md`

**Action Required:**
```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda
cp README_V1.2.0.md README.md
```

**Key Changes:**
- Update version badge from 1.1.9 to 1.2.0
- Add "What's New in v1.2.0" section
- Add GPU compatibility table with package sizes
- Update performance benchmarks (940M, T4, RTX 4090)
- Add FlashAttention feature highlights
- Update installation instructions
- Add troubleshooting section for GPU detection

### 2. setup.py (If needed)

**Current State:** Minimal setup.py that defers to pyproject.toml

**Action:** No changes needed (pyproject.toml already updated)

## 🗂️ Files to Create on GitHub

### 1. Migration Guide (Optional)

**File:** `llcuda/MIGRATING_TO_V1.2.0.md`

**Purpose:** Help users upgrade from v1.1.x to v1.2.0

**Content:**
```markdown
# Migrating to v1.2.0

## What Changed

v1.2.0 introduces GPU-specific binary bundles with automatic detection. The API remains unchanged, but the bootstrap process is enhanced.

## Do I Need to Do Anything?

**No!** If you're upgrading from v1.1.x:

1. Uninstall old version: `pip uninstall llcuda`
2. Install new version: `pip install llcuda==1.2.0`
3. First import will auto-download optimized binaries for your GPU

## What Gets Downloaded?

- **Maxwell GPUs (940M, GTX 950/960)**: 26 MB package
- **Modern GPUs (T4, RTX series, A100)**: 264 MB package with FlashAttention

## Breaking Changes

**None.** All existing code continues to work without modifications.

## New Features Available

### GPU Detection
```python
import llcuda
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"Compute: {compat['compute_capability']}")
```

### Force Specific Bundle (if needed)
```bash
export LLCUDA_FORCE_BUNDLE="940m"  # or "t4"
python your_script.py
```

## Performance Improvements

If you have a CC 7.5+ GPU (Tesla T4, RTX series), you'll automatically benefit from FlashAttention:
- **2x faster** inference
- Lower latency
- Better throughput for longer contexts

## Troubleshooting

### Old binaries still cached?
```bash
rm -rf ~/.cache/llcuda/
python -c "import llcuda"  # Re-download
```

### Want to verify which bundle was downloaded?
```bash
ls -lh ~/.cache/llcuda/
# Should show either llcuda-binaries-cuda12-940m.tar.gz or llcuda-binaries-cuda12-t4.tar.gz
```
```

## 📦 Files to Upload to GitHub Releases

### Release v1.2.0

**Tag:** `v1.2.0`

**Release Title:** `llcuda v1.2.0 - CUDA 12 Support for GeForce 940M & Tesla T4`

**Files to Upload:**
1. `release-packages/llcuda-binaries-cuda12-940m.tar.gz` (26 MB) ✅ Ready
2. `release-packages/llcuda-binaries-cuda12-t4.tar.gz` (264 MB) ✅ Ready

**Release Notes:** See template in `RELEASE_V1.2.0_SUMMARY.md`

### Actions on Old Releases

Edit the following releases to add deprecation warning:

**v1.1.9:**
Add to top of release notes:
```markdown
⚠️ **Legacy Version - Please Use v1.2.0**

This version is superseded by v1.2.0 which includes:
- GPU-specific optimizations
- FlashAttention support (2x faster on modern GPUs)
- Critical bug fixes
- 90% smaller downloads for Maxwell GPUs

👉 [Upgrade to v1.2.0](https://github.com/waqasm86/llcuda/releases/tag/v1.2.0)
```

**v1.1.8, v1.1.7, v1.1.6:**
Add same warning as above.

**v1.1.1.post1, v1.1.1:**
Mark as "Pre-release" in GitHub UI.

## 🐍 Files to Update on PyPI

### Package Build and Upload

**Steps:**
```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda

# Clean old builds
rm -rf dist/ build/ *.egg-info/

# Build package
python -m build
# or: python setup.py sdist bdist_wheel

# Verify package contents
ls -lh dist/
# Should show:
# llcuda-1.2.0.tar.gz        : < 1 MB
# llcuda-1.2.0-py3-none-any.whl : < 1 MB

# Verify no binaries included
tar -tzf dist/llcuda-1.2.0.tar.gz | grep -E "(binaries|lib|\.so|\.gguf)" || echo "✓ No binaries (correct!)"

# Check package metadata
twine check dist/*

# Upload to PyPI
twine upload dist/*
# Username: __token__
# Password: (your PyPI API token)
```

**Files Generated:**
- `dist/llcuda-1.2.0.tar.gz` (source distribution)
- `dist/llcuda-1.2.0-py3-none-any.whl` (wheel)

**PyPI Page Updates:**
- Description automatically pulls from README.md
- Project metadata from pyproject.toml
- No manual updates needed on PyPI website

## 📋 Pre-Upload Checklist

### Before Committing to GitHub

- [ ] README.md updated to v1.2.0
- [ ] CHANGELOG.md has v1.2.0 entry
- [ ] All version numbers are 1.2.0:
  - [ ] `llcuda/__init__.py`
  - [ ] `pyproject.toml`
- [ ] Bootstrap points to v1.2.0 release URL
- [ ] .gitignore excludes binaries/lib/models
- [ ] No large files staged:
  ```bash
  git status
  # Should NOT show: binaries/, lib/, models/, *.gguf, *.tar.gz
  ```

### Before Creating GitHub Release

- [ ] Both .tar.gz packages created:
  - [ ] `llcuda-binaries-cuda12-940m.tar.gz` (26 MB)
  - [ ] `llcuda-binaries-cuda12-t4.tar.gz` (264 MB)
- [ ] Packages have correct structure (bin/ and lib/ directories)
- [ ] Test extraction:
  ```bash
  cd release-packages
  tar -tzf llcuda-binaries-cuda12-940m.tar.gz | head -20
  tar -tzf llcuda-binaries-cuda12-t4.tar.gz | head -20
  ```

### Before Uploading to PyPI

- [ ] Package built successfully
- [ ] Package size < 1 MB
- [ ] No binaries in package
- [ ] twine check passes
- [ ] Test installation in clean venv:
  ```bash
  python3.11 -m venv test_env
  source test_env/bin/activate
  pip install dist/llcuda-1.2.0-py3-none-any.whl
  python -c "import llcuda; print(llcuda.__version__)"
  # Should output: 1.2.0
  deactivate
  rm -rf test_env
  ```

## 🧪 Post-Release Testing

### Test on Local System (GeForce 940M)

```bash
# Clean environment
python3.11 -m venv test_local
source test_local/bin/activate

# Install from PyPI
pip install llcuda==1.2.0

# Test GPU detection and bootstrap
python << 'EOF'
import llcuda
print(f"Version: {llcuda.__version__}")

# Check GPU detection
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"Compute: {compat['compute_capability']}")

# Bootstrap will download 940M binaries (26 MB)
print("Bootstrap starting...")
EOF

# Cleanup
deactivate
rm -rf test_local
```

**Expected Output:**
```
Version: 1.2.0
GPU: GeForce 940M
Compute: 5.0
Bootstrap starting...
🎯 llcuda First-Time Setup
🎮 GPU Detected: GeForce 940M (Compute 5.0)
📦 Selecting optimized binaries for your GPU...
   Selected: llcuda-binaries-cuda12-940m.tar.gz
📥 Downloading optimized binaries from GitHub...
   This is a one-time download (~30 MB)
✅ Setup Complete!
```

### Test on Google Colab (Tesla T4)

In a new Colab notebook:

```python
!pip install llcuda==1.2.0

import llcuda
print(f"Version: {llcuda.__version__}")

# Bootstrap will download T4 binaries (264 MB)
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)
result = engine.infer("What is 2+2?", max_tokens=50)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")
```

**Expected Output:**
```
Version: 1.2.0
🎮 GPU Detected: Tesla T4 (Compute 7.5)
📦 Selecting optimized binaries for your GPU...
   Selected: llcuda-binaries-cuda12-t4.tar.gz
📥 Downloading optimized binaries from GitHub...
   This is a one-time download (~270 MB)
✅ Setup Complete!

[Inference output...]
Speed: 45.0 tokens/sec
```

## 🚀 Deployment Steps Summary

**Day 1 - GitHub Repository:**
1. Update README.md with v1.2.0 content
2. Verify CHANGELOG.md has v1.2.0 entry
3. Commit changes:
   ```bash
   cd llcuda
   git add README.md CHANGELOG.md
   git commit -m "Update documentation for v1.2.0 release"
   git push origin main
   ```

**Day 1 - GitHub Releases:**
1. Go to https://github.com/waqasm86/llcuda/releases
2. Click "Draft a new release"
3. Tag: `v1.2.0`, Title: `llcuda v1.2.0 - CUDA 12 Support for GeForce 940M & Tesla T4`
4. Upload both .tar.gz files
5. Add release notes (template in RELEASE_V1.2.0_SUMMARY.md)
6. Publish release
7. Create and push tag:
   ```bash
   git tag v1.2.0
   git push origin v1.2.0
   ```
8. Update old releases with deprecation warnings

**Day 1 - PyPI Upload:**
1. Build package
2. Upload to PyPI
3. Verify page content at https://pypi.org/project/llcuda/

**Day 2 - Testing:**
1. Test on local system (940M)
2. Test on Google Colab (T4)
3. Verify bootstrap downloads correct binaries
4. Verify inference works on both platforms

## 📊 Success Metrics

After deployment, verify:

### GitHub Main Repository
- [ ] README shows version 1.2.0
- [ ] CHANGELOG has v1.2.0 entry at top
- [ ] Repository size < 100 MB
- [ ] No binaries in git history

### GitHub Releases
- [ ] v1.2.0 release published
- [ ] Both .tar.gz files uploaded and downloadable
- [ ] Release notes complete
- [ ] Old releases marked as legacy

### PyPI
- [ ] Version 1.2.0 published
- [ ] Package description shows new features
- [ ] Installation works: `pip install llcuda`
- [ ] Old versions still available

### Functionality
- [ ] 940M systems download 26 MB package
- [ ] T4 systems download 264 MB package
- [ ] GPU detection works correctly
- [ ] Inference successful on both platforms
- [ ] Performance matches expectations (940M: 10-20 tok/s, T4: 25-60 tok/s)

## 🎉 Completion

When all checkboxes above are marked, v1.2.0 is successfully deployed!

**Final Verification:**
```bash
# Test full workflow
pip install llcuda==1.2.0
python -c "import llcuda; engine = llcuda.InferenceEngine(); print('✅ v1.2.0 working!')"
```

---

**Last Updated:** 2025-01-04
**Target Release Date:** 2025-01-04
