# Step-by-Step Instructions: Upload llcuda v2.0.1 Release

## Overview

This guide walks you through uploading the CUDA 12 binaries for llcuda v2.0.1 to GitHub Releases.

---

## Files Ready for Upload

Located in: `C:\Users\CS-AprilVenture\Documents\Project-Waqas\Project-Waqas-Programming\Project-Nvidia\`

### Main Binary Package
- **llcuda-binaries-cuda12-t4.tar.gz** (140 MB)
  - SHA256: `54bb904f213b38aec4a2c85baa2fab8256200966aa994bb9fa4a77640d699aa4`

### Documentation
- **RELEASE_NOTES_v2.0.1.md** - Full release notes
- **GITHUB_RELEASE_DESCRIPTION_v2.0.1.md** - GitHub release description (copy-paste ready)

### Updated Project Files
- **llcuda/llcuda/_internal/bootstrap.py** - Updated to v2.0.1 URL

---

## Step 1: Verify Files

```bash
cd "C:\Users\CS-AprilVenture\Documents\Project-Waqas\Project-Waqas-Programming\Project-Nvidia"

# Check binary package
ls -lh llcuda-binaries-cuda12-t4.tar.gz
# Should show: 140 MB

# Verify checksum
sha256sum llcuda-binaries-cuda12-t4.tar.gz
# Should match: 54bb904f213b38aec4a2c85baa2fab8256200966aa994bb9fa4a77640d699aa4

# Test extraction
mkdir -p test_extract
tar -xzf llcuda-binaries-cuda12-t4.tar.gz -C test_extract
ls test_extract/bin/llama-server
ls test_extract/lib/libggml-cuda.so.0
rm -rf test_extract
```

---

## Step 2: Create GitHub Release

### 2.1 Navigate to Releases
1. Open browser: https://github.com/waqasm86/llcuda/releases
2. Click **"Draft a new release"** button

### 2.2 Configure Release

**Choose a tag:**
```
v2.0.1
```
- Select **"Create new tag: v2.0.1 on publish"**

**Target:**
- Branch: `main` (or current branch)

**Release title:**
```
llcuda v2.0.1 - Tesla T4 CUDA 12 Binaries
```

**Description:**
- Copy the entire contents of `GITHUB_RELEASE_DESCRIPTION_v2.0.1.md`
- Paste into the description box
- Preview to ensure formatting looks good

### 2.3 Upload Binary Files

In the **"Attach binaries"** section at the bottom:

1. Click **"choose your files"** or drag & drop
2. Upload: `llcuda-binaries-cuda12-t4.tar.gz` (140 MB)
3. Wait for upload to complete (green checkmark appears)
4. **Optional:** Also upload `RELEASE_NOTES_v2.0.1.md` for detailed documentation

### 2.4 Final Settings

- ✅ Check **"Set as the latest release"**
- ❌ **DO NOT** check "Set as a pre-release" (this is production-ready)

### 2.5 Publish

Click **"Publish release"** button

---

## Step 3: Verify Release

### 3.1 Check Release Page
Visit: https://github.com/waqasm86/llcuda/releases/tag/v2.0.1

Verify:
- ✅ Release is marked as "Latest"
- ✅ Binary file is listed under "Assets"
- ✅ File size shows ~140 MB
- ✅ Description is properly formatted

### 3.2 Test Download URL
```bash
# Test that download URL works
curl -I https://github.com/waqasm86/llcuda/releases/download/v2.0.1/llcuda-binaries-cuda12-t4.tar.gz

# Should return: HTTP/2 302 (redirect) or HTTP/2 200
```

---

## Step 4: Update and Commit Project Files

### 4.1 Navigate to llcuda Directory
```bash
cd "C:\Users\CS-AprilVenture\Documents\Project-Waqas\Project-Waqas-Programming\Project-Nvidia\llcuda"
```

### 4.2 Check Git Status
```bash
git status
```

Should show:
- Modified: `llcuda/_internal/bootstrap.py` (updated to v2.0.1 URL)

### 4.3 Verify Changes
```bash
git diff llcuda/_internal/bootstrap.py
```

Should show:
```diff
-GITHUB_RELEASE_URL = "https://github.com/waqasm86/llcuda/releases/download/v2.0.0"
+GITHUB_RELEASE_URL = "https://github.com/waqasm86/llcuda/releases/download/v2.0.1"
```

### 4.4 Commit Changes
```bash
# Add changes
git add llcuda/_internal/bootstrap.py

# Commit
git commit -m "Release v2.0.1: Update bootstrap to point to v2.0.1 binaries

- Updated GITHUB_RELEASE_URL to v2.0.1
- Binaries: 140 MB CUDA 12 T4-optimized package
- Includes FlashAttention, CUDA Graphs, Tensor Core optimization
- Tested on Google Colab Tesla T4
"

# Push to GitHub
git push origin main
```

### 4.5 Create Git Tag
```bash
# Create tag
git tag -a v2.0.1 -m "llcuda v2.0.1 - Tesla T4 CUDA 12 Binaries

CUDA 12 binaries optimized for Tesla T4 GPU (Google Colab).
Features: FlashAttention, CUDA Graphs, Tensor Core optimization.
Binary package: 140 MB (371 MB extracted).
"

# Push tag
git push origin v2.0.1
```

---

## Step 5: Test the Release

### 5.1 Test in Fresh Environment (Google Colab)

Create a new Colab notebook: https://colab.research.google.com/

```python
# Cell 1: Install
!pip install llcuda==2.0.1

# Cell 2: Test bootstrap download
import llcuda

# This should trigger automatic download of binaries from v2.0.1 release
# Watch for: "Downloading T4 binaries (264 MB)..." message
# (Note: Message says 264 MB but actual download is 140 MB)

# Cell 3: Verify installation
from llcuda.core import get_device_properties

props = get_device_properties(0)
print(f"✅ GPU: {props.name}")
print(f"   Compute: SM {props.compute_capability_major}.{props.compute_capability_minor}")

# Cell 4: Test inference
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", silent=True)

result = engine.infer("What is 2+2?", max_tokens=20)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

### 5.2 Expected Results
- Bootstrap should download from: `https://github.com/waqasm86/llcuda/releases/download/v2.0.1/llcuda-binaries-cuda12-t4.tar.gz`
- Download size: ~140 MB
- Extraction should create: `~/.cache/llcuda/bin/` and `~/.cache/llcuda/lib/`
- Model inference should work correctly
- Speed should be ~45 tok/s for Gemma 3-1B on T4

---

## Step 6: Update PyPI Package (Optional)

If you made code changes to llcuda Python files, update PyPI:

```bash
cd "C:\Users\CS-AprilVenture\Documents\Project-Waqas\Project-Waqas-Programming\Project-Nvidia\llcuda"

# Ensure version is 2.0.1 in pyproject.toml
cat pyproject.toml | grep version
# Should show: version = "2.0.1"

# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/llcuda-2.0.1*

# Test installation
pip install --upgrade llcuda
```

**Note:** In this case, you only updated bootstrap.py URL, so PyPI update is needed.

---

## Step 7: Announce the Release

### 7.1 Update README.md
Ensure README.md mentions v2.0.1:

```markdown
**Version**: 2.0.1
**Target GPU**: **Tesla T4 ONLY** (SM 7.5)
```

### 7.2 Optional: Create Announcement
- GitHub Discussions: https://github.com/waqasm86/llcuda/discussions
- Create a new discussion announcing v2.0.1 release
- Link to release page and highlight key features

---

## Checklist

Before publishing:
- [ ] ✅ Built binaries with cmake in Google Colab
- [ ] ✅ Created release package (llcuda-binaries-cuda12-t4.tar.gz)
- [ ] ✅ Verified package structure (bin/ and lib/ directories)
- [ ] ✅ Generated SHA256 checksum
- [ ] ✅ Updated bootstrap.py to v2.0.1 URL
- [ ] ✅ Written release notes
- [ ] ✅ Prepared GitHub release description

After publishing:
- [ ] Uploaded binary to GitHub Releases v2.0.1
- [ ] Set as latest release
- [ ] Verified download URL works
- [ ] Committed bootstrap.py changes
- [ ] Created and pushed v2.0.1 git tag
- [ ] Tested in fresh Google Colab environment
- [ ] Updated PyPI package (if needed)
- [ ] Updated README.md to mention v2.0.1

---

## Troubleshooting

### Issue: Upload fails with "file too large"
- GitHub Releases supports files up to 2 GB
- Your file is only 140 MB, so this shouldn't happen
- Try using a different browser or GitHub CLI

### Issue: Download URL returns 404
- Wait a few minutes after publishing
- GitHub may need time to propagate the release
- Verify you're using correct URL: `/releases/download/v2.0.1/`

### Issue: Bootstrap still downloads from v2.0.0
- Ensure bootstrap.py was updated and committed
- Check that you pushed changes to GitHub
- Clear pip cache: `pip cache purge`
- Reinstall: `pip uninstall llcuda && pip install llcuda`

---

## Success Criteria

Your release is successful when:
1. ✅ GitHub release page shows v2.0.1 as "Latest"
2. ✅ Binary downloads successfully from release URL
3. ✅ Fresh pip install works: `pip install llcuda`
4. ✅ Bootstrap downloads binaries from v2.0.1
5. ✅ Inference works in Google Colab with T4 GPU
6. ✅ Performance matches benchmarks (~45 tok/s for Gemma 3-1B)

---

## Files Summary

### Ready for Upload
```
llcuda-binaries-cuda12-t4.tar.gz          140 MB    Binary package
RELEASE_NOTES_v2.0.1.md                     ~15 KB    Full documentation
GITHUB_RELEASE_DESCRIPTION_v2.0.1.md         ~5 KB    GitHub description
```

### Modified Files (Need commit)
```
llcuda/llcuda/_internal/bootstrap.py       Changed    Updated release URL
```

### Build Artifacts (Can delete after upload)
```
build_t4/                                   486 MB    Original build
llcuda-t4-release/                          371 MB    Staged release dir
build_t4_colab.tar.gz                      Original   Original Colab tar
```

---

**Ready to upload!** Follow the steps above to publish your release.
