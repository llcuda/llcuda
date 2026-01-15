# PyPI Package Upload Guide for llcuda

## Overview

This guide explains how to prepare and upload the llcuda Python package to PyPI, ensuring the package stays under 100MB by excluding large binaries.

---

## Important Size Limits

### PyPI Limits:
- **Package size:** 100MB maximum (hard limit)
- **Project total:** 10GB across all versions
- **Individual file:** 100MB maximum

### Strategy:
- ✅ Upload Python code to PyPI (~1-5MB)
- ✅ Exclude binaries/libraries (use `.gitignore`)
- ✅ Download binaries from GitHub Releases on first import
- ✅ This keeps PyPI package tiny (~1-5MB)

---

## Prerequisites

### 1. Install Build Tools

```bash
pip install --upgrade pip setuptools wheel twine
```

### 2. Create PyPI Account

- Go to: https://pypi.org/account/register/
- Verify email
- Enable 2FA (required)
- Create API token: https://pypi.org/manage/account/token/

Save the token (you'll need it for upload):
```
pypi-AgEIcHlwaS5vcmc...
```

### 3. Create Test PyPI Account (Recommended for testing)

- Go to: https://test.pypi.org/account/register/
- Same process as above

---

## Package Preparation

### Step 1: Verify .gitignore

The `.gitignore` file MUST exclude large files:

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda

# Check .gitignore
cat .gitignore | grep -E "(binaries|lib|models|gguf)"

# Should see:
# llcuda/binaries/
# llcuda/lib/
# llcuda/models/
# *.gguf
```

### Step 2: Clean Build Artifacts

```bash
# Remove old builds
rm -rf build/ dist/ *.egg-info/

# Verify no large files in package
find llcuda -type f -size +10M

# Should return nothing (or only small files you expect)
```

### Step 3: Verify Directory Structure

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda

tree -L 2 -h
```

Expected structure (WITHOUT binaries/lib):
```
llcuda/
├── llcuda/
│   ├── __init__.py
│   ├── server.py
│   ├── models.py
│   ├── utils.py
│   ├── chat.py
│   ├── embeddings.py
│   ├── jupyter.py
│   ├── _internal/
│   │   ├── __init__.py
│   │   ├── bootstrap.py
│   │   └── registry.py
│   └── (NO binaries/, lib/, or models/ directories)
├── setup.py
├── setup.cfg (optional)
├── pyproject.toml (optional)
├── README.md
├── LICENSE
├── MANIFEST.in (optional)
└── requirements.txt (optional)
```

### Step 4: Update Version

Edit `llcuda/__init__.py`:

```python
__version__ = "1.2.2"
```

Edit `setup.py` (if version is hardcoded there):

```python
setup(
    name='llcuda',
    version='1.2.2',
    ...
)
```

### Step 5: Update README and Description

Ensure `README.md` has clear installation instructions:

```markdown
# llcuda

CUDA-accelerated LLM inference for Python.

## Installation

```bash
pip install llcuda
```

## Quick Start

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M")
result = engine.infer("What is AI?")
print(result.text)
```

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA 12.x
- Compute Capability 5.0 or higher
```

---

## Build the Package

### Step 1: Build Distribution Files

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda

# Build source distribution and wheel
python setup.py sdist bdist_wheel
```

This creates:
```
dist/
├── llcuda-1.2.2.tar.gz          # Source distribution
└── llcuda-1.2.2-py3-none-any.whl # Wheel (binary distribution)
```

### Step 2: Verify Package Contents

```bash
# Check what's in the source distribution
tar -tzf dist/llcuda-1.2.2.tar.gz | head -20

# Check what's in the wheel
unzip -l dist/llcuda-1.2.2-py3-none-any.whl | head -20
```

**CRITICAL:** Ensure these do NOT contain:
- `llcuda/binaries/`
- `llcuda/lib/`
- `llcuda/models/`
- Any `.gguf` files

### Step 3: Check Package Size

```bash
ls -lh dist/

# Expected sizes:
# llcuda-1.2.2.tar.gz        : < 100 KB (should be tiny!)
# llcuda-1.2.2-py3-none-any.whl : < 100 KB
```

**If > 10MB:** Something is wrong, binaries are included!

### Step 4: Run Checks

```bash
# Check for common issues
twine check dist/*

# Should output:
# Checking dist/llcuda-1.2.2.tar.gz: PASSED
# Checking dist/llcuda-1.2.2-py3-none-any.whl: PASSED
```

---

## Test on Test PyPI (Recommended)

### Step 1: Upload to Test PyPI

```bash
twine upload --repository testpypi dist/*
```

Enter credentials:
- Username: `__token__`
- Password: (paste your Test PyPI API token)

### Step 2: Test Installation

```bash
# Create clean virtual environment
python3.11 -m venv test_env
source test_env/bin/activate

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ llcuda

# Test import
python -c "import llcuda; print(llcuda.__version__)"

# Test full workflow (will download binaries from GitHub)
python << 'EOF'
import llcuda

# Check GPU
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"Compatible: {compat['compatible']}")

# Bootstrap should download binaries
print("Import successful!")
EOF

# Deactivate
deactivate
rm -rf test_env
```

---

## Upload to Production PyPI

### Step 1: Final Verification

```bash
# Verify package size
ls -lh dist/
# Must be < 100MB (should be < 1MB)

# Verify .gitignore worked
tar -tzf dist/llcuda-1.2.2.tar.gz | grep -E "(binaries|\.so|\.gguf)"
# Should return nothing

# Verify no large files
find llcuda -type f -size +1M -not -path "*/\.*"
# Should return nothing
```

### Step 2: Upload to PyPI

```bash
twine upload dist/*
```

Enter credentials:
- Username: `__token__`
- Password: (paste your production PyPI API token)

### Step 3: Verify on PyPI

Visit: https://pypi.org/project/llcuda/

Check:
- [ ] Version number correct
- [ ] Description renders properly
- [ ] File sizes shown (should be < 1MB)
- [ ] Dependencies listed correctly

### Step 4: Test Installation

```bash
# Clean environment
python3.11 -m venv verify_env
source verify_env/bin/activate

# Install from PyPI
pip install llcuda

# Verify
python -c "import llcuda; print(llcuda.__version__)"

# Cleanup
deactivate
rm -rf verify_env
```

---

## Configure .pypirc (Optional)

Create `~/.pypirc` for easier uploads:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHlwaS5vcmc...
```

**Security:** Set permissions:
```bash
chmod 600 ~/.pypirc
```

Then you can upload without typing credentials:
```bash
twine upload dist/*
```

---

## Update After Each Release

### Checklist:

1. **Update version:**
   - `llcuda/__init__.py` → `__version__ = "1.2.1"`
   - `setup.py` → `version='1.2.1'`

2. **Update CHANGELOG.md:**
   ```markdown
   ## [1.2.1] - 2026-01-04
   ### Fixed
   - Bug fix description

   ### Added
   - New feature description
   ```

3. **Clean and rebuild:**
   ```bash
   rm -rf dist/ build/ *.egg-info/
   python setup.py sdist bdist_wheel
   ```

4. **Upload:**
   ```bash
   twine check dist/*
   twine upload dist/*
   ```

5. **Tag in Git:**
   ```bash
   git tag v1.2.1
   git push origin v1.2.1
   ```

---

## Troubleshooting

### Issue 1: Package too large (> 100MB)

**Symptom:**
```
HTTPError: 400 Bad Request
File too large
```

**Cause:** Binaries included in package

**Fix:**
1. Verify .gitignore:
   ```bash
   cat .gitignore | grep binaries
   ```

2. Remove from package:
   ```bash
   git rm --cached -r llcuda/binaries/
   git rm --cached -r llcuda/lib/
   ```

3. Rebuild:
   ```bash
   rm -rf dist/ build/
   python setup.py sdist bdist_wheel
   ```

### Issue 2: MANIFEST.in including wrong files

**Symptom:** Package still large after .gitignore

**Fix:** Create/update `MANIFEST.in`:

```
include README.md
include LICENSE
include requirements.txt
recursive-include llcuda *.py
recursive-exclude llcuda/binaries *
recursive-exclude llcuda/lib *
recursive-exclude llcuda/models *
global-exclude *.pyc
global-exclude *.gguf
global-exclude *.so
```

### Issue 3: Bootstrap not downloading

**Symptom:** Users report "binaries not found"

**Cause:** GitHub Release URL wrong or files not uploaded

**Fix:**
1. Verify GitHub Release exists:
   ```
   https://github.com/waqasm86/llcuda/releases/tag/v1.2.2
   ```

2. Verify download URL works:
   ```bash
   wget https://github.com/waqasm86/llcuda/releases/download/v1.2.2/llcuda-binaries-cuda12.tar.gz
   ```

3. Update bootstrap.py URL if needed

### Issue 4: Dependencies not installing

**Symptom:** `pip install llcuda` fails due to missing deps

**Fix:** Check `setup.py`:

```python
install_requires=[
    'numpy>=1.20.0',
    'requests>=2.20.0',
    'huggingface_hub>=0.10.0',
    'tqdm>=4.60.0',
],
```

---

## Setup.py Best Practices

```python
from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name='llcuda',
    version='1.2.2',
    author='Waqas Muhammad',
    author_email='waqasm86@gmail.com',
    description='CUDA-accelerated LLM inference for Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/waqasm86/llcuda',
    project_urls={
        'Bug Tracker': 'https://github.com/waqasm86/llcuda/issues',
        'Documentation': 'https://github.com/waqasm86/llcuda#readme',
        'Source Code': 'https://github.com/waqasm86/llcuda',
        'Releases': 'https://github.com/waqasm86/llcuda/releases',
    },
    packages=find_packages(include=['llcuda', 'llcuda.*']),
    include_package_data=False,  # Don't include data files (binaries)
    python_requires='>=3.11',
    install_requires=[
        'numpy>=1.20.0',
        'requests>=2.20.0',
        'huggingface_hub>=0.10.0',
        'tqdm>=4.60.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'twine>=4.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='llm cuda gpu inference ai deep-learning',
)
```

---

## Summary Checklist

Before uploading to PyPI:
- [ ] .gitignore excludes binaries/lib/models
- [ ] Version updated in __init__.py and setup.py
- [ ] README.md is clear and helpful
- [ ] Package size < 1MB (check with `ls -lh dist/`)
- [ ] Tested on Test PyPI first
- [ ] `twine check dist/*` passes
- [ ] GitHub Release with binaries published
- [ ] bootstrap.py URL points to correct release

After uploading:
- [ ] Verify on https://pypi.org/project/llcuda/
- [ ] Test installation in clean environment
- [ ] Update documentation
- [ ] Announce in README/discussions

---

**You're now ready to upload llcuda to PyPI while keeping it under 100MB!**
