# Step-by-Step Guide: Publishing llcuda to PyPI

This guide follows the official Python packaging tutorial: https://packaging.python.org/en/latest/tutorials/packaging-projects/

## ✅ Preparation Complete

Your package is ready for PyPI! Here's what's been done:

- ✅ Python 3.11+ requirement set in all config files
- ✅ Package metadata configured (name, version, author, etc.)
- ✅ LICENSE file added (MIT)
- ✅ README.md with comprehensive documentation
- ✅ Source distribution built: `dist/llcuda-0.1.0.tar.gz`
- ✅ Package validated with `twine check` - PASSED
- ✅ All changes pushed to GitHub

## 📋 Publishing Steps

### Step 1: Create PyPI Account (5 minutes)

1. **Register on PyPI**
   - Go to: https://pypi.org/account/register/
   - Fill in your details
   - Click "Create account"

2. **Verify Your Email**
   - Check your inbox for verification email
   - Click the verification link

3. **Enable Two-Factor Authentication (REQUIRED)**
   - Go to: https://pypi.org/manage/account/
   - Click "Account settings"
   - Enable 2FA using an authenticator app (Google Authenticator, Authy, etc.)
   - **Important**: PyPI requires 2FA for all new accounts

### Step 2: Create API Token (2 minutes)

1. **Login to PyPI**
   - Go to: https://pypi.org/
   - Login with your credentials

2. **Create API Token**
   - Go to: https://pypi.org/manage/account/
   - Scroll down to "API tokens" section
   - Click "Add API token"

3. **Configure Token**
   - **Token name**: `llcuda-upload` (or any name you prefer)
   - **Scope**: Select "Entire account" (for first upload)
     - After first upload, you can create a project-specific token
   - Click "Add token"

4. **Save Your Token**
   - **CRITICAL**: Copy the token immediately!
   - Format: `pypi-AgEIcHlwaS5vcmc...` (starts with `pypi-`)
   - You won't be able to see it again
   - Store it securely (password manager recommended)

### Step 3: Configure Authentication (Choose One Method)

#### Method A: Using Keyring (Recommended for Windows)

```bash
# Store token in system keyring
python -m keyring set https://upload.pypi.org/legacy/ __token__
# When prompted, paste your API token (including the 'pypi-' prefix)
```

#### Method B: Using .pypirc File

Create file at: `C:\Users\CS-AprilVenture\.pypirc`

```ini
[distutils]
index-servers =
    pypi

[pypi]
username = __token__
password = pypi-YourActualTokenHere
```

**Security Note**: If using .pypirc:
- Replace `pypi-YourActualTokenHere` with your actual token
- Keep this file secure
- Never commit it to git

### Step 4: Upload to PyPI (1 minute)

Navigate to your project directory and upload:

```bash
cd "C:\Users\CS-AprilVenture\Documents\Project-Waqas\Project-Waqas-Programming\Project-Nvidia\llcuda"

# Upload to PyPI
python -m twine upload dist/*
```

**If using Method B (.pypirc):**
- Twine will automatically use credentials from .pypirc

**If using Method A (keyring) or no config:**
- Username: `__token__`
- Password: (paste your API token)

**Expected Output:**
```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading llcuda-0.1.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 29.5/29.5 kB • 00:00 • ?

View at:
https://pypi.org/project/llcuda/0.1.0/
```

### Step 5: Verify Publication

1. **Visit Package Page**
   - Go to: https://pypi.org/project/llcuda/
   - Verify all information displays correctly
   - Check that README renders properly

2. **Test Installation**
   ```bash
   # In a new environment
   pip install llcuda
   ```

3. **Verify Metadata**
   ```bash
   pip show llcuda
   ```

   Should show:
   ```
   Name: llcuda
   Version: 0.1.0
   Summary: CUDA-accelerated LLM inference for Python
   Home-page: https://github.com/waqasm86/llcuda
   Author: Waqas Muhammad
   Author-email: waqasm86@gmail.com
   License: MIT
   Requires: numpy
   Required-by:
   ```

## 🎉 Success!

Congratulations! Your package is now published on PyPI!

- **PyPI Page**: https://pypi.org/project/llcuda/
- **GitHub Repo**: https://github.com/waqasm86/llcuda
- **Install Command**: `pip install llcuda`

## 📊 Post-Publication Tasks

### 1. Update GitHub README

Add PyPI badge to your README.md:

```markdown
[![PyPI version](https://badge.fury.io/py/llcuda.svg)](https://badge.fury.io/py/llcuda)
[![Python versions](https://img.shields.io/pypi/pyversions/llcuda.svg)](https://pypi.org/project/llcuda/)
[![License](https://img.shields.io/pypi/l/llcuda.svg)](https://github.com/waqasm86/llcuda/blob/main/LICENSE)
```

### 2. Create GitHub Release

1. Go to: https://github.com/waqasm86/llcuda/releases
2. Click "Create a new release"
3. Tag: `v0.1.0`
4. Title: `llcuda v0.1.0 - Initial Release`
5. Description: Copy from CHANGELOG.md
6. Click "Publish release"

### 3. Create Project-Specific Token

Now that the package exists on PyPI:

1. Go to: https://pypi.org/manage/account/
2. Create new token with scope: "Project: llcuda"
3. Update your .pypirc or keyring with the new token

### 4. Monitor Package

- **PyPI Statistics**: https://pypistats.org/packages/llcuda
- **Download Stats**: Available after ~24 hours
- **GitHub Insights**: Check stars, forks, issues

## 🔄 Publishing Future Versions

When releasing version 0.2.0:

### 1. Update Version Number

Edit these files:
- `setup.py` line 87: `version='0.2.0'`
- `pyproject.toml` line 12: `version = "0.2.0"`
- `llcuda/__init__.py` line 16: `__version__ = "0.2.0"`

### 2. Update CHANGELOG.md

```markdown
## [0.2.0] - 2024-XX-XX

### Added
- New feature 1
- New feature 2

### Changed
- Updated feature 3

### Fixed
- Bug fix 1
```

### 3. Commit Changes

```bash
git add -A
git commit -m "Release version 0.2.0"
git tag -a v0.2.0 -m "Version 0.2.0"
git push origin main --tags
```

### 4. Build and Upload

```bash
# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build new distribution
python -m build --sdist

# Check package
python -m twine check dist/*

# Upload to PyPI
python -m twine upload dist/*
```

### 5. Create GitHub Release

Create release for v0.2.0 on GitHub

## ⚠️ Important Notes

### Package Name Availability

**Before uploading**, verify `llcuda` is available:
- Search: https://pypi.org/search/?q=llcuda
- If taken, you'll need a different name:
  - `llcuda-inference`
  - `llama-cuda`
  - `cuda-llm`
  - etc.

### Source Distribution Only

This package currently publishes **source distribution only** (`.tar.gz`).

Users will need:
- CUDA Toolkit (11.7+ or 12.0+)
- C++ compiler (GCC, MSVC, Clang)
- CMake 3.24+
- Python 3.11+

**For better user experience**, consider:
- Building platform-specific wheels (.whl files)
- Using GitHub Actions + cibuildwheel
- See: https://cibuildwheel.readthedocs.io/

### Version Control

- **Never delete** published versions from PyPI
- **Never re-upload** the same version number
- Always increment version for changes
- Use pre-release versions for testing: `0.2.0a1`, `0.2.0b1`, `0.2.0rc1`

### TestPyPI (Optional)

Test uploads before going to production:

1. **Register on TestPyPI**: https://test.pypi.org/account/register/
2. **Create token** on TestPyPI
3. **Upload to TestPyPI**:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```
4. **Test installation**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --no-deps llcuda
   ```

## 🆘 Troubleshooting

### Error: "Invalid or non-existent authentication"

- Verify token starts with `pypi-`
- Check token is for correct repository (PyPI not TestPyPI)
- Ensure 2FA is enabled on your account

### Error: "Filename or contents already exists"

- You're trying to upload same version again
- Solution: Increment version number or delete old dist files

### Error: "The name 'llcuda' is already claimed"

- Package name is taken
- Solution: Choose a different name and update:
  - `setup.py` (name field)
  - `pyproject.toml` (name field)
  - Rebuild distribution

### Upload is Slow

- Normal for first upload
- PyPI scans for malware
- Can take 1-5 minutes
- Be patient!

### README not rendering on PyPI

- Check markdown syntax
- Verify `long_description_content_type='text/markdown'` in setup.py
- Rebuild and re-upload (with new version number)

## 📚 Resources

- **Official Tutorial**: https://packaging.python.org/en/latest/tutorials/packaging-projects/
- **PyPI Help**: https://pypi.org/help/
- **Twine Docs**: https://twine.readthedocs.io/
- **Python Packaging Guide**: https://packaging.python.org/
- **Semantic Versioning**: https://semver.org/

## ✅ Quick Checklist

Before uploading, verify:

- [ ] PyPI account created and verified
- [ ] 2FA enabled
- [ ] API token generated and saved
- [ ] Authentication configured (keyring or .pypirc)
- [ ] Package name is available on PyPI
- [ ] Version number is correct in all files
- [ ] CHANGELOG.md updated
- [ ] `twine check dist/*` passes
- [ ] README.md renders correctly
- [ ] All changes committed and pushed to GitHub

## 🎯 Ready to Publish?

Your package is ready! Just follow Steps 1-5 above.

The entire process takes about 10 minutes for first-time setup.

**Good luck with your PyPI publication!** 🚀

---

For questions or issues, refer to:
- [PYPI_PUBLISHING_GUIDE.md](PYPI_PUBLISHING_GUIDE.md) - Detailed guide
- [QUICK_PUBLISH_STEPS.md](QUICK_PUBLISH_STEPS.md) - Quick reference
