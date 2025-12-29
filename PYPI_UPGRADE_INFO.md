# PyPI Package Upgrade Instructions

## Where Users See Installation Instructions

### 1. **PyPI Package Page** (https://pypi.org/project/llcuda/)

The PyPI page automatically renders the README.md from your package. Users will now see:

```bash
# Install or upgrade to latest version
pip install --upgrade llcuda

# Or install specific version
pip install llcuda==1.0.1
```

**Note**: PyPI renders the README.md that was included in the uploaded package. The v1.0.1 package was already uploaded, so the old README is showing. The updated README will appear in the **next release** (v1.0.2 or later).

---

### 2. **GitHub Repository** (https://github.com/waqasm86/llcuda)

✅ **Already Updated** - The README now shows:
- `pip install --upgrade llcuda` (primary installation method)
- `pip install llcuda==1.0.1` (version-specific installation)

---

### 3. **GitHub Release Page** (https://github.com/waqasm86/llcuda/releases/tag/v1.0.1)

✅ **Already Includes** - The release notes mention upgrading from v1.0.0

---

## How to Update PyPI Package Page

Unfortunately, **PyPI doesn't allow editing package metadata after publishing**. The README shown on PyPI comes from the package files uploaded during `twine upload`.

### Options:

#### Option 1: Publish v1.0.2 (Recommended)
Create a patch release with the updated README:

```bash
# 1. Update version in pyproject.toml to 1.0.2
# 2. Add small fix or documentation improvement
# 3. Build and upload
python3.11 -m build
python3.11 -m twine upload dist/*
```

This will show the updated README with `--upgrade` flag on PyPI.

#### Option 2: Wait for Next Release
The updated README will automatically appear when you publish v1.1.0 or the next version.

#### Option 3: Use "yank" (Not Recommended)
You could yank v1.0.1 and re-upload, but this is not recommended as users may have already installed it.

---

## Current Status

### ✅ Updated Locations:
1. **GitHub README** - Shows `pip install --upgrade llcuda`
2. **GitHub Release Notes** - Mentions upgrading
3. **Future PyPI Uploads** - Will include updated README

### ⏳ Not Yet Updated:
1. **PyPI v1.0.1 Page** - Shows old README (can't be changed without new release)

---

## Verification

You can verify the current PyPI page here:
https://pypi.org/project/llcuda/1.0.1/

The "Project description" section on PyPI comes from README.md in the uploaded package.

---

## Recommendation

For immediate visibility of the `--upgrade` flag on PyPI, I recommend:

1. Create a **v1.0.2 patch release** with:
   - Updated README (already done)
   - Minor documentation improvements
   - Or a small bug fix if any

2. This way, users on PyPI will see the updated installation instructions.

Would you like me to:
- Create v1.0.2 with documentation updates?
- Or wait and include these changes in the next feature release?
