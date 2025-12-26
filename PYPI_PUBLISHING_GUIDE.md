# PyPI Publishing Guide for llcuda

This guide will walk you through the process of publishing the `llcuda` package to PyPI (Python Package Index).

## Prerequisites

All prerequisites have been completed:
- ✅ Package configuration files updated
- ✅ LICENSE file added (MIT License)
- ✅ Repository URLs corrected to https://github.com/waqasm86/llcuda
- ✅ Email updated to waqasm86@gmail.com
- ✅ MANIFEST.in created for proper file inclusion
- ✅ Build tools installed (build, twine)
- ✅ Source distribution built successfully

## Step 1: Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Create a new account with your email
3. Verify your email address
4. Enable Two-Factor Authentication (2FA) - **REQUIRED** for new uploads

## Step 2: Create API Token

PyPI no longer accepts username/password for uploads. You must use API tokens:

1. Log in to PyPI: https://pypi.org/
2. Go to Account Settings: https://pypi.org/manage/account/
3. Scroll to "API tokens" section
4. Click "Add API token"
5. Give it a name (e.g., "llcuda-upload")
6. Set scope to "Entire account" (you can later create project-specific tokens)
7. Copy the token - **SAVE IT IMMEDIATELY** (you won't see it again!)
   - Format: `pypi-AgEIcHlwaS5vcmc...` (starts with `pypi-`)

## Step 3: Configure Authentication

### Option A: Using keyring (Recommended)

Store your token securely:

```bash
# Set the token for PyPI
python -m keyring set https://upload.pypi.org/legacy/ __token__
# When prompted, paste your API token (including the 'pypi-' prefix)
```

### Option B: Using .pypirc file (Alternative)

Create a file at `C:\Users\CS-AprilVenture\.pypirc` (Windows) or `~/.pypirc` (Linux/Mac):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YourActualTokenHere

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YourTestPyPITokenHere
```

**Important:** Make sure this file is readable only by you:
```bash
# On Linux/Mac
chmod 600 ~/.pypirc
```

## Step 4: Test on TestPyPI (Optional but Recommended)

TestPyPI is a separate instance of PyPI for testing:

1. Create account on TestPyPI: https://test.pypi.org/account/register/
2. Create API token on TestPyPI (same process as above)
3. Upload to TestPyPI:

```bash
cd "C:\Users\CS-AprilVenture\Documents\Project-Waqas\Project-Waqas-Programming\Project-Nvidia\llcuda"

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*
```

4. Test installation from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps llcuda
```

## Step 5: Upload to PyPI

Once you're confident everything is correct:

```bash
cd "C:\Users\CS-AprilVenture\Documents\Project-Waqas\Project-Waqas-Programming\Project-Nvidia\llcuda"

# Upload to PyPI
python -m twine upload dist/*
```

You'll see output like:
```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading llcuda-0.1.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 29.5/29.5 kB • 00:00 • ?
View at:
https://pypi.org/project/llcuda/0.1.0/
```

## Step 6: Verify Upload

1. Visit your package page: https://pypi.org/project/llcuda/
2. Check that all information displays correctly
3. Test installation:

```bash
pip install llcuda
```

## Step 7: Commit and Push Changes to GitHub

Now commit all the changes we made:

```bash
cd "C:\Users\CS-AprilVenture\Documents\Project-Waqas\Project-Waqas-Programming\Project-Nvidia\llcuda"

# Check status
git status

# Add all changes
git add -A

# Commit
git commit -m "$(cat <<'EOF'
Prepare package for PyPI publication

- Add MIT LICENSE file
- Update repository URLs from local-llama-cuda to llcuda
- Update author email to waqasm86@gmail.com
- Fix setup.py package discovery configuration
- Add MANIFEST.in for proper file inclusion in distribution
- Build source distribution (sdist)
- Add PyPI publishing guide

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"

# Push to GitHub
git push origin main
```

## Publishing Future Versions

When you're ready to publish a new version:

1. **Update version number** in:
   - [setup.py](setup.py) (line 87)
   - [pyproject.toml](pyproject.toml) (line 12)
   - [llcuda/__init__.py](llcuda/__init__.py) (line 16)

2. **Update README.md** with new features/changes

3. **Commit changes:**
   ```bash
   git add -A
   git commit -m "Release version 0.2.0"
   git tag -a v0.2.0 -m "Version 0.2.0"
   git push origin main --tags
   ```

4. **Clean previous builds:**
   ```bash
   rm -rf dist/ build/ *.egg-info
   ```

5. **Build new distribution:**
   ```bash
   python -m build --sdist
   ```

6. **Upload to PyPI:**
   ```bash
   python -m twine upload dist/*
   ```

## Important Notes

### About Binary Wheels

This package contains C++/CUDA extensions. We're currently only publishing source distributions (`.tar.gz`). Users will need:
- CUDA toolkit installed
- C++ compiler
- CMake

For better user experience, consider:
- Publishing pre-built wheels for common platforms (Linux, Windows)
- Using GitHub Actions with cibuildwheel for automated wheel building
- See: https://cibuildwheel.readthedocs.io/

### Version Numbers

Follow semantic versioning (semver):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Package Metadata

Update these in both `setup.py` and `pyproject.toml`:
- `version`
- `description`
- `classifiers` (development status, etc.)
- `keywords`

### README Best Practices

Your README.md is the package homepage on PyPI. Make sure it includes:
- ✅ Clear description
- ✅ Installation instructions
- ✅ Quick start examples
- ✅ Requirements
- ✅ License

## Troubleshooting

### "Invalid or non-existent authentication information"

Make sure your API token:
- Starts with `pypi-`
- Is for the correct repository (PyPI vs TestPyPI)
- Has the right scope (entire account or project-specific)

### "Package name already exists"

The name `llcuda` must be available on PyPI. Check: https://pypi.org/project/llcuda/

If taken, you'll need to choose a different name (e.g., `llcuda-inference`, `llama-cuda`, etc.)

### "File already exists"

You can't re-upload the same version. Either:
- Delete the file from `dist/` and rebuild
- Increment the version number

### Upload is slow or times out

- Check your internet connection
- Try uploading individual files: `twine upload dist/llcuda-0.1.0.tar.gz`
- Use `--verbose` flag: `twine upload --verbose dist/*`

## Security Best Practices

1. **Never commit `.pypirc` to git** (it contains secrets)
2. **Use API tokens, not passwords**
3. **Enable 2FA on PyPI account**
4. **Create project-specific tokens** after first upload
5. **Revoke tokens** if compromised
6. **Keep tokens secure** like passwords

## Useful Commands

```bash
# Check package before upload
python -m twine check dist/*

# View package metadata
tar -tzf dist/llcuda-0.1.0.tar.gz

# Install in development mode
pip install -e .

# Run tests before publishing
pytest tests/

# Check what files will be included
python setup.py sdist --dry-run
```

## Resources

- PyPI: https://pypi.org/
- TestPyPI: https://test.pypi.org/
- Packaging Guide: https://packaging.python.org/
- Twine docs: https://twine.readthedocs.io/
- Build docs: https://build.pypa.io/

## Support

If you encounter issues:
1. Check PyPI status: https://status.python.org/
2. PyPI support: https://pypi.org/help/
3. Packaging discussions: https://discuss.python.org/c/packaging/

---

**Ready to publish?** Start with Step 1 above!
