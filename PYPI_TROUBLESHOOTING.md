# PyPI Upload Troubleshooting Guide

## 🔴 Error: 403 Forbidden - Invalid Authentication

You're getting this error because the API token format is incorrect.

### ✅ Solution: Fix the Token Format

PyPI tokens should look like this:
```
pypi-AgEIcHlwaS5vcmcCJDg1YjUwOTZm...  (starts with "pypi-")
```

Your token appears to be missing the `pypi-` prefix.

---

## 📋 Step-by-Step Fix

### Option 1: Get a New Token (Recommended)

1. **Go to PyPI Account Settings:**
   - Visit: https://pypi.org/manage/account/token/
   - Login to your PyPI account

2. **Create New API Token:**
   - Click: "Add API token"
   - Token name: `llcuda-upload`
   - Scope: Select "Project: llcuda" (if llcuda already exists) OR "Entire account"
   - Click "Add token"

3. **Copy the Token:**
   - The token will be displayed **ONLY ONCE**
   - It should start with `pypi-`
   - Copy the ENTIRE token including `pypi-` prefix

4. **Upload with New Token:**
   ```bash
   cd /media/waqasm86/External1/Project-Nvidia/llcuda

   # Set credentials
   export TWINE_USERNAME=__token__
   export TWINE_PASSWORD=pypi-YOUR_NEW_COMPLETE_TOKEN_HERE

   # Upload
   /usr/local/bin/python3.11 -m twine upload dist/*
   ```

---

### Option 2: Use .pypirc File (Easier for Multiple Uploads)

1. **Create `~/.pypirc` file:**
   ```bash
   nano ~/.pypirc
   ```

2. **Add this content:**
   ```ini
   [pypi]
   username = __token__
   password = pypi-YOUR_COMPLETE_TOKEN_HERE
   ```

3. **Set permissions:**
   ```bash
   chmod 600 ~/.pypirc
   ```

4. **Upload:**
   ```bash
   cd /media/waqasm86/External1/Project-Nvidia/llcuda
   /usr/local/bin/python3.11 -m twine upload dist/*
   ```

---

### Option 3: Interactive Upload (Fallback)

If you prefer not to use tokens:

```bash
cd /media/waqasm86/External1/Project-Nvidia/llcuda

# This will prompt for username and password
/usr/local/bin/python3.11 -m twine upload dist/*

# When prompted:
# Username: waqasm86
# Password: <your PyPI password>
```

---

## 🔍 Common Token Issues

### Issue 1: Missing `pypi-` Prefix
**Wrong:**
```
AgEIcHlwaS5vcmcCJDg1YjUwOTZm...
```

**Correct:**
```
pypi-AgEIcHlwaS5vcmcCJDg1YjUwOTZm...
```

### Issue 2: Token Scope
If you created a project-scoped token but the project doesn't exist yet:
- First upload requires "Entire account" scope
- After first upload, create project-scoped token for llcuda

### Issue 3: Token Expired
- PyPI tokens can expire
- Create a new token if yours is old

### Issue 4: Wrong Repository
Make sure you're uploading to PyPI, not TestPyPI:
```bash
# PyPI (correct)
/usr/local/bin/python3.11 -m twine upload dist/*

# TestPyPI (wrong - only for testing)
/usr/local/bin/python3.11 -m twine upload --repository testpypi dist/*
```

---

## ✅ Verification Steps

After fixing the token, verify upload:

```bash
# 1. Set credentials correctly
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_COMPLETE_TOKEN_WITH_PREFIX

# 2. Test credentials first
/usr/local/bin/python3.11 -m twine upload --verbose dist/*

# 3. If successful, verify on PyPI
curl https://pypi.org/pypi/llcuda/json | grep version
```

---

## 🎯 Expected Successful Output

When upload succeeds, you should see:

```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading llcuda-0.2.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.0/44.0 kB • 00:00 • ?
Uploading llcuda-0.2.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.0/58.0 kB • 00:00 • ?

View at:
https://pypi.org/project/llcuda/0.2.0/
```

---

## 🆘 Still Having Issues?

### Check PyPI Status
```bash
curl https://status.python.org/
```

### Verbose Upload for Debugging
```bash
/usr/local/bin/python3.11 -m twine upload --verbose dist/*
```

### Check Token Permissions
- Go to: https://pypi.org/manage/account/token/
- Verify token exists and has correct scope
- Check expiration date

### Alternative: Use TestPyPI First
Test the upload process on TestPyPI:

```bash
# 1. Create token at: https://test.pypi.org/manage/account/token/

# 2. Upload to TestPyPI
/usr/local/bin/python3.11 -m twine upload --repository testpypi dist/*

# 3. Test installation
pip install --index-url https://test.pypi.org/simple/ llcuda==0.2.0

# 4. If works, upload to real PyPI
/usr/local/bin/python3.11 -m twine upload dist/*
```

---

## 📝 Quick Checklist

Before uploading, verify:

- [ ] Token starts with `pypi-`
- [ ] Token is copied completely (no truncation)
- [ ] Token has correct scope ("Entire account" for first upload)
- [ ] Username is set to `__token__` (exactly, not your username)
- [ ] You're in the correct directory (`/media/waqasm86/External1/Project-Nvidia/llcuda`)
- [ ] Dist files exist: `ls dist/`
- [ ] Packages verified: `/usr/local/bin/python3.11 -m twine check dist/*`

---

## 🎯 Most Likely Fix

Based on your error, the most likely issue is:

**Your token is missing the `pypi-` prefix**

**Solution:**
1. Go to https://pypi.org/manage/account/token/
2. Create a new token
3. Copy the COMPLETE token including `pypi-`
4. Use it like this:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEI...YOUR_COMPLETE_TOKEN

/usr/local/bin/python3.11 -m twine upload dist/*
```

---

**Good luck! Once uploaded, your package will be live at:**
**https://pypi.org/project/llcuda/0.2.0/**
