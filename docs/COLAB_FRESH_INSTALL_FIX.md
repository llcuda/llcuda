# Google Colab - Fix Circular Import + llama-server Crash

## Problem Summary

You're experiencing two issues in Google Colab:

1. **RuntimeWarning: Circular Import** (still showing despite fix)
   - The warning appears because Colab has cached the OLD version
   - The fix exists on GitHub but isn't installed in your environment yet

2. **RuntimeError: llama-server process died unexpectedly**
   - llama-server binary crashes when trying to start
   - Usually caused by CUDA library issues or binary path problems

---

## Solution: Complete Clean Install

The issue is that your Colab environment has a **cached/stale version** of llcuda. Follow these steps exactly:

### Step 1: Completely Uninstall (Clear Cache)

```python
# Run this in a Colab cell
import os
import shutil
import sys

print("🧹 Cleaning up llcuda installation...\n")

# Method 1: pip uninstall
!pip uninstall -y llcuda 2>/dev/null || echo "✓ pip uninstall done"

# Method 2: Remove site-packages
import site
site_packages = site.getsitepackages()[0]
llcuda_path = f"{site_packages}/llcuda"
if os.path.exists(llcuda_path):
    shutil.rmtree(llcuda_path)
    print(f"✓ Removed {llcuda_path}")

# Method 3: Clear pip cache
!pip cache purge 2>/dev/null || echo "✓ pip cache cleared"

# Method 4: Clear home directory cache
cache_dir = os.path.expanduser("~/.cache/llcuda")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"✓ Removed {cache_dir}")

print("\n✅ Cleanup complete!")
```

### Step 2: Restart Kernel

⚠️ **CRITICAL:** Restart the kernel to clear all imports from memory

```
Menu → Kernel → Restart kernel
```

Wait for it to say "Kernel restarted"

### Step 3: Fresh Installation from GitHub

```python
print("📦 Installing llcuda v2.1.1 from GitHub (fresh)...\n")

!pip install -q git+https://github.com/llcuda/llcuda.git@v2.1.1

print("✅ Installation complete!")
```

### Step 4: Verify Installation (Before using models)

```python
print("🔍 Verifying installation...\n")

import llcuda

print(f"Version: {llcuda.__version__}")
print(f"Location: {llcuda.__file__}")

# Check for the circular import warning
# EXPECTED: No RuntimeWarning in output above
print("\n✅ Installation verified!")
```

Expected output (after fix):
```
Version: 2.1.1
Location: /usr/local/lib/python3.12/dist-packages/llcuda/__init__.py
(no warnings!)

✅ Installation verified!
```

---

## Why This Works

### The Circular Import Issue

**Root Cause:** Old cached version from before commit 65d5377

**The Fix (in bootstrap.py line 21):**
- OLD: `from llcuda import __version__` ← causes circular import
- NEW: (removed) ← no circular dependency

**To Get It:** Must reinstall from GitHub with the fix

### The llama-server Crash

**Root Cause:** Usually binary compatibility or CUDA library paths

**Why Clean Install Helps:**
1. Ensures correct CUDA binary (v2.1.1)
2. Proper environment paths setup
3. Clean cache directories
4. Fresh extraction of binaries

---

## Detailed Step-by-Step (Copy & Paste Ready)

### Cell 1: Complete Cleanup
```python
import os
import shutil

print("Step 1/3: Complete cleanup of old installation")
print("=" * 70)

# Uninstall package
os.system("pip uninstall -y llcuda 2>/dev/null")

# Remove site-packages
import site
site_packages = site.getsitepackages()[0]
llcuda_path = f"{site_packages}/llcuda"
llcuda_egg_path = f"{site_packages}/llcuda.egg-info"

for path in [llcuda_path, llcuda_egg_path]:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed: {path}")

# Clear caches
os.system("pip cache purge 2>/dev/null")

cache_dir = os.path.expanduser("~/.cache/llcuda")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Removed cache: {cache_dir}")

print("\n✅ Cleanup complete - restart kernel now!")
```

Then: **Kernel → Restart Kernel** (WAIT for it to restart)

### Cell 2 (After Restart): Fresh Install
```python
print("Step 2/3: Fresh installation from GitHub")
print("=" * 70)

!pip install -q git+https://github.com/llcuda/llcuda.git@v2.1.1

print("\n✅ Installation complete!")
```

### Cell 3: Verify
```python
print("Step 3/3: Verify installation")
print("=" * 70)

import llcuda
import warnings

# Check for warnings
print(f"llcuda version: {llcuda.__version__}")

# If you see NO RuntimeWarning above, the fix is installed ✅
print("\n✅ Ready for model loading!")
```

---

## What to Expect

### Correct Output (After Fix)
```
Step 3/3: Verify installation
======================================================================
llcuda version: 2.1.1

✅ Ready for model loading!
```

### ❌ Wrong Output (Still has old version)
```
RuntimeWarning: llcuda bootstrap failed: cannot import name '__version__'
```

If you see this, it means:
- Kernel wasn't restarted properly
- Old version still cached
- Try restart again: Kernel → Restart

---

## Now Load Your Model

After verification passes, you can safely load the model:

```python
import llcuda
import time

# Initialize engine
engine = llcuda.InferenceEngine()

print("📥 Loading Gemma 3-1B-IT Q4_K_M from Unsloth...")

start_time = time.time()

engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=False,        # Show errors if they occur
    auto_start=True
)

load_time = time.time() - start_time

print(f"\n✅ Model loaded in {load_time:.1f}s!")
```

---

## Troubleshooting llama-server Crash

If you still see "llama-server process died unexpectedly":

### Debug Option 1: Show Error Messages
```python
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=False,  # ← Shows actual error instead of hiding it
    auto_start=True
)
```

### Debug Option 2: Check Binary
```python
import subprocess
import os

binary_path = os.path.expanduser("~/.cache/llcuda/llama-server")
if os.path.exists(binary_path):
    print(f"✓ Binary exists: {binary_path}")
    
    # Try to run it directly
    result = subprocess.run(
        [binary_path, "--version"],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    if result.returncode == 0:
        print(f"✓ Binary works: {result.stdout}")
    else:
        print(f"✗ Binary error: {result.stderr}")
else:
    print("✗ Binary not found - reinstall")
```

### Debug Option 3: Check CUDA
```python
!nvidia-smi

# Check library paths
import os
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')}")
```

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| RuntimeWarning: circular import | Old cached version | Restart kernel after pip install |
| llama-server crashes | Binary compatibility | Show errors with `silent=False` |
| Model loads but inference fails | CUDA library issue | Check `nvidia-smi` and LD_LIBRARY_PATH |
| Still seeing old version | Kernel not restarted | Use Kernel menu to restart (not just re-run) |

---

## GitHub Status

The fix is already live on GitHub:
- Commit: `65d5377`
- File: `llcuda/_internal/bootstrap.py`
- Change: Removed line 21 (`from llcuda import __version__`)
- Status: Merged and pushed

You just need to **reinstall to get it** in your Colab environment.

---

## Quick Command Reference

```bash
# Clean uninstall
pip uninstall -y llcuda

# Fresh install (with fix)
pip install -q git+https://github.com/llcuda/llcuda.git@v2.1.1

# Verify
python -c "import llcuda; print(f'Version: {llcuda.__version__}')"
```

---

**Expected Timeline:**
1. Run cleanup cell: 30 seconds
2. Restart kernel: 10 seconds  
3. Fresh install: 30 seconds
4. Verify: 5 seconds
5. Model loading: 2-3 minutes (first time, faster after)

**Total: ~5-10 minutes for first complete setup**

---

Let me know what error message you get from `silent=False` if llama-server still crashes!
