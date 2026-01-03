"""
Google Colab CUDA Version Diagnostic Script

This script checks the CUDA environment in Google Colab to diagnose
compatibility issues with llcuda binaries.

Run this in Google Colab to determine:
1. CUDA toolkit version
2. CUDA driver version
3. CUDA runtime version
4. GPU compute capability
5. PTX version support

Usage in Colab:
    !pip install llcuda -q
    !wget https://raw.githubusercontent.com/waqasm86/llcuda/main/examples/check_colab_cuda.py
    !python3 check_colab_cuda.py
"""

import subprocess
import sys
import os
from pathlib import Path

print("=" * 80)
print("Google Colab CUDA Diagnostic Tool")
print("=" * 80)
print()

# 1. Check CUDA Toolkit Version
print("[1] CUDA Toolkit Version (nvcc)")
print("-" * 80)
try:
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(result.stdout)
        # Extract version
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                print(f"✅ Found: {line.strip()}")
    else:
        print("⚠️  nvcc not found or error")
        print(result.stderr)
except Exception as e:
    print(f"❌ Error running nvcc: {e}")

print()

# 2. Check CUDA Driver Version (nvidia-smi)
print("[2] CUDA Driver Version (nvidia-smi)")
print("-" * 80)
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if 'CUDA Version' in line:
                print(f"✅ {line.strip()}")
                # Also print GPU info
                if i + 2 < len(lines):
                    print(lines[i+2].strip())
            elif 'Tesla T4' in line or 'GPU' in line:
                print(line.strip())
    else:
        print("⚠️  nvidia-smi not found or error")
except Exception as e:
    print(f"❌ Error running nvidia-smi: {e}")

print()

# 3. Check PyTorch CUDA Version
print("[3] PyTorch CUDA Runtime Version")
print("-" * 80)
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Compute Capability: {props.major}.{props.minor}")
                print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print("✅ PyTorch CUDA working")
    else:
        print("⚠️  CUDA not available in PyTorch")
except ImportError:
    print("⚠️  PyTorch not installed")
except Exception as e:
    print(f"❌ Error checking PyTorch: {e}")

print()

# 4. Check CUDA Libraries
print("[4] CUDA Runtime Libraries")
print("-" * 80)

cuda_paths = [
    "/usr/local/cuda/version.txt",
    "/usr/local/cuda/version.json",
    "/usr/local/cuda-12/version.txt",
    "/usr/local/cuda-11/version.txt",
]

for path in cuda_paths:
    if Path(path).exists():
        print(f"✅ Found: {path}")
        try:
            with open(path, 'r') as f:
                print(f"   Content: {f.read().strip()}")
        except:
            pass

# Check for libcudart
try:
    result = subprocess.run(
        ["find", "/usr", "-name", "libcudart.so*", "-type", "f"],
        capture_output=True,
        text=True,
        timeout=10
    )
    if result.stdout:
        libs = result.stdout.strip().split('\n')
        print(f"✅ Found {len(libs)} libcudart libraries:")
        for lib in libs[:5]:  # Show first 5
            print(f"   {lib}")
except Exception as e:
    print(f"⚠️  Could not search for CUDA libraries: {e}")

print()

# 5. Check Environment Variables
print("[5] CUDA Environment Variables")
print("-" * 80)
cuda_vars = ["CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "PATH"]
for var in cuda_vars:
    value = os.environ.get(var, "Not set")
    if var in ["LD_LIBRARY_PATH", "PATH"]:
        # Show only CUDA-related paths
        if value != "Not set":
            cuda_paths = [p for p in value.split(':') if 'cuda' in p.lower()]
            if cuda_paths:
                print(f"{var}:")
                for p in cuda_paths[:3]:  # Show first 3
                    print(f"  - {p}")
            else:
                print(f"{var}: (no CUDA paths)")
    else:
        print(f"{var}: {value}")

print()

# 6. Check llama-server if llcuda is installed
print("[6] llcuda Binary Information")
print("-" * 80)
try:
    import llcuda
    print(f"llcuda version: {llcuda.__version__}")

    from llcuda import ServerManager
    server = ServerManager()
    llama_server = server.find_llama_server()

    if llama_server:
        print(f"✅ llama-server found: {llama_server}")
        print(f"   Exists: {llama_server.exists()}")

        # Try to get version
        try:
            result = subprocess.run(
                [str(llama_server), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"   Version output:")
                for line in result.stdout.split('\n')[:10]:
                    if line.strip():
                        print(f"     {line}")
            else:
                print(f"   Error getting version:")
                for line in result.stderr.split('\n')[:10]:
                    if line.strip():
                        print(f"     {line}")
        except Exception as e:
            print(f"   ⚠️  Could not run --version: {e}")
    else:
        print("❌ llama-server not found")

except ImportError:
    print("⚠️  llcuda not installed")
except Exception as e:
    print(f"❌ Error checking llcuda: {e}")

print()

# 7. Summary and Recommendations
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("Copy the output above and share it to diagnose the CUDA PTX error.")
print()
print("Key information needed:")
print("  1. CUDA Driver Version (from nvidia-smi)")
print("  2. CUDA Toolkit Version (from nvcc)")
print("  3. GPU Compute Capability")
print("  4. llama-server binary version")
print()
print("=" * 80)
