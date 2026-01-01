#!/usr/bin/env python3.11
"""
Test script to verify llcuda installation and setup.
Run this to check if everything is working correctly.
"""

import sys
print(f"Python version: {sys.version}\n")

# Test 1: Import llcuda
print("=" * 60)
print("Test 1: Importing llcuda")
print("=" * 60)
try:
    import llcuda
    print(f"✓ llcuda version {llcuda.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Failed to import llcuda: {e}")
    sys.exit(1)

# Test 2: Check CUDA
print("\n" + "=" * 60)
print("Test 2: CUDA Availability")
print("=" * 60)
cuda_available = llcuda.check_cuda_available()
print(f"CUDA Available: {'✓ Yes' if cuda_available else '✗ No'}")

if cuda_available:
    gpu_info = llcuda.get_cuda_device_info()
    if gpu_info:
        print(f"CUDA Version: {gpu_info['cuda_version'] or 'Unknown'}")
        print(f"Number of GPUs: {len(gpu_info['gpus'])}")
        for i, gpu in enumerate(gpu_info['gpus']):
            print(f"\nGPU {i}:")
            print(f"  Name: {gpu['name']}")
            print(f"  Memory: {gpu['memory']}")
            print(f"  Driver: {gpu['driver_version']}")

# Test 3: Find llama-server
print("\n" + "=" * 60)
print("Test 3: Finding llama-server")
print("=" * 60)
from llcuda import ServerManager
manager = ServerManager()
server_path = manager.find_llama_server()
if server_path:
    print(f"✓ Found llama-server at: {server_path}")
else:
    print("✗ llama-server not found")
    print("  Set LLAMA_CPP_DIR environment variable or install llama.cpp")

# Test 4: Find GGUF models
print("\n" + "=" * 60)
print("Test 4: Finding GGUF Models")
print("=" * 60)
models = llcuda.find_gguf_models()
print(f"Found {len(models)} GGUF models")
for i, model in enumerate(models[:3]):  # Show first 3
    size_mb = model.stat().st_size / (1024 * 1024)
    print(f"  {i+1}. {model.name} ({size_mb:.0f} MB)")
if len(models) > 3:
    print(f"  ... and {len(models) - 3} more")

# Test 5: Environment setup
print("\n" + "=" * 60)
print("Test 5: Environment Setup")
print("=" * 60)
env_vars = llcuda.setup_environment()
if env_vars:
    print("✓ Environment variables set:")
    for key, value in env_vars.items():
        print(f"  {key}={value[:60]}...")
else:
    print("✓ Environment already configured")

# Test 6: Get llama-cpp-cuda path
print("\n" + "=" * 60)
print("Test 6: llama-cpp-cuda Installation")
print("=" * 60)
llama_cpp_path = llcuda.get_llama_cpp_cuda_path()
if llama_cpp_path:
    print(f"✓ Found llama-cpp-cuda at: {llama_cpp_path}")
    print(f"  Checking bin/llama-server: ", end='')
    if (llama_cpp_path / 'bin' / 'llama-server').exists():
        print("✓ Found")
    else:
        print("✗ Not found")
else:
    print("✗ llama-cpp-cuda not found in common locations")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
status = []
status.append(("llcuda import", True))
status.append(("CUDA available", cuda_available))
status.append(("llama-server found", server_path is not None))
status.append(("Models found", len(models) > 0))
status.append(("llama-cpp-cuda found", llama_cpp_path is not None))

all_good = all(s[1] for s in status)

for name, success in status:
    icon = "✓" if success else "✗"
    print(f"{icon} {name}")

if all_good:
    print("\n✓ All checks passed! Ready to use llcuda.")
    print("\nNext steps:")
    print("  1. Open JupyterLab")
    print("  2. Open examples/quickstart_jupyterlab.ipynb")
    print("  3. Run the cells to start using llcuda")
else:
    print("\n⚠ Some checks failed. Please review the output above.")
    print("  See README.md for setup instructions.")

print("\n" + "=" * 60)
