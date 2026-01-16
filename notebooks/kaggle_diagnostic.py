#!/usr/bin/env python3
"""
Kaggle System Diagnostic Script
Run this in a Kaggle notebook cell to gather complete environment info.
Copy the output and share it for debugging.
"""

print("=" * 80)
print("KAGGLE SYSTEM DIAGNOSTIC - llcuda v2.2.0 Build Environment")
print("=" * 80)

# ============================================================================
# 1. SYSTEM INFO
# ============================================================================
print("\n" + "=" * 80)
print("1. SYSTEM INFORMATION")
print("=" * 80)

import platform
import os
import subprocess

print(f"Python Version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")
print(f"Processor: {platform.processor()}")

# Check if running in Kaggle
print(f"\nKaggle Environment: {'KAGGLE_KERNEL_RUN_TYPE' in os.environ}")
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    print(f"   Run Type: {os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'N/A')}")

# ============================================================================
# 2. GPU INFORMATION
# ============================================================================
print("\n" + "=" * 80)
print("2. GPU INFORMATION")
print("=" * 80)

# nvidia-smi
result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", 
                         "--format=csv,noheader"], capture_output=True, text=True)
if result.returncode == 0:
    print("nvidia-smi GPU info:")
    for i, line in enumerate(result.stdout.strip().split('\n')):
        print(f"   GPU {i}: {line}")
else:
    print("nvidia-smi failed")

# Full nvidia-smi output
print("\nFull nvidia-smi output:")
os.system("nvidia-smi")

# ============================================================================
# 3. CUDA INFORMATION
# ============================================================================
print("\n" + "=" * 80)
print("3. CUDA TOOLKIT INFORMATION")
print("=" * 80)

# CUDA version
result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
if result.returncode == 0:
    print("nvcc --version:")
    for line in result.stdout.strip().split('\n'):
        print(f"   {line}")

# CUDA paths
print(f"\nCUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'NOT SET')}")

# Check common CUDA locations
cuda_locations = [
    "/usr/local/cuda",
    "/usr/local/cuda-12",
    "/usr/local/cuda-12.5",
    "/opt/cuda",
]
print("\nCUDA Directory Check:")
for loc in cuda_locations:
    if os.path.exists(loc):
        if os.path.islink(loc):
            target = os.readlink(loc)
            print(f"   ✅ {loc} -> {target}")
        else:
            print(f"   ✅ {loc} (directory)")
    else:
        print(f"   ❌ {loc} (not found)")

# ============================================================================
# 4. CUDA LIBRARY FILES (CRITICAL FOR CMAKE)
# ============================================================================
print("\n" + "=" * 80)
print("4. CUDA LIBRARY FILES (Critical for CMake)")
print("=" * 80)

# Check for libcuda.so in various locations
libcuda_locations = [
    "/usr/local/cuda/lib64/libcuda.so",
    "/usr/local/cuda/lib64/libcuda.so.1",
    "/usr/local/cuda/lib64/stubs/libcuda.so",
    "/usr/local/cuda/targets/x86_64-linux/lib/libcuda.so",
    "/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib/libcuda.so",
    "/usr/lib/libcuda.so.1",
]

print("libcuda.so locations:")
for loc in libcuda_locations:
    if os.path.exists(loc):
        if os.path.islink(loc):
            target = os.readlink(loc)
            print(f"   ✅ {loc} -> {target}")
        else:
            size = os.path.getsize(loc)
            print(f"   ✅ {loc} ({size} bytes)")
    else:
        print(f"   ❌ {loc}")

# List stubs directory
print("\n/usr/local/cuda/lib64/stubs/ contents:")
stubs_dir = "/usr/local/cuda/lib64/stubs"
if os.path.exists(stubs_dir):
    for f in os.listdir(stubs_dir):
        full_path = os.path.join(stubs_dir, f)
        if os.path.islink(full_path):
            target = os.readlink(full_path)
            print(f"   {f} -> {target}")
        else:
            size = os.path.getsize(full_path)
            print(f"   {f} ({size} bytes)")
else:
    print("   Directory not found")

# List main lib64 directory (libcuda files only)
print("\n/usr/local/cuda/lib64/ libcuda* files:")
lib64_dir = "/usr/local/cuda/lib64"
if os.path.exists(lib64_dir):
    for f in sorted(os.listdir(lib64_dir)):
        if 'cuda' in f.lower() and 'lib' in f.lower():
            full_path = os.path.join(lib64_dir, f)
            if os.path.islink(full_path):
                target = os.readlink(full_path)
                print(f"   {f} -> {target}")
            elif os.path.isfile(full_path):
                size = os.path.getsize(full_path)
                print(f"   {f} ({size} bytes)")

# ============================================================================
# 5. CMAKE CUDA MODULE CHECK
# ============================================================================
print("\n" + "=" * 80)
print("5. CMAKE INFORMATION")
print("=" * 80)

result = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
if result.returncode == 0:
    print(f"CMake version: {result.stdout.strip().split()[2]}")

# Check CMake modules
cmake_module_paths = [
    "/usr/share/cmake-3.22/Modules",
    "/usr/share/cmake/Modules",
    "/usr/local/share/cmake/Modules",
]
print("\nCMake FindCUDA modules:")
for path in cmake_module_paths:
    if os.path.exists(path):
        cuda_modules = [f for f in os.listdir(path) if 'cuda' in f.lower() or 'CUDA' in f]
        if cuda_modules:
            print(f"   {path}:")
            for m in sorted(cuda_modules)[:10]:
                print(f"      {m}")

# ============================================================================
# 6. ENVIRONMENT VARIABLES
# ============================================================================
print("\n" + "=" * 80)
print("6. RELEVANT ENVIRONMENT VARIABLES")
print("=" * 80)

env_vars = [
    "PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", 
    "CUDA_HOME", "CUDA_PATH", "CUDA_ROOT",
    "CMAKE_PREFIX_PATH", "CMAKE_LIBRARY_PATH",
    "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
]

for var in env_vars:
    value = os.environ.get(var, "NOT SET")
    if value != "NOT SET" and len(value) > 100:
        # Truncate long paths
        value = value[:100] + "..."
    print(f"{var}:")
    print(f"   {value}")

# ============================================================================
# 7. LDCONFIG CHECK
# ============================================================================
print("\n" + "=" * 80)
print("7. LDCONFIG CUDA LIBRARIES")
print("=" * 80)

result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
if result.returncode == 0:
    cuda_libs = [line for line in result.stdout.split('\n') if 'cuda' in line.lower()]
    print(f"Found {len(cuda_libs)} CUDA-related libraries in ldconfig:")
    for lib in cuda_libs[:20]:  # Show first 20
        print(f"   {lib.strip()}")
    if len(cuda_libs) > 20:
        print(f"   ... and {len(cuda_libs) - 20} more")

# ============================================================================
# 8. FIND ACTUAL libcuda.so.1
# ============================================================================
print("\n" + "=" * 80)
print("8. SEARCHING FOR libcuda.so.1 (the real driver library)")
print("=" * 80)

result = subprocess.run(["find", "/", "-name", "libcuda.so*", "-type", "f", "2>/dev/null"], 
                        capture_output=True, text=True, shell=True)
# Use locate or find
os.system("find /usr -name 'libcuda.so*' 2>/dev/null | head -20")
os.system("find /lib -name 'libcuda.so*' 2>/dev/null | head -10")

# ============================================================================
# 9. CHECK CUDA DRIVER VERSION
# ============================================================================
print("\n" + "=" * 80)
print("9. CUDA DRIVER VERSION (from nvidia-smi)")
print("=" * 80)

os.system("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")

# ============================================================================
# 10. RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("10. DIAGNOSTIC COMPLETE - COPY ALL OUTPUT ABOVE")
print("=" * 80)
print("""
Please copy ALL the output above and share it.
This will help diagnose the CUDA::cuda_driver CMake issue.
""")
