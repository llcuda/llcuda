#!/usr/bin/env python3.11
"""
Quick test script for llcuda installation

Tests:
1. Import llcuda
2. Check auto-configuration
3. Print system info
4. List registry models
"""

print("=" * 70)
print("llcuda Installation Test")
print("=" * 70)

# Test 1: Import
print("\n[1/4] Testing import...")
try:
    import llcuda
    print(f"✓ llcuda imported successfully")
    print(f"  Version: {llcuda.__version__}")
except ImportError as e:
    print(f"✗ Failed to import llcuda: {e}")
    exit(1)

# Test 2: Check auto-configuration
print("\n[2/4] Checking auto-configuration...")
try:
    import os
    llama_server_path = os.environ.get('LLAMA_SERVER_PATH')
    ld_library_path = os.environ.get('LD_LIBRARY_PATH')

    if llama_server_path:
        print(f"✓ LLAMA_SERVER_PATH auto-configured: {llama_server_path}")
    else:
        print(f"✗ LLAMA_SERVER_PATH not set")

    if ld_library_path and 'llcuda' in ld_library_path:
        print(f"✓ LD_LIBRARY_PATH includes llcuda libs")
    else:
        print(f"⚠ LD_LIBRARY_PATH may not include llcuda libs")
except Exception as e:
    print(f"✗ Auto-configuration check failed: {e}")

# Test 3: System info
print("\n[3/4] Checking system info...")
try:
    llcuda.print_system_info()
    print("✓ System info check complete")
except Exception as e:
    print(f"✗ System info check failed: {e}")

# Test 4: List registry models
print("\n[4/4] Listing registry models...")
try:
    from llcuda.models import list_registry_models
    models = list_registry_models()
    print(f"✓ Found {len(models)} models in registry")
    print("\nFirst 3 models:")
    for i, (name, info) in enumerate(list(models.items())[:3], 1):
        print(f"  {i}. {name} - {info['description']}")
except Exception as e:
    print(f"✗ Registry models check failed: {e}")

# Final summary
print("\n" + "=" * 70)
print("Installation Test Complete!")
print("=" * 70)
print("\nNext steps:")
print("  engine = llcuda.InferenceEngine()")
print("  engine.load_model('gemma-3-1b-Q4_K_M')  # Downloads model")
print("  result = engine.infer('What is AI?')")
print("  print(result.text)")
print("=" * 70)
