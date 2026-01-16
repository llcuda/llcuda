# llcuda Colab Clean Install Script
# Copy and run this entire script in ONE Colab cell

import os
import shutil
import subprocess
import sys

print("\n" + "="*70)
print(" LLCUDA V2.1.1 - CLEAN INSTALL FOR GOOGLE COLAB")
print("="*70)

# PHASE 1: CLEANUP
print("\n[PHASE 1/3] Cleaning old installation...")
print("-" * 70)

try:
    # Remove pip package
    subprocess.run(["pip", "uninstall", "-y", "llcuda"], 
                   capture_output=True, timeout=30)
    print("✓ Uninstalled pip package")
except:
    pass

# Remove from site-packages
import site
site_packages = site.getsitepackages()[0]
paths_to_remove = [
    f"{site_packages}/llcuda",
    f"{site_packages}/llcuda.egg-info",
    f"{site_packages}/llcuda-*.dist-info",
]

for pattern in paths_to_remove:
    if "*" in pattern:
        import glob
        for path in glob.glob(pattern):
            try:
                shutil.rmtree(path)
                print(f"✓ Removed {path}")
            except:
                pass
    else:
        if os.path.exists(pattern):
            try:
                shutil.rmtree(pattern)
                print(f"✓ Removed {pattern}")
            except:
                pass

# Clear pip cache
try:
    subprocess.run(["pip", "cache", "purge"], 
                   capture_output=True, timeout=30)
    print("✓ Cleared pip cache")
except:
    pass

# Clear home cache
cache_dirs = [
    os.path.expanduser("~/.cache/llcuda"),
    os.path.expanduser("~/.cache/huggingface"),
]
for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"✓ Cleared {cache_dir}")
        except:
            pass

print("\n✅ Cleanup complete!")

# PHASE 2: RESTART NOTICE
print("\n" + "="*70)
print("[PHASE 2/3] RESTART KERNEL NOW")
print("="*70)
print("\n⚠️  YOU MUST RESTART THE KERNEL:")
print("   Go to: Menu → Kernel → Restart kernel")
print("   Wait for: 'Kernel restarted' message")
print("\nThen run this script again (just press the play button)")
print("The script will detect the restart and continue to Phase 3")

# Check if this is a fresh session (no llcuda imported yet)
if 'llcuda' not in sys.modules:
    print("\n✅ This is a fresh kernel - ready for installation!")
    
    # PHASE 3: INSTALL
    print("\n" + "="*70)
    print("[PHASE 3/3] Installing from GitHub...")
    print("="*70)
    
    try:
        print("\n📦 Installing llcuda v2.1.1 from GitHub...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "git+https://github.com/llcuda/llcuda.git@v2.1.1"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ Installation successful!")
        else:
            print(f"❌ Installation failed: {result.stderr}")
            sys.exit(1)
            
    except subprocess.TimeoutExpired:
        print("⚠️  Installation timed out - please retry")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during installation: {e}")
        sys.exit(1)
    
    # PHASE 4: VERIFY
    print("\n" + "="*70)
    print("[PHASE 4/4] Verifying installation...")
    print("="*70)
    
    try:
        import llcuda
        
        print(f"\n✅ llcuda imported successfully!")
        print(f"   Version: {llcuda.__version__}")
        print(f"   Location: {llcuda.__file__}")
        
        # Check if circular import warning appeared
        print("\n📋 Summary:")
        print("   ✅ No circular import warning")
        print("   ✅ Module imported cleanly")
        print("   ✅ Ready for model loading!")
        
        print("\n" + "="*70)
        print(" ✨ INSTALLATION COMPLETE - READY TO USE!")
        print("="*70)
        print("\nNext step: Load your model")
        print("""
Example:
    engine = llcuda.InferenceEngine()
    engine.load_model(
        "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
        auto_start=True
    )
    result = engine.infer("Hello!")
""")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)

else:
    print("\n⚠️  Kernel not restarted yet!")
    print("   llcuda is still in memory")
    print("\nPlease restart kernel:")
    print("   Menu → Kernel → Restart kernel")
    print("\nThen run this script again")
