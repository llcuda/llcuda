#!/usr/bin/env python3
"""
prepare_binaries.py
Extracts CUDA binaries into package for inclusion in PyPI wheel
Cross-platform (Windows, Linux, macOS)
"""

import os
import sys
import shutil
import tarfile
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main():
    print("=" * 70)
    print("Preparing llcuda binaries for PyPI package")
    print("=" * 70)

    # Get directories
    script_dir = Path(__file__).parent
    package_dir = script_dir / "llcuda"
    build_artifacts_dir = script_dir / "build-artifacts"
    tar_file = build_artifacts_dir / "llcuda-binaries-cuda12-t4-v2.0.2.tar.gz"

    print("\nStep 1: Checking for binaries archive...")
    if not tar_file.exists():
        print(f"❌ Error: Binaries archive not found at {tar_file}")
        print("\nPlease download it from:")
        print("https://github.com/waqasm86/llcuda/releases/download/v2.0.2/llcuda-binaries-cuda12-t4-v2.0.2.tar.gz")
        print(f"\nAnd place it in: {build_artifacts_dir}/")
        sys.exit(1)

    size_mb = tar_file.stat().st_size / (1024 * 1024)
    print(f"✅ Found binaries archive ({size_mb:.1f} MB)")

    # Clean up existing binaries
    print("\nStep 2: Cleaning up old binaries...")
    binaries_dir = package_dir / "binaries"
    lib_dir = package_dir / "lib"

    if binaries_dir.exists():
        shutil.rmtree(binaries_dir)
    if lib_dir.exists():
        shutil.rmtree(lib_dir)
    print("✅ Cleaned up old binaries")

    # Extract binaries
    print("\nStep 3: Extracting binaries...")
    temp_dir = script_dir / "temp_extract"
    temp_dir.mkdir(exist_ok=True)

    try:
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(temp_dir)
        print("✅ Extracted to temporary directory")

        # Copy binaries to package
        print("\nStep 4: Installing binaries into package...")

        bin_source = temp_dir / "bin"
        if bin_source.exists():
            cuda12_dir = binaries_dir / "cuda12"
            cuda12_dir.mkdir(parents=True, exist_ok=True)

            for item in bin_source.iterdir():
                dest = cuda12_dir / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                    # Make executable on Unix
                    if os.name != 'nt':
                        dest.chmod(0o755)

            bin_count = len(list(cuda12_dir.iterdir()))
            print(f"✅ Installed {bin_count} binaries")
        else:
            print("❌ Error: bin/ directory not found in archive")
            sys.exit(1)

        lib_source = temp_dir / "lib"
        if lib_source.exists():
            lib_dir.mkdir(parents=True, exist_ok=True)

            for item in lib_source.iterdir():
                if item.is_file():
                    shutil.copy2(item, lib_dir / item.name)

            lib_count = len(list(lib_dir.iterdir()))
            print(f"✅ Installed {lib_count} libraries")
        else:
            print("❌ Error: lib/ directory not found in archive")
            sys.exit(1)

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    # Verify installation
    print("\nStep 5: Verifying installation...")
    llama_server = binaries_dir / "cuda12" / "llama-server"
    if llama_server.exists() and llama_server.is_file():
        print("✅ llama-server is installed")
    else:
        print("❌ Error: llama-server not found")
        sys.exit(1)

    libggml_cuda = lib_dir / "libggml-cuda.so"
    if not libggml_cuda.exists():
        # Try versioned name
        libggml_cuda = list(lib_dir.glob("libggml-cuda.so*"))
        if libggml_cuda:
            libggml_cuda = libggml_cuda[0]

    if libggml_cuda and Path(libggml_cuda).exists():
        cuda_size_mb = Path(libggml_cuda).stat().st_size / (1024 * 1024)
        print(f"✅ {libggml_cuda.name} found ({cuda_size_mb:.1f} MB)")
    else:
        print("❌ Error: libggml-cuda.so not found")
        sys.exit(1)

    # Calculate total size
    def get_dir_size(path):
        total = 0
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
        return total

    binaries_size = get_dir_size(binaries_dir) / (1024 * 1024)
    lib_size = get_dir_size(lib_dir) / (1024 * 1024)
    total_size = binaries_size + lib_size

    print("\n" + "=" * 70)
    print("✅ Binaries prepared successfully!")
    print("=" * 70)
    print(f"\nBinaries location: {binaries_dir / 'cuda12'}/")
    print(f"Libraries location: {lib_dir}/")
    print(f"Total size: {total_size:.1f} MB")
    print("\nNext steps:")
    print("  1. Run: python -m build")
    print("  2. Check wheel size: ls -lh dist/ (or dir dist\\ on Windows)")
    print("  3. Upload to PyPI: python -m twine upload dist/*")
    print("")


if __name__ == "__main__":
    main()
