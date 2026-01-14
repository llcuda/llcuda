"""
llcuda v2.1 Bootstrap Module - Tesla T4 Only

This module handles first-time setup for llcuda v2.1:
- Verifies GPU is Tesla T4 or compatible (SM 7.5+)
- Downloads T4-optimized CUDA 12 binaries (264 MB)
- Downloads llcuda v2.1 native extension if needed

Designed for Google Colab and modern GPUs with Tensor Core support.
"""

import os
import sys
import json
import shutil
import tarfile
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple
import subprocess
from llcuda import __version__

try:
    from huggingface_hub import hf_hub_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Configuration for llcuda v2.1.0 (uses v2.0.6 binaries - 100% compatible)
#GITHUB_RELEASE_URL = "https://github.com/llcuda/llcuda/releases/download/v2.1.0"
GITHUB_RELEASE_URL = f"https://github.com/llcuda/llcuda/releases/download/v{__version__}"
HF_REPO_ID = "waqasm86/llcuda-models"

# T4-only binary bundle (v2.0.6 binaries work with v2.1.0 - pure Python API layer)
#T4_BINARY_BUNDLE = "llcuda-binaries-cuda12-t4-v2.1.0.tar.gz"  # 266 MB
T4_BINARY_BUNDLE = f"llcuda-binaries-cuda12-t4-v{__version__}.tar.gz"
T4_NATIVE_BUNDLE = "llcuda-v2-native-t4.tar.gz"        # ~100 MB

# Minimum compute capability for llcuda v2.1
MIN_COMPUTE_CAPABILITY = 7.5  # Tesla T4, RTX 20xx+, A100, H100

# Paths
PACKAGE_DIR = Path(__file__).parent.parent
BINARIES_DIR = PACKAGE_DIR / "binaries"
LIB_DIR = PACKAGE_DIR / "lib"
MODELS_DIR = PACKAGE_DIR / "models"
CACHE_DIR = Path.home() / ".cache" / "llcuda"


def detect_gpu_compute_capability() -> Optional[Tuple[str, str]]:
    """
    Detect NVIDIA GPU compute capability using nvidia-smi.

    Returns:
        Tuple of (gpu_name, compute_capability) or None if no GPU found
        Example: ("Tesla T4", "7.5")
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            # Take first GPU
            line = result.stdout.strip().split("\n")[0]
            gpu_name, compute_cap = line.split(",")
            return gpu_name.strip(), compute_cap.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    return None


def detect_platform() -> str:
    """
    Detect execution platform (local, colab, kaggle).

    Returns:
        Platform name: "colab", "kaggle", or "local"
    """
    # Check for Colab
    try:
        import google.colab

        return "colab"
    except ImportError:
        pass

    # Check for Kaggle
    if os.path.exists("/kaggle"):
        return "kaggle"

    return "local"


def verify_gpu_compatibility(gpu_name: str, compute_cap: str) -> bool:
    """
    Verify GPU is compatible with llcuda v2.0 (SM 7.5+).

    Args:
        gpu_name: GPU name from nvidia-smi
        compute_cap: Compute capability (e.g., "7.5", "8.0")

    Returns:
        True if compatible, False otherwise

    Raises:
        RuntimeError if GPU is not compatible
    """
    try:
        cc_float = float(compute_cap)
    except (ValueError, TypeError):
        raise RuntimeError(f"Invalid compute capability: {compute_cap}")

    gpu_lower = gpu_name.lower()

    # Check minimum requirement
    if cc_float < MIN_COMPUTE_CAPABILITY:
        print()
        print("=" * 70)
        print("❌ INCOMPATIBLE GPU DETECTED")
        print("=" * 70)
        print()
        print(f"  Your GPU: {gpu_name} (SM {compute_cap})")
        print(f"  Required: Tesla T4 (SM 7.5)")
        print()
        print("  llcuda v2.1 is designed exclusively for Tesla T4 GPU")
        print()
        print("  Compatible environment:")
        print("    - Google Colab (free tier with Tesla T4)")
        print()
        print("  For other GPUs, use llcuda v1.2.2:")
        print()
        print("=" * 70)
        raise RuntimeError(f"GPU compute capability {compute_cap} < {MIN_COMPUTE_CAPABILITY} (minimum required)")

    # Tesla T4 verification
    if cc_float == 7.5 and "t4" in gpu_lower:
        print(f"  ✅ Tesla T4 detected - Perfect for llcuda v2.1!")
    elif cc_float == 7.5:
        print(f"  ⚠️  {gpu_name} (SM {compute_cap}) - May work but not tested")
        print(f"      llcuda v2.1 is optimized exclusively for Tesla T4")
    else:
        print(f"  ⚠️  {gpu_name} (SM {compute_cap}) - Not tested")
        print(f"      llcuda v2.1 is designed for Tesla T4 (SM 7.5)")

    return True


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> None:
    """
    Download file with progress bar.

    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress bar
    """
    import urllib.request

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            mb_downloaded = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r{desc}: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
            )
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Download failed: {e}")


def extract_tarball(tarball_path: Path, dest_dir: Path) -> None:
    """
    Extract tarball to destination directory.

    Args:
        tarball_path: Path to tarball
        dest_dir: Destination directory
    """
    print(f"📦 Extracting {tarball_path.name}...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tarball_path, "r:gz") as tar:
        # First, check what's in the tarball
        members = tar.getmembers()
        print(f"Found {len(members)} files in archive")

        # Extract all
        tar.extractall(dest_dir)

        # List extracted files for debugging
        extracted_files = list(dest_dir.rglob("*"))
        print(f"Extracted {len(extracted_files)} files to {dest_dir}")

    print("✅ Extraction complete!")


def download_t4_binaries() -> None:
    """
    Download and install T4-optimized CUDA 12 binaries for llcuda v2.0.

    Includes:
    - llama-server (6.5 MB)
    - libggml-cuda.so with FlashAttention (219 MB)
    - Supporting libraries

    Total size: 264 MB
    """
    # Check if binaries already exist
    llama_server = BINARIES_DIR / "cuda12" / "llama-server"
    if llama_server.exists() and llama_server.stat().st_size > 0:
        print("✅ T4 binaries already installed")
        return

    print("=" * 70)
    print("🎯 llcuda v2.1 First-Time Setup - Tesla T4 Optimized")
    print("=" * 70)
    print()

    # Detect GPU and verify compatibility
    gpu_info = detect_gpu_compute_capability()
    platform = detect_platform()

    if gpu_info:
        gpu_name, compute_cap = gpu_info
        print(f"🎮 GPU Detected: {gpu_name} (Compute {compute_cap})")

        # Verify SM 7.5+ compatibility
        try:
            verify_gpu_compatibility(gpu_name, compute_cap)
        except RuntimeError as e:
            # GPU not compatible, abort
            raise
    else:
        print("❌ No NVIDIA GPU detected")
        print()
        print("llcuda v2.1 requires an NVIDIA GPU with SM 7.5+ (Tesla T4 or newer)")
        print("Please ensure:")
        print("  1. NVIDIA drivers are installed")
        print("  2. nvidia-smi command is available")
        print("  3. GPU is properly detected by the system")
        print()
        raise RuntimeError("No compatible NVIDIA GPU found")

    print(f"🌐 Platform: {platform.capitalize()}")
    print()

    # Download T4 binary bundle
    print("📦 Downloading T4-optimized binaries (264 MB)...")
    print("    Features: FlashAttention + Tensor Cores + CUDA Graphs")
    print()

    cache_tarball = CACHE_DIR / T4_BINARY_BUNDLE
    bundle_url = f"{GITHUB_RELEASE_URL}/{T4_BINARY_BUNDLE}"

    if not cache_tarball.exists():
        print(f"📥 Downloading from GitHub releases...")
        print(f"   URL: {bundle_url}")
        print(f"   This is a one-time download (~264 MB)")
        print()

        download_file(bundle_url, cache_tarball, "Downloading T4 binaries")
        print()
    else:
        print(f"✅ Using cached binaries from {cache_tarball}")
        print()

    # Extract binaries
    temp_extract_dir = CACHE_DIR / "extract"
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    extract_tarball(cache_tarball, temp_extract_dir)

    print("📂 Installing binaries and libraries...")

    # The archive contains bin/ and lib/ directories
    # We need to reorganize them into binaries/cuda12/ and lib/

    bin_dir = temp_extract_dir / "bin"
    lib_dir = temp_extract_dir / "lib"

    if bin_dir.exists() and lib_dir.exists():
        # Archive has bin/ and lib/ structure (v1.1.7 format)
        print("  Found bin/ and lib/ directories")

        # Create cuda12 directory and copy all binaries
        cuda12_dir = BINARIES_DIR / "cuda12"
        cuda12_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files from bin/ to binaries/cuda12/
        for item in bin_dir.iterdir():
            if item.is_file():
                dest = cuda12_dir / item.name
                shutil.copy2(item, dest)
                # Make executable if it's a binary (no extension)
                if not item.suffix or item.suffix == '.sh':
                    try:
                        dest.chmod(0o755)
                    except:
                        pass

        print(f"  Copied {len(list(bin_dir.iterdir()))} binaries to {cuda12_dir}")

        # Copy all libraries from lib/ to lib/
        LIB_DIR.mkdir(parents=True, exist_ok=True)
        for item in lib_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, LIB_DIR / item.name)

        print(f"  Copied {len(list(lib_dir.iterdir()))} libraries to {LIB_DIR}")
        print("✅ Binaries installed successfully!")

    else:
        # Fallback: Try to find binaries anywhere in temp_extract_dir
        print("⚠️  Expected structure not found, searching for binaries...")
        server_binaries = list(temp_extract_dir.rglob("llama-server"))
        so_files = list(temp_extract_dir.rglob("*.so*"))

        if server_binaries:
            BINARIES_DIR.mkdir(parents=True, exist_ok=True)
            cuda12_dir = BINARIES_DIR / "cuda12"
            cuda12_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(server_binaries[0], cuda12_dir / "llama-server")
            server_binaries[0].chmod(0o755)
            cuda12_dir.joinpath("llama-server").chmod(0o755)
            print(f"  ✅ Found and copied llama-server")

        if so_files:
            LIB_DIR.mkdir(parents=True, exist_ok=True)
            for so_file in so_files:
                shutil.copy2(so_file, LIB_DIR / so_file.name)
            print(f"  ✅ Found and copied {len(so_files)} libraries")

        if server_binaries or so_files:
            print("✅ Binaries installed successfully!")
        else:
            print("❌ No binaries found in archive!")

    # Cleanup
    shutil.rmtree(temp_extract_dir, ignore_errors=True)
    print()


def download_default_model() -> None:
    """
    Download default model (Gemma 3 1B) from Hugging Face.
    """
    if not HF_AVAILABLE:
        print("⚠️  huggingface_hub not available, skipping model download")
        print("   Install with: pip install huggingface_hub")
        return

    # Check if model already exists
    model_file = MODELS_DIR / "google_gemma-3-1b-it-Q4_K_M.gguf"
    if model_file.exists() and model_file.stat().st_size > 700_000_000:  # > 700 MB
        print("✅ Model already downloaded")
        return

    print("📥 Downloading default model from Hugging Face...")
    print(f"   Repository: {HF_REPO_ID}")
    print(f"   Model: google_gemma-3-1b-it-Q4_K_M.gguf (769 MB)")
    print(f"   This is a one-time download")
    print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="google_gemma-3-1b-it-Q4_K_M.gguf",
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
        )
        print()
        print(f"✅ Model downloaded: {downloaded_path}")
    except Exception as e:
        print(f"⚠️  Model download failed: {e}")
        print("   You can manually download models later")

    print()


def bootstrap() -> None:
    """
    Main bootstrap function for llcuda v2.1.0 - called on first import.

    Downloads T4-optimized binaries from GitHub Releases on first import.
    Uses v2.0.6 binaries (100% compatible with v2.1.0 pure Python APIs).
    Models are downloaded on-demand when load_model() is called.

    Raises:
        RuntimeError: If GPU is not compatible (SM < 7.5) or download fails
    """
    # Check if binaries already installed
    llama_server = BINARIES_DIR / "cuda12" / "llama-server"

    if llama_server.exists() and llama_server.stat().st_size > 0:
        # Binaries already installed
        return

    # Download T4 binaries from GitHub Releases
    print()
    download_t4_binaries()

    # Verify installation
    if not llama_server.exists():
        raise RuntimeError(
            "Binary installation failed. Please check your internet connection and try again:\n"
            "pip install --no-cache-dir --force-reinstall git+https://github.com/llcuda/llcuda.git"
        )


if __name__ == "__main__":
    bootstrap()
