"""
Bootstrap module for llcuda hybrid architecture.
Downloads binaries and models on first import based on GPU detection.
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

try:
    from huggingface_hub import hf_hub_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Configuration
<<<<<<< HEAD
GITHUB_RELEASE_URL = "https://github.com/waqasm86/llcuda/releases/download/v1.2.2"
=======
GITHUB_RELEASE_URL = "https://github.com/waqasm86/llcuda/releases/download/v1.2.0"
>>>>>>> 2bf25c9922fd76c379669cd3cddcbc9feb3c3e7d
HF_REPO_ID = "waqasm86/llcuda-models"

# GPU-specific binary bundles
GPU_BUNDLES = {
    "940m": "llcuda-binaries-cuda12-940m.tar.gz",  # GeForce 940M (CC 5.0)
    "t4": "llcuda-binaries-cuda12-t4.tar.gz",      # Tesla T4 (CC 7.5)
    "default": "llcuda-binaries-cuda12-t4.tar.gz"  # Default to T4 (more common in cloud)
}

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


def select_binary_bundle(gpu_name: Optional[str], compute_cap: Optional[str]) -> str:
    """
    Select appropriate binary bundle based on GPU.

    Args:
        gpu_name: GPU name from nvidia-smi
        compute_cap: Compute capability (e.g., "5.0", "7.5")

    Returns:
        Bundle filename to download
    """
    if not gpu_name or not compute_cap:
        # No GPU detected, use default (T4)
        print("  No GPU detected, using default binaries (T4 compatible)")
        return GPU_BUNDLES["default"]

    # Parse compute capability
    try:
        cc_float = float(compute_cap)
    except (ValueError, TypeError):
        return GPU_BUNDLES["default"]

    # Map GPU to appropriate bundle
    gpu_lower = gpu_name.lower()

    # GeForce 940M and similar CC 5.0 GPUs (Maxwell architecture)
    if "940" in gpu_lower or "920" in gpu_lower or "930" in gpu_lower:
        print(f"  Detected GeForce 940M series (CC {compute_cap})")
        return GPU_BUNDLES["940m"]
    elif cc_float >= 5.0 and cc_float < 6.0:
        # Maxwell architecture (CC 5.x)
        print(f"  Detected Maxwell GPU (CC {compute_cap}), using 940M binaries")
        return GPU_BUNDLES["940m"]

    # Tesla T4 and similar CC 7.0+ GPUs (Volta/Turing/Ampere/Ada)
    elif "t4" in gpu_lower or cc_float >= 7.0:
        print(f"  Detected modern GPU (CC {compute_cap}), using T4 binaries")
        return GPU_BUNDLES["t4"]

    # Pascal architecture (CC 6.x) - use T4 binaries (they're compatible)
    elif cc_float >= 6.0 and cc_float < 7.0:
        print(f"  Detected Pascal GPU (CC {compute_cap}), using T4 binaries")
        return GPU_BUNDLES["t4"]

    # Older GPUs (CC < 5.0) - not supported, but try T4 binaries
    else:
        print(f"  GPU Compute {compute_cap} may not be fully supported")
        return GPU_BUNDLES["default"]


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


def download_binaries() -> None:
    """
    Download and install binary bundle for detected GPU.
    """
    # Check if binaries already exist
    llama_server = BINARIES_DIR / "cuda12" / "llama-server"
    if llama_server.exists() and llama_server.stat().st_size > 0:
        print("✅ Binaries already installed")
        return

    print("=" * 60)
    print("🎯 llcuda First-Time Setup")
    print("=" * 60)
    print()

    # Detect GPU
    gpu_info = detect_gpu_compute_capability()
    platform = detect_platform()

    if gpu_info:
        gpu_name, compute_cap = gpu_info
        print(f"🎮 GPU Detected: {gpu_name} (Compute {compute_cap})")
    else:
        print("⚠️  No NVIDIA GPU detected (will use CPU)")
        gpu_name = None
        compute_cap = None

    print(f"🌐 Platform: {platform.capitalize()}")
    print()

    # Select appropriate binary bundle for this GPU
    print("📦 Selecting optimized binaries for your GPU...")
    binary_bundle_name = select_binary_bundle(gpu_name, compute_cap)
    print(f"   Selected: {binary_bundle_name}")
    print()

    # Download binary bundle
    cache_tarball = CACHE_DIR / binary_bundle_name
    bundle_url = f"{GITHUB_RELEASE_URL}/{binary_bundle_name}"

    if not cache_tarball.exists():
        # Determine expected download size
        download_size = "~30 MB" if "940m" in binary_bundle_name else "~270 MB"

        print(f"📥 Downloading optimized binaries from GitHub...")
        print(f"   URL: {bundle_url}")
        print(f"   This is a one-time download ({download_size})")
        print()

        download_file(bundle_url, cache_tarball, "Downloading binaries")
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
    Main bootstrap function called on first import.
    Downloads binaries ONLY. Models are downloaded on-demand when load_model() is called.
    """
    # Check if binaries already installed
    llama_server = BINARIES_DIR / "cuda12" / "llama-server"

    if llama_server.exists():
        # Binaries already installed
        return

    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download binaries only
    try:
        download_binaries()
    except Exception as e:
        print(f"❌ Binary download failed: {e}")
        print("   llcuda may not function correctly")
        print()
        return

    print("=" * 60)
    print("✅ llcuda Setup Complete!")
    print("=" * 60)
    print()
    print("You can now use llcuda:")
    print()
    print("  import llcuda")
    print("  engine = llcuda.InferenceEngine()")
    print("  engine.load_model('gemma-3-1b-Q4_K_M')  # Downloads model on first use")
    print("  result = engine.infer('What is AI?')")
    print()


if __name__ == "__main__":
    bootstrap()
