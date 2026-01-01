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
GITHUB_RELEASE_URL = "https://github.com/waqasm86/llcuda/releases/download/v1.1.1"
HF_REPO_ID = "waqasm86/llcuda-models"
BINARY_BUNDLE_NAME = "llcuda-binaries-cuda12.tar.gz"

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
            timeout=10
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
            sys.stdout.write(f"\r{desc}: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
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
        tar.extractall(dest_dir)

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
        compute_cap = None

    print(f"🌐 Platform: {platform.capitalize()}")
    print()

    # Download binary bundle
    cache_tarball = CACHE_DIR / BINARY_BUNDLE_NAME
    bundle_url = f"{GITHUB_RELEASE_URL}/{BINARY_BUNDLE_NAME}"

    if not cache_tarball.exists():
        print(f"📥 Downloading optimized binaries from GitHub...")
        print(f"   URL: {bundle_url}")
        print(f"   This is a one-time download (~120 MB)")
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

    # Move binaries to package directory
    bundle_name = BINARY_BUNDLE_NAME.replace(".tar.gz", "")
    extracted_bundle = temp_extract_dir / bundle_name

    if extracted_bundle.exists():
        # Copy binaries
        if (extracted_bundle / "binaries").exists():
            shutil.copytree(
                extracted_bundle / "binaries",
                BINARIES_DIR,
                dirs_exist_ok=True
            )

        # Copy libraries
        if (extracted_bundle / "lib").exists():
            shutil.copytree(
                extracted_bundle / "lib",
                LIB_DIR,
                dirs_exist_ok=True
            )

        # Make binaries executable
        for binary in BINARIES_DIR.glob("**/*"):
            if binary.is_file() and not binary.suffix:
                binary.chmod(0o755)

        print("✅ Binaries installed successfully!")
    else:
        raise RuntimeError(f"Extraction failed: {extracted_bundle} not found")

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
            local_dir_use_symlinks=False
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
    """
    # Check if setup already complete
    llama_server = BINARIES_DIR / "cuda12" / "llama-server"
    model_file = MODELS_DIR / "google_gemma-3-1b-it-Q4_K_M.gguf"

    if llama_server.exists() and model_file.exists():
        # Setup already complete
        return

    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download binaries
    if not llama_server.exists():
        try:
            download_binaries()
        except Exception as e:
            print(f"❌ Binary download failed: {e}")
            print("   llcuda may not function correctly")
            print()

    # Download default model
    if not model_file.exists():
        try:
            download_default_model()
        except Exception as e:
            print(f"⚠️  Model download skipped: {e}")
            print()

    print("=" * 60)
    print("✅ llcuda Setup Complete!")
    print("=" * 60)
    print()
    print("You can now use llcuda:")
    print()
    print("  import llcuda")
    print("  engine = llcuda.InferenceEngine()")
    print("  engine.load_model('gemma-3-1b-Q4_K_M')")
    print("  result = engine.infer('What is AI?')")
    print()


if __name__ == "__main__":
    bootstrap()
