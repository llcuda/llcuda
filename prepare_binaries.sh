#!/bin/bash
# prepare_binaries.sh
# Extracts CUDA binaries into package for inclusion in PyPI wheel

set -e

echo "=================================================="
echo "Preparing llcuda binaries for PyPI package"
echo "=================================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$SCRIPT_DIR/llcuda"
BUILD_ARTIFACTS_DIR="$SCRIPT_DIR/build-artifacts"
TAR_FILE="$BUILD_ARTIFACTS_DIR/llcuda-binaries-cuda12-t4-v2.0.2.tar.gz"

echo ""
echo "Step 1: Checking for binaries archive..."
if [ ! -f "$TAR_FILE" ]; then
    echo "❌ Error: Binaries archive not found at $TAR_FILE"
    echo ""
    echo "Please download it from:"
    echo "https://github.com/waqasm86/llcuda/releases/download/v2.0.2/llcuda-binaries-cuda12-t4-v2.0.2.tar.gz"
    echo ""
    echo "And place it in: $BUILD_ARTIFACTS_DIR/"
    exit 1
fi
echo "✅ Found binaries archive ($(du -h "$TAR_FILE" | cut -f1))"

# Clean up existing binaries
echo ""
echo "Step 2: Cleaning up old binaries..."
rm -rf "$PACKAGE_DIR/binaries"
rm -rf "$PACKAGE_DIR/lib"
echo "✅ Cleaned up old binaries"

# Create temporary extraction directory
echo ""
echo "Step 3: Extracting binaries..."
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

tar -xzf "$TAR_FILE" -C "$TEMP_DIR"
echo "✅ Extracted to temporary directory"

# Copy binaries to package
echo ""
echo "Step 4: Installing binaries into package..."

if [ -d "$TEMP_DIR/bin" ]; then
    mkdir -p "$PACKAGE_DIR/binaries/cuda12"
    cp -r "$TEMP_DIR/bin/"* "$PACKAGE_DIR/binaries/cuda12/"
    chmod +x "$PACKAGE_DIR/binaries/cuda12/"*
    echo "✅ Installed $(ls "$PACKAGE_DIR/binaries/cuda12/" | wc -l) binaries"
else
    echo "❌ Error: bin/ directory not found in archive"
    exit 1
fi

if [ -d "$TEMP_DIR/lib" ]; then
    mkdir -p "$PACKAGE_DIR/lib"
    cp -r "$TEMP_DIR/lib/"* "$PACKAGE_DIR/lib/"
    echo "✅ Installed $(ls "$PACKAGE_DIR/lib/" | wc -l) libraries"
else
    echo "❌ Error: lib/ directory not found in archive"
    exit 1
fi

# Verify installation
echo ""
echo "Step 5: Verifying installation..."
LLAMA_SERVER="$PACKAGE_DIR/binaries/cuda12/llama-server"
if [ -f "$LLAMA_SERVER" ] && [ -x "$LLAMA_SERVER" ]; then
    echo "✅ llama-server is executable"
else
    echo "❌ Error: llama-server not found or not executable"
    exit 1
fi

LIBGGML_CUDA="$PACKAGE_DIR/lib/libggml-cuda.so"
if [ -f "$LIBGGML_CUDA" ]; then
    CUDA_LIB_SIZE=$(du -h "$LIBGGML_CUDA" | cut -f1)
    echo "✅ libggml-cuda.so found ($CUDA_LIB_SIZE)"
else
    echo "❌ Error: libggml-cuda.so not found"
    exit 1
fi

# Calculate total size
TOTAL_SIZE=$(du -sh "$PACKAGE_DIR/binaries" "$PACKAGE_DIR/lib" | awk '{s+=$1} END {print s}')
echo ""
echo "=================================================="
echo "✅ Binaries prepared successfully!"
echo "=================================================="
echo ""
echo "Binaries location: $PACKAGE_DIR/binaries/cuda12/"
echo "Libraries location: $PACKAGE_DIR/lib/"
echo "Total size: ~266 MB"
echo ""
echo "Next steps:"
echo "  1. Run: python -m build"
echo "  2. Check wheel size: ls -lh dist/"
echo "  3. Upload to PyPI: python -m twine upload dist/*"
echo ""
