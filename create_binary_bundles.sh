#!/bin/bash
# Create binary bundles for llcuda hybrid bootstrap architecture
# This script creates separate bundles for each SM version

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="/media/waqasm86/External1/Project-Nvidia/llama.cpp/build"
BUNDLE_DIR="${SCRIPT_DIR}/bundles"

echo "=========================================="
echo "llcuda Binary Bundle Creator"
echo "=========================================="
echo ""

# Check if build exists
if [ ! -d "$BUILD_DIR/bin" ]; then
    echo "❌ Error: Build directory not found at $BUILD_DIR"
    echo "Please build llama.cpp first with all 8 architectures"
    exit 1
fi

# Create bundle directory
mkdir -p "$BUNDLE_DIR"

# Clean old bundles
rm -rf "$BUNDLE_DIR"/*.tar.gz
rm -rf "$BUNDLE_DIR"/llcuda-bins-*

echo "📦 Creating binary bundles..."
echo ""

# For the hybrid architecture, we're creating a single bundle
# that contains binaries compiled for ALL architectures
# Users will download this based on their detected GPU

BUNDLE_NAME="llcuda-bins-multiarch"
BUNDLE_PATH="$BUNDLE_DIR/$BUNDLE_NAME"

echo "Creating bundle: $BUNDLE_NAME"
echo "  This bundle contains binaries for ALL SM versions:"
echo "  - SM 5.0 (Maxwell: GTX 900, 940M)"
echo "  - SM 6.0 (Pascal: P100)"
echo "  - SM 6.1 (Pascal: GTX 10xx)"
echo "  - SM 7.0 (Volta: V100)"
echo "  - SM 7.5 (Turing: T4, RTX 20xx)"
echo "  - SM 8.0 (Ampere: A100)"
echo "  - SM 8.6 (Ampere: RTX 30xx)"
echo "  - SM 8.9 (Ada Lovelace: RTX 40xx)"
echo ""

# Create bundle structure
mkdir -p "$BUNDLE_PATH/binaries/cuda12"
mkdir -p "$BUNDLE_PATH/lib"

# Copy binaries
echo "  Copying binaries..."
cp -v "$BUILD_DIR/bin/llama-server" "$BUNDLE_PATH/binaries/cuda12/" || true
cp -v "$BUILD_DIR/bin/llama-cli" "$BUNDLE_PATH/binaries/cuda12/" || true
cp -v "$BUILD_DIR/bin/llama-bench" "$BUNDLE_PATH/binaries/cuda12/" || true
cp -v "$BUILD_DIR/bin/llama-quantize" "$BUNDLE_PATH/binaries/cuda12/" || true

# Copy libraries
echo "  Copying libraries..."
cp -v "$BUILD_DIR/bin"/libggml*.so* "$BUNDLE_PATH/lib/" || true
cp -v "$BUILD_DIR/bin"/libllama*.so* "$BUNDLE_PATH/lib/" || true
cp -v "$BUILD_DIR/bin"/libmtmd*.so* "$BUNDLE_PATH/lib/" || true

# Create metadata
echo "  Creating metadata..."
cat > "$BUNDLE_PATH/metadata.json" <<EOF
{
  "version": "1.1.0",
  "llama_cpp_version": "0.9.4",
  "compute_capabilities": ["5.0", "6.0", "6.1", "7.0", "7.5", "8.0", "8.6", "8.9"],
  "cuda_version": "12.8",
  "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "architectures": [
    {"sm": "5.0", "name": "Maxwell", "gpus": "GTX 900, 940M"},
    {"sm": "6.0", "name": "Pascal", "gpus": "Tesla P100"},
    {"sm": "6.1", "name": "Pascal", "gpus": "GTX 10xx"},
    {"sm": "7.0", "name": "Volta", "gpus": "Tesla V100"},
    {"sm": "7.5", "name": "Turing", "gpus": "Tesla T4, RTX 20xx"},
    {"sm": "8.0", "name": "Ampere", "gpus": "A100"},
    {"sm": "8.6", "name": "Ampere", "gpus": "RTX 30xx"},
    {"sm": "8.9", "name": "Ada Lovelace", "gpus": "RTX 40xx"}
  ]
}
EOF

# Create tarball
echo "  Creating tarball..."
cd "$BUNDLE_DIR"
tar -czf "${BUNDLE_NAME}.tar.gz" "$BUNDLE_NAME"

# Generate checksum
echo "  Generating SHA256 checksum..."
sha256sum "${BUNDLE_NAME}.tar.gz" > "${BUNDLE_NAME}.tar.gz.sha256"

# Get size
BUNDLE_SIZE=$(du -h "${BUNDLE_NAME}.tar.gz" | cut -f1)

echo "  ✅ Bundle created: ${BUNDLE_NAME}.tar.gz ($BUNDLE_SIZE)"
echo ""

# Summary
echo "=========================================="
echo "✅ Bundle Creation Complete!"
echo "=========================================="
echo ""
echo "Bundle created:"
echo "  📦 ${BUNDLE_NAME}.tar.gz ($BUNDLE_SIZE)"
echo "  🔒 ${BUNDLE_NAME}.tar.gz.sha256"
echo ""
echo "Location: $BUNDLE_DIR"
echo ""
echo "Next steps:"
echo "  1. Upload to GitHub Releases (v1.1.0-runtime)"
echo "  2. Update Python code to download this bundle"
echo "  3. Test on different GPU architectures"
echo ""
