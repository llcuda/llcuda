#!/bin/bash
# Build script for llcuda native extension

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build/native"

echo "=== Building llcuda Native Extension ==="
echo "Project root: $PROJECT_ROOT"
echo "Build directory: $BUILD_DIR"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo ""
echo "=== Configuring CMake ==="
cmake ../.. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE=$(which python3.11)

# Build
echo ""
echo "=== Building ==="
make -j$(nproc)

# Copy built library to Python package
echo ""
echo "=== Installing ==="
cp llcuda_cpp*.so "${PROJECT_ROOT}/"

echo ""
echo "=== Build Complete ==="
echo "Library: ${PROJECT_ROOT}/llcuda_cpp*.so"
echo ""
echo "To test, run:"
echo "  cd ${PROJECT_ROOT}"
echo "  python3.11 -m pytest tests/test_tensor_api.py -v"
