#!/bin/bash
# Build Python wheel for llcuda package
# Usage: bash build_wheel.sh [cuda_arch]

set -e

# Default CUDA architecture (T4 GPU)
CUDA_ARCH="${1:-75}"

echo "========================================="
echo " Building llcuda Python Wheel"
echo "========================================="
echo "CUDA Architecture: ${CUDA_ARCH}"
echo ""

# Set environment variables
export CUDA_ARCHITECTURES="${CUDA_ARCH}"
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

# Create virtual environment for clean build
VENV_DIR="build_venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Upgrade build tools
echo "Upgrading build tools..."
pip install --upgrade pip setuptools wheel build

# Install build dependencies
echo "Installing build dependencies..."
pip install pybind11 numpy cmake ninja

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

# Build wheel
echo "Building wheel..."
python -m build --wheel

# Build source distribution
echo "Building source distribution..."
python -m build --sdist

# List built packages
echo ""
echo "========================================="
echo " Build Complete!"
echo "========================================="
echo "Built packages:"
ls -lh dist/

# Optional: Test installation
echo ""
read -p "Test installation? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing installation..."
    pip install dist/*.whl --force-reinstall
    
    echo "Running import test..."
    python -c "import llcuda; print(f'llcuda {llcuda.__version__} installed successfully!')"
    
    echo "Running tests..."
    pip install pytest
    pytest tests/ -v
fi

# Deactivate virtual environment
deactivate

echo ""
echo "========================================="
echo " Wheel Build Summary"
echo "========================================="
echo "CUDA Architecture: ${CUDA_ARCH}"
echo "Wheel location: dist/"
echo ""
echo "To install:"
echo "  pip install dist/llcuda-*.whl"
echo ""
echo "To upload to PyPI:"
echo "  pip install twine"
echo "  twine upload dist/*"
echo "========================================="
