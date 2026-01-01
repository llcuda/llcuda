#!/bin/bash
set -e  # Exit on error

echo "=== Building llcuda Package ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/ 2>/dev/null || true

# Update version (if provided as argument)
if [ ! -z "$1" ]; then
    echo "Updating version to $1"
    sed -i 's/version = ".*"/version = "'"$1"'"/' pyproject.toml
    sed -i 's/__version__ = ".*"/__version__ = "'"$1"'"/' llcuda/__init__.py
fi

# Build source distribution
echo "Building source distribution..."
python -m build --sdist

# Build wheel
echo "Building wheel..."
python -m build --wheel

# List built packages
echo ""
echo "=== Built Packages ==="
ls -lh dist/

# Check package contents
echo ""
echo "=== Package Contents ==="
tar -tzf dist/llcuda-*.tar.gz | head -20
echo "..."
tar -tzf dist/llcuda-*.tar.gz | tail -20

# Test installation
echo ""
echo "=== Testing Installation ==="
python -m pip install dist/llcuda-*.whl --force-reinstall
python -c "import llcuda; print(f'Successfully imported llcuda version {llcuda.__version__}')"

echo ""
echo "=== Build Complete ==="
echo "Packages available in dist/ directory"
