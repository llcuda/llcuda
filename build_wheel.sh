#!/bin/bash
# Build Python wheel for llcuda package
# Usage: bash build_wheel.sh

set -e

echo "========================================="
echo " Building llcuda Python Wheel"
echo "========================================="
echo "Python version: $(python3.11 --version)"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

# Verify version consistency - FIXED version parsing
echo "Verifying version consistency..."
# Method 1: Get project version using sed
PYPROJECT_VERSION=$(sed -n '/^\[project\]/,/^\[/p' pyproject.toml | grep 'version =' | head -1 | cut -d'"' -f2)

# Method 2: Alternative if method 1 fails
if [ -z "$PYPROJECT_VERSION" ]; then
    PYPROJECT_VERSION=$(grep '^version =' pyproject.toml | head -1 | cut -d'"' -f2)
fi

INIT_VERSION=$(grep '__version__ =' llcuda/__init__.py | cut -d'"' -f2)

if [ -z "$PYPROJECT_VERSION" ]; then
    echo "✗ Could not find version in pyproject.toml"
    exit 1
fi

if [ "$PYPROJECT_VERSION" = "$INIT_VERSION" ]; then
    echo "✓ Versions match: $PYPROJECT_VERSION"
else
    echo "✗ Version mismatch!"
    echo "  pyproject.toml: $PYPROJECT_VERSION"
    echo "  __init__.py: $INIT_VERSION"
    exit 1
fi

# Build using python3.11
echo ""
echo "Building with python3.11..."
python3.11 -m pip install --upgrade pip build setuptools wheel

# Build wheel and source distribution
echo "Building distribution..."
python3.11 -m build --wheel --sdist

# List built packages
echo ""
echo "========================================="
echo " Build Complete!"
echo "========================================="
echo "Built packages:"
ls -lh dist/

echo ""
echo "========================================="
echo " Distribution Summary"
echo "========================================="
echo "Version: $PYPROJECT_VERSION"
echo "Wheel: dist/llcuda-$PYPROJECT_VERSION-py3-none-any.whl"
echo "Source: dist/llcuda-$PYPROJECT_VERSION.tar.gz"
echo ""
echo "To install locally:"
echo "  pip install dist/llcuda-$PYPROJECT_VERSION-py3-none-any.whl"
echo ""
echo "To upload to PyPI:"
echo "  pip install twine"
echo "  twine upload dist/*"
echo "========================================="
