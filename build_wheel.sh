#!/bin/bash
# Build Python wheel for llcuda package
# Usage: bash build_wheel.sh

set -e

echo "========================================="
echo " Building llcuda v1.1.2 Python Wheel"
echo "========================================="
echo "Python version: $(python3.11 --version)"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

# Remove temporary binaries (keep source clean)
echo "Cleaning temporary files..."
rm -rf llcuda/binaries/ llcuda/lib/ llcuda/models/
rm -f llcuda-source.tar.gz llcuda-*.tar.gz llcuda-*.zip

# Create empty directories for package structure
mkdir -p llcuda/binaries/cuda12
mkdir -p llcuda/lib
mkdir -p llcuda/models
mkdir -p releases/binaries
mkdir -p releases/libraries

# Placeholder files to keep directory structure
touch llcuda/binaries/cuda12/.gitkeep
touch llcuda/lib/.gitkeep
touch llcuda/models/.gitkeep
touch releases/binaries/.gitkeep
touch releases/libraries/.gitkeep

# Verify version consistency
echo "Verifying version consistency..."
PYPROJECT_VERSION=$(grep 'version =' pyproject.toml | cut -d'"' -f2)
INIT_VERSION=$(grep '__version__ =' llcuda/__init__.py | cut -d'"' -f2)

if [ "$PYPROJECT_VERSION" = "$INIT_VERSION" ]; then
    echo "✓ Versions match: $PYPROJECT_VERSION"
else
    echo "✗ Version mismatch!"
    echo "  pyproject.toml: $PYPROJECT_VERSION"
    echo "  __init__.py: $INIT_VERSION"
    exit 1
fi

# Check for required files
echo "Checking required files..."
required_files=(
    "pyproject.toml"
    "llcuda/__init__.py"
    "llcuda/server.py"
    "llcuda/_internal/bootstrap.py"
    "README.md"
    "LICENSE"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ Missing: $file"
        exit 1
    fi
done

# Build using python3.11 specifically
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

# Test installation
echo ""
echo "Testing installation..."
python3.11 -m pip install dist/*.whl --force-reinstall --quiet

echo "Running import test..."
python3.11 -c "import llcuda; print(f'✓ llcuda {llcuda.__version__} installed successfully!')"

# Clean up placeholder files
rm -f llcuda/binaries/cuda12/.gitkeep
rm -f llcuda/lib/.gitkeep
rm -f llcuda/models/.gitkeep
rm -f releases/binaries/.gitkeep
rm -f releases/libraries/.gitkeep

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
echo ""
echo "To create GitHub release:"
echo "  git tag v$PYPROJECT_VERSION"
echo "  git push origin v$PYPROJECT_VERSION"
echo "========================================="