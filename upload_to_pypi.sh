#!/bin/bash
# Upload llcuda to PyPI

echo "========================================"
echo "Uploading llcuda to PyPI"
echo "========================================"

# Check for dist files
if [ ! -d "dist" ]; then
    echo "Error: dist directory not found. Run build_wheel.sh first."
    exit 1
fi

# Get version
VERSION=$(grep -A1 '^\[project\]' pyproject.toml | grep 'version =' | cut -d'"' -f2)
WHEEL_FILE="dist/llcuda-$VERSION-py3-none-any.whl"

if [ ! -f "$WHEEL_FILE" ]; then
    echo "Error: Wheel file not found: $WHEEL_FILE"
    exit 1
fi

echo "Version: $VERSION"
echo "Wheel: $(basename $WHEEL_FILE)"
echo ""

# Install/upgrade twine
echo "Installing twine..."
python3.11 -m pip install --upgrade twine

# Upload
echo "Uploading to PyPI..."
python3.11 -m twine upload dist/*

echo ""
echo "========================================"
echo "Upload complete!"
echo "========================================"
echo "Package available at: https://pypi.org/project/llcuda/"
echo ""
echo "Verify with:"
echo "  pip install llcuda==$VERSION"
echo "  python -c \"import llcuda; print(llcuda.__version__)\""
