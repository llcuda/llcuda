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

# Check for wheel file
WHEEL_FILE=$(ls dist/*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL_FILE" ]; then
    echo "Error: No wheel file found in dist/"
    exit 1
fi

echo "Found wheel: $(basename $WHEEL_FILE)"
echo "Found source: $(ls dist/*.tar.gz 2>/dev/null | head -1 | xargs basename)"

# Install/upgrade twine
echo "Installing/upgrading twine..."
python3.11 -m pip install --upgrade twine

# Test upload to TestPyPI first
echo ""
read -p "Upload to TestPyPI first? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Uploading to TestPyPI..."
    python3.11 -m twine upload --repository testpypi dist/*
    
    echo ""
    echo "Test upload complete! Verify at: https://test.pypi.org/project/llcuda/"
    echo ""
    read -p "Continue to real PyPI? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping at TestPyPI."
        exit 0
    fi
fi

# Upload to real PyPI
echo "Uploading to PyPI..."
python3.11 -m twine upload dist/*

echo ""
echo "========================================"
echo "Upload complete!"
echo "========================================"
echo "Package available at: https://pypi.org/project/llcuda/"
echo ""
echo "To verify installation:"
echo "  pip install llcuda==1.1.2"
echo "  python -c \"import llcuda; print(llcuda.__version__)\""