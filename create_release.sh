#!/bin/bash
# Create release for llcuda

set -e

echo "========================================"
echo "Creating llcuda Release"
echo "========================================"

# Get version from pyproject.toml
VERSION=$(grep 'version =' pyproject.toml | cut -d'"' -f2)
echo "Version: $VERSION"

# 1. Build the distribution
echo ""
echo "1. Building distribution..."
./build_wheel.sh

# 2. Verify the build
echo ""
echo "2. Verifying build..."
if [ ! -f "dist/llcuda-$VERSION-py3-none-any.whl" ]; then
    echo "Error: Wheel file not found!"
    exit 1
fi

if [ ! -f "dist/llcuda-$VERSION.tar.gz" ]; then
    echo "Error: Source distribution not found!"
    exit 1
fi

# 3. Test installation
echo ""
echo "3. Testing installation..."
python3.11 -m pip install dist/llcuda-$VERSION-py3-none-any.whl --force-reinstall --quiet
python3.11 -c "import llcuda; print(f'✓ llcuda {llcuda.__version__} installed successfully!')"

# 4. Summary
echo ""
echo "========================================"
echo "Release ready for distribution!"
echo "========================================"
echo ""
echo "Files created:"
echo "  - dist/llcuda-$VERSION-py3-none-any.whl"
echo "  - dist/llcuda-$VERSION.tar.gz"
echo ""
echo "To upload to PyPI:"
echo "  ./upload_to_pypi.sh"
echo ""
echo "To create GitHub release manually:"
echo "  1. Go to: https://github.com/waqasm86/llcuda/releases"
echo "  2. Click 'Draft a new release'"
echo "  3. Tag: v$VERSION"
echo "  4. Title: llcuda v$VERSION"
echo "  5. Copy release notes from CHANGELOG.md"
echo "  6. Upload the .whl and .tar.gz files"
echo ""
echo "Remember to commit and push changes first:"
echo "  git add ."
echo "  git commit -m 'Release v$VERSION'"
echo "  git tag v$VERSION"
echo "  git push origin main --tags"
echo "========================================"
