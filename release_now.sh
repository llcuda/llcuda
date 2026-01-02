#!/bin/bash
# Complete release process

set -e

echo "========================================"
echo "llcuda v1.1.2 Release Process"
echo "========================================"

# 1. Build
echo "1. Building package..."
./build_wheel.sh

# 2. Verify
VERSION=$(grep -A1 '^\[project\]' pyproject.toml | grep 'version =' | cut -d'"' -f2)
echo "Version: $VERSION"

if [ ! -f "dist/llcuda-$VERSION-py3-none-any.whl" ]; then
    echo "Error: Build failed!"
    exit 1
fi

# 3. Test installation
echo "2. Testing installation..."
python3.11 -m pip install dist/llcuda-$VERSION-py3-none-any.whl --force-reinstall --quiet
python3.11 -c "import llcuda; print(f'✓ llcuda {llcuda.__version__} installed successfully!')"

# 4. Commit changes
echo "3. Committing changes..."
git add .
git commit -m "Release v$VERSION: Fix Colab compatibility and binary extraction"

# 5. Create tag
echo "4. Creating git tag v$VERSION..."
git tag -a "v$VERSION" -m "Release version $VERSION"

# 6. Push to GitHub
echo "5. Pushing to GitHub..."
git push origin main
git push origin "v$VERSION"

echo ""
echo "========================================"
echo "GitHub release ready!"
echo "========================================"
echo "Go to: https://github.com/waqasm86/llcuda/releases"
echo "Edit v$VERSION release and add:"
echo "1. Release notes from CHANGELOG.md"
echo "2. Upload dist/llcuda-$VERSION-py3-none-any.whl"
echo "3. Upload dist/llcuda-$VERSION.tar.gz"
echo ""
echo "After GitHub release, run: ./upload_to_pypi.sh"
echo "========================================"
