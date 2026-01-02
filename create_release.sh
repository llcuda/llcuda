#!/bin/bash
# Create release for llcuda v1.1.2

set -e

VERSION="1.1.2"

echo "========================================"
echo "Creating llcuda v$VERSION Release"
echo "========================================"

# 1. Verify clean git state
echo "1. Checking git status..."
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Git working directory not clean"
    git status
    exit 1
fi

# 2. Update version files
echo "2. Updating version to $VERSION..."
sed -i "s/version = \"1.1.1.post1\"/version = \"$VERSION\"/" pyproject.toml
sed -i "s/__version__ = \"1.1.1.post1\"/__version__ = \"$VERSION\"/" llcuda/__init__.py
sed -i "s|download/v1.1.1|download/v$VERSION|g" llcuda/_internal/bootstrap.py

# 3. Build distribution
echo "3. Building distribution..."
bash build_wheel.sh

# 4. Commit changes
echo "4. Committing changes..."
git add .
git commit -m "Release v$VERSION - Fix Colab compatibility and binary extraction"

# 5. Create tag
echo "5. Creating git tag v$VERSION..."
git tag -a "v$VERSION" -m "Release version $VERSION"

# 6. Push to GitHub
echo "6. Pushing to GitHub..."
git push origin main
git push origin "v$VERSION"

echo ""
echo "========================================"
echo "Release v$VERSION created successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Go to: https://github.com/waqasm86/llcuda/releases"
echo "2. Edit the v$VERSION release"
echo "3. Add release notes from CHANGELOG.md"
echo "4. Upload dist/llcuda-$VERSION-py3-none-any.whl"
echo "5. Upload dist/llcuda-$VERSION.tar.gz"
echo "6. (Optional) Upload llcuda-binaries-cuda12.tar.gz if updated"
echo ""
echo "To upload to PyPI:"
echo "  pip install twine"
echo "  twine upload dist/*"