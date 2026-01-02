#!/bin/bash
# prepare_github_release.sh

VERSION="1.1.5"

# Clean previous builds
rm -rf dist/
rm -rf build/
rm -rf llcuda.egg-info/

# Build the package
python3.11 -m build

# Create source archives
tar -czvf llcuda-v${VERSION}-source.tar.gz \
  --exclude=".git" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  --exclude="dist" \
  --exclude="build" \
  --exclude="llcuda.egg-info" \
  --exclude=".pytest_cache" \
  .

zip -r llcuda-v${VERSION}-source.zip \
  --exclude="*.git*" \
  --exclude="*__pycache__*" \
  --exclude="*.pyc" \
  --exclude="dist/*" \
  --exclude="build/*" \
  --exclude="llcuda.egg-info/*" \
  --exclude=".pytest_cache/*" \
  .

# Copy binaries bundle
cp releases/llcuda-binaries-cuda12-v1.1.5.tar.gz releases/llcuda-binaries-cuda12.tar.gz

echo "Assets prepared for GitHub release v${VERSION}"
echo "1. llcuda-${VERSION}-py3-none-any.whl"
echo "2. llcuda-${VERSION}.tar.gz"
echo "3. llcuda-v${VERSION}-source.tar.gz"
echo "4. llcuda-v${VERSION}-source.zip"
echo "5. llcuda-binaries-cuda12.tar.gz"