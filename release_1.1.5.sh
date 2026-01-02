#!/bin/bash
set -e

echo "=== RELEASING llcuda v1.1.5 ==="

# 1. Update version
echo "1. Updating version to 1.1.5..."
sed -i 's/version = "1\.1\.3"/version = "1.1.5"/g' pyproject.toml
sed -i 's/__version__ = "1\.1\.3"/__version__ = "1.1.5"/g' llcuda/__init__.py
sed -i 's/download\/v1\.1\.3/download\/v1.1.5/g' llcuda/_internal/bootstrap.py 2>/dev/null || true

# 2. Update changelog
echo "2. Updating CHANGELOG.md..."
cat > /tmp/1.1.5_entry.md << 'CHANGELOG_EOF'
## [1.1.5] - 2025-01-15

### 🔧 Version Skip - PyPI Filename Resolution

This release skips to version 1.1.5 to resolve PyPI filename conflicts from previous upload attempts.

### No Functional Changes
- Contains all fixes from v1.1.2 and v1.1.3
- Binary extraction fixes for Google Colab
- Updated download URLs
- Enhanced library path detection
- PyPI upload compatibility fix
CHANGELOG_EOF

head -8 CHANGELOG.md > /tmp/header.md
cat /tmp/header.md /tmp/1.1.5_entry.md <(tail -n +9 CHANGELOG.md) > CHANGELOG.md.new
mv CHANGELOG.md.new CHANGELOG.md
rm -f /tmp/header.md /tmp/1.1.5_entry.md

# 3. Rebuild
echo "3. Rebuilding package..."
rm -rf dist/ build/ *.egg-info
./build_wheel.sh

# 4. Test
echo "4. Testing installation..."
python3.11 -m pip install dist/llcuda-1.1.5-py3-none-any.whl --force-reinstall --quiet
python3.11 -c 'import llcuda; print("✓ llcuda", llcuda.__version__, "ready for release")'

# 5. Commit
echo "5. Committing changes..."
git add .
git commit -m "Release v1.1.5: Skip version to resolve PyPI filename conflicts"

# 6. Tag
echo "6. Creating git tag v1.1.5..."
git tag -a "v1.1.5" -m "Release version 1.1.5"

# 7. Push
echo "7. Pushing to GitHub..."
git push origin main
git push origin v1.1.5

echo ""
echo "=== UPLOAD TO PYPI ==="
echo "Run this command to upload:"
echo "  python3.11 -m twine upload dist/*"
echo ""
echo "=== GITHUB RELEASE ==="
echo "1. Go to: https://github.com/waqasm86/llcuda/releases"
echo "2. Edit v1.1.5 release"
echo "3. Upload: dist/llcuda-1.1.5-py3-none-any.whl"
echo "4. Upload: dist/llcuda-1.1.5.tar.gz"
