#!/bin/bash
# Finalize llcuda v1.1.2 release

echo "=== FINALIZING llcuda v1.1.2 ==="

# Test installation (with proper quoting)
echo "1. Testing installation..."
python3.11 -c 'import llcuda; print("✓ llcuda", llcuda.__version__, "installed successfully!")'

# Commit changes
echo ""
echo "2. Committing changes..."
git add .
git commit -m "Release v1.1.2: Fix Colab compatibility and binary extraction issues" || echo "Nothing to commit"

# Create tag
echo ""
echo "3. Creating git tag..."
git tag -a "v1.1.2" -m "Release version 1.1.2"

# Push to GitHub
echo ""
echo "4. Pushing to GitHub..."
git push origin main
git push origin v1.1.2

echo ""
echo "=== GITHUB RELEASE INSTRUCTIONS ==="
echo "1. Go to: https://github.com/waqasm86/llcuda/releases"
echo "2. Click 'Draft a new release'"
echo "3. Tag: v1.1.2"
echo "4. Title: llcuda v1.1.2"
echo "5. Description: Copy 1.1.2 section from CHANGELOG.md"
echo "6. Attach files:"
echo "   - dist/llcuda-1.1.2-py3-none-any.whl"
echo "   - dist/llcuda-1.1.2.tar.gz"
echo ""
echo "=== PYPI UPLOAD ==="
echo "After GitHub release, run:"
echo "  python3.11 -m pip install --upgrade twine"
echo "  python3.11 -m twine upload dist/*"
