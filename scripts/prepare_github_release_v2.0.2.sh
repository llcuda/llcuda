#!/bin/bash
# GitHub Release Preparation Script for llcuda v2.0.2
# This script helps prepare all files for uploading to GitHub Releases

set -e

VERSION="2.0.2"
TAG="v${VERSION}"
RELEASE_TITLE="llcuda v${VERSION} - Critical Bug Fixes"
BINARIES_TAR="llcuda-binaries-cuda12-t4-v${VERSION}.tar.gz"
BINARIES_DIR="/media/waqasm86/External1/Project-Nvidia-Office/Project-Nvidia-Office/llcuda-complete-cuda12-t4-tar-file"

echo "======================================================================="
echo "📦 GitHub Release Preparation - llcuda v${VERSION}"
echo "======================================================================="
echo

# Check if binary tar exists
if [ ! -f "${BINARIES_DIR}/${BINARIES_TAR}" ]; then
    echo "❌ Error: Binary tar not found at ${BINARIES_DIR}/${BINARIES_TAR}"
    exit 1
fi

echo "✅ Binary tar found: ${BINARIES_TAR}"
echo "   Size: $(du -h "${BINARIES_DIR}/${BINARIES_TAR}" | cut -f1)"
echo

# Check SHA256
if [ -f "${BINARIES_DIR}/${BINARIES_TAR}.sha256" ]; then
    echo "✅ SHA256 checksum found"
    cat "${BINARIES_DIR}/${BINARIES_TAR}.sha256"
else
    echo "⚠️  Generating SHA256 checksum..."
    cd "${BINARIES_DIR}"
    sha256sum "${BINARIES_TAR}" > "${BINARIES_TAR}.sha256"
    cat "${BINARIES_TAR}.sha256"
fi
echo

# Verify tar structure
echo "🔍 Verifying tar file structure..."
TAR_STRUCTURE=$(tar -tzf "${BINARIES_DIR}/${BINARIES_TAR}" | head -10)
if echo "$TAR_STRUCTURE" | grep -q "^bin/"; then
    echo "✅ Tar structure is correct (bin/ and lib/ at root)"
else
    echo "❌ Error: Tar structure is incorrect!"
    echo "   Expected: bin/ and lib/ at root level"
    echo "   Got:"
    echo "$TAR_STRUCTURE"
    exit 1
fi
echo

# Create release notes
RELEASE_NOTES=$(cat <<'EOF'
## 🐛 Critical Bug Fixes - Upgrade Recommended

This release fixes installation failures on Kaggle, Colab, and other cloud platforms.

### Fixed Issues

- ✅ **HTTP 404 Error**: Fixed bootstrap failing with 404 when downloading binaries on first import
- ✅ **Version Inconsistency**: Fixed `__version__` incorrectly reporting "1.2.2" instead of "2.0.2"
- ✅ **Tar Structure**: Fixed extraction failures due to unexpected parent directory in tar file

### Upgrade Instructions

```bash
pip install --upgrade llcuda
```

### Binary Package Details

- **File**: `llcuda-binaries-cuda12-t4-v2.0.2.tar.gz`
- **Size**: 266 MB
- **SHA256**: `1dcf78936f3e0340a288950cbbc0e7bf12339d7b9dfbd1fe0344d44b6ead39b5`
- **Features**: Tesla T4 optimized with FlashAttention + CUDA Graphs + Tensor Cores

### Compatibility

- Python 3.11+
- CUDA 12.x
- Tesla T4 GPU (SM 7.5)
- Works on: Google Colab, Kaggle, local Linux

### Breaking Changes

**None** - Fully backward compatible with v2.0.0/v2.0.1

---

For full changelog, see [CHANGELOG.md](https://github.com/waqasm86/llcuda/blob/main/CHANGELOG.md)
EOF
)

echo "======================================================================="
echo "📝 GitHub Release Information"
echo "======================================================================="
echo
echo "Tag:   $TAG"
echo "Title: $RELEASE_TITLE"
echo
echo "Release Notes:"
echo "---"
echo "$RELEASE_NOTES"
echo "---"
echo

# Instructions for manual upload
echo "======================================================================="
echo "📤 Upload Instructions"
echo "======================================================================="
echo
echo "1. Go to: https://github.com/waqasm86/llcuda/releases/new"
echo
echo "2. Fill in:"
echo "   - Tag: $TAG"
echo "   - Title: $RELEASE_TITLE"
echo "   - Description: Copy the release notes above"
echo
echo "3. Upload files:"
echo "   - ${BINARIES_TAR}"
echo "   - ${BINARIES_TAR}.sha256"
echo
echo "4. Check:"
echo "   [✓] Set as the latest release"
echo
echo "5. Click 'Publish release'"
echo
echo "======================================================================="
echo "📦 Files to Upload"
echo "======================================================================="
echo
echo "Location: ${BINARIES_DIR}"
echo
ls -lh "${BINARIES_DIR}/${BINARIES_TAR}"
ls -lh "${BINARIES_DIR}/${BINARIES_TAR}.sha256" 2>/dev/null || echo "(SHA256 file will be created)"
echo

# Check if gh CLI is available
if command -v gh &> /dev/null; then
    echo "======================================================================="
    echo "🚀 Automated Upload (GitHub CLI)"
    echo "======================================================================="
    echo
    echo "GitHub CLI detected! You can also use this command to upload:"
    echo
    echo "cd ${BINARIES_DIR}"
    echo "gh release create ${TAG} \\"
    echo "  --title \"${RELEASE_TITLE}\" \\"
    echo "  --notes \"\${RELEASE_NOTES}\" \\"
    echo "  \"${BINARIES_TAR}\" \\"
    echo "  \"${BINARIES_TAR}.sha256\""
    echo
else
    echo "💡 Tip: Install GitHub CLI (gh) for automated uploads"
    echo "   https://cli.github.com/"
fi

echo
echo "✅ GitHub release preparation complete!"
echo
