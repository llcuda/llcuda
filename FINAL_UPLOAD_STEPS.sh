#!/bin/bash
# Final Upload Steps for llcuda v2.0.2
# Run this script manually to upload to PyPI

set -e

echo "========================================================================="
echo "🚀 llcuda v2.0.2 - Final Upload to PyPI"
echo "========================================================================="
echo

# Navigate to project directory
cd /media/waqasm86/External1/Project-Nvidia-Office/Project-Nvidia-Office/llcuda

echo "📦 Package Information:"
echo "   Wheel:  llcuda-2.0.2-py3-none-any.whl (54 KB)"
echo "   Source: llcuda-2.0.2.tar.gz (67 KB)"
echo

# Verify packages exist
if [ ! -f "dist/llcuda-2.0.2-py3-none-any.whl" ]; then
    echo "❌ Error: Wheel file not found!"
    exit 1
fi

if [ ! -f "dist/llcuda-2.0.2.tar.gz" ]; then
    echo "❌ Error: Source tarball not found!"
    exit 1
fi

echo "✅ Both packages found in dist/"
echo

# Check if twine is installed
if ! command -v twine &> /dev/null; then
    echo "⚠️  twine not found. Installing..."
    pip install --upgrade twine
fi

echo "========================================================================="
echo "📤 Uploading to PyPI..."
echo "========================================================================="
echo
echo "You will be prompted for your PyPI credentials:"
echo "  Username: __token__"
echo "  Password: (your PyPI API token starting with pypi-...)"
echo
echo "If you don't have a token, create one at:"
echo "  https://pypi.org/manage/account/token/"
echo
echo "========================================================================="
echo

# Upload to PyPI
python3.11 -m twine upload dist/llcuda-2.0.2*

echo
echo "========================================================================="
echo "✅ Upload Complete!"
echo "========================================================================="
echo
echo "🔍 Verify at: https://pypi.org/project/llcuda/2.0.2/"
echo
echo "🧪 Test installation:"
echo "   pip install --upgrade llcuda"
echo "   python3.11 -c 'import llcuda; print(llcuda.__version__)'"
echo
echo "========================================================================="
