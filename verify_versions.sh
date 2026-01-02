#!/bin/bash
echo "=== Version Verification ==="
echo "1. Local pyproject.toml:"
grep "version = " pyproject.toml
echo ""
echo "2. Local __init__.py:"
grep "__version__" llcuda/__init__.py
echo ""
echo "3. PyPI package (checking via pip index):"
python -c "import requests; import json; r=requests.get('https://pypi.org/pypi/llcuda/json'); print('PyPI version:', json.loads(r.text)['info']['version'])" 2>/dev/null || echo "Could not fetch PyPI info"
echo ""
echo "✓ All versions should be: 1.1.1.post1"
