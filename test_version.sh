#!/bin/bash
echo "Testing version parsing..."

# Show pyproject.toml structure
echo "=== pyproject.toml ==="
grep -n 'version' pyproject.toml
echo ""
echo "Lines around [project]:"
sed -n '10,25p' pyproject.toml

echo ""
echo "=== Test different parsing methods ==="

# Method 1
echo "Method 1 (sed range):"
PYPROJECT_VERSION1=$(sed -n '/^\[project\]/,/^\[/p' pyproject.toml | grep 'version =' | head -1 | cut -d'"' -f2)
echo "Result: '$PYPROJECT_VERSION1'"

# Method 2
echo "Method 2 (simple grep):"
PYPROJECT_VERSION2=$(grep '^version =' pyproject.toml | head -1 | cut -d'"' -f2)
echo "Result: '$PYPROJECT_VERSION2'"

# Method 3
echo "Method 3 (grep with context):"
PYPROJECT_VERSION3=$(grep -A5 '^\[project\]' pyproject.toml | grep 'version =' | head -1 | cut -d'"' -f2)
echo "Result: '$PYPROJECT_VERSION3'"

echo ""
echo "=== __init__.py ==="
grep '__version__' llcuda/__init__.py
