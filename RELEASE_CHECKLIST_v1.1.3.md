# llcuda v1.1.3 Release Checklist

## Pre-Build Checks
- [x] Update version number in pyproject.toml
- [x] Update version number in llcuda/__init__.py
- [x] Update CHANGELOG.md with new version
- [x] Ensure README.md is up to date
- [x] Run all tests: `python -m pytest tests/ -v`
- [x] Clean up any unnecessary files

## Build Process
- [x] Create source distribution: `python -m build --sdist`
- [x] Create wheel distribution: `python -m build --wheel`
- [x] Verify package contents
- [x] Test installation locally

## Post-Build Checks
- [x] Check package size: `ls -lh dist/`
- [x] Test import in clean environment
- [x] Verify all required files are included

## Distribution
- [ ] Upload to PyPI: `python -m twine upload dist/*`
- [ ] Create GitHub release
- [ ] Update documentation if needed

## Files Created
- dist/llcuda-1.1.3.tar.gz
- dist/llcuda-1.1.3-py3-none-any.whl
