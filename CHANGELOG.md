# Changelog

## [1.1.4] - 2024-01-XX

### Fixed
- Removed automatic model download when importing llcuda
- Bootstrap now only downloads binaries, not models
- Models are downloaded on-demand via `engine.load_model()`

### Added
- Explicit model download function: `download_default_model()`
- Better control over when downloads occur
- Improved .gitignore to exclude large model files

### Changed
- Updated bootstrap messages to clarify model download process
- `load_model()` now handles downloads transparently
- Better user experience with controlled downloads

### Removed
- Automatic 769MB model download on import
- Unnecessary print statements during setup
- Potential timeout issues with HuggingFace downloads

## [1.1.3] - Previous Version
- Multi-GPU architecture support
- Colab/Kaggle compatibility
- Hybrid bootstrap system
