# llcuda Runtime Binaries v1.1.4

This bundle contains the CUDA-accelerated binaries for llcuda.

## Directory Structure:
- `binaries/cuda12/` - llama-server and related binaries
- `lib/` - CUDA shared libraries
- `models/` - (Empty - models downloaded on-demand)

## Installation:
This bundle is automatically extracted by llcuda's bootstrap system.
No manual installation required.

## CUDA Requirements:
- NVIDIA GPU with Compute Capability 7.5+
- CUDA 12.8 compatible drivers
- Linux x86_64 system

## Auto-extraction:
When you first import llcuda:
```python
import llcuda
The bootstrap system will download and extract this bundle automatically.

