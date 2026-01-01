from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llcuda",
    version="1.1.4",
    author="Waqas Mahmood",
    author_email="waqas.mahmood@example.com",
    description="CUDA-Accelerated LLM Inference for Python - Self-contained with binaries and auto-download",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/waqasm86/llcuda",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "huggingface-hub>=0.19.0",
        "psutil>=5.9.0",
        "py-cpuinfo>=9.0.0",
    ],
    include_package_data=True,
    package_data={
        "llcuda": ["*.txt", "*.json", "*.md"],
    },
)
