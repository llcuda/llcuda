#!/usr/bin/env python3
"""
Setup script for llcuda-cu128 (CUDA 12.8 integrated package)

This creates a PyTorch-style wheel with bundled CUDA binaries and libraries.
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import shutil
from pathlib import Path


class PostInstallCommand(install):
    """Post-installation: Setup library paths and permissions"""

    def run(self):
        install.run(self)
        self.setup_after_install()

    def setup_after_install(self):
        """Setup after installation complete"""
        try:
            # Get installation directory
            install_dir = Path(self.install_lib) / 'llcuda'

            if not install_dir.exists():
                print(f"Warning: Installation directory not found: {install_dir}")
                return

            # Make binaries executable
            bin_dir = install_dir / 'binaries' / 'cuda12'
            if bin_dir.exists():
                for binary in bin_dir.glob('*'):
                    if binary.is_file():
                        os.chmod(binary, 0o755)
                        print(f"✓ Made executable: {binary.name}")

            print("\n" + "=" * 70)
            print("llcuda (CUDA 12.8) installed successfully!")
            print("=" * 70)
            print("\nQuick Start:")
            print("  import llcuda")
            print("  engine = llcuda.InferenceEngine()")
            print("  engine.load_model('gemma-3-1b-Q4_K_M')")
            print("  result = engine.infer('What is AI?')")
            print("  print(result.text)")
            print("\n" + "=" * 70)

        except Exception as e:
            print(f"Warning: Post-install setup encountered an error: {e}")
            print("Installation may still work, but you may need to chmod +x binaries manually")


class PostDevelopCommand(develop):
    """Post-develop: Setup library paths"""

    def run(self):
        develop.run(self)
        # Same setup as install
        install_cmd = PostInstallCommand(self.distribution)
        install_cmd.install_lib = self.install_lib
        install_cmd.setup_after_install()


def read_long_description():
    readme_path = Path(__file__).parent / 'README.md'
    if readme_path.exists():
        return readme_path.read_text(encoding='utf-8')
    return 'CUDA-accelerated LLM inference for Python with integrated CUDA 12.8 binaries'


setup(
    name='llcuda-cu128',
    version='1.0.0',
    author='Waqas Muhammad',
    author_email='waqasm86@gmail.com',
    description='PyTorch-style CUDA-accelerated LLM inference with integrated CUDA 12.8 binaries',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/waqasm86/llcuda',
    project_urls={
        'Bug Tracker': 'https://github.com/waqasm86/llcuda/issues',
        'Documentation': 'https://waqasm86.github.io/',
        'Source Code': 'https://github.com/waqasm86/llcuda',
        'Changelog': 'https://github.com/waqasm86/llcuda/blob/main/CHANGELOG.md',
    },
    packages=find_packages(include=['llcuda', 'llcuda.*']),
    include_package_data=True,
    package_data={
        'llcuda': [
            'binaries/cuda12/*',
            'lib/*.so*',
            'lib/cmake/**/*',
            'models/.gitkeep',
            '_internal/*.py',
        ]
    },
    python_requires='>=3.11',
    install_requires=[
        'numpy>=1.20.0',
        'requests>=2.20.0',
        'huggingface_hub>=0.10.0',  # For model downloads
        'tqdm>=4.60.0',              # Progress bars
    ],
    extras_require={
        'jupyter': [
            'ipywidgets>=7.6.0',
            'IPython>=7.0.0',
            'matplotlib>=3.5.0',
            'pandas>=1.3.0',
        ],
        'embeddings': [
            'scikit-learn>=1.0.0',
        ],
        'all': [
            'ipywidgets>=7.6.0',
            'IPython>=7.0.0',
            'matplotlib>=3.5.0',
            'pandas>=1.3.0',
            'scikit-learn>=1.0.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'mypy>=0.950',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Framework :: Jupyter',
        'Environment :: GPU :: NVIDIA CUDA :: 12',
    ],
    keywords='llm cuda gpu inference deep-learning llama gguf cuda12 nvidia pytorch-style',
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
    zip_safe=False,  # Important: Don't zip the package (binaries need direct access)
)
