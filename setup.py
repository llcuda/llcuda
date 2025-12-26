#!/usr/bin/env python3
"""
Setup script for local-llama-cuda Python package
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Extension using CMake build system"""
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build extension using CMake"""
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Ensure directory exists
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        
        # CMake configuration arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DENABLE_CUDA=ON',
            '-DENABLE_MPI=OFF',  # Disable MPI for pip package
            '-DENABLE_BENCHMARKS=OFF',
            '-DENABLE_TESTS=OFF',
            '-DENABLE_EXAMPLES=OFF',
            '-DBUILD_PYTHON_BINDINGS=ON'
        ]
        
        # Build type
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        
        # CUDA architecture - default to T4 (7.5) for Kaggle/Colab
        cuda_arch = os.environ.get('CUDA_ARCHITECTURES', '75')
        cmake_args += [f'-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}']
        
        # Parallel build
        if hasattr(self, 'parallel') and self.parallel:
            build_args += ['-j' + str(self.parallel)]
        else:
            import multiprocessing
            build_args += ['-j' + str(multiprocessing.cpu_count())]
        
        # Create build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        
        # Run CMake
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=str(build_temp)
        )
        
        # Build
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=str(build_temp)
        )


# Read long description from README
def read_long_description():
    readme_path = Path(__file__).parent / 'README.md'
    if readme_path.exists():
        return readme_path.read_text(encoding='utf-8')
    return ''


setup(
    name='llcuda',
    version='0.1.0',
    author='Waqas Muhammad',
    author_email='waqasm86@gmail.com',
    description='CUDA-accelerated LLM inference for Python',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/waqasm86/llcuda',
    project_urls={
        'Bug Tracker': 'https://github.com/waqasm86/llcuda/issues',
        'Documentation': 'https://github.com/waqasm86/llcuda#readme',
        'Source Code': 'https://github.com/waqasm86/llcuda',
    },
    packages=find_packages(include=['llcuda', 'llcuda.*']),
    ext_modules=[CMakeExtension('llcuda._llcuda')],
    cmdclass={'build_ext': CMakeBuild},
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'mypy>=0.950',
            'sphinx>=4.5.0',
        ],
        'jupyter': [
            'jupyter>=1.0.0',
            'ipywidgets>=7.6.0',
            'matplotlib>=3.5.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='llm cuda gpu inference deep-learning llama',
    zip_safe=False,
)
