#!/usr/bin/env python3
"""
Simple setup script for llcuda Python package (Pure Python - No C++ Extensions)
"""

from setuptools import setup, find_packages
from pathlib import Path


def read_long_description():
    readme_path = Path(__file__).parent / 'README.md'
    if readme_path.exists():
        return readme_path.read_text(encoding='utf-8')
    return ''


setup(
    name='llcuda',
    version='0.2.0',
    author='Waqas Muhammad',
    author_email='waqasm86@gmail.com',
    description='CUDA-accelerated LLM inference for Python with automatic server management',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/waqasm86/llcuda',
    project_urls={
        'Bug Tracker': 'https://github.com/waqasm86/llcuda/issues',
        'Documentation': 'https://github.com/waqasm86/llcuda#readme',
        'Source Code': 'https://github.com/waqasm86/llcuda',
    },
    packages=find_packages(include=['llcuda', 'llcuda.*']),
    python_requires='>=3.11',
    install_requires=[
        'numpy>=1.20.0',
        'requests>=2.20.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'mypy>=0.950',
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
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='llm cuda gpu inference deep-learning llama',
    zip_safe=True,
)
