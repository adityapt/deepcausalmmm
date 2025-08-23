"""
Setup script for DeepCausalMMM package.

This is a fallback setup.py for environments that don't fully support pyproject.toml.
The primary configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages

setup(
    name="deepcausalmmm",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.8",
)
