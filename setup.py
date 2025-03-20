"""
Setup script for ML Simulation Environment.

This script installs the ML Simulation Environment package and its dependencies.
"""

from setuptools import setup, find_packages

setup(
    name="ml_simulation",
    version="1.0.0",
    author="ML Simulation Team",
    description="A simulation environment for machine learning with residuals analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ml-simulation/ml-simulation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.23.0",
        "tensorflow>=2.4.0",
        "pandas>=1.1.0",
        "seaborn>=0.11.0",
        "scipy>=1.5.0",
    ],
)
