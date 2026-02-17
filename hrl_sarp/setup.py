"""
Setup script for the HRL-SARP framework.
Install via: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="hrl-sarp",
    version="1.0.0",
    description="Hierarchical RL for Sector-Aware Risk-Adaptive Portfolio Management",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="HRL-SARP Framework Contributors",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "torch>=2.0",
        "gymnasium>=0.28",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "pyyaml>=6.0",
        "matplotlib>=3.7",
    ],
    extras_require={
        "full": [
            "mlflow>=2.5",
            "streamlit>=1.25",
            "plotly>=5.15",
            "shap>=0.42",
            "transformers>=4.30",
            "pandas-ta>=0.3",
            "yfinance>=0.2",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "hrl-sarp=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
