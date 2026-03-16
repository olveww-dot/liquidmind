"""
LiquidMind - 液态神经网络
动态适应的智能系统
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="liquidmind",
    version="0.1.0",
    author="EC",
    author_email="",
    description="液态神经网络 - 动态适应的智能系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ec/liquidmind",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "matplotlib>=3.5.0",
            "pytest>=7.0.0",
        ],
        "examples": [
            "matplotlib>=3.5.0",
        ],
    },
    keywords="liquid neural networks, LTC, CfC, time series, forecasting, deep learning, pytorch",
    project_urls={
        "Bug Reports": "https://github.com/ec/liquidmind/issues",
        "Source": "https://github.com/ec/liquidmind",
    },
)
