"""Metadata describing the configuration of package"""
import os
from setuptools import find_packages, setup

BUILD_ID = os.environ.get("BUILD_BUILDID", "0")

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="diffilqrax",
    version="0.1" + "." + BUILD_ID,
    packages=find_packages(),
    description="Differentiable iLQR algorithm for dynamical systems with JAX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThomasMullen/diffilqrax",
    author="Thomas Soares Mullen, Marine Schimel",
    author_email="thomasmullen96@gmail.com, marine.schimel@hotmail.fr",
    license="MIT",
    install_requires=[
        "matplotlib",
        "numpy",
        "jax",
        "jaxopt",
        "chex",
        "pytest",
        "wheel",
        "Pillow",
    ],
    extras_require={
        "dev": ["twine>=4.0.2"],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development",
        "Topic :: Utilities",
    ],
)
