"""Setup script for MuAI Orchestration System."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="muai-orchestration",
    version="0.1.0",
    author="MuAI Team",
    description="MuAI Multi-Model Orchestration System with Consciousness Modules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "duckduckgo-search>=3.8.0",
        "trafilatura>=1.6.0",
        "httpx>=0.24.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "hypothesis>=6.82.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
