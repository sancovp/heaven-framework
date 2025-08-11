#!/usr/bin/env python3
"""
Setup script for HEAVEN framework.
This file is needed for backward compatibility and editable installs.
"""
from setuptools import setup, find_packages
import os

# Read the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="heaven-framework",
    version="0.1.11",
    description="HEAVEN - Hierarchical, Embodied, Autonomously Validating Evolution Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="HEAVEN Team",
    author_email="heaven@example.com",
    url="https://github.com/heaven-framework/heaven",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "google-adk>=0.2.0",
        "google-genai>=1.11.0",
        "langchain-anthropic>=0.3.13",
        "langchain-core>=0.3.65",
        "langchain-deepseek>=0.1.2",
        "langchain-google-genai>=2.1.5",
        "langchain-groq>=0.2.4",
        "langchain-openai>=0.3.22",
        "litellm>=1.67.5",
        "pydantic==2.10.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "black>=23.0",
            "ruff>=0.1.0",
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="agents ai llm framework heaven",
    project_urls={
        "Documentation": "https://heaven.readthedocs.io",
        "Source": "https://github.com/heaven-framework/heaven",
        "Issues": "https://github.com/heaven-framework/heaven/issues",
    },
    include_package_data=True,
    zip_safe=False,
)