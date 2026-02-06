"""
Document AI Platform - Production Setup
A complete OCR and document intelligence suite without Tesseract
"""
from setuptools import setup, find_packages

setup(
    name="document-ai-platform",
    version="1.0.0",
    description="Production-grade OCR and document intelligence platform",
    author="FAANG Engineering Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "opencv-contrib-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "onnxruntime>=1.15.0",
        "pdf2image>=1.16.0",
        "PyMuPDF>=1.23.0",
        "python-docx>=0.8.11",
        "openpyxl>=3.1.0",
        "scipy>=1.11.0",
        "scikit-image>=0.21.0",
        "albumentations>=1.3.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "loguru>=0.7.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.7.0",
        "flake8>=6.1.0",
        "mypy>=1.5.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "docai=api.cli.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
