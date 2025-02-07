from setuptools import setup, find_packages

setup(
    name="hybrid-llm-classifier",
    version="0.1.0",
    description="Zero-shot legal document classification using Mistral-7B and FAISS",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "python-jose[cryptography]>=3.3.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.5.2",
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.3",
        "httpx>=0.25.2",
        "python-dotenv>=1.0.0",
        "prometheus-client>=0.19.0",
        "pydantic-settings>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "mypy>=1.7.1",
            "flake8>=6.1.0",
        ]
    },
)
