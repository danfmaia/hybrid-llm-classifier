# Hybrid Legal Document Classifier

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1+-blue.svg)](https://fastapi.tiangolo.com)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready zero-shot legal document classification system powered by Mistral-7B and FAISS vector similarity validation. This hybrid approach combines the reasoning capabilities of Large Language Models with the precision of embedding-based validation to achieve high-accuracy document classification.

## 🚀 Features

- **Zero-Shot Classification**: Leverages Mistral-7B for flexible category inference without training data
- **Hybrid Validation**: FAISS vector store validation ensures classification accuracy
- **Production-Ready Architecture**:
  - FastAPI async endpoints with comprehensive middleware
  - Rate limiting and JWT authentication
  - Performance monitoring and logging
- **High Performance**:
  - < 2s response time at 150 RPM
  - 95% classification accuracy target on LegalBench
  - Optimized caching and GPU utilization

## 🏗️ Technical Architecture

```
src/
├── app/
│   ├── auth/          # JWT authentication and token handling
│   ├── models/        # Core classification models
│   ├── middleware/    # Auth and rate limiting
│   └── routers/      # API endpoints and routing
tests/                # Test suite
```

### Key Components

1. **Classification Engine**

   - Mistral-7B integration via Ollama
   - Zero-shot prompt engineering
   - FAISS validation layer

2. **API Layer**
   - Async endpoint structure
   - JWT authentication and rate limiting
   - Input validation schemas

## 🚦 Project Status

The project is in active development with core functionality implemented:

✅ Classification engine with Mistral-7B  
✅ FastAPI application structure  
✅ Security middleware (JWT + Rate limiting)  
✅ Basic test suite  
🚧 Performance optimization  
🚧 Production deployment

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/hybrid-llm-classifier.git
cd hybrid-llm-classifier
```

2. **Set up the environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

3. **Install and start Ollama**
   Follow instructions at [Ollama.ai](https://ollama.ai) to install and run the Mistral model

## 🚀 Quick Start

1. **Start the API server**

```bash
uvicorn app.main:app --reload
```

2. **Make a classification request**

```python
import httpx

async with httpx.AsyncClient() as client:
    # First get an access token
    auth_response = await client.post(
        "http://localhost:8000/api/v1/auth/token",
        data={"username": "your_username", "password": "your_password"}
    )
    token = auth_response.json()["access_token"]

    # Make classification request
    response = await client.post(
        "http://localhost:8000/api/v1/classify",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "text": "Your legal document text here"
        }
    )
    result = response.json()
```

## 🧪 Development

### Running Tests

```bash
make test        # Run unit tests
make test-cov    # Run tests with coverage
make lint        # Run linting checks
```

### Code Quality

- Black for code formatting
- isort for import sorting
- mypy for type checking
- pytest for testing

## 📈 Performance

Target performance metrics (see BENCHMARKS.md for testing plan):

- Response time: < 2s at 150 RPM target
- Classification accuracy: 95% target on LegalBench
- Resource utilization optimization planned

## 🛣️ Roadmap

1. **Q1 2024**

   - Performance benchmarking and optimization
   - Caching implementation
   - AWS production deployment
   - Extended test coverage

2. **Q2 2024**
   - Multi-model ensemble support
   - Real-time performance monitoring
   - Extended legal taxonomy

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

While this project is primarily for demonstration purposes, we welcome feedback and suggestions. Please open an issue to discuss potential improvements.

---

_Note: This project is under active development. Core functionality is implemented and tested, with additional features and optimizations in progress._
