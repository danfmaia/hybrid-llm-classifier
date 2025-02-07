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
  - JWT authentication and rate limiting
  - Performance monitoring and logging
- **Current Performance** (as of Feb 7, 2025):
  - Response time: ~11.37s per request
  - Classification accuracy: 85% on initial testing
  - GPU utilization: 22 layers on GPU
  - Throughput: ~5 requests per minute

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
   - GPU-accelerated inference (22 layers)
   - FAISS similarity validation
   - Automatic retry logic

2. **API Layer**
   - Async endpoint structure
   - JWT authentication
   - Rate limiting (1000 req/min)
   - Detailed error handling

## 🚦 Project Status

Current implementation status (Feb 7, 2025):

✅ Core Classification Engine

- GPU-accelerated Mistral-7B integration
- Basic FAISS validation layer
- Performance monitoring

✅ API Infrastructure

- FastAPI application structure
- JWT authentication
- Rate limiting middleware

✅ Security Features

- Input validation
- Error handling
- Request logging

🚧 Performance Optimization

- Current: 11.37s response time
- Target: <2s response time
- Planned: Response streaming, caching

## 🛠️ Installation

### Prerequisites

- NVIDIA GPU with 4GB+ VRAM
- 4+ CPU cores
- 16GB+ system RAM
- Python 3.10+

### Steps

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
   - Follow instructions at [Ollama.ai](https://ollama.ai)
   - Pull Mistral model: `ollama pull mistral`
   - Verify GPU support: `nvidia-smi`

## 🚀 Quick Start

1. **Start the API server**

```bash
uvicorn app.main:app --reload
```

2. **Get authentication token**

```python
import httpx

async with httpx.AsyncClient() as client:
    auth_response = await client.post(
        "http://localhost:8000/api/v1/auth/token",
        data={"username": "your_username", "password": "your_password"}
    )
    token = auth_response.json()["access_token"]
```

3. **Make classification request**

```python
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

### Performance Guidelines

- Keep documents under 2,048 tokens for optimal performance
- Batch requests: 5-10 documents recommended
- Memory usage: ~3.5GB VRAM baseline
- Expected latency: ~11.37s per request

## 📈 Performance

Current metrics (see BENCHMARKS.md for details):

- Average response time: 11.37s
- Classification accuracy: 85%
- GPU memory usage: 3.5GB VRAM
- Throughput: ~5 requests/minute

Optimization roadmap:

1. Response streaming implementation
2. Request caching layer
3. Batch processing optimization
4. GPU kernel tuning

## 🛣️ Roadmap

1. **Short-term (Pre-deployment)**

   - Implement response streaming
   - Add request caching
   - Optimize batch processing

2. **Medium-term**

   - Load balancing setup
   - Memory optimization
   - Warm-up strategies

3. **Long-term**
   - Distributed processing
   - Custom GPU kernels
   - Advanced caching

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

While this project is primarily for demonstration purposes, we welcome feedback and suggestions. Please open an issue to discuss potential improvements.

---

_Note: This project is under active development. Core functionality is implemented and tested, with performance optimizations in progress._
