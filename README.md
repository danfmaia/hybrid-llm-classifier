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

### Performance Characteristics

#### Development Environment (Local)

- Hardware Requirements:
  - NVIDIA GPU with 4GB+ VRAM
  - 4+ CPU cores
  - 16GB+ system RAM
- Expected Performance:
  - Response Time: ~10s acceptable
  - Throughput: 5-10 RPM
  - Classification Accuracy: 85%

#### Production Environment (AWS)

1. Minimum Configuration (g5.xlarge):

   - NVIDIA A10G GPU (24GB VRAM)
   - Response Time: 3-4s
   - Throughput: 30-40 RPM per instance
   - Classification Accuracy: 85-90%

2. Target Configuration (g5.2xlarge or higher):
   - Response Time: ~2s
   - Throughput: 150+ RPM (with load balancing)
   - Classification Accuracy: 90-95%
   - High Availability: 99.9%

### Key Components

1. **Classification Engine**

   - Mistral-7B integration via Ollama
   - GPU-accelerated inference
   - FAISS similarity validation
   - Response caching (1-hour TTL)

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

🚧 In Progress (3-Day Sprint)

Day 1 (Today):

- Optimizing Mistral-7B integration
- Finalizing FAISS validation
- Implementing response caching

Day 2:

- API security refinements
- Performance optimization
- Load testing implementation

Day 3:

- AWS deployment setup
- Documentation completion
- Final testing & benchmarks

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

Development Environment:

- Keep documents under 2,048 tokens
- Expect ~10s response time
- 5-10 requests per minute
- Memory usage: ~3.5GB VRAM

Production Environment:

- AWS g5.xlarge or higher recommended
- Load balancing for high throughput
- Auto-scaling configuration
- Regional deployment for latency optimization

## 📈 Performance

See [BENCHMARKS.md](./BENCHMARKS.md) for detailed performance analysis and optimization experiments.

Development Environment (Current):

- Average response time: ~11.37s
- Classification accuracy: 85%
- GPU memory usage: 3.5GB VRAM
- Throughput: ~5 requests/minute

Production Targets (AWS g5.2xlarge):

- Response time: <2s
- Throughput: 150+ RPM
- Accuracy: 85-90%
- High availability: 99.9%

Optimization Roadmap:

1. Response Caching

   - In-memory caching for repeated queries
   - Configurable TTL
   - Cache hit monitoring

2. Performance Optimization

   - Response streaming
   - Batch processing
   - Memory usage optimization

3. Infrastructure
   - Docker containerization
   - AWS deployment
   - Load balancing setup
   - Monitoring integration

## 🛣️ Roadmap

1. **Core Functionality** (Day 1)

   - Optimize classification engine ✅
   - Implement caching layer
   - Document performance baselines

2. **API & Performance** (Day 2)

   - Security hardening
   - Response optimization
   - Load testing

3. **Production Ready** (Day 3)
   - AWS deployment
   - Documentation
   - Final testing

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

While this project is primarily for demonstration purposes, we welcome feedback and suggestions. Please open an issue to discuss potential improvements.

---

_Note: This project is under active development. Core functionality is implemented and tested, with performance optimizations in progress._
