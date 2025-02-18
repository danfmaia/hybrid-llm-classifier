# Hybrid Legal Document Classifier

![Hybrid Legal Document Classifier](gh-cover.jpeg)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1+-blue.svg)](https://fastapi.tiangolo.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
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
- **Current Performance** (as of Feb 11, 2025):
  - Response time: ~33.18s per request
  - Classification accuracy: 100% on latest tests
  - GPU utilization: Not optimal
  - Throughput: ~1.8 requests per minute

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
  - Response Time: ~33s average
  - Throughput: 1-2 RPM
  - Classification Accuracy: 100%

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

Day 1:

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

## 🛠️ Development Setup

### Prerequisites

- NVIDIA GPU with 4GB+ VRAM
- 4+ CPU cores
- 16GB+ system RAM
- Python 3.10+
- Conda (recommended for environment management)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/hybrid-llm-classifier.git
cd hybrid-llm-classifier
```

2. **Set up the environment**

```bash
# Create and activate environment
make setup

# Install development dependencies
make install-dev
```

3. **Install and start Ollama**
   - Follow instructions at [Ollama.ai](https://ollama.ai)
   - Pull Mistral model: `ollama pull mistral`
   - Verify GPU support: `nvidia-smi`

### Development Commands

We use `make` to standardize development commands. Here are the available targets:

#### Testing

```bash
# Run basic tests
make test

# Run tests with coverage report
make test-coverage

# Run tests in watch mode (auto-rerun on changes)
make test-watch

# Run tests with verbose output
make test-verbose
```

#### Performance Testing

```bash
# Run full benchmark suite
make benchmark

# Run continuous benchmark monitoring
make benchmark-watch

# Run memory and line profiling
make benchmark-profile
```

#### Code Quality

```bash
# Format code (black + isort)
make format

# Run all linters
make lint
```

#### Development Server

```bash
# Start development server with hot reload
make run
```

#### Cleanup

```bash
# Remove all build artifacts and cache files
make clean
```

For a complete list of available commands:

```bash
make help
```

### Test Coverage

Current test suite includes:

- Unit tests for core classification
- Integration tests for API endpoints
- Authentication and rate limiting tests
- Performance metrics validation
- Error handling scenarios
- Benchmark tests

All tests are async-compatible and use pytest-asyncio for proper async testing.

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

- Average response time: ~33.18s
- Classification accuracy: 100%
- GPU utilization: Not optimal
- Throughput: ~1.8 requests/minute

Production Targets (AWS g5.2xlarge):

- Response time: <2s
- Throughput: 150+ RPM
- Accuracy: 90-95%
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
