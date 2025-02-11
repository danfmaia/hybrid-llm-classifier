.PHONY: setup test test-verbose test-coverage test-watch lint format clean install-dev run benchmark benchmark-watch benchmark-profile help

# Environment setup
setup:
	conda env create -f environment.yml
	pip install -e .

# Development dependencies
install-dev:
	pip install -r requirements-dev.txt

# Testing
test:
	pytest tests/

test-verbose:
	pytest tests/ -v --cov=app --cov-report=term-missing

test-coverage:
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

test-watch:
	ptw tests/ --onpass "echo 'All tests passed! 🎉'" --onfail "echo 'Tests failed! 😢'"

# Benchmarking
benchmark:
	@echo "Running performance benchmarks..."
	python scripts/run_benchmarks.py
	@echo "Benchmark results saved to benchmark_results/"

benchmark-watch:
	@echo "Running continuous benchmark monitoring..."
	pytest tests/ --benchmark-only --benchmark-autosave

benchmark-profile:
	@echo "Running profiling on classification engine..."
	python -m memory_profiler scripts/test_performance.py
	@echo "Memory profile completed."
	python -m line_profiler scripts/test_performance.py
	@echo "Line profile completed."

# Linting and type checking
lint:
	black src/app tests
	isort src/app tests
	mypy src/app tests
	flake8 src/app tests

# Code formatting
format:
	black src/app tests
	isort src/app tests

# Run development server
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".benchmarks" -exec rm -rf {} +

# Help target
help:
	@echo "Available targets:"
	@echo "  setup         - Create conda environment and install package"
	@echo "  install-dev   - Install development dependencies"
	@echo "  test          - Run tests"
	@echo "  test-verbose  - Run tests with coverage report"
	@echo "  test-coverage - Run tests and generate HTML coverage report"
	@echo "  test-watch    - Run tests in watch mode"
	@echo "  benchmark     - Run performance benchmarks"
	@echo "  benchmark-watch - Run continuous benchmark monitoring"
	@echo "  benchmark-profile - Run memory and line profiling"
	@echo "  lint          - Run all linters"
	@echo "  format        - Format code with black and isort"
	@echo "  run           - Run development server"
	@echo "  clean         - Remove build artifacts and cache files"
	@echo "  help          - Show this help message" 