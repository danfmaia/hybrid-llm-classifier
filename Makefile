.PHONY: setup test lint format clean

# Environment setup
setup:
	conda env create -f environment.yml
	pip install -e .

# Testing
test:
	pytest tests/ -v --cov=app --cov-report=term-missing

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