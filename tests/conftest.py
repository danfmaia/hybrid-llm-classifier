"""Test configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add the src directory to Python path for imports
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# First party imports (after path setup)
try:
    from app.config import get_settings
    from app.main import app
    from app.middleware import RateLimitMiddleware
except ImportError as e:
    pytest.exit(f"Failed to import app modules: {e}")


@pytest.fixture(scope="function")
def rate_limiter() -> RateLimitMiddleware:
    """Create a test rate limiter instance."""
    limiter = RateLimitMiddleware(app)
    limiter.test_mode = True
    return limiter


@pytest.fixture(scope="function")
def test_app(rate_limiter: RateLimitMiddleware) -> FastAPI:
    """Create a test app instance with test middleware."""
    test_app = FastAPI()
    test_app.router = app.router
    test_app.middleware_stack = None  # Clear existing middleware
    test_app.add_middleware(RateLimitMiddleware)
    return test_app


@pytest.fixture(scope="function")
def test_client(test_app: FastAPI, rate_limiter: RateLimitMiddleware) -> TestClient:
    """Create a test client instance."""
    # Ensure rate limiter is in test mode and reset
    rate_limiter.test_mode = True
    rate_limiter.reset()

    # Create client with test app
    client = TestClient(test_app)
    return client


@pytest.fixture(scope="session")
def test_settings():
    """Get application settings."""
    return get_settings()


@pytest.fixture(autouse=True)
def reset_rate_limiter(rate_limiter: RateLimitMiddleware):
    """Reset rate limiter state before each test."""
    rate_limiter.reset()
    yield
    rate_limiter.reset()  # Also reset after each test
