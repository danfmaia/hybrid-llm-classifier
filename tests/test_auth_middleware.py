"""Test authentication and rate limiting middleware."""

import pytest
import asyncio
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from app.middleware.auth import AuthMiddleware
from app.middleware.rate_limit import RateLimitMiddleware, REQUESTS_PER_MINUTE
from app.utils.auth import generate_token


@pytest.fixture
def test_app():
    """Create test FastAPI application."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)

    @app.get("/api/v1/test-auth")
    async def test_auth():
        return {"message": "Authenticated"}

    @app.get("/api/v1/test-rate-limit")
    async def test_rate_limit():
        return {"message": "Success"}

    return app


@pytest.fixture
def test_client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def valid_token():
    """Generate a valid token for testing."""
    return generate_token({"user_id": "test_user"})


@pytest.mark.asyncio
async def test_auth_no_token(test_client):
    """Test request without token."""
    response = test_client.get("/api/v1/test-auth")
    assert response.status_code == 401
    assert response.json()["detail"] == "No authentication token provided"


@pytest.mark.asyncio
async def test_auth_invalid_token(test_client):
    """Test request with invalid token."""
    response = test_client.get(
        "/api/v1/test-auth",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid authentication token"


@pytest.mark.asyncio
async def test_auth_token_generation():
    """Test token generation."""
    token = generate_token({"user_id": "test_user"})
    assert isinstance(token, str)
    assert len(token) > 0


@pytest.mark.asyncio
async def test_rate_limiting(test_client, valid_token):
    """Test rate limiting."""
    # Make requests up to the limit
    for _ in range(REQUESTS_PER_MINUTE):
        response = test_client.get(
            "/api/v1/test-rate-limit",
            headers={"Authorization": f"Bearer {valid_token}"}
        )
        assert response.status_code == 200

    # Next request should be rate limited
    response = test_client.get(
        "/api/v1/test-rate-limit",
        headers={"Authorization": f"Bearer {valid_token}"}
    )
    assert response.status_code == 429
    assert "Too many requests" in response.json()["detail"]


@pytest.mark.asyncio
async def test_protected_endpoint_with_valid_token(test_client, valid_token):
    """Test protected endpoint with valid token."""
    response = test_client.get(
        "/api/v1/test-auth",
        headers={"Authorization": f"Bearer {valid_token}"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Authenticated"


@pytest.mark.asyncio
async def test_batch_request_with_rate_limit(test_client, valid_token):
    """Test batch request handling with rate limit."""
    # Make concurrent requests
    async def make_request():
        return test_client.get(
            "/api/v1/test-rate-limit",
            headers={"Authorization": f"Bearer {valid_token}"}
        )

    tasks = [make_request() for _ in range(REQUESTS_PER_MINUTE + 5)]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Count response types
    success_count = sum(1 for r in responses if getattr(
        r, 'status_code', None) == 200)
    rate_limited_count = sum(
        1 for r in responses if getattr(r, 'status_code', None) == 429)

    assert success_count == REQUESTS_PER_MINUTE
    assert rate_limited_count == 5
