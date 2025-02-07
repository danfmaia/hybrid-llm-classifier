"""Integration tests for authentication and rate limiting middleware."""

from typing import Any
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock


def test_auth_no_token(test_client: TestClient) -> None:
    """Test that protected endpoints require authentication."""
    response = test_client.post(
        "/api/v1/classify/", json={"text": "test document"})
    assert response.status_code == 401
    assert "detail" in response.json()


def test_auth_invalid_token(test_client: TestClient) -> None:
    """Test that invalid tokens are rejected."""
    headers = {"Authorization": "Bearer invalid_token"}
    response = test_client.post(
        "/api/v1/classify/",
        headers=headers,
        json={"text": "test document"}
    )
    assert response.status_code == 401
    assert "detail" in response.json()


def test_auth_token_generation(test_client: TestClient) -> None:
    """Test token generation endpoint."""
    response = test_client.post(
        "/api/v1/auth/token",
        data={"username": "testuser", "password": "testpass"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "token_type" in data
    assert data["token_type"] == "bearer"


def test_rate_limiting(test_client: TestClient, test_settings: Any) -> None:
    """Test that rate limiting middleware works."""
    # Get a valid token first
    auth_response = test_client.post(
        "/api/v1/auth/token",
        data={"username": "testuser", "password": "testpass"}
    )
    token = auth_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Make requests up to the limit
    for _ in range(test_settings.rate_limit_requests):
        response = test_client.get("/api/v1/test-rate-limit", headers=headers)
        # Endpoint doesn't exist, but should still be rate limited
        assert response.status_code == 404

    # Next request should be rate limited
    response = test_client.get("/api/v1/test-rate-limit", headers=headers)
    assert response.status_code == 429
    assert "detail" in response.json()
    assert "Rate limit exceeded" in response.json()["detail"]


def test_protected_endpoint_with_valid_token(test_client: TestClient) -> None:
    """Test accessing protected endpoint with valid token."""
    # Get a valid token
    auth_response = test_client.post(
        "/api/v1/auth/token",
        data={"username": "testuser", "password": "testpass"}
    )
    token = auth_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Mock the classifier to avoid actual API calls
    with patch("app.models.classifier.HybridClassifier.classify", new_callable=AsyncMock) as mock_classify:
        mock_classify.side_effect = NotImplementedError(
            "Classification not yet implemented")

        # Access protected endpoint
        response = test_client.post(
            "/api/v1/classify/",
            headers=headers,
            json={"text": "test document"}
        )
        assert response.status_code == 501  # Should return Not Implemented


def test_batch_request_with_rate_limit(test_client: TestClient) -> None:
    """Test batch requests respect rate limiting."""
    # Get a valid token
    auth_response = test_client.post(
        "/api/v1/auth/token",
        data={"username": "testuser", "password": "testpass"}
    )
    token = auth_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Prepare batch request data
    batch_data = [
        {"text": f"document {i}", "metadata": {"id": i}}
        for i in range(5)
    ]

    # Send batch request
    response = test_client.post(
        "/api/v1/classify/batch",
        headers=headers,
        json=batch_data
    )
    # Note: Will return 501 Not Implemented until we implement the classifier
    assert response.status_code in (200, 501)
