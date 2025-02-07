"""Unit tests for the HybridClassifier."""

# Standard library imports
import json
from unittest.mock import patch, AsyncMock

# Third-party imports
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

# First-party imports
from app.models.classifier import HybridClassifier, ClassificationResult
from app.main import app
from app.auth.jwt import create_access_token


@pytest.fixture(name="test_token")
def fixture_test_token():
    """Create a valid test JWT token."""
    return create_access_token(data={"sub": "testuser"})


@pytest.fixture(name="auth_headers")
def fixture_auth_headers(test_token):
    """Create headers with valid JWT token."""
    return {"Authorization": f"Bearer {test_token}"}


@pytest.fixture(name="test_client")
def fixture_test_client():
    """Create a test client instance."""
    return TestClient(app)


@pytest.fixture(name="mock_embeddings")
def fixture_mock_embeddings():
    """Create mock embeddings for testing."""
    return np.random.rand(1, 384).astype(np.float32)


@pytest.fixture(name="classifier")
def fixture_classifier():
    """Create a classifier instance for testing."""
    return HybridClassifier(
        ollama_base_url="http://test-ollama:11434",
        model_name="mistral",
        embedding_dim=384,
        similarity_threshold=0.75
    )


@pytest.mark.asyncio
async def test_classifier_initialization(classifier):
    """Test classifier initialization with default parameters."""
    assert classifier.ollama_base_url == "http://test-ollama:11434"
    assert classifier.model_name == "mistral"
    assert classifier.similarity_threshold == 0.75


@pytest.mark.asyncio
async def test_classification_result_model():
    """Test ClassificationResult Pydantic model."""
    result = ClassificationResult(
        category="contract",
        confidence=0.95,
        validation_score=0.85
    )
    assert result.category == "contract"
    assert result.confidence == 0.95
    assert result.validation_score == 0.85


@pytest.mark.asyncio
async def test_classify_endpoint(test_client, auth_headers):
    """Test the classification endpoint."""
    with patch("app.models.classifier.HybridClassifier.classify", new_callable=AsyncMock) as mock_classify:
        mock_classify.return_value = ClassificationResult(
            category="Contract",
            confidence=0.85,
            validation_score=0.75
        )

        response = test_client.post(
            "/api/v1/classify/",
            headers=auth_headers,
            json={
                "text": "This contract agreement is made between...",
                "metadata": {"source": "test"}
            }
        )

        assert response.status_code == 200
        result = response.json()
        assert result["category"] == "Contract"
        assert result["confidence"] == 0.85
        assert result["validation_score"] == 0.75


@pytest.mark.asyncio
async def test_batch_classification(test_client, auth_headers):
    """Test batch classification endpoint."""
    with patch("app.models.classifier.HybridClassifier.classify", new_callable=AsyncMock) as mock_classify:
        mock_classify.return_value = ClassificationResult(
            category="Contract",
            confidence=0.85,
            validation_score=0.75
        )

        response = test_client.post(
            "/api/v1/classify/batch",
            headers=auth_headers,
            json=[
                {
                    "text": "First legal document...",
                    "metadata": {"id": 1}
                },
                {
                    "text": "Second legal document...",
                    "metadata": {"id": 2}
                }
            ]
        )

        assert response.status_code == 200
        results = response.json()
        assert len(results) == 2
        for result in results:
            assert result["category"] == "Contract"
            assert result["confidence"] == 0.85
            assert result["validation_score"] == 0.75


@pytest.mark.asyncio
async def test_llm_classification_success(classifier):
    """Test successful LLM classification."""
    mock_llm_response = {
        "response": json.dumps({
            "category": "Contract",
            "confidence": 0.85,
            "explanation": "This is a legal contract."
        })
    }

    mock_embedding_response = {
        "embedding": [0.1] * 384
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.status_code = 200

        # Track which URL is being called
        called_urls = []

        async def mock_api_call(*args, **kwargs):
            if args and isinstance(args[0], str):
                url = args[0]
            else:
                url = kwargs.get("url", "")
            called_urls.append(url)

            if "/api/generate" in str(url):
                return mock_llm_response
            if "/api/embeddings" in str(url):
                return mock_embedding_response
            # Handle direct json() calls from response objects
            if not url and hasattr(mock_post.return_value, '_mock_return_value'):
                # Get the last URL that was called
                last_url = mock_post.call_args[0][0] if mock_post.call_args else ""
                if "/api/generate" in str(last_url):
                    return mock_llm_response
                if "/api/embeddings" in str(last_url):
                    return mock_embedding_response
            raise ValueError(f"Unexpected API call: {url}")

        mock_post.return_value.json = AsyncMock(side_effect=mock_api_call)

        result = await classifier.classify("Sample contract text")

        assert isinstance(result, ClassificationResult)
        assert result.category == "Contract"
        assert result.confidence == 0.85
        assert len(called_urls) > 0


@pytest.mark.asyncio
async def test_llm_classification_invalid_response(classifier):
    """Test LLM classification with invalid response format."""
    mock_response = {
        "response": "Invalid JSON response"
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.status_code = 200

        async def mock_api_call(*args, **kwargs):
            if args and isinstance(args[0], str):
                url = args[0]
            else:
                url = kwargs.get("url", "")

            if "/api/generate" in str(url) or not url:
                return mock_response
            return {"embedding": [0.1] * 384}

        mock_post.return_value.json = AsyncMock(side_effect=mock_api_call)

        with pytest.raises(Exception) as exc_info:
            await classifier.classify("Sample text")

        assert "Invalid LLM response format" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_classification_empty_index(classifier):
    """Test validation with empty FAISS index."""
    text = "Sample contract text"

    mock_llm_response = {
        "response": json.dumps({
            "category": "Contract",
            "confidence": 0.85,
            "explanation": "This is a legal contract."
        })
    }

    mock_embedding_response = {
        "embedding": [0.1] * 384
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.status_code = 200

        async def mock_api_call(*args, **kwargs):
            if args and isinstance(args[0], str):
                url = args[0]
            else:
                url = kwargs.get("url", "")

            if "/api/generate" in str(url):
                return mock_llm_response
            if "/api/embeddings" in str(url):
                return mock_embedding_response
            # Handle direct json() calls from response objects
            if not url and hasattr(mock_post.return_value, '_mock_return_value'):
                # Get the last URL that was called
                last_url = mock_post.call_args[0][0] if mock_post.call_args else ""
                if "/api/generate" in str(last_url):
                    return mock_llm_response
                if "/api/embeddings" in str(last_url):
                    return mock_embedding_response
            raise ValueError(f"Unexpected API call: {url}")

        mock_post.return_value.json = AsyncMock(side_effect=mock_api_call)

        result = await classifier.classify(text)
        assert isinstance(result, ClassificationResult)
        assert result.validation_score is not None


@pytest.mark.asyncio
async def test_validate_classification_with_matches(classifier):
    """Test validation with matching examples in index."""
    # Add some example documents
    examples = ["Example contract 1", "Example contract 2"]

    mock_llm_response = {
        "response": json.dumps({
            "category": "Contract",
            "confidence": 0.85,
            "explanation": "This is a legal contract."
        })
    }

    mock_embedding_response = {
        "embedding": [0.1] * 384
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.status_code = 200

        async def mock_api_call(*args, **kwargs):
            if args and isinstance(args[0], str):
                url = args[0]
            else:
                url = kwargs.get("url", "")

            if "/api/generate" in str(url):
                return mock_llm_response
            if "/api/embeddings" in str(url):
                return mock_embedding_response
            # Handle direct json() calls from response objects
            if not url and hasattr(mock_post.return_value, '_mock_return_value'):
                # Get the last URL that was called
                last_url = mock_post.call_args[0][0] if mock_post.call_args else ""
                if "/api/generate" in str(last_url):
                    return mock_llm_response
                if "/api/embeddings" in str(last_url):
                    return mock_embedding_response
            raise ValueError(f"Unexpected API call: {url}")

        mock_post.return_value.json = AsyncMock(side_effect=mock_api_call)

        await classifier.add_category("Contract", examples)
        result = await classifier.classify("Test contract")

        assert isinstance(result, ClassificationResult)
        assert 0 <= result.validation_score <= 1  # Score should be normalized
        assert mock_post.call_count > 0


@pytest.mark.asyncio
async def test_add_category(classifier):
    """Test adding a new category with examples."""
    category = "Contract"
    examples = ["Example contract 1", "Example contract 2"]

    mock_embedding_response = {
        "embedding": [0.1] * 384
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.status_code = 200

        async def mock_api_call(*args, **kwargs):
            if args and isinstance(args[0], str):
                url = args[0]
            else:
                url = kwargs.get("url", "")

            if "/api/embeddings" in str(url) or not url:
                return mock_embedding_response
            raise ValueError(f"Unexpected API call: {url}")

        mock_post.return_value.json = AsyncMock(side_effect=mock_api_call)

        await classifier.add_category(category, examples)

        assert len(classifier.categories) == len(examples)
        assert all(cat == category for cat in classifier.categories)
        assert classifier.index.ntotal == len(examples)
        assert mock_post.call_count == len(examples)


@pytest.mark.asyncio
async def test_full_classification_pipeline(classifier):
    """Test the complete classification pipeline."""
    # Mock LLM response
    mock_llm_response = {
        "response": json.dumps({
            "category": "Contract",
            "confidence": 0.85,
            "explanation": "This is a legal contract."
        })
    }

    # Mock embedding response
    mock_embedding_response = {
        "embedding": [0.1] * 384
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.status_code = 200

        # Set up mock responses for different API calls
        async def mock_api_call(*args, **kwargs):
            if args and isinstance(args[0], str):
                url = args[0]
            else:
                url = kwargs.get("url", "")

            if "/api/generate" in str(url):
                return mock_llm_response
            if "/api/embeddings" in str(url):
                return mock_embedding_response
            # Handle direct json() calls from response objects
            if not url and hasattr(mock_post.return_value, '_mock_return_value'):
                # Get the last URL that was called
                last_url = mock_post.call_args[0][0] if mock_post.call_args else ""
                if "/api/generate" in str(last_url):
                    return mock_llm_response
                if "/api/embeddings" in str(last_url):
                    return mock_embedding_response
            raise ValueError(f"Unexpected API call: {url}")

        mock_post.return_value.json = AsyncMock(side_effect=mock_api_call)

        # Add some examples first
        await classifier.add_category("Contract", ["Example contract"])

        # Test full classification
        result = await classifier.classify("Test contract document")

        assert isinstance(result, ClassificationResult)
        assert result.category == "Contract"
        assert result.confidence == 0.85
        assert result.validation_score is not None
        assert 0 <= result.validation_score <= 1


@pytest.mark.asyncio
async def test_error_handling(classifier):
    """Test error handling in classification pipeline."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.status_code = 500
        mock_post.return_value.raise_for_status.side_effect = httpx.HTTPError(
            "API error")
        mock_post.return_value.json = AsyncMock(
            side_effect=httpx.HTTPError("API error"))

        with pytest.raises(Exception) as exc_info:
            await classifier.classify("Test document")

        assert "Failed to get classification: API error" in str(exc_info.value)
