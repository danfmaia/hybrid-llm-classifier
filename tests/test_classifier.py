"""Unit tests for the HybridClassifier."""

# Standard library imports
import json
from unittest.mock import patch, AsyncMock, MagicMock

# Third-party imports
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
import faiss

# First-party imports
from app.models.classifier import HybridClassifier, ClassificationResult, ValidationError
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


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    return np.random.rand(5, 384).astype(np.float32)


@pytest.fixture
def classifier():
    """Create classifier instance for testing."""
    return HybridClassifier(ollama_base_url="http://localhost:11434")


@pytest.mark.asyncio
async def test_classifier_initialization(classifier):
    """Test classifier initialization with default parameters."""
    assert classifier.ollama_base_url == "http://localhost:11434"
    assert classifier.model_name == "mistral"
    assert classifier.embedding_dim == 384


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
    mock_response = {
        "response": '{"category": "Contract", "confidence": 0.85}'
    }

    async def mock_post(*args, **kwargs):
        mock = AsyncMock()
        mock.status_code = 200
        mock.json = AsyncMock(return_value=mock_response)
        return mock

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        result = await classifier._get_llm_classification("Test document")
        assert result.category == "Contract"
        assert result.confidence == 0.85


@pytest.mark.asyncio
async def test_llm_classification_invalid_response(classifier):
    """Test handling of invalid LLM response."""
    mock_response = {
        "response": "Invalid response format"
    }

    async def mock_post(*args, **kwargs):
        mock = AsyncMock()
        mock.status_code = 200
        mock.json = AsyncMock(return_value=mock_response)
        return mock

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        with pytest.raises(Exception) as exc_info:
            await classifier._get_llm_classification("Test document")
        assert "Invalid LLM response format" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_classification_empty_index(classifier):
    """Test validation with empty index."""
    mock_embedding = np.random.rand(384).astype(np.float32)
    mock_response = {"embedding": mock_embedding.tolist()}

    async def mock_post(*args, **kwargs):
        mock = AsyncMock()
        mock.status_code = 200
        mock.json = AsyncMock(return_value=mock_response)
        return mock

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        score, similar_docs = await classifier._validate_classification(
            "Test document", "Contract"
        )
        assert score == 0.5
        assert len(similar_docs) == 0


@pytest.mark.asyncio
async def test_validate_classification_with_matches(classifier):
    """Test validation with matching documents."""
    examples = ["Example contract " + str(i) for i in range(3)]
    mock_embedding = np.random.rand(384).astype(np.float32)
    mock_response = {"embedding": mock_embedding.tolist()}

    async def mock_post(*args, **kwargs):
        mock = AsyncMock()
        mock.status_code = 200
        mock.json = AsyncMock(return_value=mock_response)
        return mock

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        await classifier.add_category("Contract", examples)
        score, similar_docs = await classifier._validate_classification(
            "Test document", "Contract"
        )
        assert 0 <= score <= 1
        assert len(similar_docs) > 0


@pytest.mark.asyncio
async def test_add_category(classifier, mock_embeddings):
    """Test adding a category with examples."""
    examples = ["Example contract " + str(i) for i in range(5)]
    mock_response = {"embedding": mock_embeddings[0].tolist()}

    async def mock_post(*args, **kwargs):
        mock = AsyncMock()
        mock.status_code = 200
        mock.json = AsyncMock(return_value=mock_response)
        return mock

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        await classifier.add_category("Contract", examples)
        assert classifier.index.ntotal == len(examples)
        assert classifier.category_counts["Contract"] == len(examples)


@pytest.mark.asyncio
async def test_train_with_examples(classifier):
    """Test training with multiple categories."""
    examples = {
        "Contract": ["Example contract 1", "Example contract 2"],
        "Legal Opinion": ["Example opinion 1", "Example opinion 2"]
    }
    mock_embedding = np.random.rand(384).astype(np.float32)
    mock_response = {"embedding": mock_embedding.tolist()}

    async def mock_post(*args, **kwargs):
        mock = AsyncMock()
        mock.status_code = 200
        mock.json = AsyncMock(return_value=mock_response)
        return mock

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        await classifier.train_with_examples(examples)
        assert classifier.index.ntotal == 4
        assert classifier.category_counts["Contract"] == 2
        assert classifier.category_counts["Legal Opinion"] == 2


@pytest.mark.asyncio
async def test_error_handling(classifier):
    """Test error handling in classification."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = Exception("API Error")
        with pytest.raises(Exception):
            await classifier.classify("Test document")


@pytest.mark.asyncio
async def test_invalid_category(classifier):
    """Test handling of invalid categories."""
    examples = {
        "Invalid Category": ["Test document"]
    }
    with pytest.raises(ValueError):
        await classifier.train_with_examples(examples)


@pytest.mark.asyncio
async def test_empty_index_validation(classifier):
    """Test validation behavior with empty index."""
    mock_embedding = np.random.rand(384).astype(np.float32)
    mock_response = {"embedding": mock_embedding.tolist()}

    async def mock_post(*args, **kwargs):
        mock = AsyncMock()
        mock.status_code = 200
        mock.json = AsyncMock(return_value=mock_response)
        return mock

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        score, similar_docs = await classifier._validate_classification(
            "Test document", "Contract"
        )
        assert score == 0.5
        assert len(similar_docs) == 0


@pytest.mark.asyncio
async def test_confidence_computation(classifier):
    """Test confidence score computation."""
    # Add some documents to train the classifier
    examples = {
        "Contract": ["Example contract 1", "Example contract 2"],
        "Legal Opinion": ["Example opinion 1", "Example opinion 2"]
    }
    mock_embedding = np.random.rand(384).astype(np.float32)
    mock_response = {"embedding": mock_embedding.tolist()}

    async def mock_post(*args, **kwargs):
        mock = AsyncMock()
        mock.status_code = 200
        mock.json = AsyncMock(return_value=mock_response)
        return mock

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        await classifier.train_with_examples(examples)

        # Test classification with validation
        mock_llm_response = {
            "response": '{"category": "Contract", "confidence": 0.8}'
        }

        async def mock_post_classify(*args, **kwargs):
            mock = AsyncMock()
            mock.status_code = 200
            if "/api/generate" in args[0]:
                mock.json = AsyncMock(return_value=mock_llm_response)
            else:
                mock.json = AsyncMock(return_value=mock_response)
            return mock

        with patch("httpx.AsyncClient.post", side_effect=mock_post_classify):
            result = await classifier.classify("Test document")
            assert 0 <= result.confidence <= 1
            assert result.validation_score is not None


@pytest.mark.asyncio
async def test_performance_metrics_tracking(classifier):
    """Test that performance metrics are properly tracked."""
    mock_llm_response = {
        "response": '{"category": "Contract", "confidence": 0.85}'
    }
    mock_embedding = np.random.rand(384).astype(np.float32)
    mock_embedding_response = {"embedding": mock_embedding.tolist()}

    async def mock_post(*args, **kwargs):
        mock = AsyncMock()
        mock.status_code = 200
        if "/api/generate" in args[0]:
            mock.json = AsyncMock(return_value=mock_llm_response)
        else:
            mock.json = AsyncMock(return_value=mock_embedding_response)
        return mock

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        result = await classifier.classify("Test document")
        assert result.category == "Contract"
        assert result.confidence == 0.85
        assert result.validation_score == 0.5


@pytest.mark.asyncio
async def test_index_size_metrics(classifier):
    """Test that index size metrics are properly updated."""
    examples = ["Example 1", "Example 2"]
    mock_embedding = np.random.rand(384).astype(np.float32)
    mock_response = {"embedding": mock_embedding.tolist()}

    async def mock_post(*args, **kwargs):
        mock = AsyncMock()
        mock.status_code = 200
        mock.json = AsyncMock(return_value=mock_response)
        return mock

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        await classifier.add_category("Contract", examples)
        assert classifier.index.ntotal == len(examples)
        assert classifier.category_counts["Contract"] == len(examples)


@pytest.mark.asyncio
async def test_validation_score_distribution(classifier, mock_embeddings):
    """Test validation score distribution metrics."""
    examples = ["Example contract " + str(i) for i in range(5)]
    mock_response = {"embedding": mock_embeddings[0].tolist()}

    async def mock_post(*args, **kwargs):
        mock = AsyncMock()
        mock.status_code = 200
        mock.json = AsyncMock(return_value=mock_response)
        return mock

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        await classifier.add_category("Contract", examples)
        assert classifier.index.ntotal == len(examples)
        assert classifier.category_counts["Contract"] == len(examples)
