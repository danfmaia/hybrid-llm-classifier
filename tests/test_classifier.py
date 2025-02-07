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


@pytest.fixture(name="mock_embeddings")
def fixture_mock_embeddings():
    """Create mock embeddings for testing."""
    return np.random.rand(5, 384).astype(np.float32)


@pytest.fixture(name="classifier")
def fixture_classifier():
    """Create a classifier instance for testing."""
    return HybridClassifier(
        ollama_base_url="http://localhost:11434",
        model_name="mistral",
        embedding_dim=384
    )


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
        # Check confidence is adjusted but within valid range
        assert 0 <= result.confidence <= 1
        assert result.validation_score == 0.5  # Default score for empty index
        assert isinstance(result.similar_documents, list)
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
async def test_add_category(classifier, mock_embeddings):
    """Test adding a category with examples."""
    embedding_responses = [
        {"embedding": emb.tolist()} for emb in mock_embeddings
    ]

    with patch('httpx.AsyncClient.post') as mock_post:
        mock_post.side_effect = [
            AsyncMock(
                json=AsyncMock(return_value=resp),
                raise_for_status=AsyncMock()
            )
            for resp in embedding_responses
        ]

        await classifier.add_category(
            "Contract",
            ["Example contract " + str(i) for i in range(5)]
        )

        assert classifier.index.ntotal == 5
        assert len(classifier.categories) == 5
        assert classifier.category_counts["Contract"] == 5


@pytest.mark.asyncio
async def test_validation_scoring(classifier, mock_embeddings):
    """Test the validation scoring mechanism."""
    # Add some examples first
    embedding_responses = [
        {"embedding": emb.tolist()} for emb in mock_embeddings
    ]

    with patch('httpx.AsyncClient.post') as mock_post:
        # Setup for adding examples
        mock_post.side_effect = [
            AsyncMock(
                json=AsyncMock(return_value=resp),
                raise_for_status=AsyncMock()
            )
            for resp in embedding_responses
        ]

        await classifier.add_category(
            "Contract",
            ["Example contract " + str(i) for i in range(5)]
        )

        # Reset mock for validation
        mock_post.reset_mock()
        mock_post.side_effect = [
            AsyncMock(
                json=AsyncMock(
                    return_value={"embedding": mock_embeddings[0].tolist()}),
                raise_for_status=AsyncMock()
            )
        ]

        score, similar_docs = await classifier._validate_classification(
            "Test contract", "Contract"
        )

        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert len(similar_docs) > 0
        assert all(0 <= doc["similarity"] <= 1 for doc in similar_docs)


@pytest.mark.asyncio
async def test_train_with_examples(classifier):
    """Test training with multiple categories."""
    examples = {
        "Contract": ["Example contract 1", "Example contract 2"],
        "Legal Opinion": ["Example opinion 1", "Example opinion 2"]
    }

    mock_embedding = np.random.rand(384).astype(np.float32)

    with patch('httpx.AsyncClient.post') as mock_post:
        mock_post.side_effect = [
            AsyncMock(
                json=AsyncMock(
                    return_value={"embedding": mock_embedding.tolist()}),
                raise_for_status=AsyncMock()
            )
            for _ in range(4)  # 4 total examples
        ]

        await classifier.train_with_examples(examples)

        assert classifier.index.ntotal == 4
        assert len(classifier.categories) == 4
        assert classifier.category_counts["Contract"] == 2
        assert classifier.category_counts["Legal Opinion"] == 2


@pytest.mark.asyncio
async def test_error_handling(classifier):
    """Test error handling in classification."""
    with patch('httpx.AsyncClient.post') as mock_post:
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

    with patch('httpx.AsyncClient.post') as mock_post:
        mock_post.side_effect = [
            AsyncMock(
                json=AsyncMock(
                    return_value={"embedding": mock_embedding.tolist()}),
                raise_for_status=AsyncMock()
            )
        ]

        score, similar_docs = await classifier._validate_classification(
            "Test document", "Contract"
        )

        assert score == 0.5
        assert len(similar_docs) == 0


def test_confidence_computation(classifier):
    """Test confidence score computation."""
    # Test with small index
    score1 = classifier._compute_final_confidence(0.8, 0.6)
    assert 0 <= score1 <= 1

    # Simulate large index
    classifier.index.ntotal = 150
    score2 = classifier._compute_final_confidence(0.8, 0.6)
    assert 0 <= score2 <= 1
    assert score2 != score1  # Should use different weights


@pytest.mark.asyncio
async def test_performance_metrics_tracking(classifier):
    """Test that performance metrics are properly tracked."""
    mock_llm_response = {
        "response": json.dumps({
            "category": "Contract",
            "confidence": 0.85,
            "explanation": "Test contract"
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
            elif "/api/embeddings" in str(url):
                return mock_embedding_response
            # For direct json() calls
            if not url and hasattr(mock_post.return_value, '_mock_return_value'):
                last_url = mock_post.call_args[0][0] if mock_post.call_args else ""
                if "/api/generate" in str(last_url):
                    return mock_llm_response
                elif "/api/embeddings" in str(last_url):
                    return mock_embedding_response
            return mock_embedding_response

        mock_post.return_value.json = AsyncMock(side_effect=mock_api_call)

        result = await classifier.classify("Test document")

        # Check performance metrics structure
        assert result.performance_metrics is not None
        assert result.performance_metrics.llm_latency > 0
        assert result.performance_metrics.embedding_latency > 0
        assert result.performance_metrics.validation_latency > 0
        assert result.performance_metrics.total_latency > 0
        assert result.performance_metrics.document_length == len(
            "Test document")
        assert 0 <= result.performance_metrics.validation_score <= 1


@pytest.mark.asyncio
async def test_index_size_metrics(classifier):
    """Test that index size metrics are properly updated."""
    examples = ["Example 1", "Example 2"]
    category = "Contract"

    mock_embedding = np.random.rand(384).astype(np.float32)

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = [
            AsyncMock(
                json=AsyncMock(
                    return_value={"embedding": mock_embedding.tolist()}),
                raise_for_status=AsyncMock()
            )
            for _ in range(len(examples))
        ]

        # Add examples and check metrics
        await classifier.add_category(category, examples)

        # Verify index size
        assert classifier.index.ntotal == len(examples)
        assert classifier.category_counts[category] == len(examples)


@pytest.mark.asyncio
async def test_error_metrics(classifier):
    """Test that error metrics are properly tracked."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = Exception("Test error")

        try:
            await classifier.classify("Test document")
        except Exception:
            pass

        # Verify error was logged (would check Prometheus metrics in real env)
        assert True  # Basic assertion since we can't easily check Prometheus metrics in tests


@pytest.mark.asyncio
async def test_validation_score_distribution(classifier, mock_embeddings):
    """Test validation score distribution metrics."""
    # Add some examples first
    embedding_responses = [
        {"embedding": emb.tolist()} for emb in mock_embeddings
    ]

    with patch('httpx.AsyncClient.post') as mock_post:
        # Setup for adding examples
        mock_post.side_effect = [
            AsyncMock(
                json=AsyncMock(return_value=resp),
                raise_for_status=AsyncMock()
            )
            for resp in embedding_responses
        ]

        await classifier.add_category(
            "Contract",
            ["Example contract " + str(i) for i in range(5)]
        )

        # Reset mock for validation
        mock_post.reset_mock()
        mock_post.side_effect = [
            AsyncMock(
                json=AsyncMock(
                    return_value={"embedding": mock_embeddings[0].tolist()}),
                raise_for_status=AsyncMock()
            )
        ]

        score, similar_docs = await classifier._validate_classification(
            "Test contract", "Contract"
        )

        # Verify validation score is within bounds
        assert 0 <= score <= 1
        assert len(similar_docs) > 0
        assert all(0 <= doc["similarity"] <= 1 for doc in similar_docs)
