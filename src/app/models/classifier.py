"""
Module for hybrid legal document classification using LLM and embedding-based validation.

This module implements a hybrid classification system that combines Mistral-7B's
zero-shot capabilities with FAISS-based embedding validation for accurate and
efficient legal document classification.
"""

from typing import List, Dict, Optional
import logging
import json
import httpx
from pydantic import BaseModel
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class ClassificationResult(BaseModel):
    """Pydantic model for classification results."""
    category: str
    confidence: float
    subcategories: Optional[List[Dict[str, float]]] = None
    validation_score: Optional[float] = None


class HybridClassifier:
    """
    Hybrid legal document classifier using Mistral-7B and FAISS validation.

    This classifier combines zero-shot classification from Mistral-7B with
    embedding-based validation using FAISS to ensure accurate and consistent
    legal document classification.
    """

    # Legal document categories for zero-shot classification
    LEGAL_CATEGORIES = [
        "Contract",
        "Court Filing",
        "Legal Opinion",
        "Legislation",
        "Regulatory Document"
    ]

    def __init__(
        self,
        ollama_base_url: str,
        model_name: str = "mistral",
        embedding_dim: int = 384,
        similarity_threshold: float = 0.75
    ):
        """
        Initialize the hybrid classifier.

        Args:
            ollama_base_url: Base URL for Ollama API
            model_name: Name of the model to use
            embedding_dim: Dimension of embeddings
            similarity_threshold: Threshold for similarity validation
        """
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.categories: List[str] = []

    async def classify(self, text: str) -> ClassificationResult:
        """
        Classify a legal document using hybrid approach.

        Args:
            text: The document text to classify

        Returns:
            ClassificationResult with category and confidence scores

        Raises:
            Exception: If classification fails
        """
        try:
            # Get initial classification from LLM
            llm_result = await self._get_llm_classification(text)

            # Validate using embeddings
            validation_score = await self._validate_classification(
                text, llm_result.category)

            # Update confidence based on validation
            llm_result.validation_score = validation_score

            return llm_result

        except Exception as e:
            logger.error("Classification failed: %s", str(e))
            raise

    async def _get_llm_classification(self, text: str) -> ClassificationResult:
        """
        Get classification from Mistral-7B using zero-shot learning.

        Args:
            text: The legal document text to classify

        Returns:
            ClassificationResult with category and confidence scores

        Raises:
            Exception: If classification fails
        """
        # Construct zero-shot prompt
        prompt = f"""You are a legal document classifier. Classify the following document into exactly one of these categories: {', '.join(self.LEGAL_CATEGORIES)}.

For the chosen category, provide a confidence score between 0 and 1.

Document text:
{text}

Respond in JSON format:
{{
    "category": "chosen_category",
    "confidence": confidence_score,
    "explanation": "brief_explanation"
}}"""

        try:
            # Make API call to Ollama
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30.0
                )
                response.raise_for_status()

                # Parse response
                result = await response.json()
                if "error" in result:
                    raise Exception(f"Ollama API error: {result['error']}")

                # Parse LLM response from JSON string
                try:
                    llm_response = json.loads(result["response"])
                except (json.JSONDecodeError, KeyError) as e:
                    raise Exception("Invalid LLM response format") from e

                return ClassificationResult(
                    category=llm_response["category"],
                    confidence=llm_response["confidence"],
                    validation_score=None
                )

        except httpx.HTTPError as e:
            logger.error("HTTP error during classification: %s", str(e))
            raise Exception(f"Failed to get classification: {str(e)}") from e

    async def _validate_classification(self, text: str, category: str) -> float:
        """
        Validate classification using FAISS similarity search.

        Args:
            text: The document text to validate
            category: The category to validate against

        Returns:
            float: Validation score between 0 and 1

        Raises:
            Exception: If validation fails
        """
        try:
            # Get embeddings for the input text
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text
                    },
                    timeout=30.0
                )
                response.raise_for_status()

                result = await response.json()
                if "error" in result:
                    raise Exception(f"Ollama API error: {result['error']}")

                embedding = np.array([result["embedding"]], dtype=np.float32)

            # If index is empty, return default score
            if self.index.ntotal == 0:
                return 0.5

            # Search for similar documents
            num_neighbors = min(5, self.index.ntotal)
            distances, indices = self.index.search(embedding, num_neighbors)

            # Convert L2 distances to similarity scores (0 to 1)
            similarities = 1 / (1 + distances)

            # Calculate validation score based on category matches
            category_matches = sum(1 for i in indices[0]
                                   if self.categories[i] == category)
            validation_score = (
                similarities[0].mean() * category_matches / num_neighbors)

            return float(validation_score)

        except httpx.HTTPError as e:
            logger.error("HTTP error during validation: %s", str(e))
            raise Exception(
                f"Failed to validate classification: {str(e)}") from e

    async def add_category(self, category: str, examples: List[str]) -> None:
        """
        Add a new category with example documents to the FAISS index.

        Args:
            category: Name of the legal document category
            examples: List of example documents for this category

        Raises:
            Exception: If adding category fails
        """
        try:
            # Get embeddings for all examples
            async with httpx.AsyncClient() as client:
                embeddings = []
                for example in examples:
                    response = await client.post(
                        f"{self.ollama_base_url}/api/embeddings",
                        json={
                            "model": self.model_name,
                            "prompt": example
                        },
                        timeout=30.0
                    )
                    response.raise_for_status()

                    result = await response.json()
                    if "error" in result:
                        raise Exception(f"Ollama API error: {result['error']}")

                    embeddings.append(result["embedding"])

            # Convert to numpy array and add to index
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)

            # Store category labels
            self.categories.extend([category] * len(examples))

            logger.info(
                "Added %d examples for category '%s'",
                len(examples),
                category
            )

        except httpx.HTTPError as e:
            logger.error("HTTP error while adding category: %s", str(e))
            raise Exception(f"Failed to get embeddings: {str(e)}") from e
        except Exception as e:
            logger.error("Failed to add category: %s", str(e))
            raise
