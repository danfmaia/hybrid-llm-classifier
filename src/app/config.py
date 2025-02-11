from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from .models.classifier import HybridClassifier


class Settings(BaseSettings):
    """Application settings managed by pydantic."""

    # API Settings
    api_title: str = "Legal Document Classifier"
    api_description: str = "Zero-shot legal document classification using Mistral-7B and FAISS"
    api_version: str = "1.0.0"

    # Security
    jwt_secret_key: str  # Loaded from environment variable JWT_SECRET_KEY
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Classifier Settings
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "mistral:7b"  # Using Mistral-7B model
    embedding_dim: int = 384
    similarity_threshold: float = 0.75

    # Ollama Model Parameters
    ollama_num_ctx: int = 2048
    ollama_num_gpu: int = 1
    ollama_num_thread: int = 4

    # Rate Limiting
    rate_limit_requests: int = 1000
    rate_limit_window_seconds: int = 60

    model_config = {
        "env_file": ".env"
    }


_classifier_instance = None


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()


def get_classifier() -> HybridClassifier:
    """Get singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        settings = get_settings()
        _classifier_instance = HybridClassifier(
            ollama_base_url=settings.ollama_base_url,
            model_name=settings.model_name,
            embedding_dim=settings.embedding_dim,
            similarity_threshold=settings.similarity_threshold
        )
    return _classifier_instance
