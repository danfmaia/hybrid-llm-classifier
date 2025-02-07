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
    jwt_secret_key: str = "your-secret-key-here"  # Change in production
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Classifier Settings
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "mistral"
    embedding_dim: int = 384
    similarity_threshold: float = 0.75

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    class Config:
        env_file = ".env"

    def get_classifier(self) -> HybridClassifier:
        """Get configured classifier instance."""
        return HybridClassifier(
            ollama_base_url=self.ollama_base_url,
            model_name=self.model_name,
            embedding_dim=self.embedding_dim,
            similarity_threshold=self.similarity_threshold
        )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
