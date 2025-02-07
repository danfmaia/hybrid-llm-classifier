"""Router module for the FastAPI application."""

from . import auth
from . import classifier

__all__ = ["auth", "classifier"]
