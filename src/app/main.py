"""
FastAPI application entry point for the Legal Document Classifier.

This module sets up the FastAPI application with CORS middleware,
authentication, and classification endpoints. It also configures
logging and global exception handling.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .routers import auth, classifier
from .middleware import RateLimitMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal Document Classifier",
    description="Zero-shot legal document classification using Mistral-7B and FAISS",
    version="1.0.0"
)

# Include routers first to ensure authentication is set up
app.include_router(auth.router)
app.include_router(classifier.router)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware (after authentication)
app.add_middleware(RateLimitMiddleware)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for all unhandled exceptions.

    Args:
        request: FastAPI request that caused the exception
        exc: The unhandled exception

    Returns:
        JSONResponse with 500 status code and error details
    """
    # Log the request method and URL along with the exception
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        str(exc),
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
