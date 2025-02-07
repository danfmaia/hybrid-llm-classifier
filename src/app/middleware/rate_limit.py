"""Rate limiting middleware for API request throttling."""

import time
from collections import defaultdict
from typing import Callable, Dict, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from ..config import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to prevent API abuse.

    Uses a rolling window approach to track requests per client IP.
    Configurable through settings for requests per window and window size.
    """

    # Endpoints that should not be rate limited
    EXCLUDED_PATHS = {
        "/api/v1/auth/token",  # Token endpoint
        "/docs",  # API documentation
        "/openapi.json",  # OpenAPI schema
        "/health",  # Health check endpoint
    }

    def __init__(self, app: ASGIApp):
        """Initialize the rate limiter with default settings."""
        super().__init__(app)
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.settings = get_settings()
        self._test_mode = False
        self._test_counter = 0

    @property
    def test_mode(self) -> bool:
        """Get test mode status."""
        return self._test_mode

    @test_mode.setter
    def test_mode(self, value: bool) -> None:
        """Set test mode status."""
        self._test_mode = value
        if value:
            self.reset()

    def reset(self) -> None:
        """Reset the rate limiter state."""
        self.requests.clear()
        if self._test_mode:
            self._test_counter += 1

    def should_rate_limit(self, request: Request) -> bool:
        """
        Determine if a request should be rate limited.

        Args:
            request: The incoming request

        Returns:
            bool: True if the request should be rate limited
        """
        path = request.url.path.rstrip(
            "/")  # Remove trailing slash for comparison
        return path not in self.EXCLUDED_PATHS

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process each request through the rate limiter.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/endpoint to call

        Returns:
            Response: The HTTP response
        """
        # Skip rate limiting for excluded paths
        if not self.should_rate_limit(request):
            return await call_next(request)

        # In test mode, use a unique client ID for each test case
        client_ip = f"test_client_{self._test_counter}" if self._test_mode else request.client.host

        # Clean up old requests outside the window
        now = time.time()
        window = self.settings.rate_limit_window_seconds
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < window
        ]

        # Check if rate limit is exceeded
        if len(self.requests[client_ip]) >= self.settings.rate_limit_requests:
            return Response(
                content='{"detail":"Rate limit exceeded"}',
                status_code=429,
                media_type="application/json"
            )

        # Add current request timestamp
        self.requests[client_ip].append(now)

        # Process the request
        response = await call_next(request)
        return response
