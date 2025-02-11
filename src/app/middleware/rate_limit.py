"""Rate limiting middleware for API request throttling."""

import time
from collections import defaultdict
from typing import Callable, Dict, List
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from ..config import get_settings

# Rate limit settings
REQUESTS_PER_MINUTE = 60
WINDOW_SIZE = 60  # seconds

# Paths that don't require rate limiting
EXCLUDED_PATHS = {
    "/docs",
    "/openapi.json",
    "/health"
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting using a rolling window approach."""

    def __init__(self, app=None):
        super().__init__(app)
        # Store timestamps of requests per client
        self.request_history: Dict[str, List[float]] = defaultdict(list)
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
        self.request_history.clear()
        if self._test_mode:
            self._test_counter += 1

    def should_rate_limit(self, request: Request) -> bool:
        """Check if request should be rate limited."""
        return request.url.path not in EXCLUDED_PATHS

    def clean_old_requests(self, client_id: str, current_time: float):
        """Remove requests older than the window size."""
        cutoff = current_time - WINDOW_SIZE
        while (self.request_history[client_id] and
               self.request_history[client_id][0] < cutoff):
            self.request_history[client_id].pop(0)

    async def dispatch(self, request: Request, call_next):
        """Process each request."""
        # Skip rate limiting for excluded paths
        if self.should_rate_limit(request):
            # Clean old requests
            self.clean_old_requests(request.client.host, time.time())

            # Get client identifier
            client_id = request.client.host

            # Get current time
            current_time = time.time()

            # Get requests in the current window
            client_requests = self.request_history.get(client_id, [])
            client_requests = [
                t for t in client_requests if current_time - t <= WINDOW_SIZE]

            # Check if rate limit is exceeded
            if len(client_requests) >= REQUESTS_PER_MINUTE:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests"}
                )

            # Add current request
            client_requests.append(current_time)
            self.request_history[client_id] = client_requests

        return await call_next(request)
