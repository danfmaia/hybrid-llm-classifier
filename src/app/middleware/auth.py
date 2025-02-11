"""Authentication middleware for FastAPI application."""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from ..utils.auth import decode_token

# Paths that don't require authentication
EXCLUDED_PATHS = {
    "/api/v1/auth/token",
    "/docs",
    "/openapi.json",
    "/health"
}


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT authentication."""

    async def dispatch(self, request: Request, call_next):
        """Process each request."""
        # Skip authentication for excluded paths
        if request.url.path in EXCLUDED_PATHS:
            return await call_next(request)

        # Get token from header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "No authentication token provided"}
            )

        try:
            # Extract token
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer":
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid authentication scheme"}
                )

            # Validate token using utility function
            payload = decode_token(token)
            if not payload:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid authentication token"}
                )

            # Add user info to request state
            request.state.user = payload
            return await call_next(request)

        except Exception as e:
            return JSONResponse(
                status_code=401,
                content={"detail": str(e)}
            )
