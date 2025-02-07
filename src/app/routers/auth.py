"""Authentication router for JWT token management."""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from ..auth.jwt import Token, create_access_token
from ..config import get_settings

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Token:
    """
    Get JWT access token for authentication.

    In a production environment, this would validate against a user database.
    For development, we accept any username/password combination.

    Args:
        form_data: OAuth2 password request form

    Returns:
        Token object containing JWT access token

    Raises:
        HTTPException: If authentication fails
    """
    # For development, accept any username/password
    # In production, validate against user database
    if not form_data.username or not form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    settings = get_settings()
    access_token_expires = timedelta(
        minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": form_data.username},
        expires_delta=access_token_expires
    )

    return Token(access_token=access_token, token_type="bearer")
