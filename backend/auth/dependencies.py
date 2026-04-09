"""
MemoryOS — FastAPI Auth Dependencies
======================================
Provides reusable Depends() callables for route protection.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from backend.auth.auth_manager import decode_access_token

_bearer = HTTPBearer(auto_error=False)

_401 = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token.",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """
    Extract and validate the Bearer JWT from the Authorization header.
    Returns the decoded payload dict: {"sub": user_id, "email": ...}
    Raises 401 if missing, malformed, or expired.
    """
    if not credentials:
        raise _401
    try:
        payload = decode_access_token(credentials.credentials)
    except JWTError:
        raise _401
    return payload
