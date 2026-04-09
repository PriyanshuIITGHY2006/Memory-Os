"""
MemoryOS — Auth Manager
========================
Handles:
  - Password hashing / verification  (bcrypt via passlib)
  - JWT access token creation / decoding  (HS256 via python-jose)
  - Refresh token creation / hashing  (stored as SHA-256 hash in DB)
  - Groq API key encryption / decryption  (Fernet symmetric encryption)
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from cryptography.fernet import Fernet, InvalidToken
from jose import JWTError, jwt
from passlib.context import CryptContext

import config

# ── Password ──────────────────────────────────────────────────────────────────
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


# ── Fernet (API key encryption) ───────────────────────────────────────────────
def _fernet() -> Fernet:
    key = config.ENCRYPTION_KEY
    if not key:
        raise RuntimeError(
            "ENCRYPTION_KEY is not set. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_api_key(plain_key: str) -> str:
    """Encrypt a Groq API key for storage in the database."""
    return _fernet().encrypt(plain_key.encode()).decode()


def decrypt_api_key(enc_key: str) -> str:
    """Decrypt a stored Groq API key."""
    try:
        return _fernet().decrypt(enc_key.encode()).decode()
    except (InvalidToken, Exception) as exc:
        raise ValueError("Failed to decrypt API key — key may be corrupt.") from exc


# ── JWT access tokens ─────────────────────────────────────────────────────────
def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=config.JWT_ACCESS_EXPIRE_MINUTES
    )
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT access token.
    Raises jose.JWTError on invalid / expired tokens.
    """
    payload = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
    if payload.get("type") != "access":
        raise JWTError("Not an access token")
    return payload


# ── Refresh tokens ────────────────────────────────────────────────────────────
def generate_refresh_token() -> tuple[str, str]:
    """
    Returns (raw_token, hashed_token).
    Store only the hash in the database; send the raw token in the cookie.
    """
    raw = secrets.token_urlsafe(64)
    hashed = hashlib.sha256(raw.encode()).hexdigest()
    return raw, hashed


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()
