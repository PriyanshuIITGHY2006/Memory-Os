"""
MemoryOS — User Manager
========================
SQLite-backed account store.  Auth data is intentionally kept separate
from the Neo4j memory graph so the two systems can be scaled independently.

Schema
------
users
  id              TEXT  PRIMARY KEY           — UUID4
  email           TEXT  UNIQUE NOT NULL
  username        TEXT  UNIQUE NOT NULL
  password_hash   TEXT                        — NULL for OAuth-only accounts
  groq_api_key_enc TEXT                       — Fernet-encrypted; NULL until set
  auth_provider   TEXT  DEFAULT 'email'       — 'email' | 'google' | 'github'
  provider_id     TEXT                        — OAuth provider's user ID
  avatar_url      TEXT
  created_at      TEXT  NOT NULL
  last_login      TEXT
  is_active       INTEGER DEFAULT 1

refresh_tokens
  id              TEXT  PRIMARY KEY           — UUID4
  user_id         TEXT  NOT NULL → users(id) ON DELETE CASCADE
  token_hash      TEXT  UNIQUE NOT NULL       — SHA-256 of the raw token
  expires_at      TEXT  NOT NULL
  created_at      TEXT  NOT NULL
"""

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import config
from backend.auth.auth_manager import (
    decrypt_api_key,
    encrypt_api_key,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class User:
    id: str
    email: str
    username: str
    password_hash: Optional[str]
    groq_api_key_enc: Optional[str]
    auth_provider: str
    provider_id: Optional[str]
    avatar_url: Optional[str]
    created_at: str
    last_login: Optional[str]
    is_active: int

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key_enc)

    def public_dict(self) -> dict:
        """Safe representation — never exposes hashes or encrypted keys."""
        return {
            "id":           self.id,
            "email":        self.email,
            "username":     self.username,
            "avatar_url":   self.avatar_url or "",
            "auth_provider":self.auth_provider,
            "has_groq_key": self.has_groq_key,
            "created_at":   self.created_at,
            "last_login":   self.last_login or "",
        }


# ── Manager ───────────────────────────────────────────────────────────────────

class UserManager:

    def __init__(self):
        config.DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        self._db_path = str(config.USERS_DB_PATH)
        self._init_schema()

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self._db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_schema(self):
        with self._conn() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id               TEXT PRIMARY KEY,
                    email            TEXT UNIQUE NOT NULL,
                    username         TEXT UNIQUE NOT NULL,
                    password_hash    TEXT,
                    groq_api_key_enc TEXT,
                    auth_provider    TEXT NOT NULL DEFAULT 'email',
                    provider_id      TEXT,
                    avatar_url       TEXT,
                    created_at       TEXT NOT NULL,
                    last_login       TEXT,
                    is_active        INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash  TEXT UNIQUE NOT NULL,
                    expires_at  TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_rt_user   ON refresh_tokens(user_id);
                CREATE INDEX IF NOT EXISTS idx_rt_hash   ON refresh_tokens(token_hash);
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
            """)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _row_to_user(self, row: sqlite3.Row) -> User:
        return User(**dict(row))

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Create ────────────────────────────────────────────────────────────────

    def create_email_user(
        self,
        email: str,
        username: str,
        password: str,
        groq_api_key: str = "",
    ) -> User:
        """Register a new user with email + password."""
        uid = str(uuid.uuid4())
        pw_hash = hash_password(password)
        key_enc = encrypt_api_key(groq_api_key) if groq_api_key else None
        now = self._now()

        with self._conn() as con:
            con.execute(
                """INSERT INTO users
                   (id, email, username, password_hash, groq_api_key_enc,
                    auth_provider, created_at, is_active)
                   VALUES (?,?,?,?,?,'email',?,1)""",
                (uid, email.lower().strip(), username.strip(),
                 pw_hash, key_enc, now),
            )
        return self.get_by_id(uid)

    def get_or_create_oauth_user(
        self,
        provider: str,
        provider_id: str,
        email: str,
        name: str,
        avatar_url: str,
    ) -> tuple[User, bool]:
        """
        Find or create a user from an OAuth callback.
        Returns (user, created: bool).
        """
        # Try by provider + provider_id first
        existing = self.get_by_provider(provider, provider_id)
        if existing:
            self._touch_login(existing.id)
            return self.get_by_id(existing.id), False

        # Fallback: email already registered (link accounts)
        existing_email = self.get_by_email(email)
        if existing_email:
            # Link the OAuth provider to the existing account
            with self._conn() as con:
                con.execute(
                    "UPDATE users SET auth_provider=?, provider_id=?, avatar_url=? WHERE id=?",
                    (provider, provider_id, avatar_url, existing_email.id),
                )
            self._touch_login(existing_email.id)
            return self.get_by_id(existing_email.id), False

        # Create new account
        uid = str(uuid.uuid4())
        username = self._unique_username(name or email.split("@")[0])
        now = self._now()

        with self._conn() as con:
            con.execute(
                """INSERT INTO users
                   (id, email, username, auth_provider, provider_id,
                    avatar_url, created_at, is_active)
                   VALUES (?,?,?,?,?,?,?,1)""",
                (uid, email.lower().strip(), username,
                 provider, provider_id, avatar_url, now),
            )
        return self.get_by_id(uid), True

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_by_id(self, user_id: str) -> Optional[User]:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM users WHERE id=? AND is_active=1", (user_id,)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get_by_email(self, email: str) -> Optional[User]:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM users WHERE email=? AND is_active=1",
                (email.lower().strip(),),
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get_by_provider(self, provider: str, provider_id: str) -> Optional[User]:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM users WHERE auth_provider=? AND provider_id=? AND is_active=1",
                (provider, provider_id),
            ).fetchone()
        return self._row_to_user(row) if row else None

    # ── Auth ──────────────────────────────────────────────────────────────────

    def authenticate(self, email: str, password: str) -> Optional[User]:
        """Return user if credentials are valid, else None."""
        user = self.get_by_email(email)
        if not user or not user.password_hash:
            return None
        if not verify_password(password, user.password_hash):
            return None
        self._touch_login(user.id)
        return self.get_by_id(user.id)

    def _touch_login(self, user_id: str):
        with self._conn() as con:
            con.execute(
                "UPDATE users SET last_login=? WHERE id=?",
                (self._now(), user_id),
            )

    # ── Refresh tokens ────────────────────────────────────────────────────────

    def create_refresh_token(self, user_id: str) -> str:
        """Create a refresh token, store its hash, return the raw token."""
        raw, hashed = generate_refresh_token()
        expires = (
            datetime.now(timezone.utc)
            + timedelta(days=config.JWT_REFRESH_EXPIRE_DAYS)
        ).isoformat()
        with self._conn() as con:
            con.execute(
                """INSERT INTO refresh_tokens (id, user_id, token_hash, expires_at, created_at)
                   VALUES (?,?,?,?,?)""",
                (str(uuid.uuid4()), user_id, hashed, expires, self._now()),
            )
        return raw

    def validate_refresh_token(self, raw_token: str) -> Optional[User]:
        """Validate a refresh token and return the associated user."""
        hashed = hash_refresh_token(raw_token)
        now = self._now()
        with self._conn() as con:
            row = con.execute(
                """SELECT user_id FROM refresh_tokens
                   WHERE token_hash=? AND expires_at > ?""",
                (hashed, now),
            ).fetchone()
        if not row:
            return None
        return self.get_by_id(row["user_id"])

    def revoke_refresh_token(self, raw_token: str):
        hashed = hash_refresh_token(raw_token)
        with self._conn() as con:
            con.execute(
                "DELETE FROM refresh_tokens WHERE token_hash=?", (hashed,)
            )

    def revoke_all_refresh_tokens(self, user_id: str):
        with self._conn() as con:
            con.execute(
                "DELETE FROM refresh_tokens WHERE user_id=?", (user_id,)
            )

    # ── Update ────────────────────────────────────────────────────────────────

    def update_groq_key(self, user_id: str, plain_key: str):
        enc = encrypt_api_key(plain_key)
        with self._conn() as con:
            con.execute(
                "UPDATE users SET groq_api_key_enc=? WHERE id=?",
                (enc, user_id),
            )

    def update_password(self, user_id: str, new_password: str):
        pw_hash = hash_password(new_password)
        with self._conn() as con:
            con.execute(
                "UPDATE users SET password_hash=? WHERE id=?",
                (pw_hash, user_id),
            )

    def get_decrypted_groq_key(self, user_id: str) -> str:
        user = self.get_by_id(user_id)
        if not user or not user.groq_api_key_enc:
            raise ValueError("No Groq API key set for this account.")
        return decrypt_api_key(user.groq_api_key_enc)

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_user(self, user_id: str):
        """Soft-delete: mark is_active=0 and wipe sensitive fields."""
        with self._conn() as con:
            con.execute(
                """UPDATE users
                   SET is_active=0, password_hash=NULL,
                       groq_api_key_enc=NULL, provider_id=NULL
                   WHERE id=?""",
                (user_id,),
            )
            con.execute(
                "DELETE FROM refresh_tokens WHERE user_id=?", (user_id,)
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _unique_username(self, base: str) -> str:
        """Ensure username is unique; append numeric suffix if needed."""
        slug = "".join(c for c in base.lower().replace(" ", "_") if c.isalnum() or c == "_")[:30] or "user"
        candidate = slug
        i = 1
        while True:
            with self._conn() as con:
                row = con.execute(
                    "SELECT 1 FROM users WHERE username=?", (candidate,)
                ).fetchone()
            if not row:
                return candidate
            candidate = f"{slug}{i}"
            i += 1

    def email_exists(self, email: str) -> bool:
        with self._conn() as con:
            row = con.execute(
                "SELECT 1 FROM users WHERE email=?", (email.lower().strip(),)
            ).fetchone()
        return row is not None

    def username_exists(self, username: str) -> bool:
        with self._conn() as con:
            row = con.execute(
                "SELECT 1 FROM users WHERE username=?", (username.strip(),)
            ).fetchone()
        return row is not None
