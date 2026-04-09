"""
MemoryOS — OAuth 2.0 Provider Helpers
=======================================
Supports: Google, GitHub.

Each provider follows the same pattern:
  1. build_authorization_url(state) → redirect URL for the user's browser
  2. exchange_code(code, state)     → OAuthUser(id, email, name, avatar_url, provider)

Requires env vars:
  GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET
  GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET
  APP_BASE_URL  (e.g. https://yourdomain.com)
"""

from dataclasses import dataclass

import httpx

import config


@dataclass
class OAuthUser:
    provider_id: str
    email: str
    name: str
    avatar_url: str
    provider: str   # "google" | "github"


# ── Google ────────────────────────────────────────────────────────────────────

_GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USER_URL  = "https://www.googleapis.com/oauth2/v3/userinfo"


def google_auth_url(state: str) -> str:
    params = {
        "client_id":     config.GOOGLE_CLIENT_ID,
        "redirect_uri":  f"{config.APP_BASE_URL}/auth/google/callback",
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         state,
        "access_type":   "online",
    }
    return _GOOGLE_AUTH_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())


async def google_exchange(code: str) -> OAuthUser:
    async with httpx.AsyncClient() as client:
        # Exchange code for token
        token_resp = await client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "code":          code,
                "client_id":     config.GOOGLE_CLIENT_ID,
                "client_secret": config.GOOGLE_CLIENT_SECRET,
                "redirect_uri":  f"{config.APP_BASE_URL}/auth/google/callback",
                "grant_type":    "authorization_code",
            },
        )
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]

        # Fetch user info
        user_resp = await client.get(
            _GOOGLE_USER_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user_resp.raise_for_status()
        u = user_resp.json()

    return OAuthUser(
        provider_id=u["sub"],
        email=u.get("email", ""),
        name=u.get("name", ""),
        avatar_url=u.get("picture", ""),
        provider="google",
    )


# ── GitHub ────────────────────────────────────────────────────────────────────

_GITHUB_AUTH_URL  = "https://github.com/login/oauth/authorize"
_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
_GITHUB_USER_URL  = "https://api.github.com/user"
_GITHUB_EMAIL_URL = "https://api.github.com/user/emails"


def github_auth_url(state: str) -> str:
    params = {
        "client_id":   config.GITHUB_CLIENT_ID,
        "redirect_uri":f"{config.APP_BASE_URL}/auth/github/callback",
        "scope":       "read:user user:email",
        "state":       state,
    }
    return _GITHUB_AUTH_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())


async def github_exchange(code: str) -> OAuthUser:
    async with httpx.AsyncClient() as client:
        # Exchange code for token
        token_resp = await client.post(
            _GITHUB_TOKEN_URL,
            headers={"Accept": "application/json"},
            data={
                "client_id":     config.GITHUB_CLIENT_ID,
                "client_secret": config.GITHUB_CLIENT_SECRET,
                "code":          code,
                "redirect_uri":  f"{config.APP_BASE_URL}/auth/github/callback",
            },
        )
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        }

        # Fetch user profile
        user_resp = await client.get(_GITHUB_USER_URL, headers=headers)
        user_resp.raise_for_status()
        u = user_resp.json()

        # GitHub may return null email on profile; fetch primary email separately
        email = u.get("email") or ""
        if not email:
            email_resp = await client.get(_GITHUB_EMAIL_URL, headers=headers)
            if email_resp.status_code == 200:
                for entry in email_resp.json():
                    if entry.get("primary") and entry.get("verified"):
                        email = entry["email"]
                        break

    return OAuthUser(
        provider_id=str(u["id"]),
        email=email,
        name=u.get("name") or u.get("login", ""),
        avatar_url=u.get("avatar_url", ""),
        provider="github",
    )
