"""
MemoryOS v2 — FastAPI Backend (multi-tenant, production-hardened)
=================================================================
Auth endpoints:
  POST /auth/register          — email + password sign-up
  POST /auth/login             — email + password login
  POST /auth/refresh           — exchange httpOnly refresh cookie for new access token
  POST /auth/logout            — revoke refresh token
  GET  /auth/me                — current user profile
  PUT  /auth/api-key           — update Groq API key
  PUT  /auth/password          — change password
  DELETE /auth/account         — soft-delete account + wipe all data
  GET  /auth/google            — begin Google OAuth
  GET  /auth/google/callback   — Google OAuth callback
  GET  /auth/github            — begin GitHub OAuth
  GET  /auth/github/callback   — GitHub OAuth callback

Memory endpoints (all require Bearer JWT):
  POST /chat
  GET  /graph
  GET  /stats
  GET  /search?q=...
  GET  /duplicates
  POST /merge
  POST /reset

Misc:
  GET  /health
  GET  /          — SPA frontend
"""

import asyncio
import concurrent.futures
import json as _json
import mimetypes
import os
import re
import secrets
import threading
from typing import Optional

from fastapi import Cookie, Depends, FastAPI, File, HTTPException, Query, Response, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, EmailStr, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

import config
from backend.auth.auth_manager import create_access_token
from backend.auth.dependencies import get_current_user
from backend.auth.oauth import github_auth_url, github_exchange, google_auth_url, google_exchange
from backend.managers.user_manager import UserManager
from backend.logic.orchestrator import Orchestrator

# ── App setup ─────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="MemoryOS", version="2.0.0")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Too many requests. Please slow down."})


# ── Security headers ──────────────────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), geolocation=()"
        if not response.headers.get("Cache-Control"):
            response.headers["Cache-Control"] = "no-store"
        return response


app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.APP_BASE_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────

user_manager = UserManager()

# Per-user Orchestrator cache  {user_id: Orchestrator}
_orchestrators: dict[str, Orchestrator] = {}
_orch_lock = threading.Lock()

# Per-user FileMemoryManager cache  {user_id: FileMemoryManager}
from backend.managers.file_manager import FileMemoryManager
_file_managers: dict[str, FileMemoryManager] = {}
_fm_lock = threading.Lock()

# Per-user WebSocket connections  {user_id: WebSocket}
_ws_connections: dict[str, WebSocket] = {}
_ws_lock = threading.Lock()


async def _broadcast_graph_update(user_id: str, event_type: str, data: dict):
    """Push a graph update event to the user's WebSocket if connected."""
    with _ws_lock:
        ws = _ws_connections.get(user_id)
    if ws:
        try:
            await ws.send_json({"type": event_type, **data})
        except Exception:
            with _ws_lock:
                _ws_connections.pop(user_id, None)


def _get_file_manager(user_id: str, groq_key: str) -> FileMemoryManager:
    with _fm_lock:
        if user_id not in _file_managers:
            orch = _get_orchestrator(user_id, groq_key)
            _file_managers[user_id] = FileMemoryManager(user_id, orch.archive)
        return _file_managers[user_id]


def _get_orchestrator(user_id: str, groq_key: str) -> Orchestrator:
    with _orch_lock:
        if user_id not in _orchestrators:
            _orchestrators[user_id] = Orchestrator(user_id, groq_key)
        return _orchestrators[user_id]


def _evict_orchestrator(user_id: str):
    with _orch_lock:
        if user_id in _orchestrators:
            try:
                _orchestrators[user_id].close()
            except Exception:
                pass
            del _orchestrators[user_id]


# ── Helpers ───────────────────────────────────────────────────────────────────

_COOKIE = "refresh_token"
_COOKIE_MAX_AGE = config.JWT_REFRESH_EXPIRE_DAYS * 86_400


def _set_refresh_cookie(response: Response, raw_token: str):
    response.set_cookie(
        key=_COOKIE,
        value=raw_token,
        httponly=True,
        secure=config.APP_BASE_URL.startswith("https"),
        samesite="lax",
        max_age=_COOKIE_MAX_AGE,
        path="/auth",
    )


def _clear_refresh_cookie(response: Response):
    response.delete_cookie(key=_COOKIE, path="/auth")


def _require_groq_key(user_id: str) -> str:
    """Return decrypted Groq API key or raise 400."""
    try:
        return user_manager.get_decrypted_groq_key(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Groq API key configured. Please add one in Settings.",
        )


# ── Request / Response models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str
    groq_api_key: Optional[str] = ""

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        if len(v) > 30:
            raise ValueError("Username too long (max 30 characters)")
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("Username may only contain letters, numbers, and underscores")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ApiKeyRequest(BaseModel):
    groq_api_key: str


class PasswordRequest(BaseModel):
    current_password: str
    new_password: str


class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        if len(v) > 4000:
            raise ValueError("Message too long (max 4000 characters)")
        return v


class MergeRequest(BaseModel):
    canonical: str
    alias: str


class ImportRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Text cannot be empty")
        if len(v) > 20_000:
            raise ValueError("Text too long (max 20,000 characters)")
        return v


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return FileResponse(html_path, media_type="text/html")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Auth — email / password ───────────────────────────────────────────────────

@app.post("/auth/register", status_code=201)
@limiter.limit("10/minute")
async def register(request: Request, body: RegisterRequest, response: Response):
    if user_manager.email_exists(body.email):
        raise HTTPException(400, "Email already registered.")
    if user_manager.username_exists(body.username):
        raise HTTPException(400, "Username already taken.")
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")

    user = user_manager.create_email_user(
        email=body.email,
        username=body.username,
        password=body.password,
        groq_api_key=body.groq_api_key or "",
    )
    access_token = create_access_token(user.id, user.email)
    raw_refresh   = user_manager.create_refresh_token(user.id)
    _set_refresh_cookie(response, raw_refresh)
    return {"access_token": access_token, "user": user.public_dict()}


@app.post("/auth/login")
@limiter.limit("20/minute")
async def login(request: Request, body: LoginRequest, response: Response):
    user = user_manager.authenticate(body.email, body.password)
    if not user:
        raise HTTPException(401, "Invalid email or password.")
    access_token = create_access_token(user.id, user.email)
    raw_refresh   = user_manager.create_refresh_token(user.id)
    _set_refresh_cookie(response, raw_refresh)
    return {"access_token": access_token, "user": user.public_dict()}


@app.post("/auth/refresh")
async def refresh_token(
    response: Response,
    refresh_token: Optional[str] = Cookie(default=None, alias=_COOKIE),
):
    if not refresh_token:
        raise HTTPException(401, "No refresh token.")
    user = user_manager.validate_refresh_token(refresh_token)
    if not user:
        raise HTTPException(401, "Invalid or expired refresh token.")
    user_manager.revoke_refresh_token(refresh_token)
    new_refresh   = user_manager.create_refresh_token(user.id)
    access_token  = create_access_token(user.id, user.email)
    _set_refresh_cookie(response, new_refresh)
    return {"access_token": access_token, "user": user.public_dict()}


@app.post("/auth/logout")
async def logout(
    response: Response,
    refresh_token: Optional[str] = Cookie(default=None, alias=_COOKIE),
    current_user: dict = Depends(get_current_user),
):
    if refresh_token:
        user_manager.revoke_refresh_token(refresh_token)
    _clear_refresh_cookie(response)
    return {"detail": "Logged out."}


@app.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    user = user_manager.get_by_id(current_user["sub"])
    if not user:
        raise HTTPException(404, "User not found.")
    return user.public_dict()


@app.put("/auth/api-key")
async def update_api_key(
    body: ApiKeyRequest,
    current_user: dict = Depends(get_current_user),
):
    if not body.groq_api_key.strip():
        raise HTTPException(400, "API key cannot be empty.")
    user_id = current_user["sub"]
    user_manager.update_groq_key(user_id, body.groq_api_key.strip())
    _evict_orchestrator(user_id)
    return {"detail": "Groq API key updated."}


@app.put("/auth/password")
async def change_password(
    body: PasswordRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["sub"]
    user = user_manager.get_by_id(user_id)
    if not user or not user_manager.authenticate(user.email, body.current_password):
        raise HTTPException(400, "Current password is incorrect.")
    if len(body.new_password) < 8:
        raise HTTPException(400, "New password must be at least 8 characters.")
    user_manager.update_password(user_id, body.new_password)
    user_manager.revoke_all_refresh_tokens(user_id)
    return {"detail": "Password changed. Please log in again."}


@app.delete("/auth/account")
async def delete_account(
    response: Response,
    refresh_token: Optional[str] = Cookie(default=None, alias=_COOKIE),
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["sub"]
    orch = _orchestrators.get(user_id)
    if orch:
        try:
            orch.graph.delete_user_graph()
            orch.archive.delete_user_collections()
        except Exception:
            pass
    _evict_orchestrator(user_id)
    user_manager.delete_user(user_id)
    _clear_refresh_cookie(response)
    return {"detail": "Account deleted."}


# ── Auth — OAuth ──────────────────────────────────────────────────────────────

_oauth_states: dict[str, str] = {}


@app.get("/auth/google")
async def google_login():
    state = secrets.token_urlsafe(16)
    _oauth_states[state] = "google"
    return RedirectResponse(google_auth_url(state))


@app.get("/auth/google/callback")
async def google_callback(code: str, state: str, response: Response):
    if state not in _oauth_states:
        raise HTTPException(400, "Invalid OAuth state.")
    del _oauth_states[state]
    try:
        oauth_user = await google_exchange(code)
    except Exception as exc:
        raise HTTPException(400, f"Google OAuth failed: {exc}")
    user, _ = user_manager.get_or_create_oauth_user(
        provider=oauth_user.provider,
        provider_id=oauth_user.provider_id,
        email=oauth_user.email,
        name=oauth_user.name,
        avatar_url=oauth_user.avatar_url,
    )
    access_token = create_access_token(user.id, user.email)
    raw_refresh   = user_manager.create_refresh_token(user.id)
    _set_refresh_cookie(response, raw_refresh)
    return RedirectResponse(f"{config.APP_BASE_URL}/#token={access_token}")


@app.get("/auth/github")
async def github_login():
    state = secrets.token_urlsafe(16)
    _oauth_states[state] = "github"
    return RedirectResponse(github_auth_url(state))


@app.get("/auth/github/callback")
async def github_callback(code: str, state: str, response: Response):
    if state not in _oauth_states:
        raise HTTPException(400, "Invalid OAuth state.")
    del _oauth_states[state]
    try:
        oauth_user = await github_exchange(code)
    except Exception as exc:
        raise HTTPException(400, f"GitHub OAuth failed: {exc}")
    user, _ = user_manager.get_or_create_oauth_user(
        provider=oauth_user.provider,
        provider_id=oauth_user.provider_id,
        email=oauth_user.email,
        name=oauth_user.name,
        avatar_url=oauth_user.avatar_url,
    )
    access_token = create_access_token(user.id, user.email)
    raw_refresh   = user_manager.create_refresh_token(user.id)
    _set_refresh_cookie(response, raw_refresh)
    return RedirectResponse(f"{config.APP_BASE_URL}/#token={access_token}")


# ── Memory endpoints (auth required) ─────────────────────────────────────────

@app.post("/chat")
@limiter.limit("30/minute")
async def chat_endpoint(
    request: Request,
    body: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)
    response_text, active_memories, debug_prompt, _ = orch.process_message(body.message)
    return {
        "response":        response_text,
        "active_memories": active_memories,
        "debug_prompt":    debug_prompt,
    }


@app.get("/graph")
async def get_graph(current_user: dict = Depends(get_current_user)):
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)
    return orch.graph.get_graph_data()


@app.get("/stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)
    raw      = orch.graph.get_stats()
    breakdown  = raw.get("entity_breakdown", {})
    prefs_list = raw.get("preferences", [])
    pref_counts: dict = {}
    for p in prefs_list:
        cat = (p.get("category") or "preference").lower()
        pref_counts[cat] = pref_counts.get(cat, 0) + 1
    return {
        "turns":           raw.get("total_turns", 0),
        "entities":        raw.get("entity_count", 0),
        "events":          raw.get("event_count", 0),
        "knowledge":       raw.get("knowledge_count", 0),
        "preferences":     raw.get("pref_count", 0),
        "people":          breakdown.get("Person", 0),
        "places":          breakdown.get("Place", 0),
        "organizations":   breakdown.get("Organization", 0),
        "pref_preference": pref_counts.get("preference", 0),
        "pref_goal":       pref_counts.get("goal", 0),
        "pref_allergy":    pref_counts.get("allergy", 0),
        "pref_constraint": pref_counts.get("constraint", 0),
        "profile":         raw.get("profile", {}),
    }


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    current_user: dict = Depends(get_current_user),
):
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)
    return orch.graph.search_entities(q)


@app.get("/duplicates")
async def get_duplicates(current_user: dict = Depends(get_current_user)):
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)
    return orch.graph.detect_duplicates()


@app.post("/merge")
async def merge_entities(
    body: MergeRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)
    result   = orch.graph.merge_entities(body.canonical, body.alias)
    return {"result": result}


@app.post("/reset")
async def reset_memory(current_user: dict = Depends(get_current_user)):
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)
    orch.graph.reset()
    orch.buffer.clear()
    return {"detail": "Memory graph reset."}


# ── Streaming chat (SSE) ──────────────────────────────────────────────────────

@app.post("/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)

    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _produce():
        try:
            for event in orch.process_message_stream(body.message):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "content": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    executor.submit(_produce)

    async def event_stream():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {_json.dumps(event)}\n\n"
            # After the done event, broadcast a graph update via WebSocket
            if event.get("type") == "done":
                asyncio.ensure_future(
                    _broadcast_graph_update(user_id, "graph_updated", {"turn": event.get("turn")})
                )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── WebSocket for live graph updates ─────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
):
    # Validate JWT passed as query param (can't send Authorization header in WS)
    if not token:
        await websocket.close(code=4001)
        return
    from backend.auth.auth_manager import decode_access_token  # noqa: PLC0415
    from jose import JWTError  # noqa: PLC0415
    try:
        payload = decode_access_token(token)
    except JWTError:
        await websocket.close(code=4001)
        return

    user_id = payload.get("sub")
    await websocket.accept()
    with _ws_lock:
        _ws_connections[user_id] = websocket

    try:
        while True:
            # Keep connection alive; client can ping with any message
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            _ws_connections.pop(user_id, None)


# ── Memory timeline ───────────────────────────────────────────────────────────

@app.get("/timeline")
async def get_timeline(
    limit: int = Query(default=100, ge=1, le=500),
    current_user: dict = Depends(get_current_user),
):
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)
    return {"items": orch.graph.get_timeline(limit=limit)}


# ── Memory import ─────────────────────────────────────────────────────────────

@app.post("/import")
@limiter.limit("10/minute")
async def import_memories(
    request: Request,
    body: ImportRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)

    import_prompt = (
        "SYSTEM IMPORT: The following text was provided by the user to be parsed "
        "and saved into memory. Extract all relevant entities, events, knowledge, "
        "and personal facts. Use the available memory tools to store everything "
        "you find. After saving, briefly summarize what was imported.\n\n"
        f"--- BEGIN IMPORT ---\n{body.text}\n--- END IMPORT ---"
    )

    loop     = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    response_text, active_memories, _, _changes = await loop.run_in_executor(
        executor, orch.process_message, import_prompt
    )

    asyncio.ensure_future(
        _broadcast_graph_update(user_id, "graph_updated", {"source": "import"})
    )

    return {"summary": response_text, "active_memories": active_memories}


# ── Knowledge gap detection ──────────────────────────────────────────────────

@app.get("/gaps")
@limiter.limit("10/minute")
async def get_knowledge_gaps(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Ask the LLM to audit the memory graph and surface what's missing."""
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)

    loop     = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _build():
        from groq import Groq
        stats    = orch.graph.get_stats()
        timeline = orch.graph.get_timeline(limit=40)
        profile  = stats.get("profile", {})
        prefs    = stats.get("preferences", [])

        summary_lines = []
        if profile:
            summary_lines.append("Profile: " + ", ".join(f"{k}={v}" for k, v in profile.items()))
        summary_lines.append(
            f"Entities: {stats.get('entity_count',0)} | "
            f"Events: {stats.get('event_count',0)} | "
            f"Knowledge: {stats.get('knowledge_count',0)}"
        )
        if prefs:
            summary_lines.append("Preferences: " + ", ".join(p["value"] for p in prefs[:8]))
        if timeline:
            summary_lines.append("Recent timeline:")
            for item in timeline[-10:]:
                summary_lines.append(f"  [{item['type']}] {item.get('title','')} (turn {item.get('turn',0)})")

        graph_text = "\n".join(summary_lines)

        client = Groq(api_key=groq_key)
        completion = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "You are analyzing a personal memory graph. Based on what is stored, "
                    "identify 5 specific knowledge gaps — important information that is "
                    "missing or incomplete. Format as a JSON array of objects: "
                    '[{"gap": "...", "why_important": "...", "suggested_question": "..."}]\n\n'
                    f"MEMORY GRAPH:\n{graph_text}"
                ),
            }],
            max_tokens=600,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Try to extract JSON
        import re as _re
        match = _re.search(r'\[.*\]', raw, _re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return [{"gap": raw, "why_important": "", "suggested_question": ""}]

    gaps = await loop.run_in_executor(executor, _build)
    return {"gaps": gaps}


# ── Ask your past self (archival-only search) ─────────────────────────────────

@app.get("/ask-past")
async def ask_past(
    q: str = Query(..., min_length=1, max_length=500),
    current_user: dict = Depends(get_current_user),
):
    """Search only archival (ChromaDB) memories — bypass the live graph."""
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)

    loop     = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _search():
        hits = orch.archive.search_memory(q, n_results=8)
        if not hits:
            return {"answer": "No relevant memories found in your archive.", "sources": []}

        context = "\n".join(
            f"[Turn {h['origin_turn']}] {h['content'][:200]}" for h in hits
        )
        from groq import Groq
        client = Groq(api_key=groq_key)
        completion = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Based ONLY on these past memory records (do not add new information), "
                    f"answer: '{q}'\n\nPast memories:\n{context}"
                ),
            }],
            max_tokens=300,
        )
        answer = (completion.choices[0].message.content or "").strip()
        return {
            "answer":  answer,
            "sources": [{"turn": h["origin_turn"], "excerpt": h["content"][:100]} for h in hits[:5]],
        }

    result = await loop.run_in_executor(executor, _search)
    return result


# ── Contradictions ────────────────────────────────────────────────────────────

@app.get("/contradictions")
async def get_contradictions(current_user: dict = Depends(get_current_user)):
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)
    return {"contradictions": orch.graph.get_pending_contradictions()}


@app.post("/contradictions/{field}/resolve")
async def resolve_contradiction(
    field: str,
    current_user: dict = Depends(get_current_user),
):
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)
    orch.graph.resolve_contradiction(field)
    return {"detail": f"Contradiction for '{field}' resolved."}


# ── Annual Report PDF export ──────────────────────────────────────────────────

@app.get("/export/report")
@limiter.limit("5/minute")
async def export_report(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Generate and return the 'Year in Memory' PDF report."""
    from backend.managers.report_manager import generate_report  # noqa: PLC0415

    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    orch     = _get_orchestrator(user_id, groq_key)

    loop     = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _build():
        stats      = orch.graph.get_stats()
        timeline   = orch.graph.get_timeline(limit=200)
        graph_data = orch.graph.get_graph_data()
        return generate_report(stats, timeline, graph_data["nodes"])

    pdf_bytes = await loop.run_in_executor(executor, _build)

    from datetime import datetime as _dt  # noqa: PLC0415
    filename = f"MemoryOS_Report_{_dt.now().year}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── File Upload & Media Memory ────────────────────────────────────────────────

@app.post("/upload")
@limiter.limit("20/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a file (image, PDF, audio, text) and store it as a memory.
    Supports: jpg/png/gif/webp, pdf, txt/md/csv, mp3/wav/m4a/ogg/webm.
    Max size: 30 MB.
    """
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    fm       = _get_file_manager(user_id, groq_key)
    orch     = _get_orchestrator(user_id, groq_key)

    data = await file.read()
    loop     = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _process():
        try:
            turn = orch.graph.get_stats().get("turns", 0)
        except Exception:
            turn = 0  # Neo4j hiccup — use 0, memory still gets stored
        return fm.store_file(
            filename=file.filename or "upload",
            data=data,
            groq_key=groq_key,
            current_turn=turn,
        )

    try:
        meta = await loop.run_in_executor(executor, _process)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    await _broadcast_graph_update(user_id, "file_added", {"file_id": meta["file_id"], "name": meta["name"]})
    return {"success": True, "file": meta}


@app.get("/files")
async def list_files(current_user: dict = Depends(get_current_user)):
    """List all uploaded files for the current user."""
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    fm       = _get_file_manager(user_id, groq_key)
    return {"files": fm.list_files()}


@app.get("/files/{file_id}")
async def serve_file(
    file_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Serve a user's uploaded file (authenticated)."""
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    fm       = _get_file_manager(user_id, groq_key)

    path = fm.get_file_path(file_id)
    if not path:
        raise HTTPException(status_code=404, detail="File not found.")

    meta      = fm.get_file_meta(file_id)
    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(
        path=str(path),
        media_type=mime_type,
        filename=meta["name"] if meta else path.name,
    )


@app.delete("/files/{file_id}")
async def delete_file(
    file_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete an uploaded file and remove it from memory."""
    user_id  = current_user["sub"]
    groq_key = _require_groq_key(user_id)
    fm       = _get_file_manager(user_id, groq_key)

    deleted = fm.delete_file(file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found.")
    return {"success": True, "detail": "File deleted."}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("Starting MemoryOS server…")
    uvicorn.run(app, host="0.0.0.0", port=8000)
