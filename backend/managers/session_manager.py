"""
MemoryOS — Session Summary Manager
=====================================
Singleton background worker that watches per-user conversation idle time.
After IDLE_THRESHOLD seconds of inactivity, it generates a 2-sentence
LLM summary of what happened in the session and stores it as a
SessionSummary node in Neo4j.

The summary is injected into the system prompt at the START of the next
session, giving the model "Last time you were here…" context.
"""

import logging
import threading
import time

import config

logger = logging.getLogger(__name__)

_IDLE_THRESHOLD = 28 * 60   # 28 minutes of silence = session over
_CHECK_INTERVAL =  5 * 60   # background thread wakes every 5 min


class SessionSummaryManager:
    """Process-wide singleton — one instance per server process."""

    _instance = None
    _class_lock = threading.Lock()

    # ── Singleton constructor ─────────────────────────────────────────────

    def __new__(cls):
        with cls._class_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._init()
                cls._instance = inst
        return cls._instance

    def _init(self):
        self._lock = threading.Lock()
        # {user_id: float}  — epoch seconds of last activity
        self._last_activity: dict[str, float] = {}
        # {user_id: {"start_turn": int, "end_turn": int, "groq_key": str}}
        self._pending: dict[str, dict] = {}
        # {user_id: int}  — last turn that was already summarised
        self._last_summarised: dict[str, int] = {}
        self._thread = threading.Thread(target=self._loop, daemon=True, name="session-summary")
        self._thread.start()
        logger.info("[SessionManager] Background thread started")

    # ── Public API ────────────────────────────────────────────────────────

    def record_activity(self, user_id: str, turn: int, groq_key: str):
        """Call once per processed message for every active user."""
        with self._lock:
            self._last_activity[user_id] = time.monotonic()
            p = self._pending.setdefault(user_id, {"start_turn": turn, "groq_key": groq_key})
            p["end_turn"]  = turn
            p["groq_key"]  = groq_key

    # ── Background loop ───────────────────────────────────────────────────

    def _loop(self):
        while True:
            try:
                time.sleep(_CHECK_INTERVAL)
                self._check_all()
            except Exception as exc:
                logger.error("[SessionManager] loop error: %s", exc)

    def _check_all(self):
        now = time.monotonic()
        with self._lock:
            candidates = [
                uid for uid, last in self._last_activity.items()
                if now - last > _IDLE_THRESHOLD
                and uid in self._pending
                and self._pending[uid].get("end_turn", 0)
                   > self._last_summarised.get(uid, 0)
            ]
        for uid in candidates:
            threading.Thread(target=self._summarise, args=(uid,), daemon=True).start()

    def _summarise(self, user_id: str):
        try:
            from groq import Groq
            from backend.managers.graph_manager import GraphManager

            with self._lock:
                info      = dict(self._pending.get(user_id, {}))
                groq_key  = info.get("groq_key", "")
                start_t   = info.get("start_turn", 0)
                end_t     = info.get("end_turn", 0)

            if not groq_key or end_t <= start_t:
                return

            gm       = GraphManager(user_id)
            timeline = gm.get_timeline(limit=60)
            items    = [t for t in timeline
                        if start_t <= (t.get("turn") or 0) <= end_t]

            if not items:
                gm.close()
                return

            bullet_text = "\n".join(
                f"• [{item['type'].upper()}] {item.get('title','')} — {item.get('content','')}"
                for item in items[:24]
            )

            client     = Groq(api_key=groq_key)
            completion = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        "Write 2-3 sentences summarising this memory session for future context. "
                        "Be specific: mention key people, events, and topics discussed. "
                        "Write in second person ('You discussed…').\n\n"
                        f"{bullet_text}"
                    ),
                }],
                max_tokens=120,
            )

            summary = (completion.choices[0].message.content or "").strip()
            if summary:
                gm.save_session_summary(summary, start_t, end_t)

            gm.close()

            with self._lock:
                self._last_summarised[user_id] = end_t
                self._pending.pop(user_id, None)

            logger.info("[SessionManager] Summarised session for user=%s (turns %d–%d)",
                        user_id, start_t, end_t)

        except Exception as exc:
            logger.error("[SessionManager] summarise failed for user=%s: %s", user_id, exc)
