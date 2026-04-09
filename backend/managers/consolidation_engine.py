"""
MemoryOS — Complementary Learning Systems (CLS) Consolidation Engine
=====================================================================
Implements the CLS theory of memory (McClelland et al., 1995) adapted
for LLM-based personal AI.

Theory:
    Biological memory has two complementary systems:
    • Hippocampus  — fast, episodic, verbatim, high-capacity, fades fast
    • Neocortex    — slow, semantic, compressed, structured, persistent

    MemoryOS analogue:
    • ChromaDB episodic store  ↔  Hippocampus
    • Neo4j knowledge graph    ↔  Neocortex

    Consolidation: high-surprise, aged episodic memories are promoted to
    the graph as structured Knowledge/Entity nodes — compressing verbatim
    episodes into persistent semantic representations.

Mathematical criterion for consolidation:
    A memory mᵢ is a consolidation candidate iff:

        surprise(mᵢ) ≥ θ_s   AND   age(mᵢ) ≥ θ_t

    where:
        θ_s = CLS_MIN_SURPRISE         (default 0.65)
        θ_t = CLS_MIN_AGE_TURNS        (default 10)
        age(mᵢ) = t_current − t_origin(mᵢ)

    This mirrors sleep-consolidation: novel information that survives
    beyond the initial encoding window is worth structural storage.

Process (runs every CLS_CHECK_INTERVAL_TURNS turns, background thread):
    1. Query ChromaDB for unconsolidated episodes with surprise ≥ θ_s
       and age ≥ θ_t (at most 10 candidates per cycle)
    2. Send batch to LLM: "Extract structured facts from these episodes"
    3. LLM returns structured JSON: entities, events, knowledge
    4. Write to Neo4j graph
    5. Mark ChromaDB episodes as consolidated = True

This is the only component that calls the LLM outside of user request
processing. It runs in a daemon thread and is rate-limited to prevent
Groq quota consumption.
"""

import json
import logging
import re
import threading
import time

import config

logger = logging.getLogger(__name__)

_CONSOLIDATION_PROMPT = """You are a memory consolidation system. Extract structured facts from these raw conversation episodes and return ONLY valid JSON.

EPISODES:
{episodes}

Return a JSON object with this exact schema:
{{
  "entities": [
    {{"name": "...", "type": "person|place|organization", "relationship": "...", "attributes": {{}}}}
  ],
  "events": [
    {{"description": "...", "date": null, "entities_involved": []}}
  ],
  "knowledge": [
    {{"topic": "...", "content": "..."}}
  ]
}}

Rules: only extract facts explicitly stated. Do not infer or hallucinate. Return [] for any category with no clear facts."""


class CLSConsolidationEngine:
    """
    Singleton background engine implementing CLS memory consolidation.
    One instance shared across all users; per-user state tracked via dicts.
    """

    _instance  = None
    _cls_lock  = threading.Lock()

    def __new__(cls):
        with cls._cls_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._init()
                cls._instance = inst
        return cls._instance

    def _init(self):
        self._lock          = threading.Lock()
        # {user_id: {"turn": int, "groq_key": str}}
        self._user_state: dict[str, dict] = {}
        # {user_id: int} — last turn consolidation ran for this user
        self._last_run: dict[str, int]    = {}
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="cls-consolidation"
        )
        self._thread.start()
        logger.info("[CLS] Consolidation engine started")

    # ── Public API ────────────────────────────────────────────────────────

    def tick(self, user_id: str, current_turn: int, groq_key: str):
        """
        Called once per processed message.  Triggers consolidation if the
        user has accumulated CLS_CHECK_INTERVAL_TURNS new turns since last run.
        """
        with self._lock:
            self._user_state[user_id] = {
                "turn":     current_turn,
                "groq_key": groq_key,
            }
            last = self._last_run.get(user_id, 0)
            if current_turn - last >= config.CLS_CHECK_INTERVAL_TURNS:
                # Spawn per-user consolidation (non-blocking)
                threading.Thread(
                    target=self._consolidate_user,
                    args=(user_id, current_turn, groq_key),
                    daemon=True,
                ).start()
                self._last_run[user_id] = current_turn

    # ── Background loop (heartbeat every 10 min) ──────────────────────────

    def _loop(self):
        while True:
            try:
                time.sleep(600)   # safety heartbeat — primary trigger is tick()
                with self._lock:
                    states = dict(self._user_state)
                for uid, info in states.items():
                    threading.Thread(
                        target=self._consolidate_user,
                        args=(uid, info["turn"], info["groq_key"]),
                        daemon=True,
                    ).start()
            except Exception as exc:
                logger.error("[CLS] heartbeat error: %s", exc)

    # ── Per-user consolidation ────────────────────────────────────────────

    def _consolidate_user(self, user_id: str, current_turn: int, groq_key: str):
        try:
            from backend.managers.archival_manager import ArchivalMemoryManager
            from backend.managers.graph_manager    import GraphManager
            from groq import Groq

            archive = ArchivalMemoryManager(user_id)
            graph   = GraphManager(user_id)

            candidates = archive.get_consolidation_candidates(
                min_surprise  = config.CLS_MIN_SURPRISE,
                min_age_turns = config.CLS_MIN_AGE_TURNS,
                current_turn  = current_turn,
                limit         = 8,
            )

            if not candidates:
                graph.close()
                return

            logger.info(
                "[CLS] Consolidating %d episodes for user=%s (turn %d)",
                len(candidates), user_id, current_turn,
            )

            # Build episode text for LLM
            episode_text = "\n\n".join(
                f"[Turn {c['turn']}, surprise={c['surprise']:.2f}]\n{c['content']}"
                for c in candidates
            )

            client     = Groq(api_key=groq_key)
            completion = client.chat.completions.create(
                model      = config.LLM_MODEL,
                messages   = [{
                    "role":    "user",
                    "content": _CONSOLIDATION_PROMPT.format(episodes=episode_text),
                }],
                max_tokens = 800,
            )

            raw = (completion.choices[0].message.content or "").strip()

            # Robustly extract JSON from response
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match:
                graph.close()
                return

            data = json.loads(match.group())

            consolidated_ids = []

            # Write entities
            for ent in (data.get("entities") or [])[:10]:
                name = (ent.get("name") or "").strip()
                if name:
                    graph.update_entity(
                        name=name,
                        relationship=ent.get("relationship"),
                        attributes=ent.get("attributes") or {},
                        entity_type=ent.get("type", "person"),
                    )

            # Write events
            for ev in (data.get("events") or [])[:6]:
                desc = (ev.get("description") or "").strip()
                if desc:
                    graph.log_event(
                        description=desc,
                        entities_involved=ev.get("entities_involved") or [],
                        date=ev.get("date"),
                    )

            # Write knowledge
            for kn in (data.get("knowledge") or [])[:6]:
                topic   = (kn.get("topic") or "").strip()
                content = (kn.get("content") or "").strip()
                if topic and content:
                    graph.add_general_knowledge(topic, content)

            # Mark source episodes as consolidated
            for c in candidates:
                consolidated_ids.append(c["id"])
            archive.mark_consolidated(consolidated_ids)

            graph.close()
            logger.info(
                "[CLS] Consolidated %d episodes → %d entities / %d events / %d knowledge for user=%s",
                len(candidates),
                len(data.get("entities") or []),
                len(data.get("events") or []),
                len(data.get("knowledge") or []),
                user_id,
            )

        except Exception as exc:
            logger.error("[CLS] consolidation failed for user=%s: %s", user_id, exc)
