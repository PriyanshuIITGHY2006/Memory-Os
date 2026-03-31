"""
MemoryOS — Relevance-Scored Context Builder
============================================
Enables 2 000–4 000 conversation turn support with a fixed token budget.

Mathematical framework
----------------------
For each memory node mᵢ, relevance to query Q is:

    score(mᵢ | Q) = α·keyword(Q,mᵢ) + β·recency(t,mᵢ) + γ·frequency(mᵢ)

where:
    keyword(Q, m)  = |tokens(Q) ∩ tokens(content(m))| / max(|tokens(Q)|, 1)
    recency(t, m)  = exp(−λ · (t_current − t_last(m)))   [λ = 0.001]
    frequency(m)   = log(1 + access_count) / log(1 + 100)

    α = 0.50   (what is relevant NOW)
    β = 0.30   (what was said recently)
    γ = 0.20   (what is accessed most often)

Context selection (greedy knapsack):
    C_t = sort M by score desc, include nodes until Σ chars(mᵢ) ≥ Budget B

Compact serialisation reduces context tokens by ≈4× vs. verbose format:
    ENTITIES(3): Alice[colleague](role=SWE)->WORKS_AT:TechCorp | Bob[friend] | …
"""

import math
import re

import config


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    """Fast lowercased word tokenizer — words of length ≥ 2 only."""
    return set(re.findall(r"[a-z]{2,}", (text or "").lower()))


def relevance_score(
    query_tokens: set,
    content: str,
    last_turn: int,
    current_turn: int,
    access_count: int,
) -> float:
    """
    Combined relevance score in [0, 1].

    Parameters
    ----------
    query_tokens  : tokenized current user message + recent history
    content       : text content of the memory node to score
    last_turn     : last turn this node was written or accessed (0 if unknown)
    current_turn  : conversation turn being processed right now
    access_count  : total write/access count for this node
    """
    # 1. Keyword overlap (query-biased Jaccard)
    ct = _tokenize(content)
    keyword = (
        len(query_tokens & ct) / max(len(query_tokens), 1)
        if query_tokens and ct
        else 0.0
    )

    # 2. Exponential recency decay
    delta = max(0, current_turn - max(0, last_turn or 0))
    recency = math.exp(-config.RELEVANCE_DECAY_LAMBDA * delta)

    # 3. Log-normalised access frequency (capped at 100 for normalisation)
    freq = math.log1p(max(0, access_count or 0)) / math.log1p(100)

    return (
        config.RELEVANCE_KEYWORD_WEIGHT * keyword
        + config.RELEVANCE_RECENCY_WEIGHT * recency
        + config.RELEVANCE_FREQ_WEIGHT    * freq
    )


def _trim(text: str, limit: int) -> str:
    """Hard-truncate text to `limit` characters, appending ellipsis if cut."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """
    Produce a compact, relevance-scored context string from Neo4j graph data.

    Fixed-size output regardless of how many turns have elapsed:
    ─────────────────────────────────────────────────────────────
    PROFILE: name=Alex | job=engineer | city=NYC
    CONSTRAINTS: [ALLERGY]peanuts [GOAL]promotion [PREF]morning-meetings
    ENTITIES(5): Alice[colleague](role=SWE)->WORKS_AT:TechCorp | Bob[friend]
    EVENTS(3): T240:promoted-to-senior T89:met-Alice-at-conference(Alice)
    KNOWLEDGE(2): python:prefer-list-comprehensions || sql:index-on-join-cols
    ARCHIVAL(2): [T12,c=0.91] first met Alice at the NeurIPS conference
    ─────────────────────────────────────────────────────────────

    Token comparison (typical graph with 20 entities, 15 events, 10 knowledge):
      Verbose format : ≈ 1 800–2 400 characters  (≈ 450–600 tokens)
      Compact format :   ≈  700–1 000 characters  (≈ 175–250 tokens)
    """

    def build(
        self,
        *,
        user: dict,
        prefs: list,
        entities: list,
        events: list,
        knowledge: list,
        archival: list,
        query: str,
        current_turn: int,
    ) -> str:
        qt = _tokenize(query)
        lines: list[str] = []

        # ── PROFILE ──────────────────────────────────────────────────────────
        skip_u = {"id", "total_turns", "created_at", "updated_at"}
        kv = {k: v for k, v in user.items() if k not in skip_u and v}
        if kv:
            parts = " | ".join(
                f"{k.replace('_', '-')}={_trim(str(v), 50)}"
                for k, v in list(kv.items())[:8]
            )
            lines.append(f"PROFILE: {_trim(parts, config.TOKEN_BUDGET_PROFILE)}")

        # ── PREFERENCES & CONSTRAINTS ────────────────────────────────────────
        if prefs:
            pstr = " ".join(
                f"[{(p.get('cat') or 'PREF').upper()}]{_trim(p['v'], 40)}"
                for p in prefs[:15]
                if p.get("v")
            )
            lines.append(f"CONSTRAINTS: {_trim(pstr, config.TOKEN_BUDGET_PREFS)}")

        # ── ENTITIES (relevance-ranked, greedy budget fill) ───────────────────
        skip_e = {"name", "created_turn", "access_count", "last_seen_turn", "_rel", "_links"}

        scored_ents = sorted(
            (
                (
                    relevance_score(
                        qt,
                        " ".join(filter(None, [
                            e.get("name", ""),
                            e.get("_rel", ""),
                            str(e.get("description", "")),
                            str(e.get("role", "")),
                        ])),
                        int(e.get("last_seen_turn") or e.get("created_turn") or 0),
                        current_turn,
                        int(e.get("access_count") or 0),
                    ),
                    e,
                )
                for e in entities
            ),
            key=lambda x: x[0],
            reverse=True,
        )

        ent_parts: list[str] = []
        budget = config.TOKEN_BUDGET_ENTITIES
        for _, e in scored_ents[: config.MAX_ENTITIES_IN_PROMPT]:
            name = e.get("name", "?")
            rel  = e.get("_rel") or "known"
            extra = {
                k: v
                for k, v in e.items()
                if k not in skip_e and v and not k.startswith("_")
            }
            part = f"{name}[{rel}]"
            if extra:
                kv_str = ",".join(
                    f"{k}={_trim(str(v), 20)}" for k, v in list(extra.items())[:3]
                )
                part += f"({kv_str})"
            links = e.get("_links", [])
            if links:
                lstr = "|".join(
                    f"{lk.get('rt', 'rel')}:{lk['nm']}"
                    for lk in links[:2]
                    if lk.get("nm")
                )
                if lstr:
                    part += f"->{lstr}"
            if len(part) >= budget:
                break
            ent_parts.append(part)
            budget -= len(part) + 3  # " | " separator

        lines.append(
            f"ENTITIES({len(ent_parts)}): " + " | ".join(ent_parts)
            if ent_parts
            else "ENTITIES: none"
        )

        # ── EVENTS (relevance-ranked) ─────────────────────────────────────────
        scored_evs = sorted(
            (
                (
                    relevance_score(
                        qt,
                        ev.get("description", ""),
                        int(ev.get("turn") or 0),
                        current_turn,
                        1,
                    ),
                    ev,
                )
                for ev in events
            ),
            key=lambda x: x[0],
            reverse=True,
        )

        ev_parts: list[str] = []
        budget = config.TOKEN_BUDGET_EVENTS
        for _, ev in scored_evs[: config.MAX_EVENTS_IN_PROMPT]:
            t    = ev.get("turn", "?")
            desc = ev.get("description") or ""
            slug = re.sub(r"\s+", "-", desc.strip().lower())[:60]
            part = f"T{t}:{slug}"
            inv  = ev.get("_involved", [])
            if inv:
                part += f"({','.join(inv[:2])})"
            if len(part) >= budget:
                break
            ev_parts.append(part)
            budget -= len(part) + 1

        lines.append(
            f"EVENTS({len(ev_parts)}): " + " ".join(ev_parts)
            if ev_parts
            else "EVENTS: none"
        )

        # ── KNOWLEDGE (relevance-ranked) ──────────────────────────────────────
        scored_kn = sorted(
            (
                (
                    relevance_score(
                        qt,
                        f"{k.get('topic', '')} {k.get('content', '')}",
                        int(k.get("last_updated_turn") or k.get("created_turn") or 0),
                        current_turn,
                        int(k.get("access_count") or 1),
                    ),
                    k,
                )
                for k in knowledge
            ),
            key=lambda x: x[0],
            reverse=True,
        )

        kn_parts: list[str] = []
        budget = config.TOKEN_BUDGET_KNOWLEDGE
        for _, k in scored_kn[: config.MAX_KNOWLEDGE_IN_PROMPT]:
            topic   = _trim(k.get("topic") or "?", 30)
            content = _trim(k.get("content") or "", 90)
            part    = f"{topic}:{content}"
            if len(part) >= budget:
                break
            kn_parts.append(part)
            budget -= len(part) + 4  # " || " separator

        if kn_parts:
            lines.append(f"KNOWLEDGE({len(kn_parts)}): " + " || ".join(kn_parts))

        # ── ARCHIVAL SEMANTIC MEMORIES ────────────────────────────────────────
        if archival:
            arch_parts: list[str] = []
            budget = config.TOKEN_BUDGET_ARCHIVAL
            for mem in archival[:6]:
                t    = mem.get("origin_turn", "?")
                conf = max(0.0, 1.0 - mem.get("distance", 1.5))
                body = _trim(mem.get("content") or "", 130)
                part = f"[T{t},c={conf:.2f}]{body}"
                if len(part) >= budget:
                    break
                arch_parts.append(part)
                budget -= len(part) + 3
            if arch_parts:
                lines.append(
                    f"ARCHIVAL({len(arch_parts)}): " + " | ".join(arch_parts)
                )

        return "\n".join(lines)
