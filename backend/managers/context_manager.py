"""
MemoryOS — Relevance-Scored Context Builder
============================================
Enables 2 000–4 000 conversation turn support with a fixed token budget.

═══════════════════════════════════════════════════════════════════════
  MEMORY RELEVANCE SCORING  (Adaptive Forgetting Extension)
═══════════════════════════════════════════════════════════════════════

Original score:
    score(mᵢ | Q) = α·keyword(Q,mᵢ) + β·recency(t,mᵢ) + γ·frequency(mᵢ)

Extended score with surprise-weighted adaptive forgetting:
    score(mᵢ | Q, t) = α·keyword(Q,mᵢ) + β·recency_adaptive(t,mᵢ)
                      + γ·frequency(mᵢ) + δ·surprise(mᵢ)

where:
    keyword(Q, m)  = |tokens(Q) ∩ tokens(content(m))| / max(|tokens(Q)|, 1)

    recency_adaptive(t, m) = exp(−λ_eff(m) · Δt)
        Δt = t_current − t_last(m)

        λ_eff(m) = λ_base · (1 − α_s · surprise(m))          [Adaptive Decay]
            surprise(m) ∈ [0,1] = cosine dist to nearest neighbour at write time
            α_s = SURPRISE_RETENTION_ALPHA  (default 0.70)

        Interpretation:
            surprise = 1  →  λ_eff = 0.30·λ  →  half-life ≈ 3.3× longer
            surprise = 0  →  λ_eff = λ        →  standard decay

    frequency(m)   = log(1 + access_count) / log(1 + 100)
    surprise(m)    = stored at write time (cosine dist to nearest neighbour)

Weights:  α = 0.45  β = 0.28  γ = 0.18  δ = 0.09    (sum = 1.00)

═══════════════════════════════════════════════════════════════════════
  PERSONALIZED PAGERANK SPREADING ACTIVATION
═══════════════════════════════════════════════════════════════════════

After initial retrieval, graph-reachable nodes are re-ranked via PPR:

    r⁽⁰⁾ᵥ = score(mᵥ | Q, t)  if mᵥ ∈ retrieved set K, else 0
    r⁽ᵏ⁺¹⁾ᵥ = (1−d)·r⁽⁰⁾ᵥ + d · Σᵤ∈N(v) [w_uv / Σⱼ w_uj] · r⁽ᵏ⁾ᵤ

    d = PPR_DAMPING (default 0.25)   k = PPR_ITERATIONS (default 4)

This is Personalized PageRank with personalisation vector = initial scores.
Nodes connected to multiple high-scoring retrieved nodes gain higher
activation, surfacing implicit associations the query didn't directly target.

═══════════════════════════════════════════════════════════════════════
  HEBBIAN CO-ACTIVATION LEARNING
═══════════════════════════════════════════════════════════════════════

When entities {e₁, e₂, …, eₙ} are jointly retrieved in the same turn:

    w_{ij}^{new} = w_{ij}^{old} + η  for all (i,j) pairs, i≠j
    η = HEBBIAN_LEARNING_RATE (default 0.10)

Edge weights decay each turn on unused edges:
    w_{ij}^{t+1} = w_{ij}^{t} · (1 − δ_H)   δ_H = HEBBIAN_DECAY

After HEBBIAN_PROMOTION_THRESH co-activations, a permanent RELATED_TO
edge is written to Neo4j.
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
    surprise: float = 0.5,
) -> float:
    """
    Extended relevance score ∈ [0, 1] with surprise-weighted adaptive decay.

    Parameters
    ----------
    query_tokens  : tokenized current user message + recent history
    content       : text content of the memory node to score
    last_turn     : last turn this node was written or accessed (0 if unknown)
    current_turn  : conversation turn being processed right now
    access_count  : total write/access count for this node
    surprise      : information-theoretic salience ∈ [0,1] computed at write time
                    defaults to 0.5 (medium salience) for legacy nodes
    """
    # 1. Keyword overlap (query-biased Jaccard)
    ct = _tokenize(content)
    keyword = (
        len(query_tokens & ct) / max(len(query_tokens), 1)
        if query_tokens and ct
        else 0.0
    )

    # 2. Surprise-weighted adaptive exponential decay
    #    λ_eff = λ_base · (1 − α_s · surprise)
    #    High surprise → slower decay → longer retention in context
    delta        = max(0, current_turn - max(0, last_turn or 0))
    lambda_eff   = config.RELEVANCE_DECAY_LAMBDA * (
        1.0 - config.SURPRISE_RETENTION_ALPHA * min(max(surprise, 0.0), 1.0)
    )
    recency      = math.exp(-lambda_eff * delta)

    # 3. Log-normalised access frequency (capped at 100 for normalisation)
    freq = math.log1p(max(0, access_count or 0)) / math.log1p(100)

    # 4. Surprise salience bonus — intrinsically important memories score higher
    sal = min(max(surprise, 0.0), 1.0)

    return (
        config.RELEVANCE_KEYWORD_WEIGHT  * keyword
        + config.RELEVANCE_RECENCY_WEIGHT  * recency
        + config.RELEVANCE_FREQ_WEIGHT     * freq
        + config.RELEVANCE_SURPRISE_WEIGHT * sal
    )


def _trim(text: str, limit: int) -> str:
    """Hard-truncate text to `limit` characters, appending ellipsis if cut."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Personalized PageRank over entity dict graph
# ---------------------------------------------------------------------------

def spreading_activation(
    scored_entities: list[tuple[float, dict]],
    damping: float = None,
    iterations: int = None,
) -> list[tuple[float, dict]]:
    """
    Apply Personalized PageRank spreading activation over the entity graph.

    Given a list of (score, entity) tuples where entity may have a `_links`
    field (list of {nm, rt} dicts from Neo4j), propagate activation through
    the link structure for `iterations` rounds.

    Returns re-ranked (score, entity) list sorted descending by final score.

    Algorithm:
        r⁽⁰⁾ᵥ = initial_score(v)
        r⁽ᵏ⁺¹⁾ᵥ = (1−d)·r⁽⁰⁾ᵥ + d · Σᵤ∈N(v) [r⁽ᵏ⁾ᵤ / |N(u)|]

    d = PPR_DAMPING,  k = PPR_ITERATIONS
    """
    d    = damping    if damping    is not None else config.PPR_DAMPING
    k    = iterations if iterations is not None else config.PPR_ITERATIONS

    if not scored_entities or k == 0:
        return scored_entities

    # Build name → (initial_score, entity) mapping
    name_to_idx   = {}
    init_scores   = []
    entities_list = []

    for i, (sc, ent) in enumerate(scored_entities):
        name = (ent.get("name") or "").strip().lower()
        if name:
            name_to_idx[name] = i
        init_scores.append(sc)
        entities_list.append(ent)

    n = len(init_scores)
    if n <= 1:
        return scored_entities

    # Current activation vector
    r = list(init_scores)

    for _ in range(k):
        r_new = [(1.0 - d) * init_scores[i] for i in range(n)]

        for i, ent in enumerate(entities_list):
            links = ent.get("_links") or []
            if not links:
                continue
            # Outgoing link weight = activation / degree
            out_weight = r[i] / len(links)
            for link in links:
                target_name = (link.get("nm") or "").strip().lower()
                j = name_to_idx.get(target_name)
                if j is not None and j != i:
                    r_new[j] += d * out_weight

        # Normalise to [0,1] to prevent score inflation
        max_r = max(r_new) or 1.0
        r     = [x / max_r for x in r_new]

    # Blend PPR score back with original (0.6 PPR + 0.4 original)
    blended = [
        (0.6 * r[i] + 0.4 * init_scores[i], entities_list[i])
        for i in range(n)
    ]
    blended.sort(key=lambda x: x[0], reverse=True)
    return blended


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
    ARCHIVAL(2): [T12,c=0.91,s=0.82] first met Alice at the NeurIPS conference
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

        # ── ENTITIES — adaptive relevance + PPR spreading activation ─────────
        skip_e = {"name", "created_turn", "access_count", "last_seen_turn",
                  "_rel", "_links", "user_id", "id", "aliases", "first_seen_turn"}

        scored_ents = [
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
                    float(e.get("surprise", 0.5)),
                ),
                e,
            )
            for e in entities
        ]
        scored_ents.sort(key=lambda x: x[0], reverse=True)

        # Apply PPR spreading activation over entity link graph
        scored_ents = spreading_activation(scored_ents)

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
            budget -= len(part) + 3

        lines.append(
            f"ENTITIES({len(ent_parts)}): " + " | ".join(ent_parts)
            if ent_parts
            else "ENTITIES: none"
        )

        # ── EVENTS (adaptive relevance-ranked) ───────────────────────────────
        scored_evs = sorted(
            [
                (
                    relevance_score(
                        qt,
                        ev.get("description", ""),
                        int(ev.get("turn") or 0),
                        current_turn,
                        1,
                        float(ev.get("surprise", 0.5)),
                    ),
                    ev,
                )
                for ev in events
            ],
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

        # ── KNOWLEDGE (adaptive relevance-ranked) ────────────────────────────
        scored_kn = sorted(
            [
                (
                    relevance_score(
                        qt,
                        f"{k.get('topic', '')} {k.get('content', '')}",
                        int(k.get("last_updated_turn") or k.get("created_turn") or 0),
                        current_turn,
                        int(k.get("access_count") or 1),
                        float(k.get("surprise", 0.5)),
                    ),
                    k,
                )
                for k in knowledge
            ],
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
            budget -= len(part) + 4

        if kn_parts:
            lines.append(f"KNOWLEDGE({len(kn_parts)}): " + " || ".join(kn_parts))

        # ── ARCHIVAL — now shows surprise score ──────────────────────────────
        if archival:
            arch_parts: list[str] = []
            budget = config.TOKEN_BUDGET_ARCHIVAL
            for mem in archival[:6]:
                t       = mem.get("origin_turn", "?")
                conf    = max(0.0, 1.0 - mem.get("distance", 1.5))
                surprise= mem.get("surprise", "?")
                s_str   = f",s={surprise:.2f}" if isinstance(surprise, float) else ""
                body    = _trim(mem.get("content") or "", 130)
                part    = f"[T{t},c={conf:.2f}{s_str}]{body}"
                if len(part) >= budget:
                    break
                arch_parts.append(part)
                budget -= len(part) + 3
            if arch_parts:
                lines.append(
                    f"ARCHIVAL({len(arch_parts)}): " + " | ".join(arch_parts)
                )

        return "\n".join(lines)
