"""
MemoryOS v2 — Neo4j Graph Manager
==================================
Replaces the flat user_state.json with a live property graph.

Node labels:   User · Person · Place · Organization · Event · Knowledge · Preference
Relationships: KNOWS · EXPERIENCED · AWARE_OF · HAS_PREFERENCE · RELATED_TO · INVOLVED_IN
"""

import re
import logging
from datetime import datetime

from neo4j import GraphDatabase
import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity resolution helpers
# ---------------------------------------------------------------------------

def _name_similarity(a: str, b: str) -> float:
    """
    Normalised Levenshtein similarity ∈ [0, 1].
    No external deps — pure Python DP.
    """
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return 1.0 - dp[n] / max(m, n)


def _possessive_rel_match(name_a: str, rel_b: str) -> bool:
    """
    True when name_a looks like a possessive descriptor whose relationship word
    matches rel_b, e.g. "Priyanshu's mother" with rel_b="mother".
    """
    if not rel_b:
        return False
    name_lower = name_a.lower()
    rel_lower  = rel_b.lower().strip()
    # strip possessive: "priyanshu's mother" → "mother"
    if "'" in name_lower:
        suffix = name_lower.split("'", 1)[-1].lstrip("s").strip()
        if rel_lower in suffix or suffix in rel_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert any string to a safe Neo4j ID."""
    return re.sub(r"[^a-z0-9]", "_", (text or "").lower().strip())[:60]


def _flatten(attrs: dict) -> dict:
    """Keep only primitive values Neo4j can store as node properties."""
    flat = {}
    for k, v in (attrs or {}).items():
        k_clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(k))
        if isinstance(v, (str, int, float, bool)):
            flat[k_clean] = v
        elif isinstance(v, list):
            flat[k_clean] = ", ".join(str(x) for x in v if x)
        elif v is None:
            pass  # skip nulls
    return flat


_VALID_LABELS = {"Person", "Place", "Organization"}


def _label(entity_type: str) -> str:
    mapping = {
        "person": "Person",
        "place": "Place",
        "location": "Place",
        "organization": "Organization",
        "company": "Organization",
        "org": "Organization",
    }
    return mapping.get((entity_type or "person").lower(), "Person")


# ---------------------------------------------------------------------------
# GraphManager
# ---------------------------------------------------------------------------

class GraphManager:
    """
    Single source of truth for structured memory.

    All public methods return a short status string so the Orchestrator
    can pass it back to the LLM as a tool result.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        self._setup_schema()
        self._ensure_user_node()
        logger.info("[GraphManager] Connected to Neo4j at %s", config.NEO4J_URI)

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _setup_schema(self):
        constraints = [
            "CREATE CONSTRAINT user_singleton   IF NOT EXISTS FOR (u:User)         REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT person_name      IF NOT EXISTS FOR (p:Person)       REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT place_name       IF NOT EXISTS FOR (p:Place)        REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT org_name         IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT knowledge_topic  IF NOT EXISTS FOR (k:Knowledge)    REQUIRE k.topic IS UNIQUE",
            "CREATE CONSTRAINT pref_id          IF NOT EXISTS FOR (p:Preference)   REQUIRE p.id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX event_turn  IF NOT EXISTS FOR (e:Event) ON (e.turn)",
            "CREATE INDEX event_date  IF NOT EXISTS FOR (e:Event) ON (e.date)",
        ]
        with self.driver.session() as s:
            for q in constraints + indexes:
                try:
                    s.run(q)
                except Exception as exc:
                    logger.debug("Schema: %s", exc)

    def _ensure_user_node(self):
        with self.driver.session() as s:
            s.run(
                """
                MERGE (u:User {id: 'main'})
                ON CREATE SET u.total_turns = 0, u.created_at = $now
                """,
                now=datetime.now().isoformat(),
            )

    # ------------------------------------------------------------------
    # Turn counter
    # ------------------------------------------------------------------

    def increment_turn(self) -> int:
        with self.driver.session() as s:
            rec = s.run(
                """
                MATCH (u:User {id: 'main'})
                SET u.total_turns = coalesce(u.total_turns, 0) + 1
                RETURN u.total_turns AS turn
                """
            ).single()
        return rec["turn"] if rec else 1

    def _current_turn(self) -> int:
        with self.driver.session() as s:
            rec = s.run(
                "MATCH (u:User {id:'main'}) RETURN coalesce(u.total_turns,0) AS t"
            ).single()
        return rec["t"] if rec else 0

    # ------------------------------------------------------------------
    # User profile
    # ------------------------------------------------------------------

    def update_profile(self, key: str, value: str) -> str:
        key_clean = re.sub(r"[^a-zA-Z0-9_]", "_", key.strip().lower())
        with self.driver.session() as s:
            s.run(
                f"MATCH (u:User {{id:'main'}}) SET u.`{key_clean}` = $v, u.updated_at = $now",
                v=value,
                now=datetime.now().isoformat(),
            )
        return f"Profile updated: {key_clean} = {value}"

    def remove_from_profile(self, key: str, value_to_remove: str = "") -> str:
        key_clean = key.strip().lower()
        if key_clean in ("preference", "preferences"):
            with self.driver.session() as s:
                rec = s.run(
                    """
                    MATCH (u:User {id:'main'})-[r:HAS_PREFERENCE]->(p:Preference)
                    WHERE toLower(p.value) CONTAINS toLower($v)
                    DELETE r, p RETURN count(p) AS n
                    """,
                    v=value_to_remove,
                ).single()
            n = rec["n"] if rec else 0
            return f"Removed {n} preference(s) matching '{value_to_remove}'"
        else:
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", key_clean)
            with self.driver.session() as s:
                s.run(f"MATCH (u:User {{id:'main'}}) REMOVE u.`{safe}`")
            return f"Profile field removed: {key_clean}"

    # ------------------------------------------------------------------
    # Entities (Person / Place / Organization)
    # ------------------------------------------------------------------

    def update_entity(
        self,
        name: str,
        relationship: str = None,
        attributes: dict = None,
        entity_type: str = "person",
    ) -> str:
        if not name:
            return "Error: entity name is required"

        clean_name = re.sub(r"\s*\(.*?\)", "", name).strip()
        label = _label(entity_type)
        attrs = _flatten(attributes)
        turn = self._current_turn()

        with self.driver.session() as s:
            s.run(
                f"""
                MERGE (e:{label} {{name: $name}})
                ON CREATE SET e.created_turn = $turn, e.access_count = 1
                ON MATCH  SET e.access_count = coalesce(e.access_count,0)+1,
                              e.last_seen_turn = $turn
                SET e += $attrs
                """,
                name=clean_name,
                turn=turn,
                attrs=attrs,
            )
            if relationship:
                s.run(
                    f"""
                    MATCH (u:User {{id:'main'}})
                    MATCH (e:{label} {{name:$name}})
                    MERGE (u)-[r:KNOWS]->(e)
                    SET r.relationship = $rel, r.last_seen = $now
                    """,
                    name=clean_name,
                    rel=relationship,
                    now=datetime.now().isoformat(),
                )
        return f"Entity synced: {clean_name} [{label}]"

    def link_entities(
        self, name_a: str, name_b: str, rel_type: str, description: str = ""
    ) -> str:
        """Create an edge between two existing entity nodes."""
        safe_rel = re.sub(r"[^A-Z0-9_]", "_", rel_type.upper().strip())
        turn = self._current_turn()
        with self.driver.session() as s:
            rec = s.run(
                """
                MATCH (a) WHERE a.name = $na AND NOT a:User AND NOT a:Event AND NOT a:Knowledge
                MATCH (b) WHERE b.name = $nb AND NOT b:User AND NOT b:Event AND NOT b:Knowledge
                MERGE (a)-[r:RELATED_TO {rel_type: $rt}]->(b)
                ON CREATE SET r.description = $desc, r.created_turn = $turn
                ON MATCH  SET r.updated_turn = $turn
                RETURN a.name AS a_name, b.name AS b_name
                """,
                na=name_a,
                nb=name_b,
                rt=safe_rel,
                desc=description,
                turn=turn,
            ).single()
        if rec:
            return f"Linked: {rec['a_name']} --[{safe_rel}]--> {rec['b_name']}"
        return f"Could not find both '{name_a}' and '{name_b}' in the graph"

    def delete_entity(self, name: str) -> str:
        with self.driver.session() as s:
            rec = s.run(
                "MATCH (e {name:$n}) WHERE NOT e:User DETACH DELETE e RETURN count(e) AS d",
                n=name,
            ).single()
        return f"Deleted entity: {name}" if rec and rec["d"] else f"Not found: {name}"

    # ------------------------------------------------------------------
    # Entity Resolution — merge duplicate nodes
    # ------------------------------------------------------------------

    def merge_entities(self, canonical_name: str, alias_name: str) -> str:
        """
        Merge alias_name into canonical_name.

        Algorithm (no APOC required):
          1. Verify both nodes exist.
          2. Re-point all KNOWS edges (User → alias) to canonical.
          3. Migrate outgoing RELATED_TO edges from alias → canonical.
          4. Migrate incoming RELATED_TO edges from alias → canonical.
          5. Migrate INVOLVED_IN edges (alias → Event) to canonical.
          6. Copy alias name into canonical.aliases list.
          7. DETACH DELETE alias.
        """
        with self.driver.session() as s:
            # 1. Verify
            check = s.run(
                """
                MATCH (c) WHERE c.name = $cn AND NOT c:User AND NOT c:Event AND NOT c:Knowledge
                MATCH (a) WHERE a.name = $an AND NOT a:User AND NOT a:Event AND NOT a:Knowledge
                RETURN c.name AS cn, a.name AS an
                """,
                cn=canonical_name, an=alias_name,
            ).single()
            if not check:
                return f"Entity resolution failed: could not find both '{canonical_name}' and '{alias_name}'"

            # 2. Re-point KNOWS (User → alias) to (User → canonical)
            s.run(
                """
                MATCH (u:User)-[r:KNOWS]->(a {name: $an})
                MATCH (c {name: $cn}) WHERE NOT c:User
                MERGE (u)-[r2:KNOWS]->(c)
                  ON CREATE SET r2.relationship = r.relationship,
                                r2.last_seen    = r.last_seen
                DELETE r
                """,
                cn=canonical_name, an=alias_name,
            )

            # 3. Outgoing RELATED_TO: alias → other  →  canonical → other
            s.run(
                """
                MATCH (a {name: $an})-[r:RELATED_TO]->(other)
                MATCH (c {name: $cn})
                WHERE other.name <> $cn AND NOT other:User
                MERGE (c)-[r2:RELATED_TO {rel_type: r.rel_type}]->(other)
                  ON CREATE SET r2.description   = r.description,
                                r2.created_turn  = r.created_turn
                DELETE r
                """,
                cn=canonical_name, an=alias_name,
            )

            # 4. Incoming RELATED_TO: other → alias  →  other → canonical
            s.run(
                """
                MATCH (other)-[r:RELATED_TO]->(a {name: $an})
                MATCH (c {name: $cn})
                WHERE other.name <> $cn AND NOT other:User
                MERGE (other)-[r2:RELATED_TO {rel_type: r.rel_type}]->(c)
                  ON CREATE SET r2.description   = r.description,
                                r2.created_turn  = r.created_turn
                DELETE r
                """,
                cn=canonical_name, an=alias_name,
            )

            # 5. INVOLVED_IN: alias → Event  →  canonical → Event
            s.run(
                """
                MATCH (a {name: $an})-[r:INVOLVED_IN]->(ev:Event)
                MATCH (c {name: $cn})
                MERGE (c)-[:INVOLVED_IN]->(ev)
                DELETE r
                """,
                cn=canonical_name, an=alias_name,
            )

            # 6. Record alias in canonical's aliases property
            s.run(
                """
                MATCH (c {name: $cn}), (a {name: $an})
                SET c.aliases = CASE
                    WHEN c.aliases IS NULL THEN a.name
                    ELSE c.aliases + '|' + a.name
                END
                SET c.access_count = coalesce(c.access_count, 0)
                                   + coalesce(a.access_count, 0)
                """,
                cn=canonical_name, an=alias_name,
            )

            # 7. Delete alias
            s.run(
                "MATCH (a {name: $an}) WHERE NOT a:User DETACH DELETE a",
                an=alias_name,
            )

        logger.info("[GraphManager] Merged '%s' → '%s'", alias_name, canonical_name)
        return f"Resolved: '{alias_name}' merged into '{canonical_name}'"

    # ------------------------------------------------------------------
    # Entity Resolution — duplicate detection
    # ------------------------------------------------------------------

    def detect_duplicates(self) -> list:
        """
        Return a ranked list of potentially duplicate entity pairs.

        Signals (each independently scored, summed to confidence):
          S1 — Same relationship_to_user string (0.40)
          S2 — Name containment: one name appears inside the other (0.50)
          S3 — Levenshtein name similarity > 0.65 (similarity × 0.40)
          S4 — Possessive-pattern match: "X's Y" where Y's rel matches (0.60)

        Pairs with combined confidence ≥ 0.40 are returned, sorted descending.
        """
        with self.driver.session() as s:
            rows = s.run(
                """
                MATCH (u:User {id:'main'})-[r:KNOWS]->(e)
                RETURN e.name AS name,
                       r.relationship AS rel,
                       labels(e)[0] AS label,
                       coalesce(e.aliases, '') AS aliases
                """
            )
            entities = [
                {
                    "name":    row["name"],
                    "rel":     (row["rel"] or "").lower().strip(),
                    "label":   row["label"],
                    "aliases": row["aliases"],
                }
                for row in rows
                if row["name"]
            ]

        duplicates = []
        seen: set = set()

        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                pair_key = tuple(sorted([e1["name"], e2["name"]]))
                if pair_key in seen:
                    continue

                reasons: list[str] = []
                score = 0.0
                n1 = e1["name"].lower()
                n2 = e2["name"].lower()

                # S1 — same relationship type to user
                if e1["rel"] and e2["rel"] and e1["rel"] == e2["rel"]:
                    reasons.append(f"same relationship to you: '{e1['rel']}'")
                    score += 0.40

                # S2 — name containment
                if n1 and n2 and (n1 in n2 or n2 in n1) and n1 != n2:
                    reasons.append("one name is contained within the other")
                    score += 0.50

                # S3 — Levenshtein similarity
                sim = _name_similarity(e1["name"], e2["name"])
                if sim > 0.65:
                    reasons.append(f"name similarity {sim:.0%}")
                    score += sim * 0.40

                # S4 — possessive pattern
                for ea, eb in [(e1, e2), (e2, e1)]:
                    if _possessive_rel_match(ea["name"], eb["rel"]):
                        reasons.append(
                            f"'{ea['name']}' is a possessive descriptor "
                            f"matching the relationship '{eb['rel']}'"
                        )
                        score += 0.60

                if score >= 0.40 and reasons:
                    seen.add(pair_key)
                    # Shorter / simpler name is suggested canonical
                    if len(e1["name"]) <= len(e2["name"]) and "'" not in e1["name"]:
                        canonical, alias = e1["name"], e2["name"]
                    else:
                        canonical, alias = e2["name"], e1["name"]

                    duplicates.append({
                        "canonical":   canonical,
                        "alias":       alias,
                        "label":       e1["label"],
                        "confidence":  round(min(score, 1.0), 2),
                        "reasons":     reasons,
                    })

        return sorted(duplicates, key=lambda x: x["confidence"], reverse=True)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def log_event(
        self, description: str, entities_involved: list = None, date: str = None
    ) -> str:
        turn = self._current_turn()
        ev_date = date or datetime.now().strftime("%Y-%m-%d")
        event_id = f"event_{turn}_{int(datetime.now().timestamp())}"

        with self.driver.session() as s:
            s.run(
                """
                MATCH (u:User {id:'main'})
                CREATE (ev:Event {
                    id: $eid, description: $desc,
                    date: $date, turn: $turn, created_at: $now
                })
                CREATE (u)-[:EXPERIENCED {turn: $turn}]->(ev)
                """,
                eid=event_id,
                desc=description,
                date=ev_date,
                turn=turn,
                now=datetime.now().isoformat(),
            )
            for ent_name in (entities_involved or []):
                s.run(
                    """
                    MATCH (ev:Event {id:$eid})
                    MATCH (e) WHERE e.name = $en AND NOT e:User
                    MERGE (e)-[:INVOLVED_IN]->(ev)
                    """,
                    eid=event_id,
                    en=ent_name,
                )
        return f"Event logged at turn {turn}: {description[:60]}"

    # ------------------------------------------------------------------
    # Knowledge
    # ------------------------------------------------------------------

    def add_general_knowledge(self, topic: str, content: str) -> str:
        turn = self._current_turn()
        with self.driver.session() as s:
            s.run(
                """
                MERGE (k:Knowledge {topic: $topic})
                ON CREATE SET k.created_turn = $turn, k.access_count = 0
                SET k.content = $content,
                    k.last_updated_turn = $turn,
                    k.access_count = coalesce(k.access_count,0)+1
                WITH k
                MATCH (u:User {id:'main'})
                MERGE (u)-[:AWARE_OF]->(k)
                """,
                topic=topic.lower().strip(),
                content=content,
                turn=turn,
            )
        return f"Knowledge saved: {topic}"

    def delete_knowledge(self, topic: str) -> str:
        with self.driver.session() as s:
            rec = s.run(
                "MATCH (k:Knowledge {topic:$t}) DETACH DELETE k RETURN count(k) AS d",
                t=topic.lower(),
            ).single()
        return f"Deleted: {topic}" if rec and rec["d"] else f"Not found: {topic}"

    # ------------------------------------------------------------------
    # Preferences / constraints / allergies
    # ------------------------------------------------------------------

    def add_preference(self, value: str, category: str = "preference") -> str:
        pref_id = f"pref_{_slugify(value)}"
        with self.driver.session() as s:
            s.run(
                """
                MERGE (p:Preference {id: $pid})
                SET p.value = $value, p.category = $cat
                WITH p
                MATCH (u:User {id:'main'})
                MERGE (u)-[:HAS_PREFERENCE]->(p)
                """,
                pid=pref_id,
                value=value,
                cat=category,
            )
        return f"Preference saved: [{category.upper()}] {value}"

    # ------------------------------------------------------------------
    # Graph traversal search (used as a tool by the LLM)
    # ------------------------------------------------------------------

    def graph_search(self, entity_name: str, depth: int = 2) -> str:
        with self.driver.session() as s:
            # 1-hop
            rec = s.run(
                """
                MATCH (start) WHERE toLower(start.name) = toLower($name) AND NOT start:User
                OPTIONAL MATCH (start)-[r]-(hop) WHERE NOT hop:User
                RETURN start,
                       collect(DISTINCT {rel: type(r), rt: r.rel_type, n: hop}) AS conns
                LIMIT 1
                """,
                name=entity_name,
            ).single()

        if not rec or not rec["start"]:
            return f"No entity found matching '{entity_name}'"

        start = dict(rec["start"])
        lines = [f"Graph: {start.get('name', entity_name)}"]
        conns = rec["conns"] or []
        if conns:
            lines.append("1-hop connections:")
            for c in conns[:12]:
                if not c.get("n"):
                    continue
                n = dict(c["n"])
                label = c.get("rt") or c.get("rel", "REL")
                target = n.get("name") or n.get("topic") or n.get("description", "?")
                lines.append(f"  --[{label}]--> {target[:70]}")

        if depth >= 2:
            with self.driver.session() as s:
                rows = s.run(
                    """
                    MATCH (start) WHERE toLower(start.name)=toLower($name)
                    MATCH (start)-[]-(h1)-[]-(h2)
                    WHERE NOT h2:User AND h2 <> start
                    RETURN DISTINCT h2 LIMIT 8
                    """,
                    name=entity_name,
                )
                second = [dict(r["h2"]) for r in rows if r["h2"]]
            if second:
                lines.append("2-hop connections:")
                for n in second:
                    lines.append(f"  {n.get('name') or n.get('topic','?')}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Core context prompt (called before every LLM turn)
    # ------------------------------------------------------------------

    def get_core_prompt(
        self,
        recent_history_text: str = "",
        archival_context: list = None,
        current_turn: int = None,
    ) -> str:
        """
        Build a compact, relevance-scored context block for the LLM system prompt.

        Uses ContextBuilder (relevance scoring + token budgeting) to maintain
        fixed-size context output regardless of conversation length, enabling
        2 000–4 000+ turn support with controlled Groq API token usage.
        """
        archival_context = archival_context or []
        turn = current_turn if current_turn is not None else self._current_turn()

        with self.driver.session() as s:
            # --- User profile + preferences ---
            p_rec = s.run(
                """
                MATCH (u:User {id:'main'})
                OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(p:Preference)
                RETURN u, collect({v: p.value, cat: p.category}) AS prefs
                """
            ).single()
            user  = dict(p_rec["u"]) if p_rec else {}
            prefs = [x for x in (p_rec["prefs"] if p_rec else []) if x.get("v")]

            # --- Entities: all (scoring will rank them) ---
            ent_rows = s.run(
                """
                MATCH (u:User {id:'main'})-[r:KNOWS]->(e)
                OPTIONAL MATCH (e)-[er:RELATED_TO]-(other)
                  WHERE NOT other:User
                RETURN e, r.relationship AS user_rel,
                       collect(DISTINCT {rt: er.rel_type, nm: other.name}) AS links
                ORDER BY coalesce(e.access_count,0) DESC LIMIT 40
                """
            )
            entities = []
            for row in ent_rows:
                e = dict(row["e"])
                e["_rel"]   = row["user_rel"]
                e["_links"] = [x for x in row["links"] if x.get("nm")]
                entities.append(e)

            # --- Events: recent 30 (scoring selects most relevant) ---
            ev_rows = s.run(
                """
                MATCH (u:User {id:'main'})-[:EXPERIENCED]->(ev:Event)
                OPTIONAL MATCH (ent)-[:INVOLVED_IN]->(ev)
                RETURN ev, collect(ent.name) AS involved
                ORDER BY ev.turn DESC LIMIT 30
                """
            )
            events = []
            for row in ev_rows:
                ev = dict(row["ev"])
                ev["_involved"] = [n for n in row["involved"] if n]
                events.append(ev)

            # --- Knowledge: all (scoring selects most relevant) ---
            kn_rows = s.run(
                """
                MATCH (u:User {id:'main'})-[:AWARE_OF]->(k:Knowledge)
                RETURN k ORDER BY k.last_updated_turn DESC LIMIT 20
                """
            )
            knowledge = [dict(r["k"]) for r in kn_rows]

        from backend.managers.context_manager import ContextBuilder
        return ContextBuilder().build(
            user=user,
            prefs=prefs,
            entities=entities,
            events=events,
            knowledge=knowledge,
            archival=archival_context,
            query=recent_history_text,
            current_turn=turn,
        )

    # ------------------------------------------------------------------
    # Visualization data (consumed by the frontend /graph endpoint)
    # ------------------------------------------------------------------

    def get_graph_data(self) -> dict:
        nodes, edges = [], []
        TYPE_COLOR = {
            "User":         "#7877c6",
            "Person":       "#4ade80",
            "Place":        "#60a5fa",
            "Organization": "#fbbf24",
            "Event":        "#f97316",
            "Knowledge":    "#a78bfa",
            "Preference":   "#f87171",
        }

        with self.driver.session() as s:
            # User node
            u_rec = s.run("MATCH (u:User {id:'main'}) RETURN u").single()
            if u_rec:
                u = dict(u_rec["u"])
                nodes.append({
                    "id": "user_main",
                    "label": u.get("name") or "You",
                    "type": "User",
                    "color": TYPE_COLOR["User"],
                    "size": 38,
                    "title": f"Turns: {u.get('total_turns',0)}",
                })

            # Entities
            for row in s.run(
                "MATCH (u:User {id:'main'})-[r:KNOWS]->(e) "
                "RETURN e, r.relationship AS rel, labels(e)[0] AS lbl"
            ):
                e = dict(row["e"])
                lbl = row["lbl"]
                nid = f"e_{_slugify(e.get('name','?'))}"
                etype_map = {"Person": "person", "Place": "place", "Organization": "organization"}
                nodes.append({
                    "id": nid,
                    "name": e.get("name", "?"),
                    "label": e.get("name", "?"),
                    "type": lbl,
                    "entity_type": etype_map.get(lbl, "person"),
                    "relationship": row["rel"] or "",
                    "attributes": {k: v for k, v in e.items()
                                   if k not in {"name","created_turn","last_seen_turn","access_count","aliases"}
                                   and v is not None},
                    "color": TYPE_COLOR.get(lbl, "#888"),
                    "size": 22,
                    "title": f"[{lbl}] Relationship: {row['rel']}",
                })
                edges.append({
                    "from": "user_main", "to": nid,
                    "label": row["rel"] or "knows",
                    "color": "rgba(120,119,198,0.55)",
                })

            # Entity ↔ entity relationships
            seen = set()
            for row in s.run(
                "MATCH (a)-[r:RELATED_TO]-(b) WHERE NOT a:User AND NOT b:User "
                "RETURN a.name AS an, b.name AS bn, r.rel_type AS rt LIMIT 50"
            ):
                a_id = f"e_{_slugify(row['an'] or '')}"
                b_id = f"e_{_slugify(row['bn'] or '')}"
                key  = tuple(sorted([a_id, b_id]))
                if key in seen:
                    continue
                seen.add(key)
                edges.append({
                    "from": a_id, "to": b_id,
                    "label": row["rt"] or "related",
                    "color": "rgba(74,222,128,0.45)",
                })

            # Events (last 12)
            for row in s.run(
                "MATCH (u:User {id:'main'})-[:EXPERIENCED]->(ev:Event) "
                "RETURN ev ORDER BY ev.turn DESC LIMIT 12"
            ):
                ev   = dict(row["ev"])
                nid  = ev.get("id", f"ev_{ev.get('turn')}")
                desc = ev.get("description", "Event")
                nodes.append({
                    "id": nid,
                    "name": desc,
                    "label": f"T{ev.get('turn')}: {desc[:28]}…" if len(desc) > 28 else f"T{ev.get('turn')}: {desc}",
                    "type": "Event",
                    "description": desc,
                    "date": ev.get("date") or "",
                    "color": TYPE_COLOR["Event"],
                    "size": 14,
                    "title": f"Turn {ev.get('turn')} | {ev.get('date')}\n{desc}",
                })
                edges.append({
                    "from": "user_main", "to": nid,
                    "label": "experienced",
                    "color": "rgba(249,115,22,0.3)",
                })
                # Event ↔ entity edges
                for er in s.run(
                    "MATCH (ent)-[:INVOLVED_IN]->(ev:Event {id:$eid}) RETURN ent.name AS n",
                    eid=nid,
                ):
                    if er["n"]:
                        edges.append({
                            "from": f"e_{_slugify(er['n'])}",
                            "to": nid,
                            "label": "involved",
                            "color": "rgba(249,115,22,0.2)",
                        })

            # Preferences
            for row in s.run(
                "MATCH (u:User {id:'main'})-[:HAS_PREFERENCE]->(p:Preference) RETURN p"
            ):
                p   = dict(row["p"])
                cat = p.get("category", "preference")
                col = "#f87171" if cat == "allergy" else "#c084fc"
                nodes.append({
                    "id": p["id"],
                    "name": p.get("value", "?"),
                    "label": p.get("value", "?")[:22],
                    "type": "Preference",
                    "value": p.get("value", ""),
                    "category": cat,
                    "color": col,
                    "size": 11,
                    "title": f"[{cat.upper()}] {p.get('value','')}",
                })
                edges.append({
                    "from": "user_main", "to": p["id"],
                    "label": cat,
                    "color": "rgba(248,113,113,0.35)",
                })

            # Knowledge
            for row in s.run(
                "MATCH (u:User {id:'main'})-[:AWARE_OF]->(k:Knowledge) "
                "RETURN k ORDER BY k.last_updated_turn DESC LIMIT 10"
            ):
                k   = dict(row["k"])
                nid = f"k_{_slugify(k.get('topic','?'))}"
                nodes.append({
                    "id": nid,
                    "name": k.get("topic", "?"),
                    "label": k.get("topic", "?")[:22],
                    "type": "Knowledge",
                    "topic": k.get("topic", ""),
                    "content": k.get("content", ""),
                    "color": TYPE_COLOR["Knowledge"],
                    "size": 13,
                    "title": k.get("content", "")[:120],
                })
                edges.append({
                    "from": "user_main", "to": nid,
                    "label": "knows",
                    "color": "rgba(167,139,250,0.35)",
                })

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Stats (sidebar / analytics dashboard)
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        with self.driver.session() as s:
            main = s.run(
                """
                MATCH (u:User {id:'main'})
                OPTIONAL MATCH (u)-[:KNOWS]->(ent)
                OPTIONAL MATCH (u)-[:EXPERIENCED]->(ev:Event)
                OPTIONAL MATCH (u)-[:AWARE_OF]->(kn:Knowledge)
                OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(pr:Preference)
                RETURN u.total_turns AS turns, u.name AS uname,
                       count(DISTINCT ent) AS ec,
                       count(DISTINCT ev)  AS evc,
                       count(DISTINCT kn)  AS kc,
                       count(DISTINCT pr)  AS pc
                """
            ).single()

            breakdown = {
                row["t"]: row["c"]
                for row in s.run(
                    "MATCH (u:User {id:'main'})-[:KNOWS]->(e) "
                    "RETURN labels(e)[0] AS t, count(e) AS c"
                )
            }

            profile = {}
            pu = s.run(
                "MATCH (u:User {id:'main'}) RETURN properties(u) AS p"
            ).single()
            if pu:
                skip = {"id", "total_turns", "created_at", "updated_at"}
                profile = {k: v for k, v in dict(pu["p"]).items() if k not in skip and v}

            preferences = [
                {"value": r["v"], "category": r["c"]}
                for r in s.run(
                    "MATCH (u:User {id:'main'})-[:HAS_PREFERENCE]->(p:Preference) "
                    "RETURN p.value AS v, p.category AS c"
                )
            ]

        if not main:
            return {"total_turns": 0, "entity_count": 0, "event_count": 0,
                    "knowledge_count": 0, "pref_count": 0}
        return {
            "total_turns":     main["turns"] or 0,
            "user_name":       main["uname"] or "User",
            "entity_count":    main["ec"],
            "event_count":     main["evc"],
            "knowledge_count": main["kc"],
            "pref_count":      main["pc"],
            "entity_breakdown": breakdown,
            "profile":         profile,
            "preferences":     preferences,
        }

    # ------------------------------------------------------------------
    # Search (text-based, used by /search endpoint)
    # ------------------------------------------------------------------

    def search_entities(self, query: str) -> list:
        with self.driver.session() as s:
            rows = s.run(
                """
                MATCH (e) WHERE (e:Person OR e:Place OR e:Organization OR e:Knowledge)
                  AND (toLower(coalesce(e.name,''))    CONTAINS toLower($q)
                    OR toLower(coalesce(e.topic,''))   CONTAINS toLower($q)
                    OR toLower(coalesce(e.content,'')) CONTAINS toLower($q))
                RETURN e, labels(e)[0] AS lbl LIMIT 12
                """,
                q=query,
            )
            return [{"node": dict(r["e"]), "label": r["lbl"]} for r in rows]

    # ------------------------------------------------------------------
    # Reset & teardown
    # ------------------------------------------------------------------

    def reset(self):
        """Wipe all data, then recreate the User node."""
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        self._ensure_user_node()
        logger.info("[GraphManager] Graph reset.")

    def close(self):
        self.driver.close()
        logger.info("[GraphManager] Driver closed.")
