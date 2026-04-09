"""
MemoryOS v2 — Neo4j Graph Manager (multi-tenant)
==================================================
Each GraphManager instance is scoped to a single user_id.  All nodes that
belong to a user are either reached through their User node (via KNOWS /
EXPERIENCED / AWARE_OF / HAS_PREFERENCE) or carry a `user_id` property on
the node itself for fast direct lookups.

Node labels:   User · Person · Place · Organization · Event · Knowledge · Preference
Relationships: KNOWS · EXPERIENCED · AWARE_OF · HAS_PREFERENCE · RELATED_TO · INVOLVED_IN
"""

import re
import logging
import warnings
from datetime import datetime

from neo4j import GraphDatabase
import config

logger = logging.getLogger(__name__)

# Suppress harmless "label/relationship does not exist yet" warnings from Neo4j
# These appear on first run before the schema is populated — not actual errors.
logging.getLogger("neo4j").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*does not exist.*")


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
    Single source of truth for one user's structured memory.

    All public methods return a short status string so the Orchestrator
    can pass it back to the LLM as a tool result.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        self._setup_schema()
        self._ensure_user_node()
        logger.info("[GraphManager] Connected for user=%s at %s", user_id, config.NEO4J_URI)

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _setup_schema(self):
        # Composite uniqueness constraints — require Neo4j 5.x
        constraints = [
            "CREATE CONSTRAINT user_singleton   IF NOT EXISTS FOR (u:User)         REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT person_user      IF NOT EXISTS FOR (p:Person)       REQUIRE (p.user_id, p.name) IS UNIQUE",
            "CREATE CONSTRAINT place_user       IF NOT EXISTS FOR (p:Place)        REQUIRE (p.user_id, p.name) IS UNIQUE",
            "CREATE CONSTRAINT org_user         IF NOT EXISTS FOR (o:Organization) REQUIRE (o.user_id, o.name) IS UNIQUE",
            "CREATE CONSTRAINT knowledge_user   IF NOT EXISTS FOR (k:Knowledge)    REQUIRE (k.user_id, k.topic) IS UNIQUE",
            "CREATE CONSTRAINT pref_id          IF NOT EXISTS FOR (p:Preference)   REQUIRE p.id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX event_turn  IF NOT EXISTS FOR (e:Event) ON (e.turn)",
            "CREATE INDEX event_date  IF NOT EXISTS FOR (e:Event) ON (e.date)",
            "CREATE INDEX event_user  IF NOT EXISTS FOR (e:Event) ON (e.user_id)",
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
                MERGE (u:User {id: $uid})
                ON CREATE SET u.total_turns = 0, u.created_at = $now
                """,
                uid=self.user_id,
                now=datetime.now().isoformat(),
            )

    # ------------------------------------------------------------------
    # Turn counter
    # ------------------------------------------------------------------

    def increment_turn(self) -> int:
        with self.driver.session() as s:
            rec = s.run(
                """
                MATCH (u:User {id: $uid})
                SET u.total_turns = coalesce(u.total_turns, 0) + 1
                RETURN u.total_turns AS turn
                """,
                uid=self.user_id,
            ).single()
        return rec["turn"] if rec else 1

    def _current_turn(self) -> int:
        with self.driver.session() as s:
            rec = s.run(
                "MATCH (u:User {id: $uid}) RETURN coalesce(u.total_turns,0) AS t",
                uid=self.user_id,
            ).single()
        return rec["t"] if rec else 0

    # ------------------------------------------------------------------
    # User profile
    # ------------------------------------------------------------------

    def update_profile(self, key: str, value: str) -> str:
        key_clean = re.sub(r"[^a-zA-Z0-9_]", "_", key.strip().lower())
        with self.driver.session() as s:
            s.run(
                f"MATCH (u:User {{id: $uid}}) SET u.`{key_clean}` = $v, u.updated_at = $now",
                uid=self.user_id,
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
                    MATCH (u:User {id: $uid})-[r:HAS_PREFERENCE]->(p:Preference)
                    WHERE toLower(p.value) CONTAINS toLower($v)
                    DELETE r, p RETURN count(p) AS n
                    """,
                    uid=self.user_id,
                    v=value_to_remove,
                ).single()
            n = rec["n"] if rec else 0
            return f"Removed {n} preference(s) matching '{value_to_remove}'"
        else:
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", key_clean)
            with self.driver.session() as s:
                s.run(
                    f"MATCH (u:User {{id: $uid}}) REMOVE u.`{safe}`",
                    uid=self.user_id,
                )
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
                MERGE (e:{label} {{name: $name, user_id: $uid}})
                ON CREATE SET e.created_turn = $turn, e.access_count = 1
                ON MATCH  SET e.access_count = coalesce(e.access_count,0)+1,
                              e.last_seen_turn = $turn
                SET e += $attrs
                """,
                name=clean_name,
                uid=self.user_id,
                turn=turn,
                attrs=attrs,
            )
            if relationship:
                s.run(
                    f"""
                    MATCH (u:User {{id: $uid}})
                    MATCH (e:{label} {{name: $name, user_id: $uid}})
                    MERGE (u)-[r:KNOWS]->(e)
                    SET r.relationship = $rel, r.last_seen = $now
                    """,
                    uid=self.user_id,
                    name=clean_name,
                    rel=relationship,
                    now=datetime.now().isoformat(),
                )
        return f"Entity synced: {clean_name} [{label}]"

    def link_entities(
        self, name_a: str, name_b: str, rel_type: str, description: str = ""
    ) -> str:
        """Create an edge between two existing entity nodes owned by this user."""
        safe_rel = re.sub(r"[^A-Z0-9_]", "_", rel_type.upper().strip())
        turn = self._current_turn()
        with self.driver.session() as s:
            rec = s.run(
                """
                MATCH (a) WHERE a.name = $na AND a.user_id = $uid
                  AND NOT a:User AND NOT a:Event AND NOT a:Knowledge
                MATCH (b) WHERE b.name = $nb AND b.user_id = $uid
                  AND NOT b:User AND NOT b:Event AND NOT b:Knowledge
                MERGE (a)-[r:RELATED_TO {rel_type: $rt}]->(b)
                ON CREATE SET r.description = $desc, r.created_turn = $turn
                ON MATCH  SET r.updated_turn = $turn
                RETURN a.name AS a_name, b.name AS b_name
                """,
                na=name_a,
                nb=name_b,
                uid=self.user_id,
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
                """
                MATCH (e {name: $n, user_id: $uid})
                WHERE NOT e:User
                DETACH DELETE e RETURN count(e) AS d
                """,
                n=name,
                uid=self.user_id,
            ).single()
        return f"Deleted entity: {name}" if rec and rec["d"] else f"Not found: {name}"

    # ------------------------------------------------------------------
    # Entity Resolution — merge duplicate nodes
    # ------------------------------------------------------------------

    def merge_entities(self, canonical_name: str, alias_name: str) -> str:
        """
        Merge alias_name into canonical_name (scoped to this user).

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
                MATCH (c) WHERE c.name = $cn AND c.user_id = $uid
                  AND NOT c:User AND NOT c:Event AND NOT c:Knowledge
                MATCH (a) WHERE a.name = $an AND a.user_id = $uid
                  AND NOT a:User AND NOT a:Event AND NOT a:Knowledge
                RETURN c.name AS cn, a.name AS an
                """,
                cn=canonical_name, an=alias_name, uid=self.user_id,
            ).single()
            if not check:
                return f"Entity resolution failed: could not find both '{canonical_name}' and '{alias_name}'"

            # 2. Re-point KNOWS (User → alias) to (User → canonical)
            s.run(
                """
                MATCH (u:User {id: $uid})-[r:KNOWS]->(a {name: $an, user_id: $uid})
                MATCH (c {name: $cn, user_id: $uid}) WHERE NOT c:User
                MERGE (u)-[r2:KNOWS]->(c)
                  ON CREATE SET r2.relationship = r.relationship,
                                r2.last_seen    = r.last_seen
                DELETE r
                """,
                uid=self.user_id, cn=canonical_name, an=alias_name,
            )

            # 3. Outgoing RELATED_TO: alias → other  →  canonical → other
            s.run(
                """
                MATCH (a {name: $an, user_id: $uid})-[r:RELATED_TO]->(other)
                MATCH (c {name: $cn, user_id: $uid})
                WHERE other.name <> $cn AND NOT other:User
                MERGE (c)-[r2:RELATED_TO {rel_type: r.rel_type}]->(other)
                  ON CREATE SET r2.description   = r.description,
                                r2.created_turn  = r.created_turn
                DELETE r
                """,
                uid=self.user_id, cn=canonical_name, an=alias_name,
            )

            # 4. Incoming RELATED_TO: other → alias  →  other → canonical
            s.run(
                """
                MATCH (other)-[r:RELATED_TO]->(a {name: $an, user_id: $uid})
                MATCH (c {name: $cn, user_id: $uid})
                WHERE other.name <> $cn AND NOT other:User
                MERGE (other)-[r2:RELATED_TO {rel_type: r.rel_type}]->(c)
                  ON CREATE SET r2.description   = r.description,
                                r2.created_turn  = r.created_turn
                DELETE r
                """,
                uid=self.user_id, cn=canonical_name, an=alias_name,
            )

            # 5. INVOLVED_IN: alias → Event  →  canonical → Event
            s.run(
                """
                MATCH (a {name: $an, user_id: $uid})-[r:INVOLVED_IN]->(ev:Event {user_id: $uid})
                MATCH (c {name: $cn, user_id: $uid})
                MERGE (c)-[:INVOLVED_IN]->(ev)
                DELETE r
                """,
                uid=self.user_id, cn=canonical_name, an=alias_name,
            )

            # 6. Record alias in canonical's aliases property
            s.run(
                """
                MATCH (c {name: $cn, user_id: $uid}), (a {name: $an, user_id: $uid})
                SET c.aliases = CASE
                    WHEN c.aliases IS NULL THEN a.name
                    ELSE c.aliases + '|' + a.name
                END
                SET c.access_count = coalesce(c.access_count, 0)
                                   + coalesce(a.access_count, 0)
                """,
                uid=self.user_id, cn=canonical_name, an=alias_name,
            )

            # 7. Delete alias
            s.run(
                "MATCH (a {name: $an, user_id: $uid}) WHERE NOT a:User DETACH DELETE a",
                an=alias_name, uid=self.user_id,
            )

        logger.info("[GraphManager] Merged '%s' → '%s' for user=%s", alias_name, canonical_name, self.user_id)
        return f"Resolved: '{alias_name}' merged into '{canonical_name}'"

    # ------------------------------------------------------------------
    # Entity Resolution — duplicate detection
    # ------------------------------------------------------------------

    def detect_duplicates(self) -> list:
        """
        Return a ranked list of potentially duplicate entity pairs for this user.

        Signals (each independently scored, summed to confidence):
          S1 — Same relationship_to_user string (0.40)
          S2 — Name containment: one name appears inside the other (0.50)
          S3 — Levenshtein name similarity > 0.65 (similarity × 0.40)
          S4 — Possessive-pattern match: "X's Y" where Y's rel matches (0.60)

        Pairs with combined confidence >= 0.40 are returned, sorted descending.
        """
        with self.driver.session() as s:
            rows = s.run(
                """
                MATCH (u:User {id: $uid})-[r:KNOWS]->(e)
                RETURN e.name AS name,
                       r.relationship AS rel,
                       labels(e)[0] AS label,
                       coalesce(e.aliases, '') AS aliases
                """,
                uid=self.user_id,
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
        event_id = f"event_{self.user_id[:8]}_{turn}_{int(datetime.now().timestamp())}"

        with self.driver.session() as s:
            s.run(
                """
                MATCH (u:User {id: $uid})
                CREATE (ev:Event {
                    id: $eid, description: $desc,
                    date: $date, turn: $turn, created_at: $now,
                    user_id: $uid
                })
                CREATE (u)-[:EXPERIENCED {turn: $turn}]->(ev)
                """,
                uid=self.user_id,
                eid=event_id,
                desc=description,
                date=ev_date,
                turn=turn,
                now=datetime.now().isoformat(),
            )
            for ent_name in (entities_involved or []):
                s.run(
                    """
                    MATCH (ev:Event {id: $eid, user_id: $uid})
                    MATCH (e) WHERE e.name = $en AND e.user_id = $uid AND NOT e:User
                    MERGE (e)-[:INVOLVED_IN]->(ev)
                    """,
                    eid=event_id,
                    en=ent_name,
                    uid=self.user_id,
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
                MERGE (k:Knowledge {topic: $topic, user_id: $uid})
                ON CREATE SET k.created_turn = $turn, k.access_count = 0
                SET k.content = $content,
                    k.last_updated_turn = $turn,
                    k.access_count = coalesce(k.access_count,0)+1
                WITH k
                MATCH (u:User {id: $uid})
                MERGE (u)-[:AWARE_OF]->(k)
                """,
                topic=topic.lower().strip(),
                uid=self.user_id,
                content=content,
                turn=turn,
            )
        return f"Knowledge saved: {topic}"

    def delete_knowledge(self, topic: str) -> str:
        with self.driver.session() as s:
            rec = s.run(
                "MATCH (k:Knowledge {topic: $t, user_id: $uid}) DETACH DELETE k RETURN count(k) AS d",
                t=topic.lower(),
                uid=self.user_id,
            ).single()
        return f"Deleted: {topic}" if rec and rec["d"] else f"Not found: {topic}"

    # ------------------------------------------------------------------
    # Preferences / constraints / allergies
    # ------------------------------------------------------------------

    def add_preference(self, value: str, category: str = "preference") -> str:
        pref_id = f"pref_{_slugify(self.user_id)}_{_slugify(value)}"
        with self.driver.session() as s:
            s.run(
                """
                MERGE (p:Preference {id: $pid})
                SET p.value = $value, p.category = $cat, p.user_id = $uid
                WITH p
                MATCH (u:User {id: $uid})
                MERGE (u)-[:HAS_PREFERENCE]->(p)
                """,
                pid=pref_id,
                value=value,
                cat=category,
                uid=self.user_id,
            )
        return f"Preference saved: [{category.upper()}] {value}"

    # ------------------------------------------------------------------
    # Graph traversal search (used as a tool by the LLM)
    # ------------------------------------------------------------------

    def graph_search(self, entity_name: str, depth: int = 2) -> str:
        with self.driver.session() as s:
            rec = s.run(
                """
                MATCH (start) WHERE toLower(start.name) = toLower($name)
                  AND start.user_id = $uid AND NOT start:User
                OPTIONAL MATCH (start)-[r]-(hop) WHERE NOT hop:User
                RETURN start,
                       collect(DISTINCT {rel: type(r), rt: r.rel_type, n: hop}) AS conns
                LIMIT 1
                """,
                name=entity_name,
                uid=self.user_id,
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
                      AND start.user_id = $uid
                    MATCH (start)-[]-(h1)-[]-(h2)
                    WHERE NOT h2:User AND h2 <> start
                    RETURN DISTINCT h2 LIMIT 8
                    """,
                    name=entity_name,
                    uid=self.user_id,
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
                MATCH (u:User {id: $uid})
                OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(p:Preference)
                RETURN u, collect({v: p.value, cat: p.category}) AS prefs
                """,
                uid=self.user_id,
            ).single()
            user  = dict(p_rec["u"]) if p_rec else {}
            prefs = [x for x in (p_rec["prefs"] if p_rec else []) if x.get("v")]

            # --- Entities: all (scoring will rank them) ---
            ent_rows = s.run(
                """
                MATCH (u:User {id: $uid})-[r:KNOWS]->(e)
                OPTIONAL MATCH (e)-[er:RELATED_TO]-(other)
                  WHERE NOT other:User
                RETURN e, r.relationship AS user_rel,
                       collect(DISTINCT {rt: er.rel_type, nm: other.name}) AS links
                ORDER BY coalesce(e.access_count,0) DESC LIMIT 40
                """,
                uid=self.user_id,
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
                MATCH (u:User {id: $uid})-[:EXPERIENCED]->(ev:Event)
                OPTIONAL MATCH (ent)-[:INVOLVED_IN]->(ev)
                RETURN ev, collect(ent.name) AS involved
                ORDER BY ev.turn DESC LIMIT 30
                """,
                uid=self.user_id,
            )
            events = []
            for row in ev_rows:
                ev = dict(row["ev"])
                ev["_involved"] = [n for n in row["involved"] if n]
                events.append(ev)

            # --- Knowledge: all (scoring selects most relevant) ---
            kn_rows = s.run(
                """
                MATCH (u:User {id: $uid})-[:AWARE_OF]->(k:Knowledge)
                RETURN k ORDER BY k.last_updated_turn DESC LIMIT 20
                """,
                uid=self.user_id,
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
            u_rec = s.run(
                "MATCH (u:User {id: $uid}) RETURN u",
                uid=self.user_id,
            ).single()
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
                "MATCH (u:User {id: $uid})-[r:KNOWS]->(e) "
                "RETURN e, r.relationship AS rel, labels(e)[0] AS lbl",
                uid=self.user_id,
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
                                   if k not in {"name","created_turn","last_seen_turn","access_count","aliases","user_id"}
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
                "MATCH (a)-[r:RELATED_TO]-(b) "
                "WHERE a.user_id = $uid AND b.user_id = $uid "
                "AND NOT a:User AND NOT b:User "
                "RETURN a.name AS an, b.name AS bn, r.rel_type AS rt LIMIT 50",
                uid=self.user_id,
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
                "MATCH (u:User {id: $uid})-[:EXPERIENCED]->(ev:Event) "
                "RETURN ev ORDER BY ev.turn DESC LIMIT 12",
                uid=self.user_id,
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
                    "MATCH (ent)-[:INVOLVED_IN]->(ev:Event {id: $eid, user_id: $uid}) "
                    "RETURN ent.name AS n",
                    eid=nid,
                    uid=self.user_id,
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
                "MATCH (u:User {id: $uid})-[:HAS_PREFERENCE]->(p:Preference) RETURN p",
                uid=self.user_id,
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
                "MATCH (u:User {id: $uid})-[:AWARE_OF]->(k:Knowledge) "
                "RETURN k ORDER BY k.last_updated_turn DESC LIMIT 10",
                uid=self.user_id,
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
                MATCH (u:User {id: $uid})
                OPTIONAL MATCH (u)-[:KNOWS]->(ent)
                OPTIONAL MATCH (u)-[:EXPERIENCED]->(ev:Event)
                OPTIONAL MATCH (u)-[:AWARE_OF]->(kn:Knowledge)
                OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(pr:Preference)
                RETURN u.total_turns AS turns, u.name AS uname,
                       count(DISTINCT ent) AS ec,
                       count(DISTINCT ev)  AS evc,
                       count(DISTINCT kn)  AS kc,
                       count(DISTINCT pr)  AS pc
                """,
                uid=self.user_id,
            ).single()

            breakdown = {
                row["t"]: row["c"]
                for row in s.run(
                    "MATCH (u:User {id: $uid})-[:KNOWS]->(e) "
                    "RETURN labels(e)[0] AS t, count(e) AS c",
                    uid=self.user_id,
                )
            }

            profile = {}
            pu = s.run(
                "MATCH (u:User {id: $uid}) RETURN properties(u) AS p",
                uid=self.user_id,
            ).single()
            if pu:
                skip = {"id", "total_turns", "created_at", "updated_at"}
                profile = {k: v for k, v in dict(pu["p"]).items() if k not in skip and v}

            preferences = [
                {"value": r["v"], "category": r["c"]}
                for r in s.run(
                    "MATCH (u:User {id: $uid})-[:HAS_PREFERENCE]->(p:Preference) "
                    "RETURN p.value AS v, p.category AS c",
                    uid=self.user_id,
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
                  AND e.user_id = $uid
                  AND (toLower(coalesce(e.name,''))    CONTAINS toLower($q)
                    OR toLower(coalesce(e.topic,''))   CONTAINS toLower($q)
                    OR toLower(coalesce(e.content,'')) CONTAINS toLower($q))
                RETURN e, labels(e)[0] AS lbl LIMIT 12
                """,
                q=query,
                uid=self.user_id,
            )
            return [{"node": dict(r["e"]), "label": r["lbl"]} for r in rows]

    # ------------------------------------------------------------------
    # Reset & teardown
    # ------------------------------------------------------------------

    def reset(self):
        """Wipe only this user's memory graph, then recreate their User node."""
        with self.driver.session() as s:
            # Delete all nodes reachable from this user (and the user node itself)
            s.run(
                """
                MATCH (u:User {id: $uid})
                OPTIONAL MATCH (u)-[*1..2]-(n)
                WHERE NOT n:User
                DETACH DELETE n
                """,
                uid=self.user_id,
            )
            s.run(
                "MATCH (u:User {id: $uid}) DETACH DELETE u",
                uid=self.user_id,
            )
            # Also clean up any orphaned user_id-scoped nodes
            s.run(
                """
                MATCH (n) WHERE n.user_id = $uid AND NOT n:User
                DETACH DELETE n
                """,
                uid=self.user_id,
            )
        self._ensure_user_node()
        logger.info("[GraphManager] Graph reset for user=%s", self.user_id)

    def get_timeline(self, limit: int = 100) -> list:
        """Return chronological timeline of events, knowledge, and entities for this user."""
        with self.driver.session() as s:
            events = s.run(
                """
                MATCH (u:User {id: $uid})-[:EXPERIENCED]->(e:Event)
                RETURN 'event' AS type,
                       e.description AS title,
                       e.description AS content,
                       e.date AS date,
                       e.turn AS turn,
                       e.created_at AS created_at
                ORDER BY e.turn ASC
                LIMIT $limit
                """,
                uid=self.user_id, limit=limit,
            ).data()

            knowledge = s.run(
                """
                MATCH (u:User {id: $uid})-[:AWARE_OF]->(k:Knowledge)
                RETURN 'knowledge' AS type,
                       k.topic AS title,
                       k.content AS content,
                       null AS date,
                       k.created_turn AS turn,
                       k.created_at AS created_at
                ORDER BY k.created_turn ASC
                LIMIT $limit
                """,
                uid=self.user_id, limit=limit,
            ).data()

            entities = s.run(
                """
                MATCH (u:User {id: $uid})-[:KNOWS]->(e)
                WHERE e:Person OR e:Place OR e:Organization
                RETURN toLower(labels(e)[0]) AS type,
                       e.name AS title,
                       e.relationship_to_user AS content,
                       null AS date,
                       e.first_seen_turn AS turn,
                       e.created_at AS created_at
                ORDER BY e.first_seen_turn ASC
                LIMIT $limit
                """,
                uid=self.user_id, limit=limit,
            ).data()

        combined = events + knowledge + entities
        combined.sort(key=lambda x: (x.get("turn") or 0))
        return combined[:limit]

    # ------------------------------------------------------------------
    # Cross-session summaries
    # ------------------------------------------------------------------

    def save_session_summary(self, summary: str, start_turn: int, end_turn: int):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (u:User {id: $uid})
                CREATE (u)-[:HAD_SESSION]->(ss:SessionSummary {
                    summary: $summary, start_turn: $start,
                    end_turn: $end, created_at: $now
                })
                """,
                uid=self.user_id, summary=summary,
                start=start_turn, end=end_turn,
                now=datetime.now().isoformat(),
            )

    def get_last_session_summary(self) -> dict | None:
        with self.driver.session() as s:
            rec = s.run(
                """
                MATCH (u:User {id: $uid})-[:HAD_SESSION]->(ss:SessionSummary)
                RETURN ss ORDER BY ss.end_turn DESC LIMIT 1
                """,
                uid=self.user_id,
            ).data()
        return dict(rec[0]["ss"]) if rec else None

    # ------------------------------------------------------------------
    # Proactive nudges — stale entities
    # ------------------------------------------------------------------

    def get_stale_entities(self, turns_threshold: int = 30) -> list:
        """Return entities not mentioned in the last N turns (good for nudges)."""
        current = self._current_turn()
        threshold = max(0, current - turns_threshold)
        with self.driver.session() as s:
            rows = s.run(
                """
                MATCH (u:User {id: $uid})-[r:KNOWS]->(e)
                WHERE coalesce(e.last_seen_turn, e.created_turn, 0) < $threshold
                  AND coalesce(e.access_count, 0) >= 2
                RETURN e.name AS name, r.relationship AS rel, labels(e)[0] AS type,
                       coalesce(e.last_seen_turn, e.created_turn, 0) AS last_seen
                ORDER BY e.access_count DESC LIMIT 4
                """,
                uid=self.user_id, threshold=threshold,
            ).data()
        return [{"name": r["name"], "rel": r["rel"],
                 "type": r["type"], "last_seen": r["last_seen"]} for r in rows]

    # ------------------------------------------------------------------
    # Lightweight graph snapshot (for turn-level memory diff)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Hebbian co-activation learning
    # ------------------------------------------------------------------

    def hebbian_strengthen(self, entity_names: list[str], current_turn: int):
        """
        Strengthen edges between entities that were co-retrieved in the same turn.

        Implements Hebb's rule: "neurons that fire together, wire together."

            w_{ij}^{new} = w_{ij}^{old} + η   for all pairs (i,j)

        where η = HEBBIAN_LEARNING_RATE. When co_activation_count reaches
        HEBBIAN_PROMOTION_THRESH, a permanent RELATED_TO edge is written.

        Runs in a background thread — non-blocking.
        """
        names = [n for n in entity_names if n]
        if len(names) < 2:
            return
        η = config.HEBBIAN_LEARNING_RATE
        thresh = config.HEBBIAN_PROMOTION_THRESH

        with self.driver.session() as s:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    try:
                        s.run(
                            """
                            MATCH (x) WHERE x.name = $a AND x.user_id = $uid
                            MATCH (y) WHERE y.name = $b AND y.user_id = $uid
                            MERGE (x)-[r:RELATED_TO]-(y)
                            ON CREATE SET r.co_activation_count = 1,
                                          r.hebbian_weight = $eta,
                                          r.rel_type = 'co_activated',
                                          r.last_seen = $turn
                            ON MATCH  SET r.co_activation_count =
                                              coalesce(r.co_activation_count, 0) + 1,
                                          r.hebbian_weight =
                                              coalesce(r.hebbian_weight, 0) + $eta,
                                          r.last_seen = $turn
                            """,
                            a=a, b=b, uid=self.user_id,
                            eta=η, turn=current_turn,
                        )
                    except Exception:
                        pass

    def get_snapshot(self) -> dict:
        with self.driver.session() as s:
            rec = s.run(
                """
                MATCH (u:User {id: $uid})
                OPTIONAL MATCH (u)-[:KNOWS]->(ent)
                OPTIONAL MATCH (u)-[:EXPERIENCED]->(ev:Event)
                OPTIONAL MATCH (u)-[:AWARE_OF]->(kn:Knowledge)
                OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(pr:Preference)
                RETURN count(DISTINCT ent) AS ec, count(DISTINCT ev) AS evc,
                       count(DISTINCT kn) AS kc, count(DISTINCT pr) AS pc
                """,
                uid=self.user_id,
            ).single()
        return dict(rec) if rec else {"ec": 0, "evc": 0, "kc": 0, "pc": 0}

    # ------------------------------------------------------------------
    # Contradictions
    # ------------------------------------------------------------------

    def save_contradiction(self, field: str, old_val: str, new_val: str):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (u:User {id: $uid})
                MERGE (u)-[:HAS_CONTRADICTION]->(c:Contradiction {field: $field})
                SET c.old_value = $old, c.new_value = $new,
                    c.detected_at = $now, c.resolved = false
                """,
                uid=self.user_id, field=field,
                old=old_val, new=new_val,
                now=datetime.now().isoformat(),
            )

    def get_pending_contradictions(self) -> list:
        with self.driver.session() as s:
            rows = s.run(
                """
                MATCH (u:User {id: $uid})-[:HAS_CONTRADICTION]->(c:Contradiction {resolved: false})
                RETURN c ORDER BY c.detected_at DESC LIMIT 5
                """,
                uid=self.user_id,
            ).data()
        return [dict(r["c"]) for r in rows]

    def resolve_contradiction(self, field: str):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (u:User {id: $uid})-[:HAS_CONTRADICTION]->(c:Contradiction {field: $field})
                SET c.resolved = true
                """,
                uid=self.user_id, field=field,
            )

    # ------------------------------------------------------------------
    # Raw profile read (for contradiction detection)
    # ------------------------------------------------------------------

    def get_profile_raw(self) -> dict:
        with self.driver.session() as s:
            rec = s.run(
                "MATCH (u:User {id: $uid}) RETURN properties(u) AS p",
                uid=self.user_id,
            ).single()
        if not rec:
            return {}
        skip = {"id", "total_turns", "created_at", "updated_at"}
        return {k: v for k, v in dict(rec["p"]).items() if k not in skip and v}

    def delete_user_graph(self):
        """Permanently delete all graph data for this user (called on account deletion)."""
        with self.driver.session() as s:
            s.run(
                """
                MATCH (n) WHERE n.user_id = $uid OR (n:User AND n.id = $uid)
                DETACH DELETE n
                """,
                uid=self.user_id,
            )
        logger.info("[GraphManager] All graph data deleted for user=%s", self.user_id)

    def close(self):
        self.driver.close()
        logger.info("[GraphManager] Driver closed for user=%s", self.user_id)
