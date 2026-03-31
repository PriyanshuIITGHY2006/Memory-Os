"""
MemoryOS v2 — One-Time Migration: JSON → Neo4j
================================================
Reads the old database/user_state.json and seeds Neo4j with:
  - user_profile      → User node properties + Preference nodes
  - entities          → Person / Place / Organization nodes + KNOWS edges
  - events            → Event nodes + EXPERIENCED edges
  - knowledge_base    → Knowledge nodes + AWARE_OF edges

Run once: python migrate_to_neo4j.py
After success, the JSON file is no longer used.
"""

import json
import sys
from pathlib import Path

# ── Locate legacy JSON file ───────────────────────────────────────────────────

LEGACY_JSON = Path(__file__).parent / "database" / "user_state.json"

if not LEGACY_JSON.exists():
    print(f"No legacy file found at {LEGACY_JSON}. Nothing to migrate.")
    sys.exit(0)

with open(LEGACY_JSON) as f:
    state = json.load(f)

print(f"Loaded {LEGACY_JSON}")

# ── Connect to Neo4j via GraphManager ────────────────────────────────────────

from backend.managers.graph_manager import GraphManager

gm = GraphManager()
print("Connected to Neo4j.")

# ── 1. User profile ───────────────────────────────────────────────────────────

profile = state.get("user_profile", {})
n_profile = 0

for key, value in profile.items():
    if key in ("preferences", "goals") or value is None:
        continue
    gm.update_profile(key, str(value))
    n_profile += 1

# Preferences list → Preference nodes
for pref_val in profile.get("preferences", []):
    gm.add_preference(pref_val, "preference")

# Allergies list (stored under "allergies" key in some versions)
for allergy in profile.get("allergies", []):
    gm.add_preference(allergy, "allergy")

print(f"  Profile: {n_profile} fields migrated, "
      f"{len(profile.get('preferences', []))} preferences, "
      f"{len(profile.get('allergies', []))} allergies.")

# ── 2. Entities ───────────────────────────────────────────────────────────────

entities = state.get("entities", {})
for key, data in entities.items():
    name         = data.get("name", key)
    relationship = data.get("relationship", "known")
    attributes   = data.get("attributes", {})
    gm.update_entity(
        name=name,
        relationship=relationship,
        attributes=attributes,
        entity_type="person",  # legacy didn't store type; assume person
    )

print(f"  Entities: {len(entities)} migrated.")

# ── 3. Events ─────────────────────────────────────────────────────────────────

# Events are logged individually; we manually set the turn counter so that
# the logged event gets the right turn number embedded.
events = state.get("events", [])
for ev in events:
    desc = ev.get("description", "")
    date = ev.get("date", "")
    turn = ev.get("turn", 0)
    if not desc:
        continue
    # Temporarily set the user's turn counter so log_event picks it up
    with gm.driver.session() as s:
        s.run(
            "MATCH (u:User {id:'main'}) SET u.total_turns = $t",
            t=turn,
        )
    gm.log_event(description=desc, date=date)

# Restore actual total turns
stats = state.get("system_stats", {})
total_turns = stats.get("total_turns", len(events))
with gm.driver.session() as s:
    s.run(
        "MATCH (u:User {id:'main'}) SET u.total_turns = $t",
        t=total_turns,
    )

print(f"  Events: {len(events)} migrated. Turn counter restored to {total_turns}.")

# ── 4. Knowledge base ─────────────────────────────────────────────────────────

knowledge = state.get("knowledge_base", {})
for topic, content in knowledge.items():
    gm.add_general_knowledge(topic, str(content))

print(f"  Knowledge: {len(knowledge)} items migrated.")

# ── Done ─────────────────────────────────────────────────────────────────────

gm.close()
print(
    f"\nMigration complete!\n"
    f"  Verify in Neo4j Browser: http://localhost:7474\n"
    f"  Query: MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count\n"
)
