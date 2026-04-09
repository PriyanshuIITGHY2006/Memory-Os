"""
MemoryOS — Migration Script: Single-User → Multi-User
======================================================
Run this ONCE after upgrading to the multi-user branch if you have existing
single-user data in Neo4j (the old `User {id: 'main'}` schema).

What this script does
---------------------
1. Drops the old global uniqueness constraints that conflict with the new
   composite (user_id, name) constraints.
2. Adds `user_id = 'main'` to all existing entity, knowledge, event, and
   preference nodes so they are scoped to the legacy user.
3. Creates the new composite constraints.

Usage
-----
  python migrate_to_multiuser.py

The script is idempotent — safe to run multiple times.
"""

import sys
from neo4j import GraphDatabase
import config

OLD_CONSTRAINTS = [
    "DROP CONSTRAINT person_name      IF EXISTS",
    "DROP CONSTRAINT place_name       IF EXISTS",
    "DROP CONSTRAINT org_name         IF EXISTS",
    "DROP CONSTRAINT knowledge_topic  IF EXISTS",
]

NEW_CONSTRAINTS = [
    "CREATE CONSTRAINT person_user    IF NOT EXISTS FOR (p:Person)       REQUIRE (p.user_id, p.name) IS UNIQUE",
    "CREATE CONSTRAINT place_user     IF NOT EXISTS FOR (p:Place)        REQUIRE (p.user_id, p.name) IS UNIQUE",
    "CREATE CONSTRAINT org_user       IF NOT EXISTS FOR (o:Organization) REQUIRE (o.user_id, o.name) IS UNIQUE",
    "CREATE CONSTRAINT knowledge_user IF NOT EXISTS FOR (k:Knowledge)    REQUIRE (k.user_id, k.topic) IS UNIQUE",
]

BACKFILL_QUERIES = [
    # Entities
    "MATCH (n:Person)       WHERE n.user_id IS NULL SET n.user_id = 'main'",
    "MATCH (n:Place)        WHERE n.user_id IS NULL SET n.user_id = 'main'",
    "MATCH (n:Organization) WHERE n.user_id IS NULL SET n.user_id = 'main'",
    # Knowledge
    "MATCH (n:Knowledge)    WHERE n.user_id IS NULL SET n.user_id = 'main'",
    # Events
    "MATCH (n:Event)        WHERE n.user_id IS NULL SET n.user_id = 'main'",
    # Preferences
    "MATCH (n:Preference)   WHERE n.user_id IS NULL SET n.user_id = 'main'",
]


def run_migration():
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )
    with driver.session() as s:
        print("Step 1 — Dropping old single-property constraints…")
        for q in OLD_CONSTRAINTS:
            try:
                s.run(q)
                print(f"  OK: {q}")
            except Exception as e:
                print(f"  WARN: {e}")

        print("\nStep 2 — Backfilling user_id='main' on legacy nodes…")
        for q in BACKFILL_QUERIES:
            try:
                result = s.run(q)
                summary = result.consume()
                print(f"  OK ({summary.counters.properties_set} props set): {q[:60]}…")
            except Exception as e:
                print(f"  WARN: {e}")

        print("\nStep 3 — Creating new composite constraints…")
        for q in NEW_CONSTRAINTS:
            try:
                s.run(q)
                print(f"  OK: {q}")
            except Exception as e:
                print(f"  WARN (may already exist): {e}")

    driver.close()
    print("\nMigration complete.")


if __name__ == "__main__":
    print("MemoryOS Multi-User Migration")
    print("=" * 40)
    ans = input("This will modify your Neo4j database. Continue? [y/N] ").strip().lower()
    if ans != "y":
        print("Aborted.")
        sys.exit(0)
    run_migration()
