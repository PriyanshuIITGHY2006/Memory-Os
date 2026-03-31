"""
MemoryOS v2 — Full Reset
=========================
Clears:
  1. Neo4j graph  (wipes all nodes / relationships, recreates User node)
  2. ChromaDB     (deletes the chroma_db/ directory entirely)

Run: python reset_state.py
"""

import shutil
import sys
import config


# ── Neo4j reset ────────────────────────────────────────────────────────────────

print("Resetting Neo4j graph…")
try:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        # Recreate the singleton User node
        session.run(
            "MERGE (u:User {id:'main'}) "
            "ON CREATE SET u.total_turns = 0"
        )
    driver.close()
    print("  Neo4j: cleared (User node recreated).")
except Exception as exc:
    print(f"  Neo4j reset failed: {exc}")
    print("  Is Neo4j running? Check NEO4J_URI / NEO4J_PASSWORD in .env")


# ── ChromaDB reset ────────────────────────────────────────────────────────────

print("Resetting ChromaDB…")
chroma_dir = config.CHROMA_DB_DIR
if chroma_dir.exists():
    try:
        shutil.rmtree(chroma_dir)
        print(f"  ChromaDB: deleted {chroma_dir}")
    except Exception as exc:
        print(f"  ChromaDB delete failed: {exc}")
else:
    print("  ChromaDB: already empty.")


print("\nReset complete. Restart the server.")
