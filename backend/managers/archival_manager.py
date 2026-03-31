"""
MemoryOS v2 — Archival (Vector) Memory Manager
================================================
ChromaDB for semantic / episodic memory search.
Works alongside Neo4j: ChromaDB finds *what is similar*,
Neo4j answers *what is connected*.

Collections
-----------
semantic_memory   : standalone facts / knowledge snippets
conversation_logs : full conversation episodes (user + assistant)
"""

import uuid
import chromadb
import config


class ArchivalMemoryManager:

    def __init__(self):
        print(f"[ArchivalManager] Connecting to ChromaDB at: {config.CHROMA_DB_DIR}")
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))

        self.semantic = self.client.get_or_create_collection(
            name="semantic_memory",
            metadata={"hnsw:space": "cosine"},
        )
        self.episodic = self.client.get_or_create_collection(
            name="conversation_logs",
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_memory(self, content: str, role: str, turn_number: int) -> str:
        """Store a single utterance in the episodic log."""
        memory_id = f"ep_{uuid.uuid4().hex[:10]}"
        self.episodic.add(
            documents=[content],
            metadatas=[{
                "role": role,
                "turn_number": turn_number,
                "origin_turn": turn_number,
            }],
            ids=[memory_id],
        )
        return memory_id

    def add_fact(self, content: str, turn_number: int, confidence: float = 1.0) -> tuple:
        """
        Store a fact with deduplication.
        Returns (memory_id, status_string).
        """
        # Deduplication: if very similar fact already exists, refresh it
        results = self.semantic.query(query_texts=[content], n_results=1)
        if (results["ids"] and results["ids"][0]
                and results["distances"][0][0] < 0.12):
            existing_id = results["ids"][0][0]
            meta = results["metadatas"][0][0]
            meta["last_used_turn"] = turn_number
            meta["count"] = meta.get("count", 1) + 1
            self.semantic.update(ids=[existing_id], metadatas=[meta])
            return existing_id, f"Refreshed existing memory {existing_id}"

        memory_id = f"fact_{uuid.uuid4().hex[:10]}"
        self.semantic.add(
            documents=[content],
            metadatas=[{
                "type": "fact",
                "origin_turn": turn_number,
                "last_used_turn": turn_number,
                "count": 1,
                "confidence": confidence,
            }],
            ids=[memory_id],
        )
        return memory_id, "New fact stored"

    def add_episode(self, user_msg: str, bot_msg: str, turn_number: int):
        """Log a complete conversation exchange."""
        content = f"User: {user_msg}\nAssistant: {bot_msg}"
        ep_id = f"ep_{uuid.uuid4().hex[:10]}"
        self.episodic.add(
            documents=[content],
            metadatas=[{"turn": turn_number}],
            ids=[ep_id],
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_memory(self, query: str, n_results: int = 4) -> list:
        """
        Search *both* semantic facts and episodic logs.
        Returns a merged, distance-sorted list.
        """
        cutoff = 1.0 - config.ARCHIVAL_RELEVANCE_CUTOFF   # convert similarity → distance

        memories = []

        # Semantic facts
        sem = self.semantic.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )
        if sem["ids"] and sem["ids"][0]:
            for i, doc in enumerate(sem["documents"][0]):
                dist = sem["distances"][0][i]
                if dist > cutoff:
                    continue
                meta = sem["metadatas"][0][i]
                memories.append({
                    "memory_id":   sem["ids"][0][i],
                    "content":     doc,
                    "origin_turn": meta.get("origin_turn", 0),
                    "last_used":   meta.get("last_used_turn", 0),
                    "distance":    dist,
                    "source":      "semantic",
                })

        # Episodic logs
        epi = self.episodic.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )
        if epi["ids"] and epi["ids"][0]:
            for i, doc in enumerate(epi["documents"][0]):
                dist = epi["distances"][0][i]
                if dist > cutoff:
                    continue
                meta = epi["metadatas"][0][i]
                memories.append({
                    "memory_id":   epi["ids"][0][i],
                    "content":     doc,
                    "origin_turn": meta.get("origin_turn", meta.get("turn", 0)),
                    "last_used":   0,
                    "distance":    dist,
                    "source":      "episodic",
                })

        memories.sort(key=lambda m: m["distance"])
        return memories[:n_results]

    def retrieve_relevant_context(
        self, query: str, turn_number: int, n_results: int = 4
    ) -> list:
        """Search + update last_used_turn on retrieved facts."""
        memories = self.search_memory(query, n_results)
        for mem in memories:
            if mem["source"] == "semantic":
                try:
                    self.semantic.update(
                        ids=[mem["memory_id"]],
                        metadatas=[{"last_used_turn": turn_number}],
                    )
                except Exception:
                    pass
        return memories
