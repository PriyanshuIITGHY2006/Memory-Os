"""
MemoryOS v2 — Archival (Vector) Memory Manager (multi-tenant)
==============================================================
ChromaDB for semantic / episodic memory search.
Works alongside Neo4j: ChromaDB finds *what is similar*,
Neo4j answers *what is connected*.

Each user gets two isolated ChromaDB collections:
  mem_{user_id[:16]}   : standalone facts / knowledge snippets
  eps_{user_id[:16]}   : full conversation episodes (user + assistant)

Surprise-Weighted Storage (Adaptive Forgetting):
  At write time, surprise is computed as the cosine distance to the
  nearest existing memory in the embedding space:

      surprise(m_new) = min(dist(m_new, m_j)) for all j ∈ collection

  ChromaDB cosine distance ∈ [0, 2], normalised to [0, 1].
  surprise = 1 → completely novel  → retained longer (lower decay rate)
  surprise = 0 → exact duplicate   → refreshed, not re-stored

  This implements information-theoretic salience: memories with higher
  KL-divergence from the existing belief state decay more slowly.
"""

import uuid
import chromadb
import config


class ArchivalMemoryManager:

    def __init__(self, user_id: str):
        self.user_id = user_id
        # Truncate user_id to keep collection names within ChromaDB's 63-char limit
        uid_slug = user_id.replace("-", "")[:16]

        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))

        self.semantic = self.client.get_or_create_collection(
            name=f"mem_{uid_slug}",
            metadata={"hnsw:space": "cosine"},
        )
        self.episodic = self.client.get_or_create_collection(
            name=f"eps_{uid_slug}",
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _compute_surprise(self, collection, content: str) -> float:
        """
        Surprise = cosine distance to nearest neighbour in the collection.
        Normalised to [0, 1] (ChromaDB cosine distance ∈ [0, 2]).

        surprise → 1: completely novel information (high salience)
        surprise → 0: near-duplicate of existing memory (low salience)
        """
        try:
            count = collection.count()
            if count == 0:
                return 1.0  # first memory ever — maximally surprising
            results = collection.query(
                query_texts=[content], n_results=1,
                include=["distances"],
            )
            if results["distances"] and results["distances"][0]:
                raw_dist = results["distances"][0][0]
                return min(raw_dist, 1.0)   # cap at 1 (cosine dist can exceed 1 for opposite vecs)
        except Exception:
            pass
        return 0.5  # safe default

    def add_memory(self, content: str, role: str, turn_number: int) -> str:
        """Store a single utterance in the episodic log with surprise metadata."""
        surprise   = self._compute_surprise(self.episodic, content)
        memory_id  = f"ep_{uuid.uuid4().hex[:10]}"
        self.episodic.add(
            documents=[content],
            metadatas=[{
                "role":        role,
                "turn_number": turn_number,
                "origin_turn": turn_number,
                "surprise":    round(surprise, 4),
                "consolidated": False,
            }],
            ids=[memory_id],
        )
        return memory_id

    def add_fact(self, content: str, turn_number: int, confidence: float = 1.0) -> tuple:
        """
        Store a fact with deduplication + surprise scoring.
        Returns (memory_id, status_string).
        """
        # Deduplication: if very similar fact already exists, refresh it
        results = self.semantic.query(query_texts=[content], n_results=1,
                                      include=["distances", "metadatas"])
        if (results["ids"] and results["ids"][0]
                and results["distances"][0][0] < 0.12):
            existing_id = results["ids"][0][0]
            meta = results["metadatas"][0][0]
            meta["last_used_turn"] = turn_number
            meta["count"] = meta.get("count", 1) + 1
            self.semantic.update(ids=[existing_id], metadatas=[meta])
            return existing_id, f"Refreshed existing memory {existing_id}"

        surprise  = self._compute_surprise(self.semantic, content)
        memory_id = f"fact_{uuid.uuid4().hex[:10]}"
        self.semantic.add(
            documents=[content],
            metadatas=[{
                "type":         "fact",
                "origin_turn":  turn_number,
                "last_used_turn": turn_number,
                "count":        1,
                "confidence":   confidence,
                "surprise":     round(surprise, 4),
                "consolidated": False,
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

    def get_consolidation_candidates(
        self, min_surprise: float, min_age_turns: int, current_turn: int, limit: int = 10
    ) -> list[dict]:
        """
        Return high-surprise, old-enough episodic memories that haven't been
        consolidated into the knowledge graph yet.

        Used by the CLS Consolidation Engine.
        """
        try:
            results = self.episodic.get(
                where={
                    "$and": [
                        {"consolidated": {"$eq": False}},
                        {"surprise":     {"$gte": min_surprise}},
                    ]
                },
                include=["documents", "metadatas"],
                limit=limit * 3,   # over-fetch then filter by age
            )
        except Exception:
            return []

        candidates = []
        for i, doc in enumerate(results.get("documents") or []):
            meta   = (results.get("metadatas") or [[]])[i] if results.get("metadatas") else {}
            mem_id = (results.get("ids") or [[]])[i] if results.get("ids") else f"ep_{i}"
            age    = current_turn - (meta.get("origin_turn") or meta.get("turn_number") or 0)
            if age >= min_age_turns:
                candidates.append({
                    "id":       mem_id,
                    "content":  doc,
                    "surprise": meta.get("surprise", 0.5),
                    "turn":     meta.get("origin_turn", 0),
                })
        # Sort by surprise descending, take top `limit`
        candidates.sort(key=lambda x: x["surprise"], reverse=True)
        return candidates[:limit]

    def mark_consolidated(self, memory_ids: list[str]):
        """Flag memories as consolidated so they aren't re-processed."""
        for mid in memory_ids:
            try:
                self.episodic.update(ids=[mid], metadatas=[{"consolidated": True}])
            except Exception:
                pass

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

    # ------------------------------------------------------------------
    # Account deletion
    # ------------------------------------------------------------------

    def delete_user_collections(self):
        """Permanently remove both collections for this user."""
        for col in (self.semantic, self.episodic):
            try:
                self.client.delete_collection(col.name)
            except Exception:
                pass
