"""
MemoryOS v2 — FastAPI Backend
================================
Endpoints:
  POST /chat          — main conversation
  GET  /graph         — full knowledge graph (nodes + edges) for visualization
  GET  /stats         — system statistics (entity/event/knowledge counts)
  GET  /search?q=...  — text search across graph entities
  GET  /health        — liveness check
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os

from backend.logic.orchestrator import Orchestrator

app = FastAPI(title="MemoryOS v2", version="2.0.0")

# Allow Streamlit frontend (same machine, different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator()


# ── Request / Response models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return FileResponse(html_path, media_type="text/html")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response_text, active_memories, debug_prompt = orchestrator.process_message(
        request.message
    )
    return {
        "response":        response_text,
        "active_memories": active_memories,
        "debug_prompt":    debug_prompt,
    }


@app.get("/graph")
async def get_graph():
    """Return the full knowledge graph for the frontend visualization."""
    return orchestrator.graph.get_graph_data()


@app.get("/stats")
async def get_stats():
    """Return system statistics for the analytics dashboard."""
    raw = orchestrator.graph.get_stats()
    breakdown = raw.get("entity_breakdown", {})
    prefs_list = raw.get("preferences", [])
    pref_counts = {}
    for p in prefs_list:
        cat = (p.get("category") or "preference").lower()
        pref_counts[cat] = pref_counts.get(cat, 0) + 1
    return {
        "turns":        raw.get("total_turns", 0),
        "entities":     raw.get("entity_count", 0),
        "events":       raw.get("event_count", 0),
        "knowledge":    raw.get("knowledge_count", 0),
        "preferences":  raw.get("pref_count", 0),
        "people":       breakdown.get("Person", 0),
        "places":       breakdown.get("Place", 0),
        "organizations":breakdown.get("Organization", 0),
        "pref_preference": pref_counts.get("preference", 0),
        "pref_goal":       pref_counts.get("goal", 0),
        "pref_allergy":    pref_counts.get("allergy", 0),
        "pref_constraint": pref_counts.get("constraint", 0),
        "profile":      raw.get("profile", {}),
    }


@app.get("/search")
async def search(q: str = Query(..., min_length=1)):
    """Text search across graph nodes."""
    return orchestrator.graph.search_entities(q)


@app.get("/duplicates")
async def get_duplicates():
    """Detect potentially duplicate entity nodes using multi-signal analysis."""
    return orchestrator.graph.detect_duplicates()


class MergeRequest(BaseModel):
    canonical: str
    alias: str


@app.post("/merge")
async def merge_entities(request: MergeRequest):
    """Merge alias entity into canonical entity (entity resolution)."""
    result = orchestrator.graph.merge_entities(request.canonical, request.alias)
    return {"result": result}


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting MemoryOS v2 server…")
    uvicorn.run(app, host="0.0.0.0", port=8000)
