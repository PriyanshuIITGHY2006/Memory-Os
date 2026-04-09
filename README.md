# MemoryOS

**AI that remembers everything about you.**

MemoryOS is a production-ready, multi-tenant personal memory system. Every conversation is stored in a living knowledge graph that grows smarter over time. Research-grade memory algorithms surface exactly the right context at every turn — within a constant token budget.

---

## Features

### Memory Architecture
- **Neo4j Knowledge Graph** — entities, relationships, events, preferences, and session summaries as a traversable graph
- **ChromaDB Vector Store** — semantic episodic and factual memory with cosine-similarity retrieval
- **Surprise-Weighted Adaptive Forgetting** — novel memories decay 3× slower: `λ_eff = λ · (1 − α · σ)`
- **Personalized PageRank Spreading Activation** — 4-iteration PPR over entity graph surfaces implicit connections
- **Hebbian Co-activation Learning** — edge weights strengthen between frequently co-retrieved entities
- **CLS Consolidation Engine** — background daemon promotes high-surprise episodes into structured graph nodes (McClelland et al., 1995)

### Chat & Streaming
- SSE streaming chat with real-time token-by-token responses
- WebSocket live graph push broadcast after each turn
- Memory diff chips showing exactly what was stored after each message
- **Ask Past Self mode** — search only archival memory, bypassing the live graph
- Voice input via Web Speech API with error toast notifications

### File & Media Memory
- **Photos** — Groq Vision (`llama-3.2-11b-vision-preview`) describes images in rich detail and stores them as memories
- **PDFs** — `pdfplumber` + `pypdf` extract and chunk text into searchable archival memories
- **Audio** — Groq Whisper (`whisper-large-v3`) transcribes audio files
- **Text / CSV / DOCX** — direct text extraction into memory
- File library with authenticated thumbnail grid, lightbox viewer, and delete
- Attach previously uploaded files to any chat message using only the summary — O(1) token cost

### Analytics & Insights
- **Knowledge gaps** — LLM audits the graph and surfaces 5 missing information areas with suggested questions
- **Contradiction detection** — background thread raises conflicts between profile values
- **Memory timeline** — chronological view of all entities, events, and knowledge grouped by turn
- **Annual PDF report** — 7-page "Year in Memory" typeset report

### Auth & Security
- JWT access tokens (15 min) + httpOnly refresh cookies (30 days) with silent rotation
- Groq API keys encrypted at rest with Fernet
- Per-user data isolation across Neo4j, ChromaDB, and the filesystem
- Google and GitHub OAuth
- Rate limiting via `slowapi`

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`, `llama-3.2-11b-vision-preview`, `whisper-large-v3`) |
| Graph DB | Neo4j AuraDB |
| Vector DB | ChromaDB (persistent) |
| Backend | FastAPI + uvicorn |
| Auth | python-jose, passlib, httpx OAuth |
| File parsing | pdfplumber, pypdf, python-docx |
| PDF export | reportlab |
| Frontend | Vanilla JS, single HTML file |

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Neo4j AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/) free instance
- [Groq](https://console.groq.com) API key

### 1. Clone and install

```bash
git clone https://github.com/your-username/memory-os.git
cd memory-os
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

JWT_SECRET=your-64-char-secret
ENCRYPTION_KEY=your-fernet-key

APP_BASE_URL=http://localhost:8001

# Optional — OAuth
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
```

Generate a Fernet encryption key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 3. Run

```bash
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8001 --reload
```

Open **http://localhost:8001** in Chrome or Edge.

> **Note:** Use `localhost`, not `0.0.0.0` — the Web Speech API requires a secure context.

### 4. Register

Create an account and enter your Groq API key. It is encrypted before storage and never logged.

---

## Docker

```bash
docker compose up --build
```

---

## MCP Server (Claude Desktop)

MemoryOS exposes a stdio MCP server so Claude Desktop can read and write your memories directly.

```bash
python backend/mcp_server.py
```

Copy `mcp_config_example.json` to your Claude Desktop config directory and update the path.

**Available tools:** `recall_memories`, `get_user_profile`, `get_recent_events`, `get_entities`, `save_memory`

---

## Memory Algorithm

At each turn the system runs a 6-stage pipeline:

```
1. Turn increment + session tracking
2. Surprise-weighted archival retrieval        σ = cosine distance to nearest neighbour
3. Personalized PageRank spreading activation  4 iterations, damping d=0.25
4. Greedy relevance-scored context assembly    α=0.45 keyword · β=0.28 recency · γ=0.18 freq · δ=0.09 surprise
5. LLM inference + tool dispatch + contradiction detection
6. Hebbian edge strengthening + CLS consolidation tick  (background threads)
```

Context budget is **constant** regardless of conversation length — O(1) tokens per turn.

---

## Project Structure

```
memory-os/
├── backend/
│   ├── auth/                        # JWT, OAuth, password hashing
│   ├── logic/
│   │   ├── orchestrator.py          # Main turn pipeline
│   │   └── prompts.py               # System prompt templates
│   └── managers/
│       ├── archival_manager.py      # ChromaDB — episodic + semantic memory
│       ├── buffer_manager.py        # In-context conversation buffer
│       ├── context_manager.py       # Relevance scoring + PPR spreading activation
│       ├── consolidation_engine.py  # CLS background consolidation
│       ├── file_manager.py          # File upload, vision, transcription
│       ├── graph_manager.py         # Neo4j — entities, events, preferences
│       ├── report_manager.py        # PDF annual report generation
│       ├── session_manager.py       # Idle-based session summarisation
│       └── user_manager.py          # Multi-tenant user management
│   ├── mcp_server.py                # Claude Desktop MCP integration
│   └── server.py                    # FastAPI app + all endpoints
├── frontend/
│   └── index.html                   # Full SPA — chat, graph, analytics, files
├── config.py                        # All tunable hyperparameters
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── mcp_config_example.json
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Get access + refresh tokens |
| POST | `/auth/refresh` | Rotate access token |
| POST | `/chat` | Send message (non-streaming) |
| POST | `/chat/stream` | Send message (SSE streaming) |
| GET | `/ws` | WebSocket live graph updates |
| GET | `/graph` | Full knowledge graph |
| GET | `/stats` | Memory statistics |
| GET | `/timeline` | Chronological memory timeline |
| GET | `/search?q=` | Semantic memory search |
| POST | `/upload` | Upload file → stored as memory |
| GET | `/files` | List uploaded files |
| GET | `/files/{id}` | Serve file (authenticated) |
| DELETE | `/files/{id}` | Delete file |
| GET | `/gaps` | Knowledge gap analysis |
| GET | `/ask-past` | Query archival memory only |
| GET | `/contradictions` | Detected contradictions |
| GET | `/export/report` | Download PDF annual report |

---

## License

MIT
