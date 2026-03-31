import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 1. PROJECT ROOT
BASE_DIR = Path(__file__).resolve().parent

# 2. CHROMADB PATH (vector search — kept alongside Neo4j)
DATABASE_DIR = BASE_DIR / "database"
CHROMA_DB_DIR = DATABASE_DIR / "chroma_db"

# 3. API KEYS
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # set in .env

# 4. LLM MODEL
LLM_MODEL = "llama-3.3-70b-versatile"

# 5. NEO4J — graph database for structured memory
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")  # change in .env

# 6. MEMORY TUNING
BUFFER_MAX_TURNS          = 15    # rolling conversation window (hot buffer)
ARCHIVAL_RELEVANCE_CUTOFF = 0.75  # cosine similarity threshold (0–1, higher = stricter)

# 7. TOKEN BUDGET — character limits per context section (chars ÷ 4 ≈ tokens)
#    Total budget: ~3 200 chars → ~800 tokens for structured memory context.
#    Fixed regardless of conversation length → supports 2 000–4 000+ turns.
TOKEN_BUDGET_PROFILE  = 400    # ~100 tokens
TOKEN_BUDGET_PREFS    = 500    # ~125 tokens
TOKEN_BUDGET_ENTITIES = 2000   # ~500 tokens  (largest section)
TOKEN_BUDGET_EVENTS   = 1200   # ~300 tokens
TOKEN_BUDGET_KNOWLEDGE= 1400   # ~350 tokens
TOKEN_BUDGET_ARCHIVAL = 800    # ~200 tokens

# 8. RELEVANCE SCORING WEIGHTS  (must sum to 1.0)
#    score(m|Q) = α·keyword(Q,m) + β·recency(t,m) + γ·frequency(m)
RELEVANCE_KEYWORD_WEIGHT = 0.50   # α — what is relevant to the current query
RELEVANCE_RECENCY_WEIGHT = 0.30   # β — what was mentioned recently
RELEVANCE_FREQ_WEIGHT    = 0.20   # γ — what is accessed most often
RELEVANCE_DECAY_LAMBDA   = 0.001  # λ — per-turn exponential decay constant

# 9. CONTEXT CAPS — hard limits for each section
MAX_ENTITIES_IN_PROMPT  = 20
MAX_EVENTS_IN_PROMPT    = 12
MAX_KNOWLEDGE_IN_PROMPT = 10