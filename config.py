import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
CHROMA_DB_DIR= DATABASE_DIR / "chroma_db"
USERS_DB_PATH= DATABASE_DIR / "users.db"     # SQLite for account records

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"

# ── Neo4j ─────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ── Auth — JWT ────────────────────────────────────────────────────────────────
JWT_SECRET          = os.getenv("JWT_SECRET", "change-me-in-production-64-chars-min")
JWT_ALGORITHM       = "HS256"
JWT_ACCESS_EXPIRE_MINUTES  = 15          # short-lived access token
JWT_REFRESH_EXPIRE_DAYS    = 30          # long-lived refresh token (httpOnly cookie)

# ── Auth — encryption (Fernet) for stored Groq API keys ──────────────────────
# Generate once:  from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")

# ── Auth — OAuth providers ────────────────────────────────────────────────────
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID",     "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GITHUB_CLIENT_ID     = os.getenv("GITHUB_CLIENT_ID",     "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")

# Base URL used to build OAuth callback URIs (e.g. https://yourdomain.com)
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")

# ── Memory tuning ─────────────────────────────────────────────────────────────
BUFFER_MAX_TURNS          = 15
ARCHIVAL_RELEVANCE_CUTOFF = 0.75

# ── Token budgets (chars; chars ÷ 4 ≈ tokens) ────────────────────────────────
TOKEN_BUDGET_PROFILE   = 400
TOKEN_BUDGET_PREFS     = 500
TOKEN_BUDGET_ENTITIES  = 2000
TOKEN_BUDGET_EVENTS    = 1200
TOKEN_BUDGET_KNOWLEDGE = 1400
TOKEN_BUDGET_ARCHIVAL  = 800

# ── Relevance weights (must sum to 1.0) ───────────────────────────────────────
RELEVANCE_KEYWORD_WEIGHT  = 0.45
RELEVANCE_RECENCY_WEIGHT  = 0.28
RELEVANCE_FREQ_WEIGHT     = 0.18
RELEVANCE_SURPRISE_WEIGHT = 0.09   # salience bonus for high-surprise memories
RELEVANCE_DECAY_LAMBDA    = 0.001

# ── Adaptive forgetting — surprise modulates decay rate ───────────────────────
# λ_eff = λ · (1 − SURPRISE_RETENTION_ALPHA · surprise)
# surprise=1 → λ_eff = λ·(1−0.70) = 0.30λ  (retained ~3× longer)
# surprise=0 → λ_eff = λ                    (normal decay)
SURPRISE_RETENTION_ALPHA  = 0.70

# ── Personalized PageRank spreading activation ─────────────────────────────────
PPR_DAMPING               = 0.25   # d: fraction of activation that spreads
PPR_ITERATIONS            = 4      # k: convergence iterations

# ── Hebbian co-activation learning ────────────────────────────────────────────
HEBBIAN_LEARNING_RATE     = 0.10   # η: edge weight increment per co-activation
HEBBIAN_DECAY             = 0.005  # per-turn weight decay on unused edges
HEBBIAN_PROMOTION_THRESH  = 3      # min co-activations before edge is written

# ── CLS Consolidation Engine ──────────────────────────────────────────────────
CLS_MIN_SURPRISE          = 0.65   # min surprise to be a consolidation candidate
CLS_MIN_AGE_TURNS         = 10     # memory must be at least N turns old
CLS_CHECK_INTERVAL_TURNS  = 20     # run consolidation every N turns

# ── Context caps ──────────────────────────────────────────────────────────────
MAX_ENTITIES_IN_PROMPT  = 20
MAX_EVENTS_IN_PROMPT    = 12
MAX_KNOWLEDGE_IN_PROMPT = 10