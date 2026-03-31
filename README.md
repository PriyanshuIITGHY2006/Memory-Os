# MemoryOS v2 — Persistent Long-Term Memory for Large Language Models

> A production-grade memory architecture that gives stateless LLMs the ability to remember, reason, and recall across thousands of conversation turns — without ever exceeding a fixed token budget.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Core Problem — LLM Stationarity](#2-the-core-problem--llm-stationarity)
3. [System Architecture](#3-system-architecture)
4. [Mathematical Foundations](#4-mathematical-foundations)
5. [Knowledge Graph Design](#5-knowledge-graph-design)
6. [Memory Retrieval Pipeline](#6-memory-retrieval-pipeline)
7. [Compact Serialization Format](#7-compact-serialization-format)
8. [Entity Resolution Engine](#8-entity-resolution-engine)
9. [Storage Layer](#9-storage-layer--neo4j-and-chromadb)
10. [LLM Tool Schema](#10-llm-tool-schema)
11. [API Reference](#11-api-reference)
12. [Frontend Architecture](#12-frontend-architecture)
13. [Installation and Setup](#13-installation-and-setup)
14. [Configuration Reference](#14-configuration-reference)
15. [Usage Examples](#15-usage-examples)
16. [Performance Benchmarks](#16-performance-benchmarks)
17. [Design Decisions and Trade-offs](#17-design-decisions-and-trade-offs)
18. [Limitations and Future Work](#18-limitations-and-future-work)
19. [Collaborators](#19-collaborators)

---

## 1. Project Overview

MemoryOS is an open-source framework that augments any large language model with a persistent, structured, and semantically searchable memory system. The system was built to address one of the most fundamental limitations of modern LLMs: they are stateless. Every conversation begins from zero. Without external scaffolding, an LLM cannot remember your name between sessions, cannot build on prior context, and cannot maintain a coherent model of you as a user across time.

MemoryOS solves this with a three-tier memory architecture:

**Tier 1 — Core Graph Memory (Neo4j)**
A labeled property graph storing a structured User World Model. Entities (people, places, organizations), life events, knowledge fragments, and preferences are stored as typed graph nodes with directional relationship edges. This layer provides structured recall, multi-hop graph traversal, and entity deduplication.

**Tier 2 — Archival Episodic Memory (ChromaDB)**
A persistent vector store containing verbatim conversation episodes encoded as dense embedding vectors. This layer provides semantic similarity search — the ability to surface relevant past exchanges even when the exact keywords are not known.

**Tier 3 — Active Context Buffer**
A sliding window buffer of the last N conversation turns held in memory for the current session. This provides immediate conversational coherence without requiring a database round-trip.

The three tiers work together in a relevance-scored pipeline. At every turn, the system assembles the most contextually relevant memory content from all three tiers, compresses it into a fixed-size context block, and injects it into the LLM system prompt — keeping token usage constant regardless of how many prior turns exist.

**Key figures:**
- Supports 2,000 to 4,000+ conversation turns with no degradation in recall quality
- Context token cost: approximately 350 tokens per turn regardless of conversation length
- Entity resolution: automatic detection and merging of duplicate graph nodes
- LLM backend: Groq API with `llama-3.3-70b-versatile`
- Graph database: Neo4j AuraDB (cloud-hosted, managed)
- Vector store: ChromaDB (local persistent)
- Frontend: single-page application in vanilla HTML/CSS/JS, served by FastAPI

---

## 2. The Core Problem — LLM Stationarity

A large language model is a stateless function. Given a sequence of tokens $x_1, x_2, \ldots, x_n$, it produces a probability distribution over the next token:

$$P(x_{n+1} \mid x_1, x_2, \ldots, x_n) = f_\theta(x_1, \ldots, x_n)$$

where $f_\theta$ is the transformer with fixed weights $\theta$. The model has no internal state that persists between API calls. Every inference is a completely independent computation.

### 2.1 The Naive Approach and Why It Fails

The most obvious solution is to pass the full conversation history on every call:

$$\text{prompt}_t = \bigl[S,\ H_1,\ H_2,\ \ldots,\ H_{t-1},\ Q_t\bigr]$$

where $S$ is the system prompt, $H_i = (u_i, a_i)$ is the $i$-th user-assistant exchange pair, and $Q_t$ is the current user query.

**Problem 1: Linear token growth**

The number of tokens in the prompt grows linearly with the number of prior turns:

$$|\text{prompt}_t| = |S| + \sum_{i=1}^{t-1} |H_i| + |Q_t| = \Theta(t)$$

At $t = 500$ turns with an average of 150 tokens per exchange, the prompt alone consumes 75,000 tokens. At $t = 4000$ turns, the naive prompt requires 600,000 tokens, which is computationally infeasible on any current model.

**Problem 2: Hard context window ceiling**

All current LLMs enforce a hard maximum context window $C_{\max}$. When the conversation history exceeds $C_{\max}$, the oldest exchanges must be silently truncated. The model literally forgets. Users who have engaged with an AI assistant for months watch their conversational history disappear without explanation.

**Problem 3: Quadratic attention cost**

The transformer self-attention mechanism computes pairwise interaction scores between every pair of tokens in the context. For a context of length $n$:

$$\text{Attention operations} = O\!\left(n^2 \cdot d\right)$$

where $d$ is the model dimension. Doubling the conversation history quadruples the inference time and cost.

### 2.2 The MemoryOS Solution

MemoryOS replaces the unbounded linear history buffer with a fixed-size, dynamically assembled context block $\mathcal{C}$:

$$\mathcal{C}_t = \text{Select}\bigl(\mathcal{M},\ Q_t,\ t,\ B\bigr)$$

where $\mathcal{M}$ is the full memory store (of arbitrary size), $Q_t$ is the current query, $t$ is the current turn, and $B$ is a fixed character budget. The selection function extracts only the most relevant memory content for the current query, so:

$$|\mathcal{C}_t| \leq B \quad \forall t$$

The token count of the context block is bounded by $B/4$ (at approximately 4 characters per token) regardless of how large $\mathcal{M}$ grows. As $t \to \infty$ and $|\mathcal{M}| \to \infty$, the token cost per turn remains constant.

---

## 3. System Architecture

The following diagram shows the complete request processing flow:

```
User Input
    │
    ▼
┌─────────────────────┐
│   Buffer Manager    │  ◄── Sliding window of last N turns (RAM)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐       ┌──────────────────────┐
│  Proactive Archival │──────►│  ChromaDB            │
│  Search             │       │  (episodic memory,   │
└─────────────────────┘       │   cosine similarity) │
    │                         └──────────────────────┘
    ▼
┌─────────────────────┐       ┌──────────────────────┐
│  Context Builder    │──────►│  Neo4j AuraDB        │
│  (relevance scored  │       │  (structured graph,  │
│   greedy budget)    │       │   property graph DB) │
└─────────────────────┘       └──────────────────────┘
    │
    ▼
┌─────────────────────┐
│  System Prompt      │  ◄── [CONTEXT] block injected here (~350 tokens)
│  Assembly           │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Groq LLM API       │  llama-3.3-70b-versatile
│  Agentic Loop       │  Up to 4 rounds, parallel tool calls enabled
└─────────────────────┘
    │
    ├──► update_entity          ──► Neo4j MERGE (Person/Place/Org node)
    ├──► link_entities          ──► Neo4j CREATE relationship edge
    ├──► log_event              ──► Neo4j Event node + EXPERIENCED edge
    ├──► save_knowledge         ──► Neo4j Knowledge node + AWARE_OF edge
    ├──► add_preference         ──► Neo4j Preference node + HAS_PREFERENCE
    ├──► update_user_profile    ──► Neo4j User node property SET
    ├──► merge_entities         ──► Neo4j 7-step entity merge
    ├──► graph_search           ──► Neo4j variable-depth traversal
    └──► archival_memory_search ──► ChromaDB cosine similarity search
    │
    ▼
┌─────────────────────┐
│  Background Archive │  ◄── Daemon thread (non-blocking)
│  Save               │       Full exchange ──► ChromaDB episode
└─────────────────────┘
    │
    ▼
Final Response ──► User
```

The architecture is intentionally modular. Each storage layer can be replaced independently:
- Neo4j AuraDB can be swapped for local Neo4j Desktop or Memgraph
- ChromaDB can be swapped for Pinecone, Weaviate, or Qdrant
- Groq can be swapped for OpenAI or Anthropic (both support the same function-calling interface)
- FastAPI can be replaced with any ASGI framework

---

## 4. Mathematical Foundations

### 4.1 Relevance Scoring

At the core of MemoryOS is a relevance scoring function that assigns a numeric priority to every memory node, determining which memories are selected for the context block at each turn.

Let $\mathcal{M} = \{m_1, m_2, \ldots, m_N\}$ be the set of all memory nodes in Neo4j, where $N$ can be arbitrarily large. Let $Q$ be the current conversational context string (the concatenation of the last 3 buffer turns).

The relevance score of memory node $m_i$ given query $Q$ at turn $t$ is:

$$\boxed{\text{score}(m_i \mid Q, t) = \alpha \cdot \kappa(m_i, Q) + \beta \cdot \rho(m_i, t) + \gamma \cdot \phi(m_i)}$$

where the three components and their weights are:

| Symbol | Weight | Component | Description |
|--------|--------|-----------|-------------|
| $\kappa(m_i, Q)$ | $\alpha = 0.50$ | Keyword overlap | Fraction of query tokens present in this memory node |
| $\rho(m_i, t)$ | $\beta = 0.30$ | Recency decay | Exponential decay since last access |
| $\phi(m_i)$ | $\gamma = 0.20$ | Frequency | Log-normalized access count |

The weights satisfy $\alpha + \beta + \gamma = 1.0$, so $\text{score}(m_i \mid Q, t) \in [0, 1]$.

**Keyword overlap component:**

Let $\mathcal{T}(s)$ denote the set of meaningful lowercase tokens in string $s$ after removing stop words. Then:

$$\kappa(m_i, Q) = \frac{\bigl|\mathcal{T}(Q) \cap \mathcal{T}(\text{serialize}(m_i))\bigr|}{|\mathcal{T}(Q)|}$$

This measures what fraction of the query's meaningful words appear in the serialized memory node. The denominator is $|\mathcal{T}(Q)|$, not $|\mathcal{T}(m_i)|$ — this is an asymmetric recall metric. A short, highly relevant node scores the same as a long node containing all the same keywords.

### 4.2 Recency Decay Function

The recency component models temporal relevance: recently accessed memories are more likely to pertain to the current conversation than memories from hundreds of turns ago.

$$\rho(m_i, t) = e^{-\lambda \cdot \Delta t_i}$$

where $\Delta t_i = t - t_{\text{last}}(m_i)$ is the number of turns elapsed since $m_i$ was last accessed or created, and $\lambda = 0.001$ is the decay rate constant.

The half-life of the recency function is:

$$t_{1/2} = \frac{\ln 2}{\lambda} = \frac{0.6931}{0.001} = 693 \text{ turns}$$

**Decay values at representative turn gaps:**

| $\Delta t$ (turns since last access) | $\rho$ value | Interpretation |
|--------------------------------------|-------------|----------------|
| 0 | 1.000 | Just accessed — maximum recency |
| 100 | 0.905 | Very recent — almost always included |
| 300 | 0.741 | Recent — likely included |
| 693 | 0.500 | Half-life reached |
| 1000 | 0.368 | Moderate — needs keyword support |
| 2000 | 0.135 | Old — needs strong keyword match |
| 4000 | 0.018 | Very old — rarely included alone |

The exponential decay is the unique distribution satisfying the memoryless property: $P(\Delta t > s + r \mid \Delta t > s) = P(\Delta t > r)$. This means the system does not penalize memories for having been old for a long time — only for not having been recently accessed.

### 4.3 Frequency Normalization

The frequency component rewards memory nodes that have been referenced, updated, or accessed many times across the entire conversation history. High-frequency nodes represent central facts in the user's life model and receive a consistent baseline boost.

$$\phi(m_i) = \frac{\ln(1 + f_i)}{\ln(1 + f_{\max})}$$

where $f_i$ is the access count of node $m_i$ and $f_{\max} = 100$ is the normalization cap.

The logarithm provides diminishing returns: the marginal value of the 50th access ($\Delta\phi \approx 0.009$) is much less than the marginal value of the 5th access ($\Delta\phi \approx 0.044$). Frequently accessed nodes stabilize quickly and do not dominate the scoring function.

### 4.4 Greedy Budget Selection

Given relevance scores for all $N$ nodes, the context selection problem is:

$$S^* = \underset{S \subseteq \mathcal{M}}{\arg\max} \sum_{m_i \in S} \text{score}(m_i \mid Q, t) \quad \text{subject to} \quad \sum_{m_i \in S} \bigl|m_i\bigr| \leq B$$

where $|m_i|$ is the serialized character length of node $m_i$ and $B$ is the total character budget.

This is an instance of the 0-1 Knapsack problem, NP-hard in general. The greedy approximation — sort by score-per-character descending, fill greedily — yields the optimal solution for the fractional relaxation and performs well in practice because memory nodes have similar sizes (30–120 characters each).

**Greedy algorithm:**

```
Input:  M = set of all memory nodes, Q = query, t = current turn, B = budget
Output: S* = selected subset

scores ← {m: score(m, Q, t) for m in M}
sorted_M ← sort M by scores[m] / len(serialize(m)) descending

budget_remaining ← B
S* ← {}

for m in sorted_M:
    if len(serialize(m)) ≤ budget_remaining:
        S* ← S* ∪ {m}
        budget_remaining ← budget_remaining − len(serialize(m))
    if budget_remaining == 0:
        break

return S*
```

Time complexity: $O(N \log N)$ for sorting, $O(N)$ for selection — total $O(N \log N)$ per turn.

**Critical property:** The output $|S^*|$ in characters is at most $B$, regardless of $N$. Even if $N$ grows to 10,000 nodes representing a multi-year memory graph, the assembled context block is always at most $B$ characters. This guarantees $O(1)$ token complexity.

### 4.5 Token Complexity Analysis

**Naive approach (full history concatenation):**

Let $\bar{h}$ be the average token count per conversation exchange. At turn $t$:

$$T_{\text{naive}}(t) = |S| + (t-1)\bar{h} + |Q_t| = \Theta(t)$$

For $\bar{h} = 150$ tokens and $t = 4000$:

$$T_{\text{naive}}(4000) = 200 + 599{,}850 + 150 = 600{,}200 \text{ tokens}$$

This exceeds every current LLM context window.

**MemoryOS approach (fixed budget):**

$$T_{\text{MemoryOS}}(t) = \underbrace{T_{\text{template}}}_{\approx 200} + \underbrace{T_{\mathcal{C}}}_{\approx 1{,}250} + \underbrace{T_{\text{buffer}}}_{\text{fixed}} = \Theta(1)$$

where $T_{\mathcal{C}} \approx B/4 \approx 1{,}250$ tokens (with $B = 5{,}000$ chars at 4 chars/token).

$$\lim_{t \to \infty} \frac{T_{\text{naive}}(t)}{T_{\text{MemoryOS}}(t)} = \lim_{t \to \infty} \frac{\Theta(t)}{\Theta(1)} = \infty$$

**Token comparison table:**

| Turn count | Naive tokens | MemoryOS tokens | Reduction factor |
|-----------|-------------|-----------------|-----------------|
| 100 | 15,200 | 1,450 | 10.5x |
| 500 | 75,200 | 1,450 | 51.9x |
| 1,000 | 150,200 | 1,450 | 103.6x |
| 2,000 | 300,200 | 1,450 | 207.0x |
| 4,000 | 600,200 (infeasible) | 1,450 | Infinite |

### 4.6 Entity Resolution — Duplicate Confidence Scoring

When the LLM stores entities about the user's world, it may create duplicate nodes under different names. For any pair of entity nodes $(e_a, e_b)$ in the graph, the duplicate confidence score is:

$$c(e_a, e_b) = s_1 \cdot \mathbf{1}\bigl[\text{rel}(e_a) = \text{rel}(e_b)\bigr] + s_2 \cdot \mathbf{1}\bigl[e_a \subset e_b \text{ or } e_b \subset e_a\bigr] + s_3 \cdot \text{lev}(e_a, e_b) \cdot \mathbf{1}\bigl[\text{lev}(e_a,e_b) > \tau\bigr] + s_4 \cdot \mathbf{1}\bigl[\text{possessive}(e_a, e_b)\bigr]$$

where:

| Signal | Symbol | Weight | Condition |
|--------|--------|--------|-----------|
| Relationship collision | $s_1$ | 0.40 | Both entities have the same relationship label to User |
| Name containment | $s_2$ | 0.50 | One name is a substring of the other |
| Levenshtein similarity | $s_3$ | 0.40 | $\text{lev}(e_a, e_b) > \tau = 0.65$ |
| Possessive pattern | $s_4$ | 0.60 | One name matches the pattern "X's [rel(other)]" |

The normalized Levenshtein similarity is:

$$\text{lev}(a, b) = 1 - \frac{d_L(a, b)}{\max(|a|, |b|)}$$

where $d_L(a, b)$ is the standard Levenshtein edit distance. Pairs with $c \geq 0.40$ are flagged as potential duplicates.

**Worked example — "Priyanshu's mother" vs. "Moumita":**

Given $\text{rel}(\text{"Priyanshu's mother"}) = \text{"mother"}$ and $\text{rel}(\text{"Moumita"}) = \text{"mother"}$:

- S1: Same relationship "mother" → $+0.40$
- S2: Name containment check → neither is a substring of the other → $+0$
- S3: $d_L = 14$, $\max(18, 7) = 18$, $\text{lev} = 0.22 < 0.65$ → $+0$
- S4: Strip "Priyanshu's " → remainder "mother" equals $\text{rel}(\text{"Moumita"})$ → $+0.60$

$$c = 0.40 + 0 + 0 + 0.60 = 1.0 \quad \text{(maximum confidence)}$$

---

## 5. Knowledge Graph Design

The User World Model is stored in Neo4j as a labeled property graph (LPG).

### Node Labels and Key Properties

| Label | Description | Key Properties |
|-------|-------------|----------------|
| `User` | Root node. Exactly one per database. | `id='main'`, `name`, `occupation`, `primary_location`, `total_turns` |
| `Person` | A named person in the user's world | `name`, `created_turn`, `last_seen_turn`, `access_count`, `aliases` |
| `Place` | A geographic location or venue | `name`, `created_turn`, `last_seen_turn`, `access_count` |
| `Organization` | A company, institution, or group | `name`, `created_turn`, `last_seen_turn`, `access_count` |
| `Event` | A significant life event | `id`, `description`, `date`, `turn` |
| `Knowledge` | A stored fact, code snippet, or reference | `topic`, `content`, `created_turn`, `last_updated_turn` |
| `Preference` | A user preference, goal, allergy, or constraint | `id`, `value`, `category` |

### Relationship Types

| Relationship | Direction | Description |
|--------------|-----------|-------------|
| `KNOWS` | `(User)→(Entity)` | User is aware of this entity. Properties: `relationship`, `last_seen` |
| `RELATED_TO` | `(Entity)↔(Entity)` | Two entities are connected. Properties: `rel_type`, `description` |
| `EXPERIENCED` | `(User)→(Event)` | User participated in this event |
| `INVOLVED_IN` | `(Entity)→(Event)` | This entity was involved in the event |
| `AWARE_OF` | `(User)→(Knowledge)` | User has stored this knowledge node |
| `HAS_PREFERENCE` | `(User)→(Preference)` | User holds this preference or constraint |

### Graph Topology Example

```
(User: Priyanshu Debnath)
        │
        ├── KNOWS [mother] ──────────► (Person: Moumita)
        │                                     │
        │                                     └── RELATED_TO ──► (Place: Kolkata)
        │
        ├── KNOWS [best friend] ──────► (Person: Alice Chen)
        │                                     │
        │                                     └── RELATED_TO (WORKS_AT) ──► (Org: Google)
        │                                                                          │
        │                                                                          └── RELATED_TO (LOCATED_IN) ──► (Place: Bangalore)
        │
        ├── KNOWS [university] ─────── ► (Org: IIT Kharagpur)
        │
        ├── EXPERIENCED ──────────────► (Event: T14 - Started internship at Accenture)
        │                                     ▲
        │                                     └── INVOLVED_IN ── (Org: Accenture)
        │
        ├── AWARE_OF ─────────────────► (Knowledge: NeuroHack competition details)
        │
        └── HAS_PREFERENCE ───────────► (Preference: [allergy] no peanuts)
```

---

## 6. Memory Retrieval Pipeline

The following sequence executes on every call to `Orchestrator.process_message()`.

### Step 1 — Turn increment

A monotonically increasing turn counter on the `User` node is incremented atomically:

```cypher
MATCH (u:User {id:'main'})
SET u.total_turns = coalesce(u.total_turns, 0) + 1
RETURN u.total_turns AS t
```

This provides a globally consistent turn ID used in all recency decay computations.

### Step 2 — Proactive archival search

Before invoking the LLM, the system performs a proactive semantic search over ChromaDB using the concatenation of the last 3 buffer turns as the query. The top-4 most semantically similar past episodes are retrieved.

Semantic similarity uses cosine distance between query embedding $\mathbf{q} \in \mathbb{R}^{384}$ and episode embeddings $\mathbf{e}_i \in \mathbb{R}^{384}$:

$$\text{sim}(\mathbf{q}, \mathbf{e}_i) = \frac{\mathbf{q} \cdot \mathbf{e}_i}{\|\mathbf{q}\|_2 \cdot \|\mathbf{e}_i\|_2}$$

ChromaDB returns distances (smaller = more relevant). The top-4 results are injected into the archival section of the context block.

### Step 3 — Relevance-scored graph context assembly

The `ContextBuilder.build()` method:

1. Fetches all node categories from Neo4j (profile, entities, events, knowledge, preferences)
2. Scores each node using the relevance function from Section 4.1
3. Applies per-section character budgets and max-node caps
4. Serializes selected nodes into compact pipe-delimited format (Section 7)
5. Merges all sections into a single context block string

Per-section budgets:

| Section | Budget (chars) | Max nodes |
|---------|---------------|-----------|
| Profile | 400 | — |
| Preferences | 500 | — |
| Entities | 2,000 | 20 |
| Events | 1,200 | 12 |
| Knowledge | 1,400 | 10 |
| Archival | 800 | 4 |

### Step 4 — System prompt construction

The context block is injected into the system prompt template at `{core_memory_block}`. The full system prompt is approximately 200 template tokens + 1,250 context tokens = 1,450 tokens total, constant across all turns.

### Step 5 — LLM agentic loop

The LLM is called via Groq with `tool_choice="auto"` and `parallel_tool_calls=True`. The loop runs up to 4 rounds:
- If the model returns tool calls → dispatch all, append results, loop again
- If the model returns a text response → exit loop

Parallel tool calls allow the model to issue multiple memory writes in a single round, e.g., calling `update_entity`, `log_event`, and `link_entities` simultaneously for a high-information user message.

### Step 6 — Background archive save

After the final response is returned, a daemon thread saves the full exchange to ChromaDB:

```python
def _bg_save():
    self.archive.add_memory(
        f"User: {user_message}\nAssistant: {final_response}",
        "assistant",
        current_turn,
    )
threading.Thread(target=_bg_save, daemon=True).start()
```

The archive write is non-blocking. The user receives the response immediately.

---

## 7. Compact Serialization Format

Verbose JSON serialization wastes tokens. A single entity in JSON:

```json
{
  "node_type": "Entity",
  "neo4j_label": "Person",
  "display_name": "Alice Chen",
  "relationship_to_user": "colleague",
  "attributes": {
    "role": "Software Engineer",
    "company": "Google",
    "city": "San Francisco"
  }
}
```

This is 188 characters. The MemoryOS compact format represents the same information as:

```
Alice Chen[colleague](role=SWE, company=Google, city=SF)
```

This is 56 characters — a 3.4x reduction per entity.

**Full example context block** (440 chars ≈ 110 tokens):

```
PROFILE: name=Priyanshu Debnath | occupation=CS Student | location=Kolkata

ENTITIES(4): Alice Chen[colleague](role=SWE)->WORKS_AT:Google |
  Moumita Debnath[mother](age=48) | IIT Kharagpur[university] |
  Raj Kumar[friend](city=Mumbai)

EVENTS(2): T14: Started internship at Accenture [2024-06-01] |
  T38: Submitted NeuroHack Challenge project

KNOWLEDGE(1): [NeuroHack] Submission due 2025-04-15. Team: Priyanshu, Sumedha.

PREFERENCES(2): [goal] Win NeuroHack Challenge | [preference] Works late nights

ARCHIVAL(1): [T8|dist=0.18] User mentioned they have a peanut allergy
```

The equivalent verbose JSON representation is approximately 1,900 characters = 475 tokens. The compact format saves approximately 365 tokens per turn — a **4.1x token reduction** for the context block.

---

## 8. Entity Resolution Engine

Duplicate entities are the most common cause of incorrect reasoning in long-running LLM memory systems. The entity resolution engine provides automatic detection and programmatic merging.

### Detection

The `detect_duplicates()` method performs $O(N^2/2)$ pairwise comparisons of all entity nodes connected to the User. For $N = 20$ entities (typical after 100 turns), this is 190 comparisons — negligible computationally.

Each pair is scored using the four-signal function from Section 4.6. Pairs with $c \geq 0.40$ are returned ranked by confidence descending.

### Merge Algorithm — 7-Step Cypher Sequence

When `merge_entities(canonical_name, alias_name)` is called, the following 7 Cypher steps execute sequentially:

**Step 1** — Verify both nodes exist. Return an error without modifying the graph if either is missing.

**Step 2** — Re-point the `KNOWS` edge from `User` to `alias` onto `User` to `canonical`:

```cypher
MATCH (u:User {id:'main'})-[r:KNOWS]->(alias {name:$alias})
MATCH (canonical {name:$canonical})
MERGE (u)-[r2:KNOWS]->(canonical)
ON CREATE SET r2 = properties(r)
DELETE r
```

**Step 3** — Migrate all outgoing `RELATED_TO` edges from `alias` to `canonical`.

**Step 4** — Migrate all incoming `RELATED_TO` edges to `alias` onto `canonical`.

**Step 5** — Migrate all `INVOLVED_IN` edges (entity → event):

```cypher
MATCH (alias {name:$alias})-[r:INVOLVED_IN]->(ev:Event)
MATCH (canonical {name:$canonical})
MERGE (canonical)-[:INVOLVED_IN]->(ev)
DELETE r
```

**Step 6** — Record the alias name in the `aliases` array on the canonical node:

```cypher
MATCH (canonical {name:$canonical})
SET canonical.aliases = coalesce(canonical.aliases, []) + [$alias]
```

**Step 7** — Delete the alias node and all remaining relationships:

```cypher
MATCH (alias {name:$alias})
DETACH DELETE alias
```

After the merge, exactly one node represents the real-world entity with all historical relationships intact. The `aliases` field preserves the full provenance of which names were merged.

---

## 9. Storage Layer — Neo4j and ChromaDB

### Neo4j AuraDB

Neo4j is a native graph database that stores nodes, relationships, and properties as first-class objects, enabling $O(1)$ traversal per relationship hop. Unlike relational databases where relationships require join operations, Neo4j stores relationship pointers directly on each node.

**MERGE semantics — idempotent writes:**

The `MERGE` clause in Cypher provides atomic upsert semantics. Calling `update_entity` twice with the same name does not create duplicate nodes:

```cypher
MERGE (e:Person {name: $name})
ON CREATE SET e.created_turn = $turn, e.access_count = 1
ON MATCH  SET e.access_count = coalesce(e.access_count, 0) + 1,
              e.last_seen_turn = $turn
SET e += $attrs
```

**URI scheme — why `neo4j+ssc://`:**

| URI Scheme | Encryption | Certificate Verification |
|------------|-----------|--------------------------|
| `neo4j://` | None | None (local dev only) |
| `neo4j+s://` | SSL/TLS | Full CA chain verification |
| `neo4j+ssc://` | SSL/TLS | Skips CA verification |

The `+ssc` scheme is required for Anaconda Python on Windows, which ships with an incomplete CA bundle that does not include the certificate authority used by Neo4j AuraDB. Attempting `neo4j+s://` results in `SSLCertVerificationError`.

### ChromaDB

ChromaDB is an open-source embedding database optimized for semantic similarity search. MemoryOS uses it as the episodic memory layer, storing complete conversation exchanges as searchable vector documents.

**Embedding model:** `all-MiniLM-L6-v2` (384-dimensional, distilled from BERT, trained on 1B sentence pairs)

**Storage schema per episode:**

```python
collection.add(
    documents=["User: ...\nAssistant: ..."],
    ids=[f"turn_{turn_id}"],
    metadatas=[{"origin_turn": turn_id, "speaker": "assistant"}],
)
```

The archival search is used in two ways:
1. **Proactively** at the start of every turn to populate the `ARCHIVAL` section of the context block
2. **Reactively** when the LLM calls `archival_memory_search` to recall something not visible in the graph context

---

## 10. LLM Tool Schema

MemoryOS exposes 9 tools to the LLM via the Groq function calling API.

| Tool | Purpose | Required Parameters |
|------|---------|-------------------|
| `update_user_profile` | Save/update a permanent user attribute | `key`, `value` |
| `delete_user_profile_field` | Remove an outdated profile field | `key`, `value_to_remove` |
| `add_preference` | Store a preference, goal, allergy, or constraint | `value`, `category` |
| `update_entity` | Create or update a Person/Place/Organization | `name`, `entity_type` |
| `link_entities` | Create a directional edge between two entities | `entity_a`, `entity_b`, `relationship_type` |
| `log_event` | Log a significant life event | `description` |
| `save_knowledge` | Store a reusable fact or reference | `topic`, `content` |
| `graph_search` | Traverse the graph from a named entity | `entity_name` |
| `merge_entities` | Merge a duplicate alias into a canonical entity | `canonical_name`, `alias_name` |
| `archival_memory_search` | Semantic search over past episodes | `query` |

The `allergy` and `constraint` preference categories are enforced as HARD RULES in the system prompt and cannot be overridden by subsequent user instructions.

**Example multi-tool turn:**

User: "I started working at Infosys last Monday. My manager is Rajesh Kumar and our office is in Bangalore."

Round 1 (parallel tool calls):
- `update_user_profile(key="occupation", value="Software Engineer at Infosys")`
- `update_entity(name="Infosys", entity_type="organization", relationship_to_user="employer")`
- `update_entity(name="Rajesh Kumar", entity_type="person", relationship_to_user="manager")`
- `update_entity(name="Bangalore", entity_type="place", relationship_to_user="work_location")`
- `log_event(description="Started working at Infosys", entities_involved=["Infosys"])`

Round 2 (linking):
- `link_entities("Rajesh Kumar", "Infosys", "WORKS_AT")`
- `link_entities("Infosys", "Bangalore", "LOCATED_IN")`

Round 3: Text response returned. 7 memory operations in 2 rounds.

---

## 11. API Reference

All endpoints are served by FastAPI at `http://localhost:8000`.

### POST `/chat`

**Request:**
```json
{"message": "My friend Bob just moved to London."}
```

**Response:**
```json
{
  "response": "Noted! I've updated Bob's location to London...",
  "active_memories": [{"content": "...", "origin_turn": 23, "distance": 0.21}],
  "debug_prompt": "You are MemoryOS...\n[CONTEXT]\n..."
}
```

### GET `/graph`

Returns all nodes and edges for the knowledge graph visualization. Node objects include `id`, `name`, `label`, `type`, `entity_type`, `relationship`, `attributes`, `color`, `size`. Edge objects include `from`, `to`, `label`, `color`.

### GET `/stats`

Returns aggregated statistics: `turns`, `entities`, `events`, `knowledge`, `preferences`, `people`, `places`, `organizations`, `pref_preference`, `pref_goal`, `pref_allergy`, `pref_constraint`, `profile`.

### GET `/search?q={query}`

Full-text search across all graph node name, topic, and content fields.

### GET `/duplicates`

Runs the entity resolution detection algorithm and returns candidate duplicate pairs with confidence scores and reasons.

### POST `/merge`

**Request:** `{"canonical": "Moumita", "alias": "Priyanshu's mother"}`

**Response:** `{"result": "Merged 'Priyanshu's mother' into 'Moumita'. 3 edges migrated."}`

### GET `/health`

Liveness probe. Returns `{"status": "ok", "version": "2.0.0"}`.

---

## 12. Frontend Architecture

The production frontend is a zero-dependency single-page application served at `GET /` by FastAPI, built with vanilla HTML, CSS, and JavaScript.

**CDN libraries:**

| Library | Version | Purpose |
|---------|---------|---------|
| `vis-network` | 9.1.9 | Interactive knowledge graph visualization |
| `Chart.js` | 4.4.0 | Analytics charts |
| `KaTeX` | 0.16.9 | LaTeX math formula rendering |
| Inter + JetBrains Mono | — | Typography |

**Pages:**

| Page | Description |
|------|-------------|
| Chat | Full chatbot UI — message bubbles, typing indicator, memory recall badges, auto-expanding textarea |
| Knowledge Graph | Interactive vis-network graph, physics simulation, color-coded node types, zoom/fit controls |
| Memory Browser | Entity cards grid, preference badges, events timeline, knowledge nodes, entity resolution panel |
| Analytics | Metric cards + doughnut/bar/horizontal-bar Chart.js charts |
| Architecture | KaTeX-rendered mathematical exposition of the system design |

The sidebar shows live memory statistics refreshed every 15 seconds and a Neo4j connection status indicator.

---

## 13. Installation and Setup

### Prerequisites

- Python 3.10 or higher
- Groq API key: [console.groq.com](https://console.groq.com) (free tier available)
- Neo4j AuraDB Free instance: [console.neo4j.io](https://console.neo4j.io) (free, 200k node limit)

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/memory-os.git
cd memory-os
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

On Anaconda (Windows), use the full pip path:

```bash
/c/Users/YOUR_USERNAME/anaconda3/Scripts/pip.exe install -r requirements.txt
```

### Step 3 — Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_actual_groq_key_here

# Use neo4j+ssc:// (not neo4j+s://) — required on Anaconda Python
NEO4J_URI=neo4j+ssc://xxxxxxxx.databases.neo4j.io
NEO4J_USER=xxxxxxxx
NEO4J_PASSWORD=your_auradb_password_here
```

### Step 4 — Start the server

```bash
python -m backend.server
```

Open `http://localhost:8000` in your browser. The production frontend loads immediately — no Streamlit or separate frontend server required.

**Development mode with auto-reload:**

```bash
uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
```

### Step 5 — Reset state (optional)

To wipe all stored memories and start fresh:

```bash
python reset_state.py
```

---

## 14. Configuration Reference

All tunable parameters are in `config.py`:

```python
# LLM
LLM_MODEL          = "llama-3.3-70b-versatile"
BUFFER_MAX_TURNS   = 20

# Per-section character budgets
TOKEN_BUDGET_PROFILE   = 400
TOKEN_BUDGET_PREFS     = 500
TOKEN_BUDGET_ENTITIES  = 2000
TOKEN_BUDGET_EVENTS    = 1200
TOKEN_BUDGET_KNOWLEDGE = 1400
TOKEN_BUDGET_ARCHIVAL  = 800

# Relevance score weights (alpha + beta + gamma = 1.0)
RELEVANCE_KEYWORD_WEIGHT = 0.50   # alpha
RELEVANCE_RECENCY_WEIGHT = 0.30   # beta
RELEVANCE_FREQ_WEIGHT    = 0.20   # gamma

# Recency decay (half-life = ln(2)/lambda turns)
RELEVANCE_DECAY_LAMBDA   = 0.001  # half-life = 693 turns

# Max nodes per section
MAX_ENTITIES_IN_PROMPT   = 20
MAX_EVENTS_IN_PROMPT     = 12
MAX_KNOWLEDGE_IN_PROMPT  = 10
```

**Tuning guidance:**

| Parameter | Increase | Decrease |
|-----------|----------|----------|
| `RELEVANCE_KEYWORD_WEIGHT` | More responsive to current query | More weight on background context |
| `RELEVANCE_DECAY_LAMBDA` | Memories decay faster | Memories persist longer |
| `TOKEN_BUDGET_ENTITIES` | More entities in context | Less budget for events/knowledge |
| `BUFFER_MAX_TURNS` | Richer immediate context | Lower per-turn token cost |

---

## 15. Usage Examples

### Example 1 — Profile setup and cross-session recall

```
Turn 1:
User:   Hi, I am Priyanshu Debnath. I study at IIT Kharagpur.
Tools:  update_user_profile(key="name", value="Priyanshu Debnath")
        update_entity(name="IIT Kharagpur", entity_type="organization",
                      relationship_to_user="university")

Turn 150:
User:   What university do I attend?
AI:     You attend IIT Kharagpur.
        [IIT Kharagpur node: recency = e^(-0.001*149) = 0.861. Present in context.]
```

### Example 2 — Entity linking and multi-hop reasoning

```
User:   My best friend Alice works at Google as a senior engineer in Bangalore.
Tools:  update_entity(name="Alice", entity_type="person",
                      relationship_to_user="best friend",
                      attributes={"role": "Senior Engineer"})
        update_entity(name="Google", entity_type="organization")
        update_entity(name="Bangalore", entity_type="place")
        link_entities("Alice", "Google", "WORKS_AT")
        link_entities("Google", "Bangalore", "LOCATED_IN")

Later:
User:   What city does Alice work in?
AI:     Alice works at Google's office in Bangalore.
        [Graph traversal: Alice -WORKS_AT-> Google -LOCATED_IN-> Bangalore]
```

### Example 3 — Allergy safety enforcement

```
[Stored: Preference(value="no peanuts", category="allergy")]

User:   Suggest snacks for a road trip.
AI:     Great road trip snacks: trail mix (sunflower seeds and raisins),
        rice cakes with hummus, dark chocolate, granola bars, roasted chickpeas.
        [All suggestions are peanut-free. The [ALLERGY] tag activates the SAFETY rule,
         which the model cannot override regardless of user framing.]
```

### Example 4 — Long-range recall via archival search

```
Turn 850:
User:   What was that biryani restaurant I mentioned?

AI:     [Calls archival_memory_search(query="biryani restaurant")]
        ChromaDB returns:
          Turn 23: "User: I love eating at Arsalan, best biryani in Kolkata"
          distance=0.19

AI:     The restaurant you mentioned was Arsalan — you described it as
        the best biryani in Kolkata.
```

### Example 5 — Automatic entity deduplication

```
Turn 12: Entity("Priyanshu's mother", relationship="mother") stored.
Turn 58: User mentions "Her name is Moumita."
         Entity("Moumita", relationship="mother") stored.
         detect_duplicates() → confidence = 1.0
         RESOLVE rule triggers → merge_entities("Moumita", "Priyanshu's mother")
         7-step Cypher merge executes.
         "Priyanshu's mother" node deleted. Moumita.aliases = ["Priyanshu's mother"]

Turn 100:
User:   How old is my mom?
AI:     Your mother Moumita is 48 years old.
        [Single clean node in context. No duplicate confusion.]
```

---

## 16. Performance Benchmarks

### Token Usage

| Turn count | MemoryOS tokens | Naive tokens | Savings |
|-----------|----------------|-------------|---------|
| 100 | 1,450 | 15,200 | 90.5% |
| 500 | 1,450 | 75,200 | 98.1% |
| 1,000 | 1,450 | 150,200 | 99.0% |
| 4,000 | 1,450 | 600,200 (infeasible) | 99.8% |

### Per-Turn Latency

| Component | Approximate time |
|-----------|-----------------|
| Neo4j turn increment | 50–120 ms |
| ChromaDB proactive search | 20–80 ms |
| Neo4j context fetch + scoring | 80–200 ms |
| Groq inference (no tools) | 300–800 ms |
| Groq inference (1-2 tool rounds) | 600–1,500 ms |
| Neo4j tool writes | 30–100 ms per tool |
| **Total (simple turn)** | **450–1,200 ms** |
| **Total (complex turn)** | **900–2,500 ms** |

### Serialization Efficiency

| Format | Chars per entity | Total for 15 entities + 5 events + 3 knowledge |
|--------|----------------|-------------------------------------------------|
| Verbose JSON | 188 | ~2,800 |
| MemoryOS compact | 56 | ~840 |
| Reduction | 3.4x | 3.3x |

---

## 17. Design Decisions and Trade-offs

### Why Neo4j instead of a relational database?

The User World Model is a heterogeneous, schema-flexible property graph with rich inter-entity relationships. In SQL, finding all entities connected to a user within 2 hops requires a 4-table join. In Neo4j, it is a single Cypher pattern match with $O(1)$ traversal cost per hop. The flexibility to add arbitrary attributes to any node without schema migrations is also essential for a system that stores whatever facts the user happens to mention.

### Why greedy budget selection instead of a learned retriever?

A learned dense retriever would theoretically provide better relevance ranking. However, it requires labeled training data (expensive to collect), adds embedding model inference latency per turn (20-100 ms), and is opaque in its ranking decisions. The greedy scoring function is interpretable, has zero inference cost, and performs well in practice for the node counts typical of a personal memory graph.

### Why ChromaDB for episodic memory instead of more Neo4j?

Storing verbatim conversation episodes as Neo4j nodes would degrade graph traversal performance as the episode count grows into the thousands. ChromaDB uses HNSW (Hierarchical Navigable Small World) indexing for approximate nearest-neighbor search, scaling to millions of vectors with sub-millisecond query latency. Using specialized tools for specialized tasks keeps both systems performant.

### Why the compact serialization format?

The LLM does not need to parse the context block as structured data — it only needs to read and reason over it as natural language. The pipe-delimited compact format is readable English that the LLM processes as efficiently as JSON, but at 3-4x fewer tokens. This is the single highest-leverage optimization in the system after the budget selection algorithm.

---

## 18. Limitations and Future Work

### Current Limitations

**Entity resolution recall:** The four-signal confidence function catches common patterns but will miss cases where two entities share neither a similar name nor the same relationship type. A future version could use the LLM itself as a zero-shot duplicate classifier.

**No multi-user support:** The current architecture assumes a single `User` node with `id='main'`. Supporting multiple users requires per-user database isolation.

**English-only tokenization:** The keyword scoring function uses an English stop-word list. Other languages require language-aware tokenization.

**No entity deletion:** There is currently no mechanism for the LLM to delete entities — only to update or merge them. A `delete_entity` tool could be added for explicit user-requested forgetting.

**ChromaDB is local-only:** The current setup uses a local persistent ChromaDB client, unsuitable for multi-instance or cloud deployments. A server-mode ChromaDB or managed vector database would be required at scale.

### Future Work

**Hierarchical memory compression:** For very long conversations, periodic compression of old episodes into higher-level summaries would reduce archival search space while retaining thematic recall.

**Temporal reasoning:** Adding explicit date parsing and a timeline query interface would allow queries like "What was happening in my life in January 2024?"

**Memory confidence and provenance:** Each stored fact could carry a confidence score and a provenance (the turn it came from). Low-confidence facts could be flagged for verification; outdated facts could be automatically demoted.

**Graph attention for context selection:** The current scoring is node-level. A graph attention mechanism could score nodes based on structural centrality — nodes with many connections to other high-scoring nodes receive a structural bonus.

---

## 19. Collaborators

**Priyanshu Debnath**
Primary architect and developer. Designed the three-tier memory architecture, implemented the Neo4j graph integration, relevance scoring engine, entity resolution algorithm, compact serialization format, agentic LLM orchestration loop, FastAPI backend, and production HTML/CSS/JS frontend.

**Sumedha Bhattacharya**
Co-developer and collaborator. Contributed to system design, testing, and refinement of the memory retrieval pipeline and context assembly logic.

---

*Built for the NeuroHack Challenge. MIT License.*
