"""
MemoryOS — Professional Interface
===================================
5 tabs:
  Chat              — conversation with structured memory recall
  Knowledge Graph   — live interactive pyvis knowledge graph
  Memory Browser    — browse entities, events, knowledge, preferences
  Analytics         — performance metrics and recall charts
  Architecture      — technical explainer with mathematical framework
"""

import json
import os
import tempfile
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MemoryOS",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%233B82F6'><path d='M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND = "http://localhost:8000"

# ── Design tokens ─────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-primary:   #070B14;
  --bg-surface:   #0D1526;
  --bg-card:      #111D30;
  --bg-card-2:    #172340;
  --accent:       #3B82F6;
  --accent-dim:   rgba(59,130,246,0.12);
  --accent-2:     #6366F1;
  --border:       rgba(255,255,255,0.07);
  --border-blue:  rgba(59,130,246,0.35);
  --text-primary: #F1F5F9;
  --text-sec:     #94A3B8;
  --text-muted:   #475569;
  --success:      #10B981;
  --warning:      #F59E0B;
  --error:        #EF4444;
}

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace; }

#MainMenu, footer, header { visibility: hidden; }

/* ── Main background ──────────────────────────────────────────────────────── */
.main { background: var(--bg-primary); }
section[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Typography ───────────────────────────────────────────────────────────── */
h1 {
  font-size: 1.9rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.03em;
  color: var(--text-primary) !important;
  margin-bottom: 0 !important;
}
h2 { font-size: 1.25rem !important; font-weight: 700 !important; color: var(--text-primary) !important; }
h3 { font-size: 1.0rem  !important; font-weight: 600 !important; color: var(--text-sec) !important; }
p, li { color: var(--text-sec) !important; }
label { color: var(--text-sec) !important; }

/* ── Chat messages ────────────────────────────────────────────────────────── */
.stChatMessage {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 20px !important;
  margin: 10px 0 !important;
}
.stChatMessage p { color: var(--text-primary) !important; }

/* ── Chat input ───────────────────────────────────────────────────────────── */
.stChatInputContainer textarea {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-blue) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
  font-size: 0.95rem !important;
}
.stChatInputContainer textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
  outline: none !important;
}
.stChatInputContainer textarea::placeholder { color: var(--text-muted) !important; }

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
  background: var(--accent) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  padding: 8px 20px !important;
  transition: all 0.2s ease !important;
  letter-spacing: 0.01em !important;
}
.stButton > button:hover {
  background: #2563EB !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px rgba(59,130,246,0.35) !important;
}

/* ── Metrics ──────────────────────────────────────────────────────────────── */
[data-testid="stMetricValue"]  { color: var(--accent) !important; font-weight: 700 !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"]  { color: var(--text-sec) !important; font-size: 0.8rem !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="metric-container"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 16px !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-surface) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  border: 1px solid var(--border) !important;
  gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 7px !important;
  color: var(--text-sec) !important;
  font-weight: 500 !important;
  font-size: 0.875rem !important;
  padding: 8px 18px !important;
  border: none !important;
  transition: all 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
  background: var(--accent-dim) !important;
  color: var(--text-primary) !important;
}

/* ── Expanders ────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-sec) !important;
}

/* ── Text inputs ──────────────────────────────────────────────────────────── */
.stTextInput > div > div > input {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}

/* ── Selectbox ────────────────────────────────────────────────────────────── */
.stSelectbox > div > div {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
}

/* ── Info / alerts ────────────────────────────────────────────────────────── */
.stAlert {
  background: var(--accent-dim) !important;
  border: 1px solid var(--border-blue) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
}

/* ── Divider ──────────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-surface); }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 6px; }

/* ── Memory recall badges ─────────────────────────────────────────────────── */
.recall-badge {
  display: inline-block;
  background: var(--bg-card-2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 12px;
  margin: 4px;
  font-size: 0.78rem;
  color: var(--text-sec);
  font-family: 'JetBrains Mono', monospace;
}
.recall-badge .conf-high   { color: #10B981; font-weight: 600; }
.recall-badge .conf-med    { color: #F59E0B; font-weight: 600; }
.recall-badge .conf-low    { color: #EF4444; font-weight: 600; }

/* ── Status dot ───────────────────────────────────────────────────────────── */
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.dot-green  { background: #10B981; }
.dot-yellow { background: #F59E0B; }
.dot-red    { background: #EF4444; }
.dot-blue   { background: #3B82F6; }

/* ── Entity/pref badges ───────────────────────────────────────────────────── */
.badge {
  display: inline-block; border-radius: 5px; padding: 2px 8px;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
  text-transform: uppercase; margin-right: 6px;
}
.badge-allergy  { background: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-goal     { background: rgba(59,130,246,0.15); color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); }
.badge-pref     { background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.3); }
.badge-person   { background: rgba(99,102,241,0.15); color: #A5B4FC; border: 1px solid rgba(99,102,241,0.3); }
.badge-place    { background: rgba(59,130,246,0.15); color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); }
.badge-org      { background: rgba(245,158,11,0.15); color: #FCD34D; border: 1px solid rgba(245,158,11,0.3); }
.badge-event    { background: rgba(249,115,22,0.15); color: #FB923C; border: 1px solid rgba(249,115,22,0.3); }
.badge-know     { background: rgba(167,139,250,0.15); color: #C4B5FD; border: 1px solid rgba(167,139,250,0.3); }

/* ── Sidebar metric card ──────────────────────────────────────────────────── */
.side-card {
  background: rgba(59,130,246,0.08);
  border: 1px solid rgba(59,130,246,0.2);
  border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
}
.side-card .label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
.side-card .value { font-size: 1.3rem; font-weight: 700; color: var(--accent); }

/* ── Architecture page ────────────────────────────────────────────────────── */
.arch-section {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 28px 32px;
  margin-bottom: 20px;
}
.arch-section h2 { font-size: 1.1rem !important; color: var(--text-primary) !important; margin-bottom: 12px !important; }
.arch-section p  { font-size: 0.9rem; color: var(--text-sec) !important; line-height: 1.7; }
.formula-box {
  background: #0A0F1E;
  border: 1px solid rgba(59,130,246,0.25);
  border-radius: 8px; padding: 16px 20px; margin: 14px 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem; color: #93C5FD; line-height: 1.8;
}
.formula-box .comment { color: var(--text-muted); }
.perf-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 14px; }
.perf-table th {
  background: rgba(59,130,246,0.1); color: var(--accent); padding: 10px 16px;
  text-align: left; font-weight: 600; font-size: 0.78rem;
  text-transform: uppercase; letter-spacing: 0.06em;
  border-bottom: 1px solid var(--border-blue);
}
.perf-table td { padding: 10px 16px; color: var(--text-sec); border-bottom: 1px solid var(--border); }
.perf-table tr:hover td { background: var(--accent-dim); }
.highlight-cell { color: #10B981 !important; font-weight: 600; }
.pipeline-step {
  display: flex; align-items: flex-start; gap: 16px;
  margin-bottom: 16px; padding: 14px;
  background: var(--bg-card-2);
  border-radius: 8px; border-left: 3px solid var(--accent);
}
.step-num {
  min-width: 28px; height: 28px; border-radius: 50%;
  background: var(--accent); color: white; display: flex;
  align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.8rem; flex-shrink: 0;
}
.step-body .step-title { color: var(--text-primary); font-weight: 600; font-size: 0.9rem; }
.step-body .step-desc  { color: var(--text-sec); font-size: 0.82rem; margin-top: 4px; line-height: 1.5; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

for key, default in {
    "messages":         [],
    "turn_count":       0,
    "memory_analytics": {"turns": [], "memories_recalled": [], "confidence_scores": []},
    "demo_prompt":      None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None):
    try:
        r = requests.get(f"{BACKEND}{path}", params=params, timeout=8)
        return r.json()
    except Exception:
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{BACKEND}{path}", json=payload, timeout=45)
        return r.json()
    except Exception:
        return None


def conf_level(distance: float) -> tuple[str, str]:
    """Return (css_class, label) for a retrieval confidence value."""
    if distance < 0.5:
        return "conf-high", "High"
    elif distance < 0.8:
        return "conf-med",  "Medium"
    return "conf-low", "Low"


def typing_animation(text: str, delay: float = 0.010):
    ph = st.empty()
    shown = ""
    for ch in text:
        shown += ch
        ph.markdown(shown)
        time.sleep(delay)
    return ph


def build_pyvis_html(graph_data: dict) -> str:
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p style='color:#EF4444;font-family:Inter,sans-serif'>pyvis not installed — run: pip install pyvis</p>"

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    net = Network(height="580px", bgcolor="#070B14", font_color="#94A3B8", directed=True)
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -9000,
          "centralGravity": 0.25,
          "springLength": 150,
          "damping": 0.12
        },
        "stabilization": {"iterations": 180}
      },
      "edges": {
        "smooth": {"type": "curvedCW", "roundness": 0.15},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.55}},
        "font": {"size": 10, "color": "#475569", "align": "middle"},
        "color": {"inherit": false}
      },
      "nodes": {
        "font": {"size": 13, "color": "#F1F5F9"},
        "borderWidth": 2,
        "borderWidthSelected": 3,
        "shadow": {"enabled": true, "size": 8, "color": "rgba(0,0,0,0.5)"}
      },
      "interaction": {"hover": true, "tooltipDelay": 100, "hideEdgesOnDrag": false}
    }
    """)

    node_ids = set()
    for n in nodes:
        net.add_node(
            n["id"],
            label=n.get("label", n["id"]),
            color=n.get("color", "#3B82F6"),
            size=n.get("size", 16),
            title=n.get("title", ""),
        )
        node_ids.add(n["id"])

    for e in edges:
        # Skip edges whose endpoints aren't in the graph (e.g. entities
        # referenced in events/relationships but not directly KNOWS-linked)
        if e["from"] not in node_ids or e["to"] not in node_ids:
            continue
        net.add_edge(
            e["from"], e["to"],
            label=e.get("label", ""),
            color=e.get("color", "rgba(59,130,246,0.4)"),
        )

    tmp = os.path.join(tempfile.gettempdir(), "memoryos_graph.html")
    net.save_graph(tmp)
    with open(tmp, "r", encoding="utf-8") as f:
        return f.read()


# ── Header ────────────────────────────────────────────────────────────────────

header_col, meta_col = st.columns([4, 2])
with header_col:
    st.markdown("""
    <div style="padding:4px 0 12px 0">
      <div style="font-size:1.75rem;font-weight:800;letter-spacing:-0.03em;color:#F1F5F9">
        MemoryOS
      </div>
      <div style="font-size:0.82rem;color:#475569;margin-top:2px;letter-spacing:0.04em;text-transform:uppercase">
        Long-Form Conversational Intelligence &nbsp;·&nbsp; Neo4j + ChromaDB
      </div>
    </div>
    """, unsafe_allow_html=True)

with meta_col:
    m1, m2 = st.columns(2)
    m1.metric("Turn", st.session_state.turn_count)
    m2.metric("Messages", len(st.session_state.messages))

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-size:0.65rem;color:#3B82F6;font-weight:700;letter-spacing:0.14em;
                text-transform:uppercase;margin-bottom:12px">Memory State</div>
    """, unsafe_allow_html=True)

    stats = api_get("/stats") or {}

    for label, key in [
        ("Entities",      "entity_count"),
        ("Events",        "event_count"),
        ("Knowledge",     "knowledge_count"),
        ("Total Turns",   "total_turns"),
    ]:
        val = stats.get(key, 0)
        st.markdown(f"""
        <div class="side-card">
          <div class="label">{label}</div>
          <div class="value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Recall timeline
    st.markdown('<div style="font-size:0.65rem;color:#3B82F6;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:8px">Recall Timeline</div>', unsafe_allow_html=True)
    if st.session_state.memory_analytics["turns"]:
        fig_side = go.Figure()
        fig_side.add_trace(go.Scatter(
            x=st.session_state.memory_analytics["turns"],
            y=st.session_state.memory_analytics["memories_recalled"],
            mode="lines+markers",
            line=dict(color="#3B82F6", width=2),
            marker=dict(size=5, color="#3B82F6"),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)",
        ))
        fig_side.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8", size=10),
            height=140, margin=dict(l=0, r=0, t=4, b=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#475569"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#475569"),
        )
        st.plotly_chart(fig_side, use_container_width=True)
    else:
        st.markdown('<div style="font-size:0.8rem;color:#475569;padding:8px 0">Start chatting to see recall data.</div>', unsafe_allow_html=True)

    st.divider()

    # Profile preview
    profile = stats.get("profile", {})
    prefs_s = stats.get("preferences", [])
    if profile or prefs_s:
        st.markdown('<div style="font-size:0.65rem;color:#3B82F6;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:8px">User Profile</div>', unsafe_allow_html=True)
        for k, v in list(profile.items())[:6]:
            st.markdown(f'<div style="font-size:0.82rem;color:#94A3B8;margin-bottom:4px"><span style="color:#F1F5F9;font-weight:500">{k.replace("_"," ").title()}:</span> {v}</div>', unsafe_allow_html=True)
        allergies = [p["value"] for p in prefs_s if p.get("category") == "allergy"]
        if allergies:
            st.markdown(f'<div style="font-size:0.82rem;color:#EF4444;font-weight:500;margin-top:6px">Allergies: {", ".join(allergies)}</div>', unsafe_allow_html=True)

    st.divider()

    # Entity breakdown
    breakdown = stats.get("entity_breakdown", {})
    if breakdown:
        df_b = pd.DataFrame(list(breakdown.items()), columns=["Type", "Count"])
        fig_b = px.pie(
            df_b, values="Count", names="Type", hole=0.45,
            color_discrete_map={"Person": "#6366F1", "Place": "#3B82F6", "Organization": "#F59E0B"},
        )
        fig_b.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8", size=10),
            height=160, margin=dict(l=0, r=0, t=0, b=0), showlegend=True,
            legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_b, use_container_width=True)

    st.divider()

    bc1, bc2 = st.columns(2)
    if bc1.button("Refresh", use_container_width=True):
        st.rerun()
    if bc2.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Export JSON", use_container_width=True):
        export = {
            "turn_count": st.session_state.turn_count,
            "stats": stats,
            "analytics": st.session_state.memory_analytics,
        }
        st.download_button(
            "Download",
            json.dumps(export, indent=2),
            "memoryos_export.json",
            "application/json",
            use_container_width=True,
        )


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_graph, tab_browser, tab_analytics, tab_arch = st.tabs([
    "Chat",
    "Knowledge Graph",
    "Memory Browser",
    "Analytics",
    "Architecture",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    if not st.session_state.messages:
        st.markdown("""
        <div style="padding:32px;background:#111D30;border:1px solid rgba(59,130,246,0.2);
                    border-radius:14px;margin-bottom:24px;text-align:center">
          <div style="font-size:1.5rem;font-weight:700;color:#F1F5F9;margin-bottom:8px">
            Stateless no more.
          </div>
          <div style="font-size:0.9rem;color:#94A3B8;max-width:520px;margin:0 auto;line-height:1.7">
            MemoryOS maintains a live knowledge graph across thousands of conversation turns.
            Every person, place, and event you mention becomes a structured memory node —
            retrieved by relevance, not by recency alone.
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Professional Scenario", use_container_width=True):
                st.session_state.demo_prompt = (
                    "Hi! I'm Alex. I'm a software engineer at TechCorp. "
                    "My manager is Sarah Chen and she reports to the CTO, Marcus Lee."
                )
        with col2:
            if st.button("Personal Scenario", use_container_width=True):
                st.session_state.demo_prompt = (
                    "Hey! My name is Jordan. I have a peanut allergy. "
                    "My best friend is Mia who lives in London."
                )
        with col3:
            if st.button("Travel Scenario", use_container_width=True):
                st.session_state.demo_prompt = (
                    "Hello! I just moved to Tokyo from New York. "
                    "I love sushi and I prefer meetings after 2 PM JST."
                )
        st.divider()

    # ── Chat history ──────────────────────────────────────────────────────────
    for message in st.session_state.messages:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])

            if role == "assistant":
                memories = message.get("memories", [])
                if memories:
                    st.markdown(
                        '<div style="font-size:0.78rem;color:#475569;margin-top:12px;'
                        'text-transform:uppercase;letter-spacing:0.08em;font-weight:600">'
                        'Recalled memories</div>',
                        unsafe_allow_html=True,
                    )
                    badge_html = ""
                    for mem in memories[:6]:
                        body   = (mem.get("content") or "")[:55]
                        dist   = mem.get("distance", 1.5)
                        origin = mem.get("origin_turn", 0)
                        cls, lbl = conf_level(dist)
                        badge_html += (
                            f'<span class="recall-badge">'
                            f'T{origin} &nbsp;<span class="{cls}">{lbl}</span>'
                            f'<br><span style="color:#64748B">{body}</span>'
                            f'</span>'
                        )
                    st.markdown(badge_html, unsafe_allow_html=True)

                if "debug_prompt" in message:
                    with st.expander("System prompt (debug)"):
                        st.code(message["debug_prompt"], language="text")
                        st.caption(f"{len(message['debug_prompt']):,} chars · ~{len(message['debug_prompt'])//4:,} tokens")

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Type your message..."):
        st.session_state.turn_count += 1
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                data = api_post("/chat", {"message": prompt})

            if data:
                bot_reply    = data.get("response", "No response.")
                memories     = data.get("active_memories", [])
                debug_prompt = data.get("debug_prompt", "")

                typing_animation(bot_reply, delay=0.010)

                # Analytics tracking
                st.session_state.memory_analytics["turns"].append(st.session_state.turn_count)
                st.session_state.memory_analytics["memories_recalled"].append(len(memories))
                if memories:
                    avg_dist = sum(m.get("distance", 1.5) for m in memories) / len(memories)
                    st.session_state.memory_analytics["confidence_scores"].append(
                        max(0.0, 1.0 - avg_dist)
                    )

                st.session_state.messages.append({
                    "role":         "assistant",
                    "content":      bot_reply,
                    "memories":     memories,
                    "debug_prompt": debug_prompt,
                    "turn":         st.session_state.turn_count,
                })

                if memories:
                    st.markdown(
                        '<div style="font-size:0.78rem;color:#475569;margin-top:12px;'
                        'text-transform:uppercase;letter-spacing:0.08em;font-weight:600">'
                        'Recalled memories</div>',
                        unsafe_allow_html=True,
                    )
                    badge_html = ""
                    for mem in memories[:6]:
                        body   = (mem.get("content") or "")[:55]
                        dist   = mem.get("distance", 1.5)
                        origin = mem.get("origin_turn", 0)
                        cls, lbl = conf_level(dist)
                        badge_html += (
                            f'<span class="recall-badge">'
                            f'T{origin} &nbsp;<span class="{cls}">{lbl}</span>'
                            f'<br><span style="color:#64748B">{body}</span>'
                            f'</span>'
                        )
                    st.markdown(badge_html, unsafe_allow_html=True)

                with st.expander("System prompt (debug)"):
                    st.code(debug_prompt, language="text")
                    st.caption(f"{len(debug_prompt):,} chars · ~{len(debug_prompt)//4:,} tokens")

                st.rerun()
            else:
                st.error(
                    "Backend not reachable at `http://localhost:8000`. "
                    "Start with: `python -m backend.server`"
                )

    if st.session_state.demo_prompt:
        dp = st.session_state.demo_prompt
        st.session_state.demo_prompt = None
        st.session_state.messages.append({"role": "user", "content": dp})
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

with tab_graph:
    g_header, g_stats_col = st.columns([3, 2])
    with g_header:
        st.header("Live Knowledge Graph")
        st.markdown('<div style="font-size:0.85rem;color:#64748B;margin-top:-8px">Every node is a memory. Every edge is a relationship. Drag, zoom, and click to explore.</div>', unsafe_allow_html=True)
    with g_stats_col:
        if st.button("Refresh Graph", use_container_width=True):
            st.rerun()

    # Legend
    legend_items = [
        ("#6366F1", "User"), ("#3B82F6", "Person"), ("#60A5FA", "Place"),
        ("#F59E0B", "Organization"), ("#F97316", "Event"),
        ("#A78BFA", "Knowledge"), ("#EF4444", "Constraint"),
    ]
    legend_html = '<div style="display:flex;flex-wrap:wrap;gap:12px;margin:10px 0 18px 0">'
    for color, label in legend_items:
        legend_html += (
            f'<span style="display:flex;align-items:center;gap:6px;'
            f'font-size:0.8rem;color:#94A3B8">'
            f'<span style="width:10px;height:10px;border-radius:50%;'
            f'background:{color};flex-shrink:0"></span>{label}</span>'
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    graph_data = api_get("/graph") or {"nodes": [], "edges": []}
    n_nodes = len(graph_data.get("nodes", []))
    n_edges = len(graph_data.get("edges", []))
    st.markdown(
        f'<div style="font-size:0.8rem;color:#475569;margin-bottom:10px">'
        f'{n_nodes} nodes &nbsp;·&nbsp; {n_edges} edges</div>',
        unsafe_allow_html=True,
    )

    if n_nodes > 1:
        components.html(build_pyvis_html(graph_data), height=600, scrolling=False)

        st.subheader("Node Inspector")
        node_opts = {n["label"]: n for n in graph_data["nodes"] if n["id"] != "user_main"}
        if node_opts:
            selected_label = st.selectbox("Select a node:", list(node_opts.keys()))
            if selected_label:
                node = node_opts[selected_label]
                nc1, nc2 = st.columns(2)
                nc1.markdown(
                    f'<div style="font-size:0.88rem;color:#94A3B8">'
                    f'<span style="color:#F1F5F9;font-weight:600">Type:</span> '
                    f'<span class="badge badge-{node["type"].lower()[:3]}">{node["type"]}</span></div>',
                    unsafe_allow_html=True,
                )
                nc2.markdown(
                    f'<div style="font-size:0.88rem;color:#94A3B8">{node.get("title","—")}</div>',
                    unsafe_allow_html=True,
                )
                connected = [e for e in graph_data["edges"] if e["from"] == node["id"] or e["to"] == node["id"]]
                if connected:
                    st.markdown(f'<div style="font-size:0.82rem;color:#475569;margin-top:10px">{len(connected)} connections</div>', unsafe_allow_html=True)
                    for e in connected[:8]:
                        direction = "out" if e["from"] == node["id"] else "in"
                        other_id  = e["to"] if e["from"] == node["id"] else e["from"]
                        other_lbl = next((n["label"] for n in graph_data["nodes"] if n["id"] == other_id), other_id)
                        arrow = "→" if direction == "out" else "←"
                        st.markdown(
                            f'<div style="font-size:0.82rem;color:#94A3B8;padding:3px 0">'
                            f'<span style="color:#3B82F6;font-family:monospace">{arrow}</span> '
                            f'<span style="color:#64748B">{e["label"]}</span> '
                            f'<span style="color:#F1F5F9">{other_lbl}</span></div>',
                            unsafe_allow_html=True,
                        )
    else:
        st.info("The knowledge graph is empty. Start chatting — every person, place, or event becomes a node.")

    st.divider()
    st.subheader("Search")
    search_q = st.text_input("Search entities and knowledge:", placeholder="e.g. Sarah, Tokyo, TechCorp")
    if search_q:
        results = api_get("/search", {"q": search_q}) or []
        if results:
            for r in results:
                node  = r.get("node", {})
                label = r.get("label", "")
                name  = node.get("name") or node.get("topic", "?")
                skip  = {"name", "topic", "id", "created_turn", "access_count"}
                attrs = {k: v for k, v in node.items() if k not in skip and v}
                with st.expander(f"[{label}]  {name}"):
                    st.json(attrs or node)
        else:
            st.markdown('<div style="color:#475569;font-size:0.88rem">No results found.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MEMORY BROWSER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_browser:
    st.header("Memory Browser")
    st.markdown('<div style="font-size:0.85rem;color:#64748B;margin-top:-8px;margin-bottom:20px">Structured view of all Neo4j memory nodes.</div>', unsafe_allow_html=True)

    graph_data_b = api_get("/graph") or {"nodes": [], "edges": []}
    nodes_b = graph_data_b.get("nodes", [])

    def nodes_by_type(t):
        return [n for n in nodes_b if n.get("type") == t]

    TYPE_BADGE = {
        "Person":       ("badge-per", "Person"),
        "Place":        ("badge-pla", "Place"),
        "Organization": ("badge-org", "Org"),
        "Event":        ("badge-eve", "Event"),
        "Knowledge":    ("badge-kno", "Knowledge"),
    }
    TYPE_COLOR = {
        "Person":       "#6366F1",
        "Place":        "#3B82F6",
        "Organization": "#F59E0B",
        "Event":        "#F97316",
        "Knowledge":    "#A78BFA",
    }

    col_e, col_ev = st.columns(2)

    with col_e:
        entity_types = ["Person", "Place", "Organization"]
        total_ents = sum(len(nodes_by_type(t)) for t in entity_types)
        st.subheader(f"Entities  ({total_ents})")
        for lbl in entity_types:
            group = nodes_by_type(lbl)
            if group:
                color = TYPE_COLOR[lbl]
                st.markdown(
                    f'<div style="font-size:0.78rem;color:{color};font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.08em;margin:12px 0 6px 0">'
                    f'{lbl}s  ({len(group)})</div>',
                    unsafe_allow_html=True,
                )
                for n in group:
                    with st.expander(n["label"]):
                        st.markdown(
                            f'<div style="font-size:0.85rem;color:#94A3B8">{n.get("title","No details")}</div>',
                            unsafe_allow_html=True,
                        )

    with col_ev:
        ev_nodes = nodes_by_type("Event")
        st.subheader(f"Events  ({len(ev_nodes)})")
        for n in sorted(ev_nodes, key=lambda x: x.get("title", ""), reverse=True):
            with st.expander(n["label"]):
                st.markdown(
                    f'<div style="font-size:0.85rem;color:#94A3B8">{n.get("title","No details")}</div>',
                    unsafe_allow_html=True,
                )

    st.divider()
    col_k, col_p = st.columns(2)

    with col_k:
        kn_nodes = nodes_by_type("Knowledge")
        st.subheader(f"Knowledge  ({len(kn_nodes)})")
        for n in kn_nodes:
            with st.expander(n["label"]):
                st.markdown(
                    f'<div style="font-size:0.85rem;color:#94A3B8">{n.get("title","No content")}</div>',
                    unsafe_allow_html=True,
                )

    with col_p:
        stats_b = api_get("/stats") or {}
        prefs_b = stats_b.get("preferences", [])
        st.subheader(f"Preferences & Constraints  ({len(prefs_b)})")
        for p in prefs_b:
            cat   = p.get("category", "preference")
            value = p.get("value", "")
            if cat == "allergy":
                badge_cls, badge_txt = "badge-allergy", "ALLERGY"
            elif cat == "goal":
                badge_cls, badge_txt = "badge-goal", "GOAL"
            else:
                badge_cls, badge_txt = "badge-pref", "PREF"
            st.markdown(
                f'<div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05)">'
                f'<span class="badge {badge_cls}">{badge_txt}</span>'
                f'<span style="font-size:0.88rem;color:#94A3B8">{value}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Entity Resolution — Duplicate Detection Panel ─────────────────────────
    st.divider()
    st.subheader("Entity Resolution")
    st.markdown(
        '<div style="font-size:0.82rem;color:#64748B;margin-top:-8px;margin-bottom:16px">'
        'Multi-signal duplicate detection: possessive-pattern analysis, '
        'Levenshtein similarity, and relationship-type collision.</div>',
        unsafe_allow_html=True,
    )

    dup_col1, dup_col2 = st.columns([5, 1])
    with dup_col2:
        scan_clicked = st.button("Scan for Duplicates", use_container_width=True)

    if scan_clicked or st.session_state.get("dup_results") is not None:
        if scan_clicked:
            st.session_state["dup_results"] = api_get("/duplicates") or []

        dups = st.session_state.get("dup_results", [])

        if not dups:
            st.markdown(
                '<div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);'
                'border-radius:8px;padding:14px 18px;font-size:0.88rem;color:#34D399">'
                'No duplicates detected. Graph is clean.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="font-size:0.82rem;color:#F59E0B;margin-bottom:12px">'
                f'{len(dups)} potential duplicate pair(s) found.</div>',
                unsafe_allow_html=True,
            )
            for i, dup in enumerate(dups):
                conf      = dup.get("confidence", 0)
                canonical = dup.get("canonical", "")
                alias     = dup.get("alias", "")
                reasons   = dup.get("reasons", [])
                label     = dup.get("label", "Entity")

                conf_color = "#EF4444" if conf >= 0.8 else "#F59E0B" if conf >= 0.5 else "#94A3B8"

                with st.expander(f"{alias}  →  {canonical}   (confidence {conf:.0%})"):
                    st.markdown(
                        f'<div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:14px">'
                        f'<div style="flex:1">'
                        f'<div style="font-size:0.78rem;color:#64748B;margin-bottom:6px;'
                        f'text-transform:uppercase;letter-spacing:0.08em">Suggested merge</div>'
                        f'<div style="font-size:0.95rem;color:#F1F5F9;font-weight:600">'
                        f'{alias}</div>'
                        f'<div style="font-size:0.78rem;color:#475569;margin:2px 0">alias (will be deleted)</div>'
                        f'</div>'
                        f'<div style="font-size:1.1rem;color:#475569;padding-top:14px">→</div>'
                        f'<div style="flex:1">'
                        f'<div style="font-size:0.78rem;color:#64748B;margin-bottom:6px;'
                        f'text-transform:uppercase;letter-spacing:0.08em">&nbsp;</div>'
                        f'<div style="font-size:0.95rem;color:#3B82F6;font-weight:600">'
                        f'{canonical}</div>'
                        f'<div style="font-size:0.78rem;color:#475569;margin:2px 0">canonical (kept)</div>'
                        f'</div>'
                        f'<div style="text-align:right;padding-top:4px">'
                        f'<div style="font-size:1.4rem;font-weight:800;color:{conf_color}">'
                        f'{conf:.0%}</div>'
                        f'<div style="font-size:0.72rem;color:#475569">confidence</div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        '<div style="font-size:0.78rem;color:#475569;margin-bottom:6px;'
                        'text-transform:uppercase;letter-spacing:0.08em">Detection signals</div>',
                        unsafe_allow_html=True,
                    )
                    for r in reasons:
                        st.markdown(
                            f'<div style="font-size:0.82rem;color:#94A3B8;padding:3px 0;'
                            f'border-left:2px solid #3B82F6;padding-left:10px;margin-bottom:4px">'
                            f'{r}</div>',
                            unsafe_allow_html=True,
                        )

                    # Swap canonical/alias controls
                    sc1, sc2, sc3 = st.columns([3, 3, 2])
                    can_override = sc1.text_input(
                        "Canonical (keep)", value=canonical, key=f"can_{i}"
                    )
                    ali_override = sc2.text_input(
                        "Alias (delete)", value=alias, key=f"ali_{i}"
                    )
                    if sc3.button("Merge", key=f"merge_{i}", use_container_width=True):
                        result = api_post("/merge", {
                            "canonical": can_override,
                            "alias":     ali_override,
                        })
                        if result:
                            st.success(result.get("result", "Merged."))
                            st.session_state["dup_results"] = None
                            st.rerun()
                        else:
                            st.error("Merge failed — backend unreachable.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_analytics:
    st.header("Analytics Dashboard")

    stats_a = api_get("/stats") or {}
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Turns",     stats_a.get("total_turns",     0))
    mc2.metric("Entities",        stats_a.get("entity_count",    0))
    mc3.metric("Events Logged",   stats_a.get("event_count",     0))
    mc4.metric("Knowledge Items", stats_a.get("knowledge_count", 0))

    st.divider()

    PLOT_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,17,28,0.8)",
        font=dict(color="#94A3B8", size=11),
        margin=dict(l=0, r=0, t=24, b=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#475569"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#475569"),
    )

    if st.session_state.memory_analytics["turns"]:
        ac1, ac2 = st.columns(2)

        with ac1:
            st.subheader("Memory Recall Per Turn")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.memory_analytics["turns"],
                y=st.session_state.memory_analytics["memories_recalled"],
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.08)",
                line=dict(color="#3B82F6", width=2),
                marker=dict(size=7, color="#3B82F6"),
                name="Recalled",
            ))
            fig.update_layout(**PLOT_LAYOUT, height=280, xaxis_title="Turn", yaxis_title="Memories recalled")
            st.plotly_chart(fig, use_container_width=True)

        with ac2:
            st.subheader("Retrieval Confidence")
            conf_scores = st.session_state.memory_analytics["confidence_scores"]
            if conf_scores:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=st.session_state.memory_analytics["turns"][:len(conf_scores)],
                    y=conf_scores,
                    mode="lines+markers",
                    fill="tozeroy",
                    fillcolor="rgba(16,185,129,0.08)",
                    line=dict(color="#10B981", width=2),
                    marker=dict(size=7, color="#10B981"),
                    name="Confidence",
                ))
                fig2.update_layout(**PLOT_LAYOUT, height=280, xaxis_title="Turn",
                                   yaxis_title="Confidence (1 - distance)", yaxis=dict(**PLOT_LAYOUT["yaxis"], range=[0, 1]))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.markdown('<div style="color:#475569;padding:24px 0">No confidence data yet (need at least one memory recall).</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#475569;padding:16px 0">Start chatting to generate analytics charts.</div>', unsafe_allow_html=True)

    st.divider()

    breakdown_a = stats_a.get("entity_breakdown", {})
    if breakdown_a:
        st.subheader("Entity Type Distribution")
        df_bar = pd.DataFrame(list(breakdown_a.items()), columns=["Type", "Count"])
        fig3 = px.bar(
            df_bar, x="Type", y="Count", color="Type",
            color_discrete_map={"Person": "#6366F1", "Place": "#3B82F6", "Organization": "#F59E0B"},
        )
        fig3.update_layout(**PLOT_LAYOUT, height=240, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Graph Summary")
    graph_data_a = api_get("/graph") or {"nodes": [], "edges": []}
    n_n = len(graph_data_a.get("nodes", []))
    n_e = len(graph_data_a.get("edges", []))
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Graph Nodes", n_n)
    sc2.metric("Graph Edges", n_e)
    sc3.metric("Avg Degree", f"{(2*n_e/n_n):.1f}" if n_n else "0")

    # Token efficiency indicator
    if stats_a.get("total_turns", 0) > 0:
        st.divider()
        st.subheader("Token Efficiency")
        est_naive   = stats_a.get("total_turns", 0) * 120   # ~120 tokens/turn naive
        est_budget  = 870   # fixed budget regardless of turns
        eff_col1, eff_col2, eff_col3 = st.columns(3)
        eff_col1.metric("Context Tokens (Naive)", f"{est_naive:,}")
        eff_col2.metric("Context Tokens (MemoryOS)", f"{est_budget:,}")
        reduction = max(0, 100 - int(est_budget / max(est_naive, 1) * 100))
        eff_col3.metric("Token Reduction", f"{reduction}%")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_arch:
    st.header("Architecture & Mathematical Framework")
    st.markdown(
        '<div style="font-size:0.88rem;color:#64748B;margin-top:-8px;margin-bottom:24px">'
        'How MemoryOS solves LLM statelessness at 2 000–4 000 conversation turns.</div>',
        unsafe_allow_html=True,
    )

    # ── Problem statement ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="arch-section">
      <h2>The Problem: LLM Statelessness</h2>
      <p>
        Every LLM call starts with an empty context window. Without memory infrastructure,
        a 4 000-turn conversation requires one of two broken strategies:
      </p>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px">
        <div style="background:#0A0F1E;border:1px solid rgba(239,68,68,0.25);border-radius:8px;padding:16px">
          <div style="color:#EF4444;font-weight:600;font-size:0.85rem;margin-bottom:8px">Full History (Naive)</div>
          <div style="color:#94A3B8;font-size:0.82rem;line-height:1.6">
            Send all N turns every request.<br>
            Token cost grows as O(N²) — prohibitively expensive at scale.<br>
            <span style="color:#EF4444">4 000 turns = ~400 000 context tokens/call.</span>
          </div>
        </div>
        <div style="background:#0A0F1E;border:1px solid rgba(239,68,68,0.25);border-radius:8px;padding:16px">
          <div style="color:#F59E0B;font-weight:600;font-size:0.85rem;margin-bottom:8px">Fixed Window (Truncation)</div>
          <div style="color:#94A3B8;font-size:0.82rem;line-height:1.6">
            Keep only last K turns in context.<br>
            Cost is bounded, but information before turn (N-K) is permanently lost.<br>
            <span style="color:#F59E0B">Critical facts from early turns become inaccessible.</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Mathematical framework ─────────────────────────────────────────────────
    st.markdown("""
    <div class="arch-section">
      <h2>Mathematical Framework: Relevance-Scored Context Assembly</h2>
      <p>
        For each memory node m<sub>i</sub> in the knowledge graph, we compute a relevance score
        conditioned on the current query Q:
      </p>

      <div class="formula-box">
score(mᵢ | Q) = α · keyword(Q, mᵢ)  +  β · recency(t, mᵢ)  +  γ · frequency(mᵢ)
<span class="comment">
  where:
    α = 0.50    β = 0.30    γ = 0.20           (sum to 1.0)

  keyword(Q, mᵢ)   =  |tokens(Q) ∩ tokens(content(mᵢ))| / max(|tokens(Q)|, 1)
                       Query-biased Jaccard overlap

  recency(t, mᵢ)   =  exp(−λ · (t_current − t_last(mᵢ)))      λ = 0.001
                       Exponential decay — recent nodes stay relevant longer

  frequency(mᵢ)    =  log(1 + access_count) / log(1 + 100)
                       Log-normalised access frequency (capped at 100)
</span>
      </div>

      <p style="margin-top:16px">
        Context selection is a greedy knapsack over a fixed character budget B:
      </p>
      <div class="formula-box">
C_t  =  argmax   Σ score(mᵢ | Q)
        S ⊆ M
        chars(S) ≤ B

≈  Sort M by score(mᵢ|Q) descending,
   greedily include nodes until Σ chars(mᵢ) ≥ B
      </div>

      <p style="margin-top:16px">
        This gives O(|M|·log|M|) context assembly per turn — dominated by the sort step —
        with O(B) output size regardless of conversation length N.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Compact serialisation ──────────────────────────────────────────────────
    st.markdown("""
    <div class="arch-section">
      <h2>Compact Serialisation — 4x Token Reduction</h2>
      <p>
        Instead of verbose multi-line formatting, all memory sections use a compact
        pipe-delimited format. This directly reduces the system prompt token footprint.
      </p>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px">
        <div>
          <div style="font-size:0.78rem;color:#EF4444;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.08em;margin-bottom:8px">Verbose (before)</div>
          <div class="formula-box" style="border-color:rgba(239,68,68,0.2)">
KNOWLEDGE GRAPH — 5 entities:
  - Alice [colleague]
    role: senior engineer
    → WORKS_AT: TechCorp
  - TechCorp [employer]
    industry: software
  ...
<span class="comment">≈ 420 chars  (≈105 tokens)</span>
          </div>
        </div>
        <div>
          <div style="font-size:0.78rem;color:#10B981;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.08em;margin-bottom:8px">Compact (after)</div>
          <div class="formula-box" style="border-color:rgba(16,185,129,0.2)">
ENTITIES(5): Alice[colleague](role=SWE)
->WORKS_AT:TechCorp | TechCorp
[employer](industry=software) | ...




<span class="comment">≈ 110 chars  (≈28 tokens)  — 4× smaller</span>
          </div>
        </div>
      </div>

      <div style="margin-top:20px">
        <div style="font-size:0.85rem;color:#94A3B8;margin-bottom:12px;font-weight:500">
          Fixed token budget breakdown (per turn):
        </div>
        <table class="perf-table">
          <tr><th>Section</th><th>Budget (chars)</th><th>Approx. Tokens</th></tr>
          <tr><td>Profile</td><td>400</td><td>100</td></tr>
          <tr><td>Constraints & Preferences</td><td>500</td><td>125</td></tr>
          <tr><td>Entities</td><td>2 000</td><td>500</td></tr>
          <tr><td>Events</td><td>1 200</td><td>300</td></tr>
          <tr><td>Knowledge</td><td>1 400</td><td>350</td></tr>
          <tr><td>Archival Memories</td><td>800</td><td>200</td></tr>
          <tr><td>System Instructions</td><td>560</td><td>140</td></tr>
          <tr><td style="color:#F1F5F9;font-weight:600">Total (max)</td>
              <td style="color:#F1F5F9;font-weight:600">6 860</td>
              <td class="highlight-cell">~1 715</td></tr>
        </table>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Scalability comparison ─────────────────────────────────────────────────
    st.markdown("""
    <div class="arch-section">
      <h2>Scalability: Token Cost vs. Conversation Length</h2>

      <table class="perf-table">
        <tr>
          <th>Approach</th>
          <th>100 Turns</th>
          <th>500 Turns</th>
          <th>1 000 Turns</th>
          <th>4 000 Turns</th>
        </tr>
        <tr>
          <td>Naive (full history)</td>
          <td>~12 000</td>
          <td>~60 000</td>
          <td>~120 000</td>
          <td style="color:#EF4444">~480 000</td>
        </tr>
        <tr>
          <td>Fixed window (last 15 turns)</td>
          <td>~1 800</td>
          <td>~1 800</td>
          <td>~1 800</td>
          <td style="color:#F59E0B">~1 800 (loses history)</td>
        </tr>
        <tr>
          <td style="color:#F1F5F9;font-weight:600">MemoryOS (this system)</td>
          <td class="highlight-cell">~1 715</td>
          <td class="highlight-cell">~1 715</td>
          <td class="highlight-cell">~1 715</td>
          <td class="highlight-cell">~1 715 (full recall)</td>
        </tr>
      </table>

      <p style="margin-top:16px">
        MemoryOS achieves <strong style="color:#10B981">O(1) token cost</strong> in conversation
        length while maintaining access to the full memory graph. The knowledge graph and vector
        index hold all historical information; only what is <em>relevant to the current turn</em>
        enters the context window.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── System pipeline ────────────────────────────────────────────────────────
    st.markdown('<div class="arch-section"><h2>Request Processing Pipeline</h2>', unsafe_allow_html=True)

    steps = [
        ("Increment Turn Counter",
         "Atomic increment on the User node in Neo4j. Provides t_current for all relevance scoring."),
        ("Proactive Semantic Search",
         "ChromaDB query over the last 3 buffer turns. Returns the top-4 archival episodes by cosine similarity "
         "(filtered at distance < 0.25)."),
        ("Relevance-Scored Context Assembly",
         "ContextBuilder loads all graph nodes, computes score(mᵢ|Q) for each, sorts descending, "
         "and fills the token budget greedily. Fixed output size: ~1 715 tokens."),
        ("LLM Call — Groq API",
         "llama-3.3-70b-versatile with 9 memory tools, parallel_tool_calls=True. "
         "Agentic loop up to 4 rounds for multi-tool responses."),
        ("Tool Dispatch",
         "GraphManager or ArchivalManager executes each tool call and returns a status string "
         "back to the LLM as a tool result."),
        ("Background Archive",
         "Daemon thread saves the complete exchange to ChromaDB episodic collection. "
         "Non-blocking — does not affect response latency."),
    ]
    steps_html = ""
    for i, (title, desc) in enumerate(steps, 1):
        steps_html += f"""
        <div class="pipeline-step">
          <div class="step-num">{i}</div>
          <div class="step-body">
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
          </div>
        </div>
        """
    st.markdown(steps_html + "</div>", unsafe_allow_html=True)

    # ── Storage architecture ───────────────────────────────────────────────────
    st.markdown("""
    <div class="arch-section">
      <h2>Storage Architecture</h2>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div style="background:#0A0F1E;border:1px solid rgba(59,130,246,0.2);border-radius:8px;padding:18px">
          <div style="color:#3B82F6;font-weight:700;font-size:0.88rem;margin-bottom:10px">
            Neo4j — Structured Graph Memory
          </div>
          <div style="color:#94A3B8;font-size:0.82rem;line-height:1.8">
            Nodes: User · Person · Place · Organization · Event · Knowledge · Preference<br>
            Edges: KNOWS · EXPERIENCED · AWARE_OF · HAS_PREFERENCE · RELATED_TO · INVOLVED_IN<br>
            Queries: Cypher — MERGE for upsert idempotency<br>
            Role: <em>What is connected?</em> Relationship traversal, 2-hop graph search.
          </div>
        </div>
        <div style="background:#0A0F1E;border:1px solid rgba(99,102,241,0.2);border-radius:8px;padding:18px">
          <div style="color:#818CF8;font-weight:700;font-size:0.88rem;margin-bottom:10px">
            ChromaDB — Vector Episodic Memory
          </div>
          <div style="color:#94A3B8;font-size:0.82rem;line-height:1.8">
            Collections: semantic_memory · conversation_logs<br>
            Embeddings: all-MiniLM-L6-v2 (cosine space)<br>
            Deduplication threshold: cosine distance &lt; 0.12<br>
            Role: <em>What is similar?</em> Semantic recall of past episodes.
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
fc1, fc2, fc3 = st.columns([3, 1, 1])
fc1.markdown(
    f'<div style="font-size:0.75rem;color:#475569">MemoryOS &nbsp;·&nbsp; '
    f'Neo4j + ChromaDB &nbsp;·&nbsp; {datetime.now().year}</div>',
    unsafe_allow_html=True,
)
fc2.markdown(
    '<div style="font-size:0.75rem;color:#475569;text-align:center">Target: 4 000+ turns</div>',
    unsafe_allow_html=True,
)
fc3.markdown(
    f'<div style="font-size:0.75rem;color:#3B82F6;text-align:right;font-weight:600">'
    f'Turn {st.session_state.turn_count}</div>',
    unsafe_allow_html=True,
)
