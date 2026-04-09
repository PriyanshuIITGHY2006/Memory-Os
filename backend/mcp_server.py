#!/usr/bin/env python
"""
MemoryOS MCP Server
====================
Exposes MemoryOS as a Model Context Protocol (MCP) tool server over stdio.
Claude Desktop can connect to this server to give any Claude conversation
direct access to the user's personal memory graph.

Usage (add to Claude Desktop's claude_desktop_config.json):
  {
    "mcpServers": {
      "memoryos": {
        "command": "python",
        "args": ["-m", "backend.mcp_server"],
        "env": {
          "MEMORYOS_USER_ID":   "<your-user-id>",
          "MEMORYOS_GROQ_KEY":  "<your-groq-api-key>"
        }
      }
    }
  }

Tools exposed:
  recall_memories(query)          — semantic search across graph + archive
  get_user_profile()              — structured profile + preferences
  get_recent_events(limit)        — last N events from timeline
  get_entities(type)              — people / places / organizations
  save_memory(content, category)  — store a fact or event
"""

import json
import os
import sys
import logging

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("memoryos-mcp")

# ── JSON-RPC 2.0 helpers ───────────────────────────────────────────────────────

def _send(obj: dict):
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _ok(req_id, result):
    _send({"jsonrpc": "2.0", "id": req_id, "result": result})


def _err(req_id, code: int, message: str):
    _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


# ── MCP protocol constants ────────────────────────────────────────────────────

TOOLS = [
    {
        "name":        "recall_memories",
        "description": "Search the user's personal memory graph and archive for information relevant to a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"}
            },
            "required": ["query"],
        },
    },
    {
        "name":        "get_user_profile",
        "description": "Return the user's stored profile (name, occupation, location, etc.) and preferences.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name":        "get_recent_events",
        "description": "Return the most recent events and activities from the user's memory timeline.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of events (default 10)", "default": 10}
            },
        },
    },
    {
        "name":        "get_entities",
        "description": "Return entities (people, places, organizations) the user knows.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_type": {
                    "type": "string",
                    "enum": ["person", "place", "organization", "all"],
                    "default": "all",
                }
            },
        },
    },
    {
        "name":        "save_memory",
        "description": "Save a new fact, event, or observation to the user's memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content":  {"type": "string", "description": "What to remember"},
                "category": {
                    "type": "string",
                    "enum": ["fact", "event", "preference", "knowledge"],
                    "default": "fact",
                },
            },
            "required": ["content"],
        },
    },
]


# ── Lazy manager initialisation ───────────────────────────────────────────────

_graph    = None
_archive  = None
_user_id  = None
_groq_key = None


def _init():
    global _graph, _archive, _user_id, _groq_key

    _user_id  = os.environ.get("MEMORYOS_USER_ID", "").strip()
    _groq_key = os.environ.get("MEMORYOS_GROQ_KEY", "").strip()

    if not _user_id or not _groq_key:
        logger.error("MEMORYOS_USER_ID and MEMORYOS_GROQ_KEY must be set")
        sys.exit(1)

    # Lazy import to avoid loading Neo4j driver until needed
    from backend.managers.graph_manager import GraphManager
    from backend.managers.archival_manager import ArchivalMemoryManager

    _graph   = GraphManager(_user_id)
    _archive = ArchivalMemoryManager(_user_id)
    logger.info("MemoryOS MCP server ready for user=%s", _user_id)


# ── Tool handlers ─────────────────────────────────────────────────────────────

def _recall(query: str) -> str:
    hits = _archive.search_memory(query, n_results=6)
    if not hits:
        return "No relevant memories found."
    lines = [
        f"[Turn {h['origin_turn']} | {h['source']}] {h['content'][:200]}"
        for h in hits
    ]
    return "\n".join(lines)


def _profile() -> str:
    stats = _graph.get_stats()
    profile = stats.get("profile", {})
    prefs   = stats.get("preferences", [])
    lines   = ["=== User Profile ==="]
    for k, v in profile.items():
        lines.append(f"  {k}: {v}")
    if prefs:
        lines.append("=== Preferences ===")
        for p in prefs:
            lines.append(f"  [{p.get('category','preference')}] {p.get('value','')}")
    return "\n".join(lines) if len(lines) > 1 else "No profile data stored yet."


def _events(limit: int = 10) -> str:
    timeline = _graph.get_timeline(limit=limit * 2)
    events   = [t for t in timeline if t.get("type") == "event"][:limit]
    if not events:
        return "No events stored yet."
    lines = [
        f"[Turn {e.get('turn',0)}] {e.get('title','')} — {e.get('date','')}"
        for e in events
    ]
    return "\n".join(lines)


def _entities(entity_type: str = "all") -> str:
    data  = _graph.get_graph_data()
    nodes = data.get("nodes", [])
    type_map = {"person": "Person", "place": "Place", "organization": "Organization"}
    wanted   = type_map.get(entity_type.lower(), None)
    filtered = [n for n in nodes if wanted is None or n.get("type") == wanted]
    filtered = [n for n in filtered if n.get("type") in {"Person", "Place", "Organization"}]
    if not filtered:
        return "No entities found."
    lines = [
        f"[{n['type']}] {n.get('name','')} — {n.get('relationship','')}"
        for n in filtered[:40]
    ]
    return "\n".join(lines)


def _save(content: str, category: str = "fact") -> str:
    from groq import Groq
    import config as cfg

    turn = _graph._current_turn()
    if category == "event":
        _graph.log_event(description=content, entities_involved=[], date=None)
        return f"Event saved: {content[:60]}"
    if category == "knowledge":
        # Quick LLM extract of topic vs content
        try:
            client = Groq(api_key=_groq_key)
            r = client.chat.completions.create(
                model=cfg.LLM_MODEL,
                messages=[{"role": "user",
                           "content": f'Extract a short topic (3-5 words) and a one-sentence fact from: "{content}". Respond as JSON: {{"topic":"...","fact":"..."}}'}],
                max_tokens=60,
            )
            import re
            m = re.search(r'\{.*\}', r.choices[0].message.content or "", re.DOTALL)
            if m:
                d = json.loads(m.group())
                _graph.add_general_knowledge(d.get("topic", content[:30]), d.get("fact", content))
                return f"Knowledge saved: {d.get('topic', '')}"
        except Exception:
            pass
        _graph.add_general_knowledge(content[:40], content)
        return f"Knowledge saved."
    if category == "preference":
        _graph.add_preference(content, "preference")
        return f"Preference saved: {content[:60]}"
    # default: fact → archive
    _archive.add_fact(content, turn)
    return f"Fact saved: {content[:60]}"


# ── Main dispatch ─────────────────────────────────────────────────────────────

def _call_tool(name: str, args: dict) -> str:
    if name == "recall_memories":
        return _recall(args.get("query", ""))
    if name == "get_user_profile":
        return _profile()
    if name == "get_recent_events":
        return _events(args.get("limit", 10))
    if name == "get_entities":
        return _entities(args.get("entity_type", "all"))
    if name == "save_memory":
        return _save(args.get("content", ""), args.get("category", "fact"))
    return f"Unknown tool: {name}"


# ── MCP request router ────────────────────────────────────────────────────────

def _handle(req: dict):
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params") or {}

    if method == "initialize":
        _ok(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities":    {"tools": {}},
            "serverInfo":      {"name": "memoryos", "version": "2.0.0"},
        })
        return

    if method == "notifications/initialized":
        return  # no response needed

    if method == "tools/list":
        _ok(req_id, {"tools": TOOLS})
        return

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments") or {}
        try:
            result = _call_tool(tool_name, tool_args)
            _ok(req_id, {
                "content": [{"type": "text", "text": result}],
                "isError": False,
            })
        except Exception as exc:
            _ok(req_id, {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            })
        return

    _err(req_id, -32601, f"Method not found: {method}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    _init()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _err(None, -32700, "Parse error")
            continue
        try:
            _handle(req)
        except Exception as exc:
            _err(req.get("id"), -32603, f"Internal error: {exc}")


if __name__ == "__main__":
    main()
