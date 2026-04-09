"""
MemoryOS v2 — Orchestrator (multi-tenant)
==========================================
Each Orchestrator is scoped to one (user_id, groq_api_key) pair.
The server maintains a dict[str, Orchestrator] cache keyed by user_id.

Request processing pipeline:
  1. Increment turn counter (Neo4j)
  2. Record session activity (SessionSummaryManager)
  3. Proactive archival search (ChromaDB)
  4. Build graph context + session context block (Neo4j)
  5. Call LLM with 9-tool schema
  6. Dispatch tool calls → graph / archive updates  (track memory diff)
  7. Contradiction detection (background, non-blocking)
  8. Background-save episode to ChromaDB
  9. Return (response_text, active_memories, debug_prompt, memory_diff)
"""

import json
import threading

from groq import Groq, BadRequestError, RateLimitError

import config
from backend.logic.prompts import SYSTEM_PROMPT_TEMPLATE
from backend.logic.tools import TOOLS_SCHEMA
from backend.managers.graph_manager import GraphManager
from backend.managers.archival_manager import ArchivalMemoryManager
from backend.managers.buffer_manager import BufferManager
from backend.managers.session_manager import SessionSummaryManager
from backend.managers.consolidation_engine import CLSConsolidationEngine


# ── Human-readable tool labels for the diff display ──────────────────────────
_TOOL_LABELS = {
    "update_user_profile":      "Profile updated",
    "delete_user_profile_field":"Profile field removed",
    "add_preference":           "Preference added",
    "update_entity":            "Entity saved",
    "link_entities":            "Relationship linked",
    "log_event":                "Event logged",
    "save_knowledge":           "Knowledge saved",
    "merge_entities":           "Entities merged",
    "graph_search":             "Graph searched",
    "archival_memory_search":   "Archival searched",
    # legacy aliases
    "core_memory_update":       "Profile updated",
    "delete_core_memory":       "Profile field removed",
    "update_entity_memory":     "Entity saved",
}


class Orchestrator:

    def __init__(self, user_id: str, groq_api_key: str):
        self.user_id  = user_id
        self._client  = Groq(api_key=groq_api_key)
        self.graph    = GraphManager(user_id)
        self.archive  = ArchivalMemoryManager(user_id)
        self.buffer   = BufferManager(max_turns=config.BUFFER_MAX_TURNS)
        self._session = SessionSummaryManager()   # singleton
        self._cls     = CLSConsolidationEngine()  # singleton

    # ------------------------------------------------------------------
    # Session context block builder
    # ------------------------------------------------------------------

    def _build_session_context(self, current_turn: int) -> str:
        lines = []

        # Last session summary
        summary = self.graph.get_last_session_summary()
        if summary:
            lines.append(
                f"[LAST SESSION — Turns {summary.get('start_turn')}–{summary.get('end_turn')}]\n"
                f"{summary.get('summary', '')}"
            )

        # Proactive nudges — entities not mentioned recently
        stale = self.graph.get_stale_entities(turns_threshold=30)
        if stale:
            nudge_parts = [
                f"{e['name']} ({e['rel'] or e['type']})" for e in stale
            ]
            lines.append(
                "[NUDGE — Not mentioned recently]\n"
                + ", ".join(nudge_parts)
            )

        # Pending contradictions
        contradictions = self.graph.get_pending_contradictions()
        if contradictions:
            c_lines = [
                f"• {c['field']}: was '{c.get('old_value','')}', now '{c.get('new_value','')}'"
                for c in contradictions
            ]
            lines.append("[CONTRADICTIONS FLAGGED]\n" + "\n".join(c_lines))

        if not lines:
            return ""
        return "\n\n" + "\n\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Contradiction detector (background, non-blocking)
    # ------------------------------------------------------------------

    def _detect_contradiction_bg(self, key: str, new_value: str):
        """Compare new profile value against stored one. Runs in background thread."""
        def _check():
            try:
                profile = self.graph.get_profile_raw()
                old_val = str(profile.get(key, "")).strip()
                if old_val and old_val.lower() != new_value.strip().lower():
                    # Heuristic: flag if values are substantially different
                    # (not just an update like "30 years old" → "31 years old")
                    if len(old_val) > 2 and len(new_value) > 2:
                        self.graph.save_contradiction(key, old_val, new_value)
            except Exception:
                pass
        threading.Thread(target=_check, daemon=True).start()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_message(self, user_message: str):
        try:
            current_turn = self.graph.increment_turn()
            self._session.record_activity(self.user_id, current_turn,
                                           self._client.api_key)
            self.buffer.add_turn("user", user_message)

            # 1. Proactive semantic search
            search_query  = " ".join(
                m["content"] for m in self.buffer.get_messages()[-3:]
            )
            archival_hits = self.archive.search_memory(search_query, n_results=4)

            # 2. Context
            core_ctx = self.graph.get_core_prompt(
                recent_history_text=search_query,
                archival_context=archival_hits,
                current_turn=current_turn,
            )
            session_ctx = self._build_session_context(current_turn)

            # 3. Messages
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                core_memory_block=core_ctx,
                session_context_block=session_ctx,
            )
            messages = [{"role": "system", "content": system_prompt}]
            for msg in self.buffer.get_messages():
                if msg["content"] != user_message:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_message})

            active_memories = list(archival_hits)
            memory_changes  = []
            final_response  = ""

            # 4. Agentic loop
            for _round in range(4):
                try:
                    completion = self._client.chat.completions.create(
                        model=config.LLM_MODEL,
                        messages=messages,
                        tools=TOOLS_SCHEMA,
                        tool_choice="auto",
                        parallel_tool_calls=True,
                        max_tokens=1024,
                    )
                except RateLimitError:
                    return "Rate limit reached — please wait a moment.", [], "", []
                except BadRequestError as exc:
                    if "tool_use_failed" in str(exc):
                        messages.append({
                            "role": "user",
                            "content": "SYSTEM: Invalid tool format. Please retry with valid JSON.",
                        })
                        continue
                    raise

                response_msg = completion.choices[0].message

                if response_msg.tool_calls:
                    messages.append(response_msg)
                    for tc in response_msg.tool_calls:
                        fname = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                        except Exception:
                            args = {}
                        result = self._dispatch(fname, args, current_turn,
                                                active_memories, memory_changes)
                        messages.append({
                            "tool_call_id": tc.id,
                            "role":         "tool",
                            "name":         fname,
                            "content":      str(result),
                        })
                else:
                    text = (response_msg.content or "").strip()
                    if "<function" in text or (
                        "{" in text and "type" in text and "function" in text
                    ):
                        messages.append({
                            "role": "user",
                            "content": "SYSTEM: Raw function code detected. Respond in plain text.",
                        })
                        continue
                    final_response = text
                    break

            if not final_response:
                final_response = "I'm having trouble responding right now. Please try again."

            self.buffer.add_turn("assistant", final_response)

            def _bg_save():
                self.archive.add_memory(
                    f"User: {user_message}\nAssistant: {final_response}",
                    "assistant", current_turn,
                )
            threading.Thread(target=_bg_save, daemon=True).start()

            return final_response, active_memories, system_prompt, memory_changes

        except Exception as exc:
            import traceback; traceback.print_exc()
            return f"System error: {exc}", [], "", []

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, fname: str, args: dict, turn: int,
                  active_memories: list, memory_changes: list):
        """Route a tool call to the correct manager method and record the change."""
        try:
            result = self._dispatch_inner(fname, args, turn, active_memories)
            label  = _TOOL_LABELS.get(fname)
            if label:
                detail = ""
                if fname in ("update_user_profile", "core_memory_update"):
                    detail = f"{args.get('key','')} → {args.get('value','')}"
                elif fname == "update_entity":
                    detail = args.get("name", "")
                elif fname == "add_preference":
                    detail = args.get("value", "")
                elif fname == "log_event":
                    detail = (args.get("description", "") or "")[:60]
                elif fname == "save_knowledge":
                    detail = args.get("topic", "")
                elif fname == "merge_entities":
                    detail = f"{args.get('alias_name','')} → {args.get('canonical_name','')}"
                memory_changes.append({"label": label, "detail": detail})
            return result
        except Exception as exc:
            return f"Tool error ({fname}): {exc}"

    def _dispatch_inner(self, fname: str, args: dict, turn: int, active_memories: list):
        # ── Profile ──────────────────────────────────────────────────
        if fname == "update_user_profile":
            key = args["key"]
            val = args["value"]
            self._detect_contradiction_bg(key, val)
            return self.graph.update_profile(key, val)

        if fname == "delete_user_profile_field":
            return self.graph.remove_from_profile(
                args["key"], args.get("value_to_remove", "")
            )

        # ── Preferences ──────────────────────────────────────────────
        if fname == "add_preference":
            return self.graph.add_preference(
                args["value"], args.get("category", "preference")
            )

        # ── Entities ─────────────────────────────────────────────────
        if fname == "update_entity":
            return self.graph.update_entity(
                name=args["name"],
                relationship=args.get("relationship_to_user"),
                attributes=args.get("attributes", {}),
                entity_type=args.get("entity_type", "person"),
            )

        if fname == "link_entities":
            return self.graph.link_entities(
                args["entity_a"], args["entity_b"],
                args["relationship_type"], args.get("description", ""),
            )

        # ── Events ───────────────────────────────────────────────────
        if fname == "log_event":
            return self.graph.log_event(
                description=args["description"],
                entities_involved=args.get("entities_involved", []),
                date=args.get("date"),
            )

        # ── Knowledge ────────────────────────────────────────────────
        if fname == "save_knowledge":
            return self.graph.add_general_knowledge(
                args["topic"], args["content"]
            )

        # ── Entity resolution ────────────────────────────────────────
        if fname == "merge_entities":
            return self.graph.merge_entities(
                args["canonical_name"], args["alias_name"]
            )

        # ── Graph traversal ──────────────────────────────────────────
        if fname == "graph_search":
            return self.graph.graph_search(
                args["entity_name"], args.get("depth", 2)
            )

        # ── Archival vector search ────────────────────────────────────
        if fname == "archival_memory_search":
            hits = self.archive.search_memory(args["query"], n_results=5)
            active_memories.extend(hits)
            if hits:
                lines = [
                    f"[Turn {h['origin_turn']} | dist={h['distance']:.2f}] {h['content'][:120]}"
                    for h in hits
                ]
                return "Archival results:\n" + "\n".join(lines)
            return "No relevant archival memories found."

        # ── Legacy aliases ────────────────────────────────────────────
        if fname == "core_memory_update":
            self._detect_contradiction_bg(args["key"], args["value"])
            return self.graph.update_profile(args["key"], args["value"])
        if fname == "delete_core_memory":
            return self.graph.remove_from_profile(
                args["key"], args.get("value_to_remove", "")
            )
        if fname == "update_entity_memory":
            return self.graph.update_entity(
                args["name"], args.get("relationship"), args.get("attributes", {})
            )

        return f"Unknown tool: {fname}"

    # ------------------------------------------------------------------
    # Streaming entry point (yields SSE event dicts)
    # ------------------------------------------------------------------

    def process_message_stream(self, user_message: str):
        """
        Generator that yields SSE event dicts for token-by-token streaming.

        Event types:
          {"type": "tool",   "name": <str>,  "detail": <str>}
          {"type": "token",  "content": <str>}
          {"type": "done",   "active_memories": [...], "turn": <int>,
                             "memory_changes": [...]}
          {"type": "error",  "content": <str>}
        """
        try:
            current_turn = self.graph.increment_turn()
            self._session.record_activity(self.user_id, current_turn,
                                           self._client.api_key)
            self.buffer.add_turn("user", user_message)

            search_query  = " ".join(
                m["content"] for m in self.buffer.get_messages()[-3:]
            )
            archival_hits = self.archive.search_memory(search_query, n_results=4)
            core_ctx      = self.graph.get_core_prompt(
                recent_history_text=search_query,
                archival_context=archival_hits,
                current_turn=current_turn,
            )
            session_ctx = self._build_session_context(current_turn)

            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                core_memory_block=core_ctx,
                session_context_block=session_ctx,
            )
            messages = [{"role": "system", "content": system_prompt}]
            for msg in self.buffer.get_messages():
                if msg["content"] != user_message:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_message})

            active_memories = list(archival_hits)
            memory_changes  = []

            # Phase 1: tool call rounds (non-streaming)
            for _round in range(3):
                try:
                    completion = self._client.chat.completions.create(
                        model=config.LLM_MODEL,
                        messages=messages,
                        tools=TOOLS_SCHEMA,
                        tool_choice="auto",
                        parallel_tool_calls=True,
                        max_tokens=1024,
                    )
                except RateLimitError:
                    yield {"type": "error", "content": "Rate limit reached — please wait a moment."}
                    return

                response_msg = completion.choices[0].message
                if not response_msg.tool_calls:
                    break

                messages.append(response_msg)
                for tc in response_msg.tool_calls:
                    fname = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                    label = _TOOL_LABELS.get(fname, fname.replace("_", " "))
                    yield {"type": "tool", "name": fname, "label": label}
                    result = self._dispatch(fname, args, current_turn,
                                            active_memories, memory_changes)
                    messages.append({
                        "tool_call_id": tc.id,
                        "role":         "tool",
                        "name":         fname,
                        "content":      str(result),
                    })

            # Phase 2: streaming final response
            final_response = ""
            try:
                stream = self._client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    max_tokens=1024,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        final_response += delta.content
                        yield {"type": "token", "content": delta.content}
            except RateLimitError:
                yield {"type": "error", "content": "Rate limit reached — please wait a moment."}
                return

            if not final_response:
                final_response = "I'm having trouble responding right now. Please try again."
                yield {"type": "token", "content": final_response}

            self.buffer.add_turn("assistant", final_response)

            # Collect entity names that were retrieved/activated this turn
            # for Hebbian co-activation strengthening
            retrieved_names = [
                m.get("content", "").split("\n")[0].strip()
                for m in active_memories
                if m.get("content")
            ]
            # Also grab entities from memory_changes
            for ch in memory_changes:
                if ch.get("label") in ("Entity saved", "Relationship linked"):
                    name = ch.get("detail", "").split("→")[0].strip()
                    if name:
                        retrieved_names.append(name)

            def _bg(turn=current_turn, names=retrieved_names, msg=user_message, resp=final_response):
                self.archive.add_memory(
                    f"User: {msg}\nAssistant: {resp}",
                    "assistant", turn,
                )
                # Hebbian co-activation learning
                if len(names) >= 2:
                    self.graph.hebbian_strengthen(names, turn)
                # CLS consolidation tick
                self._cls.tick(self.user_id, turn, self._client.api_key)

            threading.Thread(target=_bg, daemon=True).start()

            yield {
                "type":            "done",
                "active_memories": active_memories,
                "turn":            current_turn,
                "memory_changes":  memory_changes,
            }

        except Exception as exc:
            import traceback; traceback.print_exc()
            yield {"type": "error", "content": f"System error: {exc}"}

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self):
        self.graph.close()
