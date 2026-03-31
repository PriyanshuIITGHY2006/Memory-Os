"""
MemoryOS v2 — Orchestrator
============================
Request processing pipeline:
  1. Increment turn counter (Neo4j)
  2. Proactive archival search (ChromaDB)
  3. Build graph context (Neo4j traversal)
  4. Call LLM with 9-tool schema
  5. Dispatch tool calls → graph / archive updates
  6. Background-save episode to ChromaDB
  7. Return (response_text, active_memories, debug_prompt)
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

_client = Groq(api_key=config.GROQ_API_KEY)


class Orchestrator:

    def __init__(self):
        self.graph   = GraphManager()          # Neo4j — structured graph memory
        self.archive = ArchivalMemoryManager() # ChromaDB — semantic / episodic memory
        self.buffer  = BufferManager(max_turns=config.BUFFER_MAX_TURNS)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_message(self, user_message: str):
        try:
            current_turn = self.graph.increment_turn()
            self.buffer.add_turn("user", user_message)

            # 1. Proactive semantic search (ChromaDB)
            search_query    = " ".join(
                m["content"] for m in self.buffer.get_messages()[-3:]
            )
            archival_hits   = self.archive.search_memory(search_query, n_results=4)

            # 2. Build relevance-scored context (fixed token budget regardless of turn count)
            core_ctx = self.graph.get_core_prompt(
                recent_history_text=search_query,
                archival_context=archival_hits,
                current_turn=current_turn,
            )

            # 3. Compose messages for the LLM
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(core_memory_block=core_ctx)
            messages = [{"role": "system", "content": system_prompt}]
            for msg in self.buffer.get_messages():
                if msg["content"] != user_message:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_message})

            active_memories   = list(archival_hits)
            final_response    = ""

            # 4. Agentic loop (up to 4 rounds for multi-tool turns)
            for _round in range(4):
                try:
                    completion = _client.chat.completions.create(
                        model=config.LLM_MODEL,
                        messages=messages,
                        tools=TOOLS_SCHEMA,
                        tool_choice="auto",
                        parallel_tool_calls=True,
                        max_tokens=1024,
                    )
                except RateLimitError:
                    return "Rate limit reached — please wait a moment.", [], ""
                except BadRequestError as exc:
                    if "tool_use_failed" in str(exc):
                        messages.append({
                            "role": "user",
                            "content": "SYSTEM: Invalid tool format. Please retry with valid JSON.",
                        })
                        continue
                    raise

                response_msg = completion.choices[0].message

                # ── Tool calls ───────────────────────────────────────────────
                if response_msg.tool_calls:
                    messages.append(response_msg)
                    for tc in response_msg.tool_calls:
                        fname = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                        except Exception:
                            args = {}

                        result = self._dispatch(fname, args, current_turn, active_memories)
                        messages.append({
                            "tool_call_id": tc.id,
                            "role":         "tool",
                            "name":         fname,
                            "content":      str(result),
                        })

                # ── Text response ────────────────────────────────────────────
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

            # 5. Save to buffer + background archive
            self.buffer.add_turn("assistant", final_response)

            def _bg_save():
                self.archive.add_memory(
                    f"User: {user_message}\nAssistant: {final_response}",
                    "assistant",
                    current_turn,
                )

            threading.Thread(target=_bg_save, daemon=True).start()

            return final_response, active_memories, system_prompt

        except Exception as exc:
            import traceback
            traceback.print_exc()
            return f"System error: {exc}", [], ""

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, fname: str, args: dict, turn: int, active_memories: list):
        """Route a tool call to the correct manager method."""
        try:
            # ── Profile ──────────────────────────────────────────────────
            if fname == "update_user_profile":
                return self.graph.update_profile(args["key"], args["value"])

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
                    args["entity_a"],
                    args["entity_b"],
                    args["relationship_type"],
                    args.get("description", ""),
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

            # ── Legacy aliases (backward compat) ─────────────────────────
            if fname == "core_memory_update":
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

        except Exception as exc:
            return f"Tool error ({fname}): {exc}"

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self):
        self.graph.close()
