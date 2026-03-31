"""
MemoryOS v2 — LLM Tool Definitions
====================================
Nine tools that give the model full read/write access to the memory graph.
"""

TOOLS_SCHEMA = [
    # ── Profile ──────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": (
                "Save or update a permanent attribute of the user: name, occupation, "
                "primary_location, greeting, or any other personal fact. "
                "Call this the first time you learn something permanent about the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key":   {"type": "string",
                              "description": "Attribute name, e.g. 'name', 'occupation', 'primary_location'"},
                    "value": {"type": "string",
                              "description": "The value to store"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_user_profile_field",
            "description": (
                "Remove an outdated profile field or delete a specific preference. "
                "Use key='preferences' + value_to_remove to remove a single preference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key":            {"type": "string",
                                       "description": "Field name, or 'preferences' to remove a preference value"},
                    "value_to_remove": {"type": "string",
                                       "description": "The specific value to remove (for list fields)"},
                },
                "required": ["key", "value_to_remove"],
            },
        },
    },

    # ── Preferences / constraints ─────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "add_preference",
            "description": (
                "Save a user preference, personal constraint, dietary restriction, "
                "allergy, or goal as a typed node in the graph. "
                "Use category='allergy' for medical constraints — these are HARD RULES."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value":    {"type": "string",
                                 "description": "E.g. 'no peanuts', 'prefers morning meetings', 'wants to learn guitar'"},
                    "category": {"type": "string",
                                 "enum": ["preference", "goal", "allergy", "constraint"],
                                 "description": "Type of preference"},
                },
                "required": ["value", "category"],
            },
        },
    },

    # ── Entities ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "update_entity",
            "description": (
                "Create or update a Person, Place, or Organization in the knowledge graph. "
                "Use this for ANY named entity the user mentions. "
                "Provide relationship_to_user to link it to the user node."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name":                {"type": "string",
                                            "description": "Full name, e.g. 'Sarah Chen'"},
                    "entity_type":         {"type": "string",
                                            "enum": ["person", "place", "organization"],
                                            "description": "Node label"},
                    "relationship_to_user":{"type": "string",
                                            "description": "E.g. 'boss', 'friend', 'home', 'workplace'"},
                    "attributes":          {"type": "object",
                                            "description": "Key-value attributes: role, age, location, status, etc."},
                },
                "required": ["name", "entity_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "link_entities",
            "description": (
                "Create a directional relationship edge between two existing entities in the graph. "
                "Use when the user reveals how two entities are connected, e.g. "
                "'Sarah works at TechCorp' → link_entities('Sarah Chen', 'TechCorp', 'WORKS_AT'). "
                "This is how the knowledge graph grows richer over time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_a":         {"type": "string", "description": "Name of the source entity"},
                    "entity_b":         {"type": "string", "description": "Name of the target entity"},
                    "relationship_type":{"type": "string",
                                         "description": "E.g. 'WORKS_AT', 'LOCATED_IN', 'REPORTS_TO', 'MARRIED_TO'"},
                    "description":      {"type": "string",
                                         "description": "Human-readable description (optional)"},
                },
                "required": ["entity_a", "entity_b", "relationship_type"],
            },
        },
    },

    # ── Events ────────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "log_event",
            "description": (
                "Log a significant life event to the timeline graph. "
                "Only use for major events: job changes, moves, achievements, medical events, etc. "
                "Do NOT log casual questions or small talk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description":      {"type": "string",
                                         "description": "Clear description of the event"},
                    "entities_involved":{"type": "array", "items": {"type": "string"},
                                         "description": "Names of entities involved (creates INVOLVED_IN edges)"},
                    "date":             {"type": "string",
                                         "description": "ISO date YYYY-MM-DD, or leave empty for today"},
                },
                "required": ["description"],
            },
        },
    },

    # ── Knowledge ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "save_knowledge",
            "description": (
                "Save a reusable fact, code snippet, recipe, or reference under a unique topic key. "
                "Overwrites previous content for the same topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic":   {"type": "string", "description": "Short unique topic identifier"},
                    "content": {"type": "string", "description": "Full content to store"},
                },
                "required": ["topic", "content"],
            },
        },
    },

    # ── Graph traversal ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "graph_search",
            "description": (
                "Traverse the knowledge graph outward from a named entity to discover "
                "all connected nodes (other people, events, places, knowledge). "
                "Use this when you want to understand everything connected to a specific person or place."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string",
                                    "description": "Name of the entity to start from"},
                    "depth":       {"type": "integer", "default": 2,
                                    "description": "Graph traversal depth (1 or 2)"},
                },
                "required": ["entity_name"],
            },
        },
    },

    # ── Entity resolution ─────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "merge_entities",
            "description": (
                "Merge a duplicate entity (alias) into its canonical entity. "
                "Call this whenever you detect that two stored entities refer to the same "
                "real-world person, place, or organisation — for example, "
                "'Priyanshu's mother' and 'Moumita' are the same person. "
                "All relationships of the alias are migrated to the canonical node "
                "and the alias is deleted, keeping the graph clean."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "canonical_name": {
                        "type": "string",
                        "description": "The correct, primary name to keep (e.g. 'Moumita')",
                    },
                    "alias_name": {
                        "type": "string",
                        "description": "The redundant name to merge and delete (e.g. 'Priyanshu\\'s mother')",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One-sentence explanation of why these are the same entity",
                    },
                },
                "required": ["canonical_name", "alias_name"],
            },
        },
    },

    # ── Archival vector search ────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "archival_memory_search",
            "description": (
                "Search long-term conversation history using semantic similarity. "
                "Use ONLY when the answer is not visible in [MEMORY CONTEXT]. "
                "Do NOT call this if the information is already in the context block."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                               "description": "Natural language search query"},
                },
                "required": ["query"],
            },
        },
    },
]
