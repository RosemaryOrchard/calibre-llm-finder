"""Tool schemas for the book-finding agent.

Exposed in Ollama's OpenAI-style ``tools`` format so the model can decide,
on its own, when to call the library search vs. ask the user a clarifying
question vs. answer directly. The exact same schema/handlers are reused by
the MCP server (:mod:`calibre_llm_finder.mcp.server`) so there is only one
definition of "what searching the library means" in the whole codebase.
"""

from __future__ import annotations

SEARCH_LIBRARY_TOOL = {
    "type": "function",
    "function": {
        "name": "search_library",
        "description": (
            "Semantically search the user's Calibre library for books matching a "
            "free-text description. Use this whenever the user describes a book "
            "they're trying to remember, even vaguely (plot fragments, character "
            "names, a vibe, a half-remembered setting). Returns candidate books "
            "ranked by similarity, each with title, author, tags, series and a "
            "short blurb."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What the user remembers about the book, in their own words.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of candidates to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

GET_BOOK_DETAILS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_book_details",
        "description": (
            "Fetch full metadata for one specific book by its Calibre library ID. "
            "Use this after search_library to confirm details before answering, "
            "e.g. to check the exact series position or publication date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "calibre_id": {
                    "type": "integer",
                    "description": "The Calibre book ID, as returned by search_library.",
                }
            },
            "required": ["calibre_id"],
        },
    },
}

ALL_TOOLS = [SEARCH_LIBRARY_TOOL, GET_BOOK_DETAILS_TOOL]
