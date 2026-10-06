"""MCP server: exposes this library's search as standard MCP tools.

Any MCP-compatible client (Claude Desktop, an agent framework, a custom
harness) can attach to this process over stdio and call ``search_library`` /
``get_book_details`` exactly as the built-in agent does — same underlying
:class:`~calibre_llm_finder.search.engine.SearchEngine`, so there's one
source of truth for "what does searching this library mean", reused by both
the in-app agent and the outside world.

Run with:  ``clf mcp-serve``  (see :mod:`calibre_llm_finder.cli`).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from calibre_llm_finder.config import get_settings
from calibre_llm_finder.indexing.index_store import IndexStore
from calibre_llm_finder.llm.ollama_client import OllamaClient
from calibre_llm_finder.search.engine import SearchEngine

mcp = FastMCP(
    "calibre-llm-finder",
    instructions=(
        "Tools for searching a specific person's personal Calibre ebook "
        "library by fuzzy, half-remembered description, and for fetching "
        "full metadata for a specific book once you've identified a "
        "candidate."
    ),
)


def _build_search_engine() -> SearchEngine:
    settings = get_settings()
    ollama = OllamaClient(settings.ollama_host, timeout=settings.request_timeout_seconds)
    index = IndexStore(settings.index_db_path, embedding_dimensions=settings.embedding_dimensions)
    return SearchEngine(ollama, index, embedding_model=settings.embedding_model)


_engine: SearchEngine | None = None


def _engine_singleton() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = _build_search_engine()
    return _engine


@mcp.tool()
async def search_library(query: str, limit: int = 5) -> list[dict]:
    """Semantically search the Calibre library for books matching a free-text
    description of something the user half-remembers about a book (plot,
    characters, setting, vibe). Returns ranked candidates with metadata."""
    results = await _engine_singleton().search(query, limit=limit)
    return [
        {
            "calibre_id": r.book.calibre_id,
            "title": r.book.title,
            "authors": r.book.authors,
            "tags": r.book.tags,
            "series": r.book.series,
            "comments": r.book.comments,
        }
        for r in results
    ]


@mcp.tool()
async def get_book_details(calibre_id: int) -> dict:
    """Fetch full metadata for one book by its Calibre library ID."""
    book = await _engine_singleton().get_book(calibre_id)
    if book is None:
        return {"error": f"No book with calibre_id={calibre_id} found."}
    return book.model_dump(mode="json")


def run() -> None:
    """Entry point used by ``clf mcp-serve``."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
