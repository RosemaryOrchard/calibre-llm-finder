"""Search engine: embeds a query and finds nearest-neighbour candidates.

This is the thing both the agent's ``search_library`` tool and the MCP
server call into — kept separate from both so neither has to know about
embeddings, sqlite-vec, or Ollama directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from calibre_llm_finder.calibre.models import Book
from calibre_llm_finder.indexing.index_store import IndexStore
from calibre_llm_finder.llm.ollama_client import OllamaClient


@dataclass(frozen=True)
class SearchResult:
    book: Book


class SearchEngine:
    def __init__(self, ollama: OllamaClient, index: IndexStore, *, embedding_model: str) -> None:
        self._ollama = ollama
        self._index = index
        self._embedding_model = embedding_model

    async def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        if not query.strip():
            return []
        embedding = await self._ollama.embed(self._embedding_model, query)
        books = await self._index.search(embedding, limit=limit)
        return [SearchResult(book=book) for book in books]

    async def get_book(self, calibre_id: int) -> Book | None:
        return await self._index.get_book(calibre_id)
