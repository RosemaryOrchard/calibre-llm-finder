"""Tests for the SearchEngine orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from calibre_llm_finder.calibre.models import Book
from calibre_llm_finder.search.engine import SearchEngine


@dataclass
class _FakeOllama:
    embedding: list[float] = field(default_factory=lambda: [1.0, 0.0])
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def embed(self, model: str, text: str) -> list[float]:
        self.calls.append((model, text))
        return self.embedding


@dataclass
class _FakeIndex:
    books: list[Book]
    searched_with: list[tuple[list[float], int]] = field(default_factory=list)

    async def search(self, embedding: list[float], *, limit: int) -> list[Book]:
        self.searched_with.append((embedding, limit))
        return self.books[:limit]

    async def get_book(self, calibre_id: int) -> Book | None:
        return next((b for b in self.books if b.calibre_id == calibre_id), None)


def _book(calibre_id: int, title: str) -> Book:
    return Book(calibre_id=calibre_id, title=title, authors=["A"], path=f"A/{title}")


async def test_search_embeds_query_and_delegates_to_index() -> None:
    ollama = _FakeOllama(embedding=[0.5, 0.5])
    index = _FakeIndex(books=[_book(1, "Dracula"), _book(2, "Frankenstein")])
    engine = SearchEngine(ollama, index, embedding_model="nomic-embed-text")  # type: ignore[arg-type]

    results = await engine.search("a vampire story", limit=1)

    assert ollama.calls == [("nomic-embed-text", "a vampire story")]
    assert index.searched_with == [([0.5, 0.5], 1)]
    assert [r.book.title for r in results] == ["Dracula"]


async def test_search_with_blank_query_returns_no_results_without_calling_backend() -> None:
    ollama = _FakeOllama()
    index = _FakeIndex(books=[_book(1, "Dracula")])
    engine = SearchEngine(ollama, index, embedding_model="m")  # type: ignore[arg-type]

    results = await engine.search("   ")

    assert results == []
    assert ollama.calls == []
    assert index.searched_with == []


async def test_get_book_delegates_to_index() -> None:
    ollama = _FakeOllama()
    index = _FakeIndex(books=[_book(1, "Dracula")])
    engine = SearchEngine(ollama, index, embedding_model="m")  # type: ignore[arg-type]

    assert (await engine.get_book(1)).title == "Dracula"  # type: ignore[union-attr]
    assert await engine.get_book(999) is None
