"""Tests for concurrent embedding computation."""

from __future__ import annotations

import asyncio

from calibre_llm_finder.calibre.models import Book
from calibre_llm_finder.indexing.embeddings import EmbeddedBook, EmbeddingError, embed_books
from calibre_llm_finder.llm.ollama_client import OllamaError


class _FakeClient:
    """Stand-in for OllamaClient that tracks concurrency and can fail on demand."""

    def __init__(self, *, fail_titles: set[str] | None = None, max_concurrency: int = 2) -> None:
        self.fail_titles = fail_titles or set()
        self.max_concurrency = max_concurrency
        self._in_flight = 0
        self._peak_in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def peak_in_flight(self) -> int:
        return self._peak_in_flight

    async def embed(self, model: str, text: str) -> list[float]:
        async with self._lock:
            self._in_flight += 1
            self._peak_in_flight = max(self._peak_in_flight, self._in_flight)
        try:
            await asyncio.sleep(0.01)
            if any(title in text for title in self.fail_titles):
                raise OllamaError("simulated failure")
            return [float(len(text))]
        finally:
            async with self._lock:
                self._in_flight -= 1


def _book(calibre_id: int, title: str) -> Book:
    return Book(calibre_id=calibre_id, title=title, authors=["A"], path=f"A/{title}")


async def test_embed_books_respects_concurrency_limit() -> None:
    client = _FakeClient(max_concurrency=3)
    books = [_book(i, f"Book {i}") for i in range(10)]

    results = await embed_books(client, books, model="m", concurrency=3)  # type: ignore[arg-type]

    assert len(results) == 10
    assert all(isinstance(r, EmbeddedBook) for r in results)
    assert client.peak_in_flight <= 3


async def test_embed_books_collects_individual_failures_without_aborting() -> None:
    client = _FakeClient(fail_titles={"Bad Book"})
    books = [_book(1, "Good Book"), _book(2, "Bad Book"), _book(3, "Another Good Book")]

    results = await embed_books(client, books, model="m", concurrency=2)  # type: ignore[arg-type]

    successes = [r for r in results if isinstance(r, EmbeddedBook)]
    failures = [r for r in results if isinstance(r, EmbeddingError)]
    assert len(successes) == 2
    assert len(failures) == 1
    assert failures[0].book.title == "Bad Book"


async def test_embed_books_reports_progress() -> None:
    client = _FakeClient()
    books = [_book(i, f"Book {i}") for i in range(5)]
    progress_calls: list[tuple[int, int]] = []

    await embed_books(
        client,  # type: ignore[arg-type]
        books,
        model="m",
        concurrency=2,
        progress_callback=lambda done, total: progress_calls.append((done, total)),
    )

    assert len(progress_calls) == 5
    assert progress_calls[-1] == (5, 5)
