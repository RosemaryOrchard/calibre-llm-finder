"""Concurrent embedding computation against a local Ollama server.

This is the asyncio half of the "asyncio AND threading" story: computing
embeddings for a whole library is a textbook I/O-bound fan-out — dozens of
independent HTTP requests to the same local server. We bound concurrency
with a semaphore rather than firing everything at once (Ollama queues
requests server-side anyway, so unbounded concurrency buys nothing but a
pile of timed-out sockets).

Compare with :mod:`calibre_llm_finder.indexing.index_store`, which uses a
background *thread* (not asyncio) for writes, because the thing it wraps
(``sqlite3``) is a blocking, non-async-native API that must only ever be
touched from one thread at a time.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from calibre_llm_finder.calibre.models import Book
from calibre_llm_finder.llm.ollama_client import OllamaClient, OllamaError

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class EmbeddedBook:
    book: Book
    embedding: list[float]


@dataclass(frozen=True)
class EmbeddingError:
    book: Book
    error: str


EmbeddingResult = EmbeddedBook | EmbeddingError


async def embed_books(
    client: OllamaClient,
    books: list[Book],
    *,
    model: str,
    concurrency: int = 8,
    progress_callback: ProgressCallback | None = None,
) -> list[EmbeddingResult]:
    """Embed every book's search document, with bounded concurrency.

    A failure on one book (model down mid-run, one pathological document)
    does not abort the batch — it's collected as an :class:`EmbeddingError`
    so the caller can report "indexed 1,203 of 1,210 books; 7 failed" instead
    of losing all progress on the first bad row.
    """
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(books)
    lock = asyncio.Lock()

    async def _embed_one(book: Book) -> EmbeddingResult:
        nonlocal completed
        async with semaphore:
            try:
                vector = await client.embed(model, book.to_search_document())
                result: EmbeddingResult = EmbeddedBook(book=book, embedding=vector)
            except OllamaError as exc:
                logger.warning("Failed to embed %r: %s", book.title, exc)
                result = EmbeddingError(book=book, error=str(exc))
        async with lock:
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
        return result

    return await asyncio.gather(*(_embed_one(book) for book in books))
