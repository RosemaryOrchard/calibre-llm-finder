"""Tests for the sqlite-vec index store and its background writer thread."""

from __future__ import annotations

from pathlib import Path

import pytest

from calibre_llm_finder.calibre.models import Book
from calibre_llm_finder.indexing.index_store import IndexStore

DIM = 4


def _book(calibre_id: int, title: str) -> Book:
    return Book(
        calibre_id=calibre_id,
        title=title,
        authors=["Test Author"],
        path=f"Test Author/{title}",
    )


@pytest.fixture
async def store(tmp_path: Path):
    s = IndexStore(tmp_path / "index.db", embedding_dimensions=DIM)
    yield s
    await s.close()


async def test_upsert_and_count(store: IndexStore) -> None:
    await store.upsert_book(_book(1, "Alpha"), [1.0, 0.0, 0.0, 0.0])
    await store.upsert_book(_book(2, "Beta"), [0.0, 1.0, 0.0, 0.0])
    assert await store.count() == 2


async def test_search_returns_nearest_neighbour_first(store: IndexStore) -> None:
    await store.upsert_book(_book(1, "Close match"), [1.0, 0.0, 0.0, 0.0])
    await store.upsert_book(_book(2, "Far match"), [0.0, 0.0, 0.0, 1.0])

    results = await store.search([0.9, 0.1, 0.0, 0.0], limit=2)

    assert [b.title for b in results] == ["Close match", "Far match"]


async def test_get_book_returns_none_for_unknown_id(store: IndexStore) -> None:
    assert await store.get_book(999) is None


async def test_upsert_is_idempotent_on_calibre_id(store: IndexStore) -> None:
    await store.upsert_book(_book(1, "Original Title"), [1.0, 0.0, 0.0, 0.0])
    await store.upsert_book(_book(1, "Updated Title"), [1.0, 0.0, 0.0, 0.0])

    assert await store.count() == 1
    book = await store.get_book(1)
    assert book is not None
    assert book.title == "Updated Title"
