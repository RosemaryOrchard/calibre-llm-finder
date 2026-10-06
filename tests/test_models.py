"""Tests for the Book domain model."""

from __future__ import annotations

from calibre_llm_finder.calibre.models import Book


def _book(**overrides: object) -> Book:
    defaults: dict[str, object] = {
        "calibre_id": 1,
        "title": "The Hobbit",
        "authors": ["J. R. R. Tolkien"],
        "path": "J. R. R. Tolkien/The Hobbit (1)",
    }
    defaults.update(overrides)
    return Book(**defaults)  # type: ignore[arg-type]


def test_author_display_joins_multiple_authors() -> None:
    book = _book(authors=["Terry Pratchett", "Neil Gaiman"])
    assert book.author_display == "Terry Pratchett & Neil Gaiman"


def test_author_display_handles_no_authors() -> None:
    book = _book(authors=[])
    assert book.author_display == "Unknown"


def test_to_search_document_includes_series_tags_and_comments() -> None:
    book = _book(
        series="Middle-earth",
        series_index=1.0,
        tags=["Fantasy", "Adventure"],
        comments="A hobbit goes on an unexpected journey.",
    )
    doc = book.to_search_document()

    assert "The Hobbit" in doc
    assert "J. R. R. Tolkien" in doc
    assert "Book 1.0 of the Middle-earth series" in doc
    assert "Genres: Fantasy, Adventure" in doc
    assert "A hobbit goes on an unexpected journey." in doc


def test_to_search_document_omits_absent_optional_fields() -> None:
    book = _book()
    doc = book.to_search_document()
    assert "series" not in doc.lower()
    assert "Genres" not in doc


def test_to_search_document_truncates_long_comments() -> None:
    # Long enough to exceed a typical embedding model's context window
    # (e.g. Ollama's default 2048 tokens for nomic-embed-text) if left
    # untruncated, which previously caused indexing to fail outright.
    book = _book(comments="word " * 5000)
    doc = book.to_search_document()

    assert len(doc) < len(book.comments or "")
    assert doc.endswith("…")
