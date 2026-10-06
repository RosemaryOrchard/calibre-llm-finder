"""Tests for read-only Calibre metadata.db parsing."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from calibre_llm_finder.calibre.reader import open_calibre_db, read_all_books

_SCHEMA = """
CREATE TABLE books (
    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, path TEXT NOT NULL,
    pubdate TEXT, last_modified TEXT, series_index REAL
);
CREATE TABLE authors (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL);
CREATE TABLE books_authors_link (
    id INTEGER PRIMARY KEY AUTOINCREMENT, book INTEGER, author INTEGER
);
CREATE TABLE tags (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL);
CREATE TABLE books_tags_link (id INTEGER PRIMARY KEY AUTOINCREMENT, book INTEGER, tag INTEGER);
CREATE TABLE series (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL);
CREATE TABLE books_series_link (id INTEGER PRIMARY KEY AUTOINCREMENT, book INTEGER, series INTEGER);
CREATE TABLE comments (id INTEGER PRIMARY KEY AUTOINCREMENT, book INTEGER, text TEXT);
"""


@pytest.fixture
def sample_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)

    conn.execute(
        "INSERT INTO books (id, title, path, pubdate, series_index) VALUES "
        "(1, 'Good Omens', 'Authors/Good Omens (1)', '1990-05-01T00:00:00+00:00', 1.0)"
    )
    conn.execute("INSERT INTO authors (id, name) VALUES (1, 'Terry Pratchett'), (2, 'Neil Gaiman')")
    conn.execute("INSERT INTO books_authors_link (book, author) VALUES (1, 1), (1, 2)")
    conn.execute("INSERT INTO tags (id, name) VALUES (1, 'Fantasy'), (2, 'Comedy')")
    conn.execute("INSERT INTO books_tags_link (book, tag) VALUES (1, 1), (1, 2)")
    conn.execute(
        "INSERT INTO comments (book, text) VALUES "
        "(1, 'An angel and a demon try to stop the apocalypse.')"
    )
    conn.commit()
    conn.close()
    return db_path


def test_read_all_books_parses_joins_correctly(sample_db: Path) -> None:
    books = read_all_books(sample_db)

    assert len(books) == 1
    book = books[0]
    assert book.calibre_id == 1
    assert book.title == "Good Omens"
    assert book.authors == ["Terry Pratchett", "Neil Gaiman"]
    assert set(book.tags) == {"Fantasy", "Comedy"}
    assert book.series is None
    assert book.comments == "An angel and a demon try to stop the apocalypse."
    assert book.pubdate is not None and book.pubdate.year == 1990


def test_open_calibre_db_is_strictly_read_only(sample_db: Path) -> None:
    with open_calibre_db(sample_db) as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO tags (name) VALUES ('Should Fail')")


def test_missing_metadata_db_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="metadata.db"):
        read_all_books(tmp_path / "metadata.db")
