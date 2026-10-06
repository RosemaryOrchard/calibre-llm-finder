"""Read-only access to a Calibre library's ``metadata.db``.

We deliberately never write to Calibre's own database — this tool only
*reads* it (via a read-only SQLite URI connection) and keeps all of its own
derived data (embeddings, cache) in a completely separate SQLite file under
``settings.data_dir``. That means there's no risk of corrupting your library,
and Calibre itself can keep running normally alongside this tool.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from calibre_llm_finder.calibre.models import Book

_FIELD_SEP = "\x1f"  # unit separator: vanishingly unlikely to appear in a tag/author name

_BOOKS_QUERY = f"""
SELECT
    b.id,
    b.title,
    b.path,
    b.pubdate,
    b.last_modified,
    b.series_index,
    (
        SELECT group_concat(a.name, '{_FIELD_SEP}')
        FROM books_authors_link bal
        JOIN authors a ON a.id = bal.author
        WHERE bal.book = b.id
        ORDER BY bal.id
    ) AS authors,
    (
        SELECT group_concat(t.name, '{_FIELD_SEP}')
        FROM books_tags_link btl
        JOIN tags t ON t.id = btl.tag
        WHERE btl.book = b.id
        ORDER BY t.name
    ) AS tags,
    (
        SELECT s.name
        FROM books_series_link bsl
        JOIN series s ON s.id = bsl.series
        WHERE bsl.book = b.id
        LIMIT 1
    ) AS series,
    (
        SELECT c.text FROM comments c WHERE c.book = b.id LIMIT 1
    ) AS comments
FROM books b
ORDER BY b.id
"""


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _row_to_book(row: sqlite3.Row) -> Book:
    authors = row["authors"].split(_FIELD_SEP) if row["authors"] else []
    tags = row["tags"].split(_FIELD_SEP) if row["tags"] else []
    return Book(
        calibre_id=row["id"],
        title=row["title"],
        authors=authors,
        tags=tags,
        series=row["series"],
        series_index=row["series_index"],
        comments=row["comments"],
        pubdate=_parse_datetime(row["pubdate"]),
        last_modified=_parse_datetime(row["last_modified"]),
        path=row["path"],
    )


@contextmanager
def open_calibre_db(metadata_db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open Calibre's ``metadata.db`` strictly read-only.

    Uses a ``file:`` URI with ``mode=ro`` so SQLite refuses to write even if
    our own code has a bug — Calibre's database is never at risk.
    """
    if not metadata_db_path.exists():
        raise FileNotFoundError(
            f"No Calibre metadata.db found at {metadata_db_path}. "
            "Check CLF_CALIBRE_LIBRARY_PATH points at your library root "
            "(the folder that directly contains metadata.db)."
        )
    uri = f"file:{metadata_db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def read_all_books(metadata_db_path: Path) -> list[Book]:
    """Read every book's metadata from a Calibre library in one pass."""
    with open_calibre_db(metadata_db_path) as conn:
        rows = conn.execute(_BOOKS_QUERY).fetchall()
        return [_row_to_book(row) for row in rows]
