"""Build a sample Calibre library from Project Gutenberg books.

Produces a directory that looks exactly like a real Calibre library —
``Author/Title (id)/Title - Author.txt`` per book, plus a ``metadata.db``
with the same schema Calibre itself uses for the handful of tables our
read-only reader (:mod:`calibre_llm_finder.calibre.reader`) queries. This
lets the whole project (and anyone cloning the repo) be tried out in
minutes without owning a real Calibre library or any non-public-domain
content.

Concurrent download of the book texts is a genuine asyncio use case: ~20
independent HTTPS GETs to gutenberg.org, bounded by a semaphore so we're a
good citizen of their servers rather than a stress test.

Usage:
    python scripts/build_sample_library.py [--out ~/Calibre\\ Library\\ Sample] [--concurrency 5]
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sqlite3
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from sample_books import SAMPLE_BOOKS, SampleBook  # noqa: E402

_SCHEMA = """
CREATE TABLE books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    sort TEXT,
    pubdate TIMESTAMP,
    series_index REAL NOT NULL DEFAULT 1.0,
    path TEXT NOT NULL,
    last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE authors (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, sort TEXT);
CREATE TABLE books_authors_link (
    id INTEGER PRIMARY KEY AUTOINCREMENT, book INTEGER NOT NULL, author INTEGER NOT NULL
);
CREATE TABLE tags (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL);
CREATE TABLE books_tags_link (
    id INTEGER PRIMARY KEY AUTOINCREMENT, book INTEGER NOT NULL, tag INTEGER NOT NULL
);
CREATE TABLE series (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL);
CREATE TABLE books_series_link (
    id INTEGER PRIMARY KEY AUTOINCREMENT, book INTEGER NOT NULL, series INTEGER NOT NULL
);
CREATE TABLE comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, book INTEGER NOT NULL, text TEXT NOT NULL
);
"""


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "", value).strip()
    return re.sub(r"\s+", " ", cleaned)


async def _download_book_text(
    client: httpx.AsyncClient, book: SampleBook, semaphore: asyncio.Semaphore
) -> str:
    async with semaphore:
        try:
            response = await client.get(book.gutenberg_txt_url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as exc:
            print(f"  ! failed to download {book.title!r} ({exc}); writing a placeholder instead")
            return f"[Could not download text for {book.title} from Project Gutenberg: {exc}]"


async def _download_all(books: list[SampleBook], concurrency: int) -> dict[int, str]:
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        texts = await asyncio.gather(*(_download_book_text(client, b, semaphore) for b in books))
    return dict(zip((b.gutenberg_id for b in books), texts, strict=True))


def _build_library(out_dir: Path, books: list[SampleBook], texts: dict[int, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "metadata.db"
    db_path.unlink(missing_ok=True)

    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)

    author_ids: dict[str, int] = {}
    tag_ids: dict[str, int] = {}
    series_ids: dict[str, int] = {}

    def get_or_create(table: dict[str, int], table_name: str, name: str) -> int:
        if name not in table:
            cur = conn.execute(f"INSERT INTO {table_name} (name) VALUES (?)", (name,))
            table[name] = cur.lastrowid
        return table[name]

    for book in books:
        author_dir = _safe_path_component(book.authors[0])
        title_dir = _safe_path_component(f"{book.title} ({book.gutenberg_id})")
        book_dir = out_dir / author_dir / title_dir
        book_dir.mkdir(parents=True, exist_ok=True)

        text_filename = f"{_safe_path_component(book.title)} - {author_dir}.txt"
        (book_dir / text_filename).write_text(texts[book.gutenberg_id], encoding="utf-8")

        rel_path = f"{author_dir}/{title_dir}"
        series_index = float(book.series[1]) if book.series else 1.0
        cur = conn.execute(
            "INSERT INTO books (title, sort, pubdate, series_index, path) VALUES (?, ?, ?, ?, ?)",
            (book.title, book.title, book.pubdate, series_index, rel_path),
        )
        book_id = cur.lastrowid

        for author in book.authors:
            author_id = get_or_create(author_ids, "authors", author)
            conn.execute(
                "INSERT INTO books_authors_link (book, author) VALUES (?, ?)", (book_id, author_id)
            )

        for tag in book.tags:
            tag_id = get_or_create(tag_ids, "tags", tag)
            conn.execute("INSERT INTO books_tags_link (book, tag) VALUES (?, ?)", (book_id, tag_id))

        if book.series:
            series_id = get_or_create(series_ids, "series", book.series[0])
            conn.execute(
                "INSERT INTO books_series_link (book, series) VALUES (?, ?)", (book_id, series_id)
            )

        conn.execute("INSERT INTO comments (book, text) VALUES (?, ?)", (book_id, book.comments))

    conn.commit()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "Calibre Library Sample",
        help="Directory to create the sample library in (default: ~/Calibre Library Sample)",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent downloads.")
    args = parser.parse_args()

    print(
        f"Downloading {len(SAMPLE_BOOKS)} books from Project Gutenberg "
        f"(concurrency={args.concurrency})..."
    )
    texts = asyncio.run(_download_all(SAMPLE_BOOKS, args.concurrency))

    print(f"Building Calibre-style library at {args.out}...")
    _build_library(args.out, SAMPLE_BOOKS, texts)

    print(f"Done. Set CLF_CALIBRE_LIBRARY_PATH={args.out} (or edit .env) and run `clf index`.")


if __name__ == "__main__":
    main()
