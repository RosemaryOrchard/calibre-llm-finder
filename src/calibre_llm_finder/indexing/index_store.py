"""Our own derived vector index — entirely separate from Calibre's database.

Design note on *threading* (as distinct from asyncio, used everywhere else in
this project): ``sqlite3`` connections are blocking and must only be used
from the thread that created them. Rather than sprinkle ``asyncio.to_thread``
calls across every query (which would open/close a connection per call, or
require a thread-unsafe connection shared across arbitrary threads), this
store follows the standard pattern for mixing a synchronous, stateful
resource into an async application: a single dedicated background thread
owns the one SQLite connection for its entire lifetime, and all work is
marshalled to it through a plain ``queue.Queue``. Async callers get back an
``asyncio.Future`` they can ``await`` normally, so the rest of the app never
has to know the storage layer isn't native asyncio.

This is a genuine, justified use of threading alongside asyncio (not just
"asyncio.to_thread everywhere"): one long-lived worker thread, a thread-safe
handoff queue, and a connection that is never touched from any other thread.
"""

from __future__ import annotations

import asyncio
import json
import queue
import sqlite3
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlite_vec

from calibre_llm_finder.calibre.models import Book

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    calibre_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    tags TEXT NOT NULL,
    series TEXT,
    series_index REAL,
    comments TEXT,
    pubdate TEXT,
    last_modified TEXT,
    path TEXT NOT NULL
);
"""


def _serialize_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


@dataclass
class _Job:
    fn: Callable[[sqlite3.Connection], Any]
    future: asyncio.Future[Any]
    loop: asyncio.AbstractEventLoop


class IndexStore:
    """Async-friendly handle onto a background SQLite/sqlite-vec writer thread."""

    def __init__(self, db_path: Path, *, embedding_dimensions: int) -> None:
        self._db_path = db_path
        self._embedding_dimensions = embedding_dimensions
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="index-store-writer", daemon=True)
        self._started = threading.Event()
        self._thread.start()
        self._started.wait()

    # -- lifecycle -----------------------------------------------------

    def _run(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(_SCHEMA)
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS book_vectors USING vec0("
            f"calibre_id INTEGER PRIMARY KEY, embedding FLOAT[{self._embedding_dimensions}])"
        )
        conn.commit()
        self._started.set()

        while True:
            job = self._queue.get()
            if job is None:  # shutdown sentinel
                conn.close()
                return
            try:
                result = job.fn(conn)
                job.loop.call_soon_threadsafe(job.future.set_result, result)
            except Exception as exc:  # noqa: BLE001 - propagate to the awaiting coroutine
                job.loop.call_soon_threadsafe(job.future.set_exception, exc)

    async def _submit(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._queue.put(_Job(fn=fn, future=future, loop=loop))
        return await future

    async def close(self) -> None:
        self._queue.put(None)
        await asyncio.to_thread(self._thread.join, timeout=5.0)

    # -- writes ----------------------------------------------------------

    async def upsert_book(self, book: Book, embedding: list[float]) -> None:
        def _write(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO books
                    (calibre_id, title, authors, tags, series, series_index,
                     comments, pubdate, last_modified, path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(calibre_id) DO UPDATE SET
                    title=excluded.title, authors=excluded.authors, tags=excluded.tags,
                    series=excluded.series, series_index=excluded.series_index,
                    comments=excluded.comments, pubdate=excluded.pubdate,
                    last_modified=excluded.last_modified, path=excluded.path
                """,
                (
                    book.calibre_id,
                    book.title,
                    json.dumps(book.authors),
                    json.dumps(book.tags),
                    book.series,
                    book.series_index,
                    book.comments,
                    book.pubdate.isoformat() if book.pubdate else None,
                    book.last_modified.isoformat() if book.last_modified else None,
                    book.path,
                ),
            )
            # vec0 virtual tables don't support INSERT OR REPLACE / ON CONFLICT,
            # so re-upserting a book's vector means delete-then-insert.
            conn.execute("DELETE FROM book_vectors WHERE calibre_id = ?", (book.calibre_id,))
            conn.execute(
                "INSERT INTO book_vectors (calibre_id, embedding) VALUES (?, ?)",
                (book.calibre_id, _serialize_vector(embedding)),
            )
            conn.commit()

        await self._submit(_write)

    # -- reads -------------------------------------------------------------

    async def search(self, query_embedding: list[float], *, limit: int) -> list[Book]:
        def _read(conn: sqlite3.Connection) -> list[Book]:
            rows = conn.execute(
                """
                SELECT b.*
                FROM book_vectors v
                JOIN books b ON b.calibre_id = v.calibre_id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (_serialize_vector(query_embedding), limit),
            ).fetchall()
            return [_row_to_book(row) for row in rows]

        return await self._submit(_read)

    async def get_book(self, calibre_id: int) -> Book | None:
        def _read(conn: sqlite3.Connection) -> Book | None:
            row = conn.execute("SELECT * FROM books WHERE calibre_id = ?", (calibre_id,)).fetchone()
            return _row_to_book(row) if row else None

        return await self._submit(_read)

    async def count(self) -> int:
        def _read(conn: sqlite3.Connection) -> int:
            return conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]

        return await self._submit(_read)


def _row_to_book(row: sqlite3.Row) -> Book:
    from datetime import datetime

    def _parse_dt(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    return Book(
        calibre_id=row["calibre_id"],
        title=row["title"],
        authors=json.loads(row["authors"]),
        tags=json.loads(row["tags"]),
        series=row["series"],
        series_index=row["series_index"],
        comments=row["comments"],
        pubdate=_parse_dt(row["pubdate"]),
        last_modified=_parse_dt(row["last_modified"]),
        path=row["path"],
    )
