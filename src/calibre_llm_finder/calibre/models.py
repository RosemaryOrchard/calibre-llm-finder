"""Domain model for a book, independent of Calibre's storage format."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class Book(BaseModel):
    """A single book as read from a Calibre library.

    ``calibre_id`` is the primary key in Calibre's own ``books`` table and is
    what we use to correlate rows across our own index and Calibre's.
    """

    calibre_id: int
    title: str
    authors: list[str]
    tags: list[str] = []
    series: str | None = None
    series_index: float | None = None
    comments: str | None = None
    pubdate: datetime | None = None
    last_modified: datetime | None = None
    path: str
    """Relative path (within the library root) to the book's folder."""

    @property
    def author_display(self) -> str:
        return " & ".join(self.authors) if self.authors else "Unknown"

    #: Hard cap on the free-text blurb fed to the embedding model. Local
    #: embedding models are typically served with a small context window
    #: (e.g. Ollama defaults ``nomic-embed-text`` to 2048 tokens), and a long
    #: Calibre "comments" field — some are several thousand words — can blow
    #: past that and get rejected outright, rather than just truncated
    #: server-side. Trimming here keeps every book embeddable regardless of
    #: the serving model's configured context, and a short, information-dense
    #: blurb embeds about as well as the full one anyway.
    _MAX_COMMENTS_CHARS = 2000

    def to_search_document(self) -> str:
        """Flatten the book's metadata into a single text blob for embedding.

        Order matters a little: title and author first (highest signal),
        then series/tags, then the free-text comments/blurb last since it's
        usually the longest field.
        """
        parts = [self.title, self.author_display]
        if self.series:
            parts.append(f"Book {self.series_index or '?'} of the {self.series} series")
        if self.tags:
            parts.append("Genres: " + ", ".join(self.tags))
        if self.comments:
            comments = self.comments
            if len(comments) > self._MAX_COMMENTS_CHARS:
                comments = comments[: self._MAX_COMMENTS_CHARS].rsplit(" ", 1)[0] + "…"
            parts.append(comments)
        return "\n".join(parts)
