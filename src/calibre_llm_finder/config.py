"""Application configuration.

Settings are sourced from environment variables / a local ``.env`` file so the
same code runs unchanged whether invoked from the CLI or the web server. See
``.env.example`` for the full list of knobs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLF_", env_file=".env", extra="ignore")

    # Path to the root of a Calibre library, i.e. the directory containing
    # ``metadata.db``. This is opened read-only; we never write to it.
    calibre_library_path: Path = Field(default=Path.home() / "Calibre Library")

    # Where the app stores its own derived index (embeddings + cache). Kept
    # entirely separate from Calibre's own database.
    data_dir: Path = Field(default=Path.home() / ".calibre_llm_finder")

    # Ollama connection + models. Any Ollama-compatible embedding/chat model
    # pair can be used; defaults are small, fast, and commonly pre-pulled.
    ollama_host: str = Field(default="http://localhost:11434")
    embedding_model: str = Field(default="nomic-embed-text")
    chat_model: str = Field(default="llama3.1")
    embedding_dimensions: int = Field(default=768)

    # How many concurrent embedding requests to fire at the local Ollama
    # server when (re)indexing a library. Ollama serialises GPU work anyway,
    # but concurrency still overlaps request/response overhead and lets us
    # saturate CPU-only setups with multiple model instances.
    embedding_concurrency: int = Field(default=8)

    # Number of vector-search candidates handed to the LLM for re-ranking.
    search_candidate_count: int = Field(default=20)
    search_result_count: int = Field(default=5)

    request_timeout_seconds: float = Field(default=60.0)

    @property
    def calibre_metadata_db(self) -> Path:
        return self.calibre_library_path / "metadata.db"

    @property
    def index_db_path(self) -> Path:
        return self.data_dir / "index.db"


def get_settings() -> Settings:
    """Factory (not a singleton) so tests can override via env vars freely."""
    return Settings()
