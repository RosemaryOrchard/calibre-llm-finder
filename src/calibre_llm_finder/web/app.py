"""FastAPI web UI: a single search box, served over Server-Sent Events.

Streaming matters here for the same reason it does in the CLI — the agent
loop can take several seconds (embedding + vector search + model
generation, sometimes twice if it re-queries), and SSE lets the browser show
tokens and tool-call activity as they happen instead of a long blank wait.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from calibre_llm_finder.config import get_settings
from calibre_llm_finder.indexing.index_store import IndexStore
from calibre_llm_finder.llm.agent import AgentSession, BookFindingAgent
from calibre_llm_finder.llm.ollama_client import OllamaClient
from calibre_llm_finder.search.engine import SearchEngine

_HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    ollama = OllamaClient(settings.ollama_host, timeout=settings.request_timeout_seconds)
    index_store = IndexStore(
        settings.index_db_path, embedding_dimensions=settings.embedding_dimensions
    )
    engine = SearchEngine(ollama, index_store, embedding_model=settings.embedding_model)
    agent = BookFindingAgent(ollama, engine, chat_model=settings.chat_model)
    _state["ollama"] = ollama
    _state["index_store"] = index_store
    _state["agent"] = agent
    try:
        yield
    finally:
        await ollama.aclose()
        await index_store.close()


app = FastAPI(title="Calibre LLM Finder", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/search")
async def search(request: Request, q: str):
    """SSE endpoint: streams :class:`AgentEvent`s as JSON lines."""
    agent: BookFindingAgent = _state["agent"]  # type: ignore[assignment]
    session = AgentSession()

    async def event_stream() -> AsyncIterator[str]:
        async for event in agent.ask(session, q):
            payload = {"type": event.type, "text": event.text}
            if event.tool_name:
                payload["tool_name"] = event.tool_name
            if event.payload is not None:
                payload["payload"] = event.payload
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
