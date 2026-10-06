"""Thin async client for a local Ollama server.

Deliberately hand-rolled on top of ``httpx.AsyncClient`` rather than the
official ``ollama`` package: we need fine-grained control over streaming
chunks and tool-call payloads for the agent loop in
:mod:`calibre_llm_finder.llm.agent`, and keeping the HTTP contract explicit
here makes it obvious exactly what the app depends on.

Everything is async because every call here is I/O-bound (a request to a
local or remote model server) — the one place in this project that does
real CPU-bound work (writing to the vector index) deliberately uses a
background thread instead; see :mod:`calibre_llm_finder.indexing.index_store`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns a malformed response.

    Callers (the agent loop, the CLI, the web layer) are expected to catch
    this and degrade gracefully — e.g. surface "local model unavailable" to
    the user — rather than letting a bare connection error bubble up.
    """


class ToolCall(BaseModel):
    """A single tool invocation requested by the model."""

    id: str = ""
    name: str
    arguments: dict[str, Any]


class ChatChunk(BaseModel):
    """One piece of a streamed chat response.

    Ollama streams newline-delimited JSON objects; each one carries either a
    fragment of assistant text, a (non-streamed, but delivered at some point
    mid-stream) list of tool calls, or the final ``done`` marker with usage
    stats. We normalise all three into this one shape.
    """

    content: str = ""
    tool_calls: list[ToolCall] = []
    done: bool = False


class OllamaClient:
    """Async client for the subset of the Ollama HTTP API this project uses."""

    def __init__(
        self,
        host: str,
        *,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=self._host, timeout=timeout)

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def aclose(self) -> None:
        """Explicit async close, for callers that don't use the ``async with`` form."""
        if self._owns_client:
            await self._client.aclose()

    async def embed(self, model: str, text: str, *, retries: int = 2) -> list[float]:
        """Compute a single embedding vector for ``text``.

        Transient server errors (Ollama returns a bare ``500`` while a model
        is still loading, or under heavy concurrent load) are retried a
        couple of times with a short backoff before giving up — this is a
        real failure mode we saw in practice on first-run cold starts, not a
        hypothetical one.
        """
        last_exc: BaseException | None = None
        for attempt in range(retries + 1):
            try:
                response = await self._client.post(
                    "/api/embeddings", json={"model": model, "prompt": text}
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = exc.response.text.strip()
                last_exc = OllamaError(
                    f"Failed to get embedding from Ollama at {self._host}: "
                    f"{exc.response.status_code} {body or exc}"
                )
            except httpx.HTTPError as exc:
                last_exc = OllamaError(
                    f"Failed to get embedding from Ollama at {self._host}: {exc}"
                )
            else:
                payload = response.json()
                embedding = payload.get("embedding")
                if not isinstance(embedding, list):
                    raise OllamaError(
                        f"Ollama embeddings response missing 'embedding' field: {payload!r}"
                    )
                return embedding

            if attempt < retries:
                await asyncio.sleep(0.5 * (attempt + 1))

        assert last_exc is not None
        raise last_exc

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Stream a chat completion, yielding one :class:`ChatChunk` per line.

        Tool calls, when the model decides to make them, arrive as a complete
        (unstreamed) message part — we surface that as a single chunk with
        ``tool_calls`` populated and empty ``content``.
        """
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if tools:
            body["tools"] = tools

        try:
            async with self._client.stream("POST", "/api/chat", json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    yield _parse_chat_line(line)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Failed to stream chat from Ollama at {self._host}: {exc}") from exc

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatChunk:
        """Non-streaming convenience wrapper: collect the full response."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        async for chunk in self.chat_stream(model, messages, tools=tools):
            content_parts.append(chunk.content)
            tool_calls.extend(chunk.tool_calls)
        return ChatChunk(content="".join(content_parts), tool_calls=tool_calls, done=True)

    async def list_models(self) -> list[str]:
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Failed to list models from Ollama at {self._host}: {exc}") from exc
        return [m["name"] for m in response.json().get("models", [])]


def _parse_chat_line(line: str) -> ChatChunk:
    import json

    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Ollama returned a non-JSON stream line: {line!r}") from exc

    message = payload.get("message") or {}
    raw_tool_calls = message.get("tool_calls") or []
    tool_calls = [
        ToolCall(
            id=str(tc.get("id", "")),
            name=tc["function"]["name"],
            arguments=tc["function"].get("arguments", {}),
        )
        for tc in raw_tool_calls
        if "function" in tc
    ]
    return ChatChunk(
        content=message.get("content", ""),
        tool_calls=tool_calls,
        done=bool(payload.get("done", False)),
    )
