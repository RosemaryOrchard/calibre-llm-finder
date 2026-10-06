"""Tests for the async Ollama HTTP client, with the network mocked via respx."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from calibre_llm_finder.llm.ollama_client import OllamaClient, OllamaError


@respx.mock
async def test_embed_returns_vector() -> None:
    respx.post("http://fake-ollama/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})
    )
    async with OllamaClient("http://fake-ollama") as client:
        vector = await client.embed("nomic-embed-text", "some text")
    assert vector == [0.1, 0.2, 0.3]


@respx.mock
async def test_embed_raises_ollama_error_on_connection_failure() -> None:
    respx.post("http://fake-ollama/api/embeddings").mock(side_effect=httpx.ConnectError("refused"))
    async with OllamaClient("http://fake-ollama") as client:
        with pytest.raises(OllamaError, match="Failed to get embedding"):
            await client.embed("nomic-embed-text", "some text")


@respx.mock
async def test_chat_stream_yields_token_chunks() -> None:
    lines = [
        json.dumps({"message": {"role": "assistant", "content": "Hello"}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": " world"}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}),
    ]
    respx.post("http://fake-ollama/api/chat").mock(
        return_value=httpx.Response(200, content="\n".join(lines) + "\n")
    )
    async with OllamaClient("http://fake-ollama") as client:
        chunks = [
            c async for c in client.chat_stream("llama3.1", [{"role": "user", "content": "hi"}])
        ]

    assert "".join(c.content for c in chunks) == "Hello world"
    assert chunks[-1].done is True


@respx.mock
async def test_chat_stream_surfaces_tool_calls() -> None:
    line = json.dumps(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "search_library", "arguments": {"query": "x"}},
                    }
                ],
            },
            "done": False,
        }
    )
    respx.post("http://fake-ollama/api/chat").mock(
        return_value=httpx.Response(200, content=line + "\n")
    )
    async with OllamaClient("http://fake-ollama") as client:
        chunks = [
            c async for c in client.chat_stream("llama3.1", [{"role": "user", "content": "hi"}])
        ]

    assert len(chunks) == 1
    assert chunks[0].tool_calls[0].name == "search_library"
    assert chunks[0].tool_calls[0].arguments == {"query": "x"}
