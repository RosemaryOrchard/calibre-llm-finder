"""Tests for the tool-calling agent loop, including failure-mode handling."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from calibre_llm_finder.calibre.models import Book
from calibre_llm_finder.llm.agent import AgentSession, BookFindingAgent
from calibre_llm_finder.llm.ollama_client import ChatChunk, OllamaError, ToolCall


@dataclass
class _ScriptedOllama:
    """Replays a fixed sequence of chat turns, one list[ChatChunk] per call."""

    turns: list[list[ChatChunk]]
    calls: list[list[dict]] = field(default_factory=list)

    async def chat_stream(
        self, model: str, messages: list[dict], *, tools=None
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append(messages)
        turn = self.turns[len(self.calls) - 1]
        for chunk in turn:
            yield chunk


@dataclass
class _FailingOllama:
    async def chat_stream(self, model, messages, *, tools=None) -> AsyncIterator[ChatChunk]:
        raise OllamaError("connection refused")
        yield  # pragma: no cover - makes this an async generator


class _FakeSearchEngine:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str, *, limit: int = 5):
        self.queries.append(query)
        from calibre_llm_finder.search.engine import SearchResult

        return [
            SearchResult(
                book=Book(calibre_id=1, title="Dracula", authors=["Bram Stoker"], path="x")
            )
        ]

    async def get_book(self, calibre_id: int):
        if calibre_id == 1:
            return Book(calibre_id=1, title="Dracula", authors=["Bram Stoker"], path="x")
        return None


async def test_agent_answers_directly_with_no_tool_calls() -> None:
    ollama = _ScriptedOllama(turns=[[ChatChunk(content="Hello there!", done=True)]])
    agent = BookFindingAgent(ollama, _FakeSearchEngine(), chat_model="m")  # type: ignore[arg-type]

    events = [e async for e in agent.ask(AgentSession(), "hi")]

    assert events[-1].type == "done"
    assert "Hello there!" in events[-1].text


async def test_agent_executes_tool_call_then_answers() -> None:
    ollama = _ScriptedOllama(
        turns=[
            [
                ChatChunk(
                    tool_calls=[
                        ToolCall(id="1", name="search_library", arguments={"query": "vampire"})
                    ]
                )
            ],
            [ChatChunk(content="It's Dracula.", done=True)],
        ]
    )
    engine = _FakeSearchEngine()
    agent = BookFindingAgent(ollama, engine, chat_model="m")  # type: ignore[arg-type]

    events = [e async for e in agent.ask(AgentSession(), "a count who drinks blood")]

    assert engine.queries == ["vampire"]
    tool_events = [e for e in events if e.type == "tool_result"]
    assert tool_events[0].payload["results"][0]["title"] == "Dracula"
    assert events[-1].text == "It's Dracula."


async def test_agent_handles_malformed_tool_call_arguments_gracefully() -> None:
    ollama = _ScriptedOllama(
        turns=[
            [ChatChunk(tool_calls=[ToolCall(id="1", name="search_library", arguments={})])],
            [ChatChunk(content="Could you clarify?", done=True)],
        ]
    )
    agent = BookFindingAgent(ollama, _FakeSearchEngine(), chat_model="m")  # type: ignore[arg-type]

    events = [e async for e in agent.ask(AgentSession(), "something")]

    tool_results = [e for e in events if e.type == "tool_result"]
    assert "error" in tool_results[0].payload
    assert events[-1].text == "Could you clarify?"


async def test_agent_handles_unreachable_model_as_error_event() -> None:
    agent = BookFindingAgent(_FailingOllama(), _FakeSearchEngine(), chat_model="m")  # type: ignore[arg-type]

    events = [e async for e in agent.ask(AgentSession(), "hi")]

    assert events[-1].type == "error"
    assert "Ollama" in events[-1].text or "model" in events[-1].text.lower()


async def test_agent_gives_up_gracefully_after_max_turns() -> None:
    # Every turn asks for another search — never converges.
    loop_turn = [
        ChatChunk(tool_calls=[ToolCall(id="1", name="search_library", arguments={"query": "x"})])
    ]
    ollama = _ScriptedOllama(turns=[loop_turn] * 10)
    agent = BookFindingAgent(ollama, _FakeSearchEngine(), chat_model="m", max_turns=3)  # type: ignore[arg-type]

    events = [e async for e in agent.ask(AgentSession(), "something impossible")]

    assert len(ollama.calls) == 3
    assert events[-1].type == "done"
    assert "detail" in events[-1].text.lower()


async def test_agent_unknown_tool_name_returns_error_result() -> None:
    ollama = _ScriptedOllama(
        turns=[
            [ChatChunk(tool_calls=[ToolCall(id="1", name="not_a_real_tool", arguments={})])],
            [ChatChunk(content="Sorry, let me try again.", done=True)],
        ]
    )
    agent = BookFindingAgent(ollama, _FakeSearchEngine(), chat_model="m")  # type: ignore[arg-type]

    events = [e async for e in agent.ask(AgentSession(), "something")]

    tool_results = [e for e in events if e.type == "tool_result"]
    assert "Unknown tool" in tool_results[0].payload["error"]
