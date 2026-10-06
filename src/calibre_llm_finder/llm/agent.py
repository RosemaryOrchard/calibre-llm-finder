"""The book-finding agent loop.

This is a genuine multi-turn agent loop, not a single prompt-and-done call:
the model is given tools, decides for itself whether to call them, we
execute whatever it asks for, feed the results back, and let it either call
another tool or produce a final streamed answer — bounded by ``max_turns``
so a model stuck in a tool-call loop can't run forever.

Local LLMs fail in predictable ways, and each one is handled explicitly
rather than left to surface as a stack trace:
  * **malformed tool calls** — unknown tool name, or arguments that don't
    parse/validate — are turned into a tool-result error message fed back
    to the model, instead of raising and killing the whole request. Models
    routinely recover from this when told plainly what went wrong.
  * **timeouts / unreachable local model** — surfaced as a single clear
    :class:`AgentEvent` the UI layer can render, rather than an unhandled
    exception.
  * **tool-call loops** — capped by ``max_turns``; if exceeded we return
    whatever the model has said so far plus a note, rather than hanging.
  * **hallucinated book IDs** — ``get_book_details`` on an unknown
    ``calibre_id`` returns a explicit "not found" tool result rather than
    `None`/a stack trace, so the model can self-correct.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from calibre_llm_finder.llm.ollama_client import OllamaClient, OllamaError
from calibre_llm_finder.llm.tools import ALL_TOOLS
from calibre_llm_finder.search.engine import SearchEngine

SYSTEM_PROMPT = """\
You are a helpful assistant that helps a reader find a specific book in \
their own Calibre ebook library, based on things they half-remember about \
it (plot fragments, characters, vibe, setting — rarely the exact title).

Always use the search_library tool to look in the library before answering \
— never rely on general knowledge about books that might not be in this \
specific library. If the first search doesn't turn up an obvious match, \
try again with a rephrased query before giving up. If several candidates \
are plausible, briefly explain why you picked the top one and mention the \
runners-up. If nothing plausible turns up, say so plainly and suggest what \
extra detail would help — do not invent a book that isn't in the results.
"""


@dataclass
class AgentEvent:
    """One step of the agent's work, surfaced to the CLI/web layer as it happens."""

    type: Literal["token", "tool_call", "tool_result", "error", "done"]
    text: str = ""
    tool_name: str = ""
    payload: Any = None


@dataclass
class AgentSession:
    """Conversation state for one back-and-forth with the agent."""

    messages: list[dict[str, Any]] = field(
        default_factory=lambda: [{"role": "system", "content": SYSTEM_PROMPT}]
    )


class BookFindingAgent:
    def __init__(
        self,
        ollama: OllamaClient,
        search_engine: SearchEngine,
        *,
        chat_model: str,
        max_turns: int = 4,
    ) -> None:
        self._ollama = ollama
        self._search = search_engine
        self._chat_model = chat_model
        self._max_turns = max_turns

    async def ask(self, session: AgentSession, user_message: str) -> AsyncIterator[AgentEvent]:
        session.messages.append({"role": "user", "content": user_message})

        for _turn in range(self._max_turns):
            try:
                assistant_text = ""
                tool_calls: list[Any] = []
                async for chunk in self._ollama.chat_stream(
                    self._chat_model, session.messages, tools=ALL_TOOLS
                ):
                    if chunk.content:
                        assistant_text += chunk.content
                        yield AgentEvent(type="token", text=chunk.content)
                    tool_calls.extend(chunk.tool_calls)
            except OllamaError as exc:
                yield AgentEvent(
                    type="error",
                    text=f"Couldn't reach the local model ({exc}). Is Ollama running?",
                )
                return

            session.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                    **({"tool_calls": _serialize_tool_calls(tool_calls)} if tool_calls else {}),
                }
            )

            if not tool_calls:
                yield AgentEvent(type="done", text=assistant_text)
                return

            for call in tool_calls:
                yield AgentEvent(type="tool_call", tool_name=call.name, payload=call.arguments)
                result = await self._execute_tool(call.name, call.arguments)
                yield AgentEvent(type="tool_result", tool_name=call.name, payload=result)
                session.messages.append(
                    {"role": "tool", "content": json.dumps(result), "name": call.name}
                )

        yield AgentEvent(
            type="done",
            text=(
                "I've tried a few searches but haven't converged on an answer yet. "
                "Could you give me one more distinguishing detail (a character name, "
                "setting, or how the story ends)?"
            ),
        )

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            if name == "search_library":
                query = arguments["query"]
                limit = int(arguments.get("limit", 5))
                results = await self._search.search(query, limit=limit)
                return {
                    "results": [
                        {
                            "calibre_id": r.book.calibre_id,
                            "title": r.book.title,
                            "authors": r.book.authors,
                            "tags": r.book.tags,
                            "series": r.book.series,
                            "comments": r.book.comments,
                        }
                        for r in results
                    ]
                }
            if name == "get_book_details":
                book = await self._search.get_book(int(arguments["calibre_id"]))
                if book is None:
                    return {"error": f"No book with calibre_id={arguments['calibre_id']!r} found."}
                return {"book": book.model_dump(mode="json")}
            return {"error": f"Unknown tool {name!r}."}
        except (KeyError, TypeError, ValueError) as exc:
            # Malformed tool call (missing/mis-typed arguments) — tell the
            # model plainly instead of raising, so it can retry correctly.
            return {"error": f"Malformed arguments for tool {name!r}: {exc}"}


def _serialize_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    return [
        {"id": tc.id, "function": {"name": tc.name, "arguments": tc.arguments}} for tc in tool_calls
    ]
