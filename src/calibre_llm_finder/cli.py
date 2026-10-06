"""Command-line interface.

Every command shares the same core (config, reader, embeddings, index,
agent) as the web UI and the MCP server — the CLI is just another
presentation layer on top of the same engine.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from calibre_llm_finder.calibre.reader import read_all_books
from calibre_llm_finder.config import get_settings
from calibre_llm_finder.indexing.embeddings import EmbeddedBook, EmbeddingError, embed_books
from calibre_llm_finder.indexing.index_store import IndexStore
from calibre_llm_finder.llm.agent import AgentSession, BookFindingAgent
from calibre_llm_finder.llm.ollama_client import OllamaClient
from calibre_llm_finder.search.engine import SearchEngine

app = typer.Typer(
    help="Find books in your Calibre library from fuzzy, half-remembered descriptions."
)
console = Console()


@app.command()
def index(
    library_path: str = typer.Option(None, help="Override CLF_CALIBRE_LIBRARY_PATH for this run."),
) -> None:
    """(Re)build the local embedding index from the Calibre library."""
    asyncio.run(_index(library_path))


async def _index(library_path: str | None) -> None:
    settings = get_settings()
    if library_path:
        settings.calibre_library_path = settings.calibre_library_path.__class__(library_path)

    books = read_all_books(settings.calibre_metadata_db)
    console.print(f"Read [bold]{len(books)}[/bold] books from {settings.calibre_metadata_db}")

    index_store = IndexStore(
        settings.index_db_path, embedding_dimensions=settings.embedding_dimensions
    )
    async with OllamaClient(
        settings.ollama_host, timeout=settings.request_timeout_seconds
    ) as ollama:
        with console.status("Embedding library...") as status:

            def on_progress(done: int, total: int) -> None:
                status.update(f"Embedding library... {done}/{total}")

            results = await embed_books(
                ollama,
                books,
                model=settings.embedding_model,
                concurrency=settings.embedding_concurrency,
                progress_callback=on_progress,
            )

        failures = [r for r in results if isinstance(r, EmbeddingError)]
        for result in results:
            if isinstance(result, EmbeddedBook):
                await index_store.upsert_book(result.book, result.embedding)

    await index_store.close()

    console.print(f"[green]Indexed {len(results) - len(failures)}/{len(results)} books.[/green]")
    if failures:
        console.print(f"[yellow]{len(failures)} books failed to embed:[/yellow]")
        for failure in failures[:10]:
            console.print(f"  - {failure.book.title}: {failure.error}")


@app.command()
def search(query: str) -> None:
    """One-shot: run the full agent loop for a single query and print the result."""
    asyncio.run(_search(query))


async def _search(query: str) -> None:
    settings = get_settings()
    index_store = IndexStore(
        settings.index_db_path, embedding_dimensions=settings.embedding_dimensions
    )
    async with OllamaClient(
        settings.ollama_host, timeout=settings.request_timeout_seconds
    ) as ollama:
        engine = SearchEngine(ollama, index_store, embedding_model=settings.embedding_model)
        agent = BookFindingAgent(ollama, engine, chat_model=settings.chat_model)
        session = AgentSession()
        async for event in agent.ask(session, query):
            if event.type == "token":
                console.print(event.text, end="")
            elif event.type == "tool_call":
                console.print(f"\n[dim]→ calling {event.tool_name}({event.payload})[/dim]")
            elif event.type == "error":
                console.print(f"\n[red]{event.text}[/red]")
            elif event.type == "done" and event.text:
                console.print(event.text)
        console.print()
    await index_store.close()


@app.command()
def chat() -> None:
    """Interactive REPL version of ``search`` that keeps conversation history."""
    asyncio.run(_chat())


async def _chat() -> None:
    settings = get_settings()
    index_store = IndexStore(
        settings.index_db_path, embedding_dimensions=settings.embedding_dimensions
    )
    async with OllamaClient(
        settings.ollama_host, timeout=settings.request_timeout_seconds
    ) as ollama:
        engine = SearchEngine(ollama, index_store, embedding_model=settings.embedding_model)
        agent = BookFindingAgent(ollama, engine, chat_model=settings.chat_model)
        session = AgentSession()
        console.print("[bold]Describe the book you're trying to find[/bold] (Ctrl-D to quit)\n")
        while True:
            try:
                query = console.input("[bold cyan]you>[/bold cyan] ")
            except EOFError:
                break
            if not query.strip():
                continue
            async for event in agent.ask(session, query):
                if event.type == "token":
                    console.print(event.text, end="")
                elif event.type == "tool_call":
                    console.print(f"\n[dim]→ calling {event.tool_name}({event.payload})[/dim]")
                elif event.type == "error":
                    console.print(f"\n[red]{event.text}[/red]")
                elif event.type == "done" and event.text:
                    console.print(event.text)
            console.print()
    await index_store.close()


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the browser UI (FastAPI + a single-page search UI)."""
    import uvicorn

    uvicorn.run("calibre_llm_finder.web.app:app", host=host, port=port)


@app.command(name="mcp-serve")
def mcp_serve() -> None:
    """Run the MCP server over stdio, for any MCP-compatible client."""
    from calibre_llm_finder.mcp.server import run

    run()


@app.command()
def eval(report: bool = typer.Option(True, help="Print a per-case pass/fail table.")) -> None:
    """Run the evaluation suite against the current index and chat model."""
    asyncio.run(_eval(report))


async def _eval(report: bool) -> None:
    from calibre_llm_finder.eval.runner import run_eval

    settings = get_settings()
    index_store = IndexStore(
        settings.index_db_path, embedding_dimensions=settings.embedding_dimensions
    )
    async with OllamaClient(
        settings.ollama_host, timeout=settings.request_timeout_seconds
    ) as ollama:
        engine = SearchEngine(ollama, index_store, embedding_model=settings.embedding_model)
        agent = BookFindingAgent(ollama, engine, chat_model=settings.chat_model)
        results = await run_eval(agent)
    await index_store.close()

    passed = sum(1 for r in results if r.passed)
    if report:
        table = Table(title=f"Eval results: {passed}/{len(results)} passed")
        table.add_column("Query")
        table.add_column("Expected")
        table.add_column("Pass?")
        for r in results:
            table.add_row(r.case.query, r.case.expected_title, "✅" if r.passed else "❌")
        console.print(table)
    else:
        console.print(f"{passed}/{len(results)} passed")


if __name__ == "__main__":
    app()
