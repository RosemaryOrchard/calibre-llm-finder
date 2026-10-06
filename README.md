# Calibre LLM Finder

Find a book in your own [Calibre](https://calibre-ebook.com/) library from
**things you half-remember about it** — a plot fragment, a character name, a
vibe, "the one where the doctor turns into someone evil at night" — rather
than its exact title or author.

Runs entirely **self-hosted, on your own machine**, against your own
library, using **local LLMs via [Ollama](https://ollama.com/)** by default.
No ebook content or metadata ever has to leave your network.

**Stack:** Python 3.11+ · asyncio · FastAPI · Typer · Pydantic · SQLite /
[`sqlite-vec`](https://github.com/asg017/sqlite-vec) · [Ollama](https://ollama.com/)
· [MCP](https://modelcontextprotocol.io/) · pytest · ruff · mypy

## Example

```
$ clf search "a doctor who drinks a potion and turns into someone evil at night"

→ searching your library for: a scientist whose formula releases his evil side

That's "The Strange Case of Dr. Jekyll and Mr. Hyde" by Robert Louis
Stevenson — a respected London doctor develops a potion that splits him
into a second, monstrous identity he can no longer fully control. The
closest runner-up in your library is "The Picture of Dorian Gray", which
is about hidden corruption too, but doesn't involve a potion.
```

## What it actually does

1. Reads your Calibre library's `metadata.db` **read-only** (never writes to
   it — Calibre can keep running normally alongside this tool).
2. Embeds each book's title/author/series/tags/blurb with a local embedding
   model and stores the vectors in its own separate SQLite + [`sqlite-vec`](https://github.com/asg017/sqlite-vec)
   index.
3. When you describe a book, an **agent** (not a single prompt) decides for
   itself whether to search the library, re-query with different wording, or
   ask you a clarifying question — using real tool calling against a local
   model, with streamed output.
4. The same search capability is also exposed as an **MCP server**, so any
   MCP-compatible client (Claude Desktop, your own agent harness, etc.) can
   call `search_library` / `get_book_details` directly.
5. A small **evaluation suite** scores the agent's answers against a set of
   hand-written "fuzzy memory" queries with known expected books, so changes
   to the prompt/model/index can be checked for regressions instead of
   eyeballed.


## How it's put together

A few decisions shape most of the codebase, each driven by a concrete problem
with the obvious simpler alternative:

- **An agent, not a single prompt.** A one-shot "embed the query, hand the
  top-5 to an LLM, print its answer" pipeline can't recover when the first
  search comes back empty or ambiguous. [`llm/agent.py`](src/calibre_llm_finder/llm/agent.py)
  instead gives the model `search_library` / `get_book_details` as tools and
  lets it decide, turn by turn, whether to rephrase the query, pull more
  detail on a candidate, or ask the user a clarifying question — with the
  loop bounded so a model that can't converge fails gracefully instead of
  hanging. The same module handles the ways local models actually misbehave
  in practice: malformed tool-call arguments, an unreachable Ollama server,
  and a model that hallucinates a book ID that doesn't exist.
- **Streaming end to end.** A multi-turn agent call with local inference can
  take several seconds; both the CLI and the browser UI ([`web/app.py`](src/calibre_llm_finder/web/app.py),
  over Server-Sent Events) surface tokens and tool-call activity as they
  happen rather than showing a blank screen until the whole thing resolves.
- **An MCP server alongside the CLI/browser UI.** `search_library` and
  `get_book_details` ([`mcp/server.py`](src/calibre_llm_finder/mcp/server.py))
  are the same two tools the built-in agent uses, exposed over the
  [Model Context Protocol](https://modelcontextprotocol.io/) so any
  MCP-aware client — Claude Desktop, a different agent framework, a one-off
  script — can search the library too, without duplicating the search logic.
- **A real evaluation suite.** [`eval/`](src/calibre_llm_finder/eval/) runs
  a small set of hand-written "fuzzy memory" queries with known expected
  answers through the agent and scores them, so a prompt tweak, model swap,
  or re-ranking change can be checked for regressions rather than eyeballed
  against a couple of manual tries.
- **Local inference by default.** [`llm/ollama_client.py`](src/calibre_llm_finder/llm/ollama_client.py)
  talks to Ollama over plain HTTP, so your library contents and queries
  never have to leave your machine; pointing it at a different
  OpenAI/Ollama-compatible endpoint is a one-line config change if you want
  a cloud model instead.
- **`asyncio` for fan-out, one background thread for SQLite.** Indexing a
  library means many independent, I/O-bound embedding calls to the same
  local server — a natural fit for `asyncio` with a bounded semaphore
  ([`indexing/embeddings.py`](src/calibre_llm_finder/indexing/embeddings.py)).
  Writing those vectors back out is a different kind of problem: `sqlite3`
  is a blocking API tied to whichever thread opened the connection, so
  [`indexing/index_store.py`](src/calibre_llm_finder/indexing/index_store.py)
  gives it exactly one dedicated writer thread behind a queue, and hands
  async callers back a real `asyncio.Future` — using each concurrency model
  where it's actually the right tool, rather than forcing one everywhere.
- **Read-only, non-destructive by construction.** [`calibre/reader.py`](src/calibre_llm_finder/calibre/reader.py)
  opens Calibre's `metadata.db` with SQLite's `mode=ro` URI flag, so even a
  bug in this code can't write to it — Calibre can keep running normally
  alongside this tool, and all of this project's own derived data lives in a
  completely separate database.

See each module's docstring for more detail on the reasoning behind it.

## Architecture

```
┌──────────┐   ┌──────────────┐   ┌────────────────────┐
│   CLI    │   │  Browser UI  │   │     MCP server      │
│ (typer)  │   │  (FastAPI +  │   │ (any MCP client,     │
│          │   │   SSE)       │   │  e.g. Claude Desktop)│
└────┬─────┘   └──────┬───────┘   └──────────┬──────────┘
     │                │                      │
     └──────────┬─────┴──────────┬───────────┘
                ▼                ▼
         BookFindingAgent   SearchEngine
         (tool-calling loop)  (embed + vector search)
                │                │
                ▼                ▼
          OllamaClient      IndexStore (sqlite-vec,
     (async, streaming,    background writer thread)
      tool calls)                │
                                  ▼
                        Calibre's metadata.db
                        (read-only, never written to)
```

## Quickstart (macOS)

```bash
# 1. Install Ollama and pull the two models used by default
brew install ollama
ollama pull nomic-embed-text
ollama pull llama3.1

# 2. Install this project
git clone https://github.com/<you>/calibre-llm-finder.git
cd calibre-llm-finder
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. Build a sample library from Project Gutenberg (public domain books),
#    or point CLF_CALIBRE_LIBRARY_PATH at your real one instead.
python scripts/build_sample_library.py --out ~/Calibre\ Library\ Sample
cp .env.example .env
echo 'CLF_CALIBRE_LIBRARY_PATH=/Users/you/Calibre Library Sample' >> .env

# 4. Build the search index
clf index

# 5. Find a book
clf search "a doctor who drinks a potion and turns into someone evil at night"
# or the interactive REPL:
clf chat
# or the browser UI, at http://127.0.0.1:8000:
clf serve
# or an MCP server any MCP client can attach to over stdio:
clf mcp-serve
```

## Using it as an MCP server (e.g. with Claude Desktop)

`clf mcp-serve` exposes `search_library` and `get_book_details` as MCP
tools over stdio, so any MCP-compatible client can call them directly —
including asking Claude Desktop to search your Calibre library in plain
chat.

1. Find your venv's `clf` executable (Claude Desktop launches it directly,
   without activating the venv, so you need the full path):

   ```bash
   # from the project directory, with the venv created as in Quickstart
   which clf   # after `source .venv/bin/activate`
   # -> e.g. /Users/you/Projects/calibre-llm-finder/.venv/bin/clf
   ```

2. Open Claude Desktop's config file (macOS):

   ```bash
   open -e ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

3. Add an entry under `mcpServers`, using the absolute path from step 1
   and your real settings in `env` — Claude Desktop doesn't load `.env` or
   your shell profile, so these need to be spelled out explicitly rather
   than relying on the file picked up by `clf` when you run it yourself:

   ```json
   {
     "mcpServers": {
       "calibre-llm-finder": {
         "command": "/Users/you/Projects/calibre-llm-finder/.venv/bin/clf",
         "args": ["mcp-serve"],
         "env": {
           "CLF_CALIBRE_LIBRARY_PATH": "/Users/you/Calibre Library",
           "CLF_DATA_DIR": "/Users/you/.calibre_llm_finder",
           "CLF_OLLAMA_HOST": "http://localhost:11434"
         }
       }
     }
   }
   ```

4. Restart Claude Desktop, then look for the 🔌 tool icon in a new chat to
   confirm `calibre-llm-finder` is connected. Ask something like *"Use
   calibre-llm-finder to find a book about a doctor who turns into
   someone evil at night"* — Claude will call `search_library` itself and
   reason over the results.

The same `clf mcp-serve` command works with any other MCP client (Claude
Code's `claude mcp add`, a custom agent harness, etc.) — point it at the
same executable and environment variables.

## Running the evaluation suite

```bash
clf eval
```

Prints a pass/fail table comparing the agent's answer on each hand-written
"fuzzy memory" query against the expected book title.

## Development

```bash
uv pip install -e ".[dev]"
pytest
ruff check .
mypy src
```

## Quality & testing

- **Unit tests** cover the parts of the system that are actually worth
  testing in isolation — Calibre metadata parsing, the vector index, the
  agent's tool-call handling and failure paths, concurrency limits on
  embedding calls — with the network mocked (`respx`) so the suite runs
  offline and deterministically.
- **Static checks**: `ruff` (lint + formatting) and `mypy` both run clean
  against fully type-hinted code, with Pydantic models enforcing the data
  contracts at the boundaries (Calibre rows, config, API payloads).
- **CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint,
  format-check, type-check, and the test suite with coverage on every push.

## Design notes / trade-offs

- **Standalone app, not a Calibre plugin.** Calibre plugins run inside
  Calibre's bundled Qt/Python environment, whose event loop doesn't mix
  cleanly with `asyncio`, and install as a `.zip` via `calibre-customize`
  rather than a Python package you can `pip install` and run on its own.
  Reading `metadata.db` directly, read-only, gets the same integration with
  a normal Python packaging/dependency story instead.

## License

MIT — see [LICENSE](LICENSE). All sample library content is sourced from
[Project Gutenberg](https://www.gutenberg.org/) (public domain).
