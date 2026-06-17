<p align="center">

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Version](https://img.shields.io/badge/version-v0.1.0-orange)

</p>

# NeuroMem

A unified hybrid memory engine for AI agents that combines graph-based reasoning and semantic vector search to build persistent, evolving cognitive memory.

NeuroMem integrates Kuzu and ChromaDB to enable agents to learn, recall, reason, detect contradictions, track failures, and propagate knowledge across namespaces.

## Features

* Hybrid memory retrieval using graph relationships and vector similarity
* Confidence-based beliefs with logical-time decay and **automatic belief reinforcement** on duplicate observations
* Automatic contradiction detection and resolution
* Negative memory for recording failed actions and dead ends (now with **Regex and Fuzzy** pattern matching)
* Reasoning trace capture for auditability and debugging
* Trust-aware knowledge propagation between agent namespaces
* **Built-in Embedding Providers**: Out-of-the-box support for OpenAI, local Ollama, and offline Sentence-Transformers
* **Async-Native Client**: Thread-safe `asyncio` wrapper for seamless FastAPI and LangGraph integration
* **MCP Server**: Built-in Model Context Protocol server (`neuromem serve`) for drop-in integration with Cursor, Claude Code, and Continue
* **Token Usage Reducer (Context Compression)** with content-type routing, reversible storage, and domain-specific summarisation
* **Full CLI** for integration with shells, Claude Code, Gemini CLI, and any subprocess-capable agent
* Persistent storage powered by Kuzu and ChromaDB

## Core Concepts

### BeliefNodes

Semantic statements stored with confidence scores. Beliefs decay over logical time (ticks), allowing stale information to gradually lose influence.

### ContradictionEvents

Triggered when new observations conflict with existing beliefs. NeuroMem can resolve contradictions by splitting reasoning paths or deprecating outdated beliefs.

### NegativeMemory

Records failed actions, rejected decisions, and undesirable outcomes to help agents avoid repeating ineffective behaviors.

### ReasoningTraces

Step-by-step records of retrieval, inference, contradiction handling, and decay operations, providing full transparency into agent reasoning.

### PropagationRecords

Track how knowledge is shared across namespaces while preserving trust scores and decay state.

### Token Usage Reducer (Context Compression)

A context optimization pipeline designed to compress raw logs, conversation history, source code, and RAG retrieval chunks to fit within LLM context windows while preserving crucial details.

### CompressionEngine

The central orchestrator that manages context routing, compression, and reversible archiving. It tracks real-time statistics like `tokens_saved`, `compression_ratio`, and `stored_memories_count`.

### ReversibleStore

A lossless archiving layer that stores the original uncompressed text on disk and allows it to be retrieved later using a unique snapshot ID.

### ContentRouter

A classification component that automatically routes raw text to the appropriate summarization strategy (e.g. parsing Python AST, extracting logs severity, or merging conversational turns).


## Installation

> [!NOTE]
> Since this project uses native graph database dependencies (`kuzu`), it is recommended to use **Python 3.11 to 3.13**. On these versions, pre-built binary wheels are automatically fetched and installation will succeed instantly. On newer Python versions (such as Python 3.14+), pip may attempt to compile these dependencies from source, which requires CMake and C++ build tools.

### From PyPI

To install the latest release directly from PyPI:

```bash
pip install neuromem-ai
```

### From Source (For Development)

Using pip:

```bash
pip install -r requirements.txt
pip install -e .
```

Using Poetry:

```bash
poetry install
```

After installation, the `neuromem` command is available on your PATH.

## Quick Start

### Python API

```python
from neuromem import NeuroMemClient

with NeuroMemClient.create("./agent_memory") as client:

    belief = client.learn(
        "The sky is blue",
        confidence=0.9
    )

    print(f"Learned: {belief.claim}")

    results = client.recall("sky colour")

    for result in results:
        print(
            f"{result.claim} | Score: {result.fused_score:.3f}"
        )

    client.guard(
        "tool_run_failed: api_timeout",
        block_threshold=2
    )

### Async API

For async applications (FastAPI, LangGraph), use the native wrapper which safely serialises writes to the graph database:

```python
import asyncio
from neuromem import AsyncNeuroMemClient

async def main():
    async with await AsyncNeuroMemClient.create("./agent_memory") as client:
        belief = await client.learn("The sky is blue", confidence=0.9)
        results = await client.recall("sky colour")
        print(results[0].claim)

asyncio.run(main())
```
```

### CLI

The `neuromem` CLI outputs JSON by default so agents and scripts can parse results directly. Every command exits `0` on success and `1` on error.

```bash
# Learn a fact
neuromem learn "The Eiffel Tower is in Paris" --confidence 0.95 --tags "geography,europe"

# Recall related beliefs
neuromem recall "French landmarks" --n 5 --pretty

# Record a guardrail
neuromem guard "never call tool X without arguments" --severity warning

# Check if a pattern is blocked
neuromem is-blocked "never call tool X without arguments"

# List all beliefs
neuromem list --pretty

# Get a specific belief
neuromem get <belief_id>

# View unified stats
neuromem stats --pretty

# Apply temporal decay
neuromem decay --ticks 3

# Compress text (from argument, --file, or stdin)
neuromem compress "ERROR 2026-06-17 Connection timeout..."
cat server.log | neuromem compress

# Retrieve the original uncompressed text
neuromem retrieve <snapshot_id>

# Share a belief to another namespace
neuromem propagate <belief_id> agent_b --trust-factor 0.8

# Deprecate a belief
neuromem forget <belief_id>

# Start the MCP Server (Model Context Protocol)
neuromem serve
```

**Global flags** (available on all commands):

| Flag | Description |
|---|---|
| `--data-dir PATH` | Storage directory (default: `./neuromem_data`) |
| `--namespace NS` | Agent namespace (default: `default`) |
| `--embed-provider` | Provider name: `openai`, `ollama`, or `sentence-transformers` |
| `--embed-model` | Model string to use for the specified provider |
| `--pretty` | Pretty-print JSON output |
| `--quiet` | Suppress loguru diagnostic output on stderr |
| `--format {json,text}` | Output format (default: `json`) |

**Agent integration example** (Claude Code, Gemini CLI, shell scripts):

```bash
# Pipe JSON output for programmatic use
result=$(neuromem recall "French landmarks" --quiet)
echo "$result" | python -c "import sys,json; [print(r['claim']) for r in json.load(sys.stdin)['results']]"
```

## Practical Usage & AI Agent Integration Guide

NeuroMem is designed as a long-term persistent memory for AI agents (such as **Claude Code**, **Gemini CLI**, or custom chatbots). Since terminal-based agents have shell execution capabilities, you can teach them to use `neuromem` to remember context across conversations and prevent loop failures.

### 1. Teaching Your Agent to Use NeuroMem (System Prompt)

Paste this system instruction into your terminal agent or include it in your project's `.clauderc`, `.cursorrules`, or custom agent instructions:

> You have access to a persistent hybrid memory CLI tool called `neuromem`. Use it to preserve context across sessions and avoid repeating command/tool failures:
> 
> * **Retrieve past context** at the start of a task:
>   `neuromem --quiet --pretty recall "<keywords matching task>"`
> * **Save important discoveries** (configurations, files, API keys, credentials, logic decisions):
>   `neuromem --quiet --pretty learn "<fact to remember>"`
> * **Prevent failure loops**: If a command or API fails, register a guardrail block:
>   `neuromem --quiet guard "<failing command description>" --severity error`
> * **Check block status** before executing commands:
>   `neuromem --quiet is-blocked "<command to check>"`

---

### 2. MCP Server Integration (Model Context Protocol)

Instead of relying on the CLI, you can connect NeuroMem directly to any MCP-compatible agent (like **Claude Desktop**, **Cursor**, or **Continue**) using the built-in MCP server. This gives the agent native, zero-configuration access to the `learn`, `recall`, `guard`, and `compress` tools.

1. Ensure you have the `mcp` extra installed:
   ```bash
   pip install "neuromem-ai[mcp]"
   ```
2. Add the server to your agent's configuration file (for example, `~/.claude/claude_desktop_config.json` for Claude Desktop):

   ```json
   {
     "mcpServers": {
       "neuromem": {
         "command": "neuromem",
         "args": ["--data-dir", "./agent_memory", "serve"]
       }
     }
   }
   ```
3. Restart your agent. It will now automatically store and retrieve memories through NeuroMem without needing any system prompts!

---

### 3. Practical Examples

#### Scenario A: Persistent Developer Preferences

Register user choices so the agent remembers them forever:
```bash
# Save a choice
neuromem --quiet learn "User prefers Python for backend development and HSL for styling" --confidence 1.0

# Recall preference later
neuromem --quiet --pretty recall "styling preferences"
```

#### Scenario B: Preventing Loop Failures (Negative Memory Guardrails)

Prevent an agent from repeating a command that always fails:
```bash
# Register a block after an npm failure
neuromem --quiet guard "npm run dev fails on port 3000 due to occupation" --severity error

# Check before running
result=$(neuromem --quiet is-blocked "npm run dev on port 3000")
# Returns: {"pattern": "npm run dev on port 3000", "blocked": true}
```

#### Scenario C: Context Window Management (Log Compression)

If you have massive logs but want to save context space, compress them first:
```bash
# Compress long output
cat server.log | neuromem --quiet --pretty compress

# Returns a short summary + ID:
# { "id": "snapshot_123", "summary": "Database timeout after 10 retries" }

# Retrieve exact details only when fixing the bug
neuromem --quiet retrieve snapshot_123
```

---

## Running Tests

Run the full suite:

```bash
poetry run pytest
```

Run a focused subset (e.g. the compression layer) with coverage:

```bash
poetry run pytest tests/test_router.py tests/test_reversible_memory.py tests/test_compression.py tests/test_stats.py --cov=neuromem.compression --cov-report=term-missing --cov-branch
```

The test suite is fully deterministic and offline. The compression tests use a built-in `MockLLMProvider`, so no network access or API keys are required.

### Test layout

| File | Covers |
| --- | --- |
| `tests/test_router.py` | Heuristic content-type detection (markdown tables, JSON, Python code, logs) and detection priority |
| `tests/test_reversible_memory.py` | Lossless write/read round-trips through the structural id, SHA-256 tamper detection, and I/O fault handling |
| `tests/test_compression.py` | Per-strategy semantic extraction (logs, conversation, code AST, RAG dedup) and the pure helper functions |
| `tests/test_stats.py` | Exact token accounting for `tokens_before` / `tokens_after` and the global compression ratio |

### Throughput benchmark

A small benchmark measures tokens processed per second across the four compression strategies:

```bash
python tests/benchmark.py            # default 1000 iterations/strategy
python tests/benchmark.py -n 200     # fewer iterations for a quick run
```

It is also importable for programmatic use:

```python
from tests.benchmark import run_benchmark

results = run_benchmark(iterations=500, strategies=["code", "logs"])
for r in results:
    print(r.strategy, f"{r.throughput_tokens_per_sec:.0f} tok/sec")
```

## License

Licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
