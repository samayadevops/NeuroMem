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
* Confidence-based beliefs with logical-time decay
* Automatic contradiction detection and resolution
* Negative memory for recording failed actions and dead ends
* Reasoning trace capture for auditability and debugging
* Trust-aware knowledge propagation between agent namespaces
* Context compression with content-type routing, reversible storage, and per-strategy summarisation (logs, conversation, code, RAG)
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

## Installation

> [!NOTE]
> Since this project uses native graph database dependencies (`kuzu`), it is recommended to use **Python 3.11 to 3.13**. On these versions, pre-built binary wheels are automatically fetched and installation will succeed instantly. On newer Python versions (such as Python 3.14+), pip may attempt to compile these dependencies from source, which requires CMake and C++ build tools.

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
```

**Global flags** (available on all commands):

| Flag | Description |
|---|---|
| `--data-dir PATH` | Storage directory (default: `./neuromem_data`) |
| `--namespace NS` | Agent namespace (default: `default`) |
| `--pretty` | Pretty-print JSON output |
| `--quiet` | Suppress loguru diagnostic output on stderr |
| `--format {json,text}` | Output format (default: `json`) |

**Agent integration example** (Claude Code, Gemini CLI, shell scripts):

```bash
# Pipe JSON output for programmatic use
result=$(neuromem recall "French landmarks" --quiet)
echo "$result" | python -c "import sys,json; [print(r['claim']) for r in json.load(sys.stdin)['results']]"
```

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
