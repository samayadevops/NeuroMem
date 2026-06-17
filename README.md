<p align="center">

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
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

Using pip:

```bash
pip install -r requirements.txt
```

Using Poetry:

```bash
poetry install
```

## Quick Start

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

Licensed under the MIT License. See the LICENSE file for details.
