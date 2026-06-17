"""Throughput benchmark for the NeuroMem compression pipeline.

Measures **tokens processed per second** across the four content-type
strategies (logs, conversation, code, text/RAG) using the offline
:class:`MockLLMProvider`, so the benchmark is fully reproducible and
needs no network access or API keys.

Usage
-----
Run directly as a script::

    python tests/benchmark.py
    python tests/benchmark.py --iterations 2000

Or import :func:`run_benchmark` programmatically to integrate the
results into a CI gate / dashboard.

Output
------
A table of per-strategy metrics plus a global aggregate:

* ``iterations``  — compress calls made for that strategy
* ``tokens_in``   — total input tokens processed
* ``tokens_out``  — total output (summary) tokens produced
* ``throughput``  — ``tokens_in / elapsed_seconds`` (tokens/sec)
* ``mean_latency``— mean microseconds per ``compress`` call
* ``ratio``       — mean ``tokens_out / tokens_in`` (higher = less compression)

The benchmark is intentionally **deterministic**: identical inputs and
iteration counts yield comparable numbers across runs, making it usable
for regression detection (e.g. a refactor that drops throughput by 50%).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Allow running as a standalone script from anywhere by ensuring the
# project root (the parent of ``tests/``) is importable.  pytest already
# adds it via ``pythonpath`` in pyproject.toml, so this is a no-op there.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from neuromem.compression import CompressionEngine, ReversibleStore, estimate_tokens


# ═══════════════════════════════════════════════════════════════════════
# Representative payloads (one per content type)
# ═══════════════════════════════════════════════════════════════════════

LOG_PAYLOAD = "\n".join(
    f"2024-01-15 14:{m:02d}:{s:02d} {lvl} event detail line {i}"
    for i, (m, s, lvl) in enumerate(
        ([(i // 60 % 60, i % 60, "INFO") for i in range(20)]
         + [(20, 0, "ERROR"), (21, 0, "WARN"), (22, 0, "CRITICAL")])
    )
)
"""A ~23-line structured log batch with mixed severities."""

CONVERSATION_PAYLOAD = (
    "User: What is NeuroMem and why should we use it?\n"
    "Assistant: NeuroMem is a hybrid memory engine. It supports graph and vector search.\n"
    "User: We decided to use it for the project. Task: configure the database.\n"
    "Assistant: Understood. NeuroMem integrates Kuzu and ChromaDB for storage.\n"
    "User: How does it handle contradictions?\n"
    "Assistant: NeuroMem detects contradictions and can split reasoning paths.\n"
)
"""A 6-turn conversation transcript."""

CODE_PAYLOAD = '''\
"""Example module for benchmarking code compression."""
import os
from typing import List, Dict, Optional


def add(a: int, b: int = 2) -> int:
    """Add two integers together."""
    return a + b


async def fetch(url: str) -> str:
    """Fetch a resource asynchronously."""
    return url


class Engine:
    """A small example engine."""

    name: str = "engine"

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, count: int) -> List[str]:
        """Run the engine count times."""
        return [self.name for _ in range(count)]
'''
"""A small Python module with imports, functions, and a class."""

TEXT_PAYLOAD = (
    "NeuroMem is a unified hybrid memory engine for AI agents. "
    "It combines graph-based reasoning and semantic vector search to build "
    "persistent, evolving cognitive memory. NeuroMem integrates Kuzu and "
    "ChromaDB to enable agents to learn, recall, reason, detect contradictions, "
    "track failures, and propagate knowledge across namespaces."
)
"""A prose / RAG-style passage."""

PAYLOADS: dict[str, str] = {
    "logs": LOG_PAYLOAD,
    "conversation": CONVERSATION_PAYLOAD,
    "code": CODE_PAYLOAD,
    "text": TEXT_PAYLOAD,
}


# ═══════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BenchmarkResult:
    """Throughput metrics for a single content-type strategy."""

    strategy: str
    iterations: int
    tokens_in: int
    tokens_out: int
    elapsed_seconds: float
    mean_latency_us: float
    throughput_tokens_per_sec: float
    mean_ratio: float

    def as_row(self) -> tuple[str, int, int, int, float, float, float, float]:
        return (
            self.strategy,
            self.iterations,
            self.tokens_in,
            self.tokens_out,
            self.elapsed_seconds,
            self.mean_latency_us,
            self.throughput_tokens_per_sec,
            self.mean_ratio,
        )


# ═══════════════════════════════════════════════════════════════════════
# Core benchmark loop
# ═══════════════════════════════════════════════════════════════════════

def _benchmark_strategy(
    engine: CompressionEngine,
    payload: str,
    iterations: int,
    label: str,
) -> BenchmarkResult:
    """Run ``iterations`` compress calls on *payload* and collect timing."""
    tokens_in_per_call = estimate_tokens(payload)
    latencies_us: list[float] = []
    total_tokens_out = 0

    # Warm-up call (not timed) — primes caches / first-import paths.
    engine.compress(payload, memory_id=f"{label}_warmup")

    for i in range(iterations):
        start = time.perf_counter()
        snapshot = engine.compress(payload, memory_id=f"{label}_{i}")
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        latencies_us.append(elapsed_us)
        total_tokens_out += estimate_tokens(snapshot.summary)

    elapsed_seconds = sum(latencies_us) / 1_000_000
    tokens_in_total = tokens_in_per_call * iterations
    mean_latency_us = statistics.fmean(latencies_us)
    throughput = tokens_in_total / elapsed_seconds if elapsed_seconds > 0 else 0.0
    mean_ratio = (
        total_tokens_out / tokens_in_total if tokens_in_total > 0 else 0.0
    )

    return BenchmarkResult(
        strategy=label,
        iterations=iterations,
        tokens_in=tokens_in_total,
        tokens_out=total_tokens_out,
        elapsed_seconds=elapsed_seconds,
        mean_latency_us=mean_latency_us,
        throughput_tokens_per_sec=throughput,
        mean_ratio=mean_ratio,
    )


def run_benchmark(
    iterations: int = 1000,
    strategies: Sequence[str] | None = None,
) -> list[BenchmarkResult]:
    """Run the throughput benchmark and return per-strategy results.

    Parameters
    ----------
    iterations:
        Number of ``compress`` calls per strategy.
    strategies:
        Subset of strategy labels to run. Defaults to all four.
    """
    selected = list(strategies) if strategies else list(PAYLOADS)
    results: list[BenchmarkResult] = []

    with tempfile.TemporaryDirectory(prefix="neuromem_bench_") as tmp:
        with ReversibleStore(tmp) as store:
            for label in selected:
                if label not in PAYLOADS:
                    raise ValueError(
                        f"Unknown strategy {label!r}; "
                        f"choose from {sorted(PAYLOADS)}"
                    )
                engine = CompressionEngine(reversible_store=store)
                results.append(
                    _benchmark_strategy(engine, PAYLOADS[label], iterations, label)
                )
    return results


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

_HEADERS = (
    "strategy", "iters", "tokens_in", "tokens_out",
    "elapsed_s", "latency_us", "tok/sec", "ratio",
)


def _format_table(results: list[BenchmarkResult]) -> str:
    """Render results as a fixed-width ASCII table."""
    rows = [_HEADERS] + [r.as_row() for r in results]

    # Aggregate row.
    total_in = sum(r.tokens_in for r in results)
    total_out = sum(r.tokens_out for r in results)
    total_elapsed = sum(r.elapsed_seconds for r in results)
    agg_throughput = total_in / total_elapsed if total_elapsed > 0 else 0.0
    agg_ratio = total_out / total_in if total_in > 0 else 0.0
    rows.append((
        "TOTAL", results[0].iterations if results else 0,
        total_in, total_out, total_elapsed,
        0.0, agg_throughput, agg_ratio,
    ))

    widths = [max(len(str(row[c])) for row in rows) for c in range(len(_HEADERS))]
    sep = "+".join("-" * (w + 2) for w in widths)
    sep = f"+{sep}+"

    def fmt(row: tuple) -> str:
        cells = []
        for val, w, idx in zip(row, widths, range(len(row))):
            # Right-align numeric columns (everything except the first).
            s = f"{val:.2f}" if isinstance(val, float) else str(val)
            cells.append(s.rjust(w) if idx else s.ljust(w))
        return "| " + " | ".join(cells) + " |"

    lines = [sep, fmt(rows[0]), sep]
    for row in rows[1:-1]:
        lines.append(fmt(row))
    lines.append(sep)
    lines.append(fmt(rows[-1]))
    lines.append(sep)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def main(argv: Sequence[str] | None = None) -> int:
    # Silence the engine's DEBUG/INFO logging so only the result table shows.
    import logging

    try:
        from loguru import logger

        logger.remove()
    except ImportError:  # pragma: no cover — loguru is a hard dep
        logging.disable(logging.CRITICAL)

    parser = argparse.ArgumentParser(
        description="Benchmark NeuroMem compression throughput (tokens/sec).",
    )
    parser.add_argument(
        "-n", "--iterations", type=int, default=1000,
        help="compress calls per strategy (default: 1000)",
    )
    parser.add_argument(
        "--strategies", nargs="*", default=None,
        choices=sorted(PAYLOADS),
        help="subset of strategies to benchmark (default: all)",
    )
    args = parser.parse_args(argv)

    if args.iterations < 1:
        parser.error("--iterations must be >= 1")

    print(f"NeuroMem compression benchmark — {args.iterations} iterations/strategy\n")
    results = run_benchmark(iterations=args.iterations, strategies=args.strategies)
    print(_format_table(results))

    # Exit non-zero if any strategy failed to process (throughput == 0).
    if any(r.throughput_tokens_per_sec <= 0 for r in results):
        print("\nERROR: one or more strategies produced zero throughput.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
