"""Context compression layer for NeuroMem.

This sub-package provides models and utilities for compressing raw
memory data (logs, conversations, belief snapshots) into compact,
retrievable representations suitable for long-term storage in ChromaDB
and Kuzu.

Public API
----------
- :class:`CompressionEngine` — top-level orchestrator that routes,
  compresses, persists, and returns :class:`MemorySnapshot` objects
  while tracking cumulative token metrics.
- :class:`ContentRouter` — heuristic content-type detection and routing.
- :class:`ContextCompressor` — per-content-type compression strategies.
- :class:`ReversibleStore` — durable archive of original (uncompressed)
  content, making every snapshot reversible.
- :mod:`models` — Pydantic v2 schemas for compression outputs and metrics.
"""

from neuromem.compression.compressor import CompressionEngine, estimate_tokens
from neuromem.compression.reversible_store import (
    InvalidMemoryIdError,
    MemoryNotFoundError,
    ReversibleStore,
    ReversibleStoreError,
)
from neuromem.compression.router import ContentRouter, ContentType
from neuromem.compression.summarizer import (
    BaseLLMProvider,
    ContextCompressor,
    MockLLMProvider,
)

__all__: list[str] = [
    # Engine (primary entry point)
    "CompressionEngine",
    "estimate_tokens",
    # Router
    "ContentRouter",
    "ContentType",
    # Strategies
    "ContextCompressor",
    "BaseLLMProvider",
    "MockLLMProvider",
    # Reversible store
    "ReversibleStore",
    "ReversibleStoreError",
    "MemoryNotFoundError",
    "InvalidMemoryIdError",
]
