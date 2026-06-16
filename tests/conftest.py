"""Shared pytest fixtures for the NeuroMem test suite.

Provides:
- A fresh ``tmp_path``-isolated storage directory per test.
- Initialised :class:`KuzuGraphEngine` and :class:`ChromaVectorEngine`.
- A wired :class:`NeuroMemEngine` and :class:`NeuroMemClient`.

All storage is ephemeral — cleaned up automatically by pytest's
``tmp_path`` fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuromem.client import NeuroMemClient
from neuromem.core.engine import EngineConfig, NeuroMemEngine
from neuromem.storage.chroma_vector import ChromaVectorEngine
from neuromem.storage.kuzu_graph import KuzuGraphEngine


@pytest.fixture()
def storage_dir(tmp_path: Path) -> Path:
    """Return a fresh ephemeral storage root directory."""
    d = tmp_path / "neuromem_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def graph_engine(storage_dir: Path) -> KuzuGraphEngine:
    """Return an initialised KuzuGraphEngine backed by tmp storage."""
    engine = KuzuGraphEngine(str(storage_dir / "graph"))
    engine.initialize()
    yield engine
    engine.close()


@pytest.fixture()
def vector_engine(storage_dir: Path) -> ChromaVectorEngine:
    """Return an initialised ChromaVectorEngine backed by tmp storage."""
    engine = ChromaVectorEngine(str(storage_dir / "vectors"))
    engine.initialize()
    yield engine
    engine.close()


@pytest.fixture()
def engine_config() -> EngineConfig:
    """Return an EngineConfig with test-friendly thresholds."""
    return EngineConfig(
        contradiction_threshold=0.5,
        decay_floor=0.1,
        trust_threshold=0.2,
        fusion_vector_weight=0.5,
        default_gamma=0.99,
        default_trust_factor=0.8,
    )


@pytest.fixture()
def engine(
    graph_engine: KuzuGraphEngine,
    vector_engine: ChromaVectorEngine,
    engine_config: EngineConfig,
) -> NeuroMemEngine:
    """Return a fully-wired NeuroMemEngine with bootstrapped schemas."""
    return NeuroMemEngine(
        graph_engine=graph_engine,
        vector_engine=vector_engine,
        config=engine_config,
        namespace="test",
    )


@pytest.fixture()
def client(
    graph_engine: KuzuGraphEngine,
    vector_engine: ChromaVectorEngine,
    engine_config: EngineConfig,
) -> NeuroMemClient:
    """Return a NeuroMemClient wrapping a fresh engine."""
    eng = NeuroMemEngine(
        graph_engine=graph_engine,
        vector_engine=vector_engine,
        config=engine_config,
        namespace="test",
    )
    return NeuroMemClient(eng)


# A deterministic fake embedding function for tests.
def _fake_embed(text: str) -> list[float]:
    """Deterministic embedding: hash-based 8-dim vector in [0, 1]."""
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:8]]


@pytest.fixture()
def embed_fn():
    """Return a deterministic fake embedding function for tests."""
    return _fake_embed


@pytest.fixture()
def client_with_embeddings(
    graph_engine: KuzuGraphEngine,
    vector_engine: ChromaVectorEngine,
    engine_config: EngineConfig,
    embed_fn,
):
    """Return a NeuroMemClient with an embedding function attached."""
    eng = NeuroMemEngine(
        graph_engine=graph_engine,
        vector_engine=vector_engine,
        config=engine_config,
        namespace="test",
    )
    return NeuroMemClient(eng, embed_fn=embed_fn)
