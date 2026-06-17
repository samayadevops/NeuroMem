"""Async-native wrapper around :class:`NeuroMemClient`.

All methods are coroutines that delegate blocking I/O to the default
thread-pool via :func:`asyncio.to_thread`, making NeuroMem compatible
with FastAPI, LangGraph, Chainlit, and any other asyncio-based framework.

Because Kuzu's Python driver is *not* thread-safe for concurrent writes,
an :class:`asyncio.Lock` serialises all engine calls so that concurrent
``await client.learn(...)`` coroutines never race.

Usage::

    import asyncio
    from neuromem import AsyncNeuroMemClient

    async def main():
        async with await AsyncNeuroMemClient.create(\"./agent_memory\") as client:
            belief = await client.learn(\"The sky is blue\", confidence=0.9)
            results = await client.recall(\"sky colour\")
            for r in results:
                print(r.claim, r.confidence)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from neuromem.client import EmbedFn, NeuroMemClient, RecallResult, SharedMemoryRecord
from neuromem.compression.models import MemorySnapshot
from neuromem.core.engine import EngineConfig
from neuromem.core.models import (
    BeliefNode,
    BeliefStatus,
    ContradictionEvent,
    NegativeMemory,
    NegativeMemorySeverity,
    PropagationRecord,
)


class AsyncNeuroMemClient:
    """Async-native NeuroMem client.

    Every public method is a coroutine.  Internally, the synchronous
    :class:`NeuroMemClient` is used via :func:`asyncio.to_thread`.

    Parameters
    ----------
    sync_client:
        A fully-initialised synchronous :class:`NeuroMemClient`.  Use
        the :meth:`create` factory instead of constructing this directly.

    Example
    -------
    ::

        async with await AsyncNeuroMemClient.create(\"./data\") as client:
            belief = await client.learn(\"Paris is in France\")
            results = await client.recall(\"French cities\")
    """

    def __init__(self, sync_client: NeuroMemClient) -> None:
        self._sync = sync_client
        # Serialise writes to protect Kuzu's non-thread-safe write path.
        self._lock: asyncio.Lock = asyncio.Lock()

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    async def create(
        cls,
        storage_dir: str | Path = "./neuromem_data",
        *,
        namespace: str = "default",
        embed_fn: EmbedFn | None = None,
        config: EngineConfig | None = None,
    ) -> "AsyncNeuroMemClient":
        """Async factory that mirrors :meth:`NeuroMemClient.create`.

        Parameters
        ----------
        storage_dir:
            Root directory for all NeuroMem data.
        namespace:
            Default namespace for this client.
        embed_fn:
            Optional embedding function.
        config:
            Optional :class:`EngineConfig` for cognitive tuning.

        Returns
        -------
        AsyncNeuroMemClient
            A ready-to-use async client.
        """
        sync = await asyncio.to_thread(
            NeuroMemClient.create,
            storage_dir,
            namespace=namespace,
            embed_fn=embed_fn,
            config=config,
        )
        return cls(sync)

    # ── Context manager ───────────────────────────────────────────────────

    async def __aenter__(self) -> "AsyncNeuroMemClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        await asyncio.to_thread(self._sync.__exit__, exc_type, exc_val, exc_tb)

    # ── Core cognitive operations ─────────────────────────────────────────

    async def learn(
        self,
        claim: str,
        *,
        confidence: float = 0.5,
        source: str = "agent",
        namespace: str | None = None,
        tags: list[str] | None = None,
        gamma: float | None = None,
    ) -> BeliefNode:
        """Learn a belief (async).  See :meth:`NeuroMemClient.learn`."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync.learn,
                claim,
                confidence=confidence,
                source=source,
                namespace=namespace,
                tags=tags,
                gamma=gamma,
            )

    async def recall(
        self,
        query: str | None = None,
        *,
        n_results: int = 10,
        namespace: str | None = None,
        min_confidence: float = 0.0,
        apply_decay: bool = True,
        auto_embed: bool = True,
    ) -> list[RecallResult]:
        """Recall beliefs (async).  See :meth:`NeuroMemClient.recall`."""
        return await asyncio.to_thread(
            self._sync.recall,
            query,
            n_results=n_results,
            namespace=namespace,
            min_confidence=min_confidence,
            apply_decay=apply_decay,
            auto_embed=auto_embed,
        )

    async def forget(
        self,
        belief_id: str,
        *,
        namespace: str | None = None,
    ) -> bool:
        """Deprecate a belief (async).  See :meth:`NeuroMemClient.forget`."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync.forget,
                belief_id,
                namespace=namespace,
            )

    async def guard(
        self,
        pattern: str,
        *,
        severity: NegativeMemorySeverity | str = NegativeMemorySeverity.WARNING,
        block_threshold: int = 1,
        context: dict[str, Any] | None = None,
        related_belief_id: str | None = None,
        namespace: str | None = None,
        pattern_type: str = "exact",
        fuzzy_threshold: float = 0.8,
    ) -> NegativeMemory:
        """Record a guardrail (async).  See :meth:`NeuroMemClient.guard`."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync.guard,
                pattern,
                severity=severity,
                block_threshold=block_threshold,
                context=context,
                related_belief_id=related_belief_id,
                namespace=namespace,
                pattern_type=pattern_type,
                fuzzy_threshold=fuzzy_threshold,
            )

    async def is_blocked(
        self,
        pattern: str,
        *,
        namespace: str | None = None,
    ) -> bool:
        """Check if a pattern is blocked (async).  See :meth:`NeuroMemClient.is_blocked`."""
        return await asyncio.to_thread(
            self._sync.is_blocked,
            pattern,
            namespace=namespace,
        )

    async def propagate(
        self,
        belief_id: str,
        target_namespace: str,
        *,
        trust_factor: float | None = None,
        namespace: str | None = None,
    ) -> PropagationRecord:
        """Propagate a belief (async).  See :meth:`NeuroMemClient.propagate`."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync.propagate,
                belief_id,
                target_namespace,
                trust_factor=trust_factor,
                namespace=namespace,
            )

    async def check_contradiction(
        self,
        belief_id: str,
        incoming_claim: str,
        *,
        incoming_embedding: list[float] | None = None,
        namespace: str | None = None,
        auto_embed: bool = True,
    ) -> ContradictionEvent | None:
        """Check for contradiction (async).  See :meth:`NeuroMemClient.check_contradiction`."""
        return await asyncio.to_thread(
            self._sync.check_contradiction,
            belief_id,
            incoming_claim,
            incoming_embedding=incoming_embedding,
            namespace=namespace,
            auto_embed=auto_embed,
        )

    # ── Compression ───────────────────────────────────────────────────────

    async def compress(self, text: str, *, importance: float = 0.5) -> MemorySnapshot:
        """Compress text (async).  See :meth:`NeuroMemClient.compress`."""
        return await asyncio.to_thread(self._sync.compress, text, importance=importance)

    async def learn_compressed(
        self,
        text: str,
        *,
        confidence: float = 0.5,
        source: str = "agent",
        namespace: str | None = None,
        tags: list[str] | None = None,
    ) -> tuple[BeliefNode, MemorySnapshot, list[Any]]:
        """Compress and learn in one step (async).  See :meth:`NeuroMemClient.learn_compressed`."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync.learn_compressed,
                text,
                confidence=confidence,
                source=source,
                namespace=namespace,
                tags=tags,
            )

    async def retrieve_original(self, memory_id: str) -> str:
        """Retrieve original text (async).  See :meth:`NeuroMemClient.retrieve_original`."""
        return await asyncio.to_thread(self._sync.retrieve_original, memory_id)

    async def share_memory(
        self,
        memory_id: str,
        target_namespace: str,
        *,
        trust_factor: float | None = None,
    ) -> SharedMemoryRecord:
        """Share a memory snapshot (async).  See :meth:`NeuroMemClient.share_memory`."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync.share_memory,
                memory_id,
                target_namespace,
                trust_factor=trust_factor,
            )

    # ── Stats & management ────────────────────────────────────────────────

    async def stats(self) -> dict[str, Any]:
        """Return unified stats (async).  See :meth:`NeuroMemClient.stats`."""
        return await asyncio.to_thread(self._sync.stats)

    async def decay(
        self,
        *,
        namespace: str | None = None,
        advance_ticks: int = 1,
    ) -> int:
        """Apply temporal decay (async).  See :meth:`NeuroMemClient.decay`."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync.decay,
                namespace=namespace,
                advance_ticks=advance_ticks,
            )

    async def get_belief(
        self,
        belief_id: str,
        *,
        namespace: str | None = None,
    ) -> BeliefNode | None:
        """Fetch a single belief by ID (async).  See :meth:`NeuroMemClient.get_belief`."""
        return await asyncio.to_thread(
            self._sync.get_belief,
            belief_id,
            namespace=namespace,
        )

    async def list_beliefs(
        self,
        *,
        namespace: str | None = None,
        status: BeliefStatus | None = None,
    ) -> list[BeliefNode]:
        """List beliefs (async).  See :meth:`NeuroMemClient.list_beliefs`."""
        return await asyncio.to_thread(
            self._sync.list_beliefs,
            namespace=namespace,
            status=status,
        )

    # ── Pass-through properties ────────────────────────────────────────────

    @property
    def namespace(self) -> str:
        """The default namespace for this client."""
        return self._sync._engine.namespace

    def __repr__(self) -> str:
        return (
            f"AsyncNeuroMemClient(namespace={self.namespace!r}, "
            f"sync={self._sync!r})"
        )
