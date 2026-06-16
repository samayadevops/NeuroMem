"""Clean, user-facing API for NeuroMem.

The :class:`NeuroMemClient` is the single entry point most users will
interact with.  It wraps the lower-level :class:`NeuroMemEngine` with:

- Sensible defaults (one-line initialisation with a storage directory).
- A factory classmethod (:meth:`create`) for the most common setup.
- Optional embedding-function injection so callers can supply their own
  embedding model (OpenAI, sentence-transformers, etc.) without touching
  the engine internals.
- Context-manager support for automatic resource cleanup.
- Convenience methods that mirror the cognitive model names
  (``learn``, ``recall``, ``forget``, ``propagate``, ``guard``).

Example
-------
::

    from neuromem import NeuroMemClient

    # One-line setup — creates Kuzu + ChromaDB in ./neuromem_data
    with NeuroMemClient.create("./neuromem_data") as client:
        # Teach the agent a fact
        belief = client.learn("The sky is blue", confidence=0.9)

        # Ask it to recall related facts
        results = client.recall("sky colour")
        for r in results:
            print(r.claim, r.confidence)

        # Record a guardrail
        client.guard("never call tool X without arguments")

        # Share knowledge with another agent
        client.propagate(belief.id, "agent_b")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Union

from loguru import logger

from neuromem.core.engine import (
    DEFAULT_FUSION_VECTOR_WEIGHT,
    EngineConfig,
    FusedResult,
    NeuroMemEngine,
)
from neuromem.core.exceptions import (
    ConfigurationError,
    NeuroMemError,
)
from neuromem.core.models import (
    BeliefNode,
    BeliefStatus,
    ContradictionEvent,
    NegativeMemory,
    NegativeMemorySeverity,
    PropagationRecord,
    ReasoningTrace,
)
from neuromem.storage.chroma_vector import ChromaVectorEngine
from neuromem.storage.kuzu_graph import KuzuGraphEngine

# ═══════════════════════════════════════════════════════════════════════
# Type aliases
# ═══════════════════════════════════════════════════════════════════════

#: A callable that converts text into a dense float vector.
EmbedFn = Callable[[str], list[float]]


# ═══════════════════════════════════════════════════════════════════════
# RecallResult — a friendlier wrapper around FusedResult
# ═══════════════════════════════════════════════════════════════════════

class RecallResult:
    """A user-friendly recall result exposing belief fields directly.

    Instead of forcing users to dig into ``result.belief.confidence``,
    this proxy exposes the most common fields at the top level while
    retaining access to the full :class:`BeliefNode` via ``.belief``.
    """

    __slots__ = (
        "_belief", "_fused_score", "_graph_confidence",
        "_vector_distance", "_similarity",
    )

    def __init__(self, fused: FusedResult) -> None:
        self._belief: BeliefNode = fused.belief
        self._fused_score: float = fused.fused_score
        self._graph_confidence: float = fused.graph_confidence
        self._vector_distance: float | None = fused.vector_distance
        # Convert distance to similarity for user convenience
        if fused.vector_distance is not None:
            self._similarity: float | None = max(0.0, 1.0 - fused.vector_distance)
        else:
            self._similarity = None

    # ── Direct belief accessors ────────────────────────────────────────

    @property
    def id(self) -> str:
        return self._belief.id

    @property
    def claim(self) -> str:
        return self._belief.claim

    @property
    def confidence(self) -> float:
        """Effective (decay-adjusted) confidence from the graph."""
        return self._graph_confidence

    @property
    def raw_confidence(self) -> float:
        """Raw stored confidence (before decay adjustment)."""
        return self._belief.confidence

    @property
    def status(self) -> BeliefStatus:
        return self._belief.status

    @property
    def source(self) -> str:
        return self._belief.source

    @property
    def tags(self) -> list[str]:
        return list(self._belief.tags)

    @property
    def namespace(self) -> str:
        return self._belief.namespace

    @property
    def created_at(self) -> datetime:
        return self._belief.created_at

    @property
    def evidence_count(self) -> int:
        return self._belief.evidence_count

    # ── Fusion-specific accessors ─────────────────────────────────────

    @property
    def fused_score(self) -> float:
        """The final fused score combining graph confidence + vector similarity."""
        return self._fused_score

    @property
    def similarity(self) -> float | None:
        """Vector similarity score in ``[0, 1]``, or ``None`` if no vector."""
        return self._similarity

    @property
    def vector_distance(self) -> float | None:
        """Raw vector distance from the query (lower = closer)."""
        return self._vector_distance

    @property
    def belief(self) -> BeliefNode:
        """Access the full underlying :class:`BeliefNode`."""
        return self._belief

    # ── Dunder ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"RecallResult(id={self.id!r}, claim={self.claim[:40]!r}, "
            f"fused={self._fused_score:.3f}, confidence={self._graph_confidence:.3f})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "id": self.id,
            "claim": self.claim,
            "confidence": self.confidence,
            "raw_confidence": self.raw_confidence,
            "fused_score": self._fused_score,
            "similarity": self._similarity,
            "vector_distance": self._vector_distance,
            "status": self.status.value if isinstance(self.status, BeliefStatus) else str(self.status),
            "source": self.source,
            "tags": self.tags,
            "namespace": self.namespace,
            "evidence_count": self.evidence_count,
            "created_at": self.created_at.isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════
# NeuroMemClient
# ═══════════════════════════════════════════════════════════════════════

class NeuroMemClient:
    """The primary user-facing API for NeuroMem.

    This client wraps :class:`NeuroMemEngine` with ergonomic defaults,
    optional embedding-function injection, and context-manager lifecycle
    management.

    Parameters
    ----------
    engine:
        A configured :class:`NeuroMemEngine` instance.
    embed_fn:
        Optional callable that converts a text string into a dense
        embedding vector.  When provided, ``learn()`` and ``recall()``
        automatically embed text inputs.

    Example
    -------
    ::

        client = NeuroMemClient.create("./my_agent_memory")
        belief = client.learn("Paris is the capital of France")
        results = client.recall("French capital")
    """

    def __init__(
        self,
        engine: NeuroMemEngine,
        *,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        self._engine: NeuroMemEngine = engine
        self._embed_fn: EmbedFn | None = embed_fn
        self._closed: bool = False

    # ══════════════════════════════════════════════════════════════════
    # Factory methods
    # ══════════════════════════════════════════════════════════════════

    @classmethod
    def create(
        cls,
        storage_dir: str | Path = "./neuromem_data",
        *,
        namespace: str = "default",
        embed_fn: EmbedFn | None = None,
        config: EngineConfig | None = None,
        auto_bootstrap: bool = True,
    ) -> "NeuroMemClient":
        """Create a fully-wired client with default Kuzu + ChromaDB backends.

        This is the recommended entry point for most users.  It:

        1. Resolves the storage directory.
        2. Creates a :class:`KuzuGraphEngine` in ``<storage_dir>/graph``.
        3. Creates a :class:`ChromaVectorEngine` in ``<storage_dir>/vectors``.
        4. Initialises both engines.
        5. Wires them into a :class:`NeuroMemEngine`.
        6. Returns a :class:`NeuroMemClient` wrapping the engine.

        Parameters
        ----------
        storage_dir:
            Root directory for all NeuroMem data.  Subdirectories
            ``graph/`` and ``vectors/`` are created within it.
        namespace:
            Default namespace for this client.
        embed_fn:
            Optional embedding function for automatic text→vector
            conversion.
        config:
            Optional :class:`EngineConfig` for cognitive tuning.
        auto_bootstrap:
            If ``True``, automatically register graph schemas.

        Returns
        -------
        NeuroMemClient
            A ready-to-use client.
        """
        storage_path = Path(storage_dir).resolve()
        graph_path = storage_path / "graph"
        vector_path = storage_path / "vectors"

        graph_path.mkdir(parents=True, exist_ok=True)
        vector_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Creating NeuroMemClient",
            storage_dir=str(storage_path),
            namespace=namespace,
        )

        try:
            graph = KuzuGraphEngine(str(graph_path))
            vector = ChromaVectorEngine(str(vector_path))

            graph.initialize()
            vector.initialize()

            engine = NeuroMemEngine(
                graph_engine=graph,
                vector_engine=vector,
                config=config,
                namespace=namespace,
                auto_bootstrap=auto_bootstrap,
            )

            return cls(engine, embed_fn=embed_fn)

        except NeuroMemError:
            raise
        except Exception as exc:
            raise ConfigurationError(
                "storage_dir",
                f"Failed to create NeuroMemClient: {exc}",
            ) from exc

    @classmethod
    def from_engine(
        cls,
        engine: NeuroMemEngine,
        *,
        embed_fn: EmbedFn | None = None,
    ) -> "NeuroMemClient":
        """Wrap an existing :class:`NeuroMemEngine` in a client.

        Use this when you need fine-grained control over the engine
        configuration or custom storage backends.
        """
        return cls(engine, embed_fn=embed_fn)

    # ══════════════════════════════════════════════════════════════════
    # Core cognitive operations
    # ══════════════════════════════════════════════════════════════════

    def learn(
        self,
        claim: str,
        *,
        confidence: float = 0.5,
        embedding: list[float] | None = None,
        source: str = "agent",
        namespace: str | None = None,
        tags: list[str] | None = None,
        gamma: float | None = None,
        auto_embed: bool = True,
    ) -> BeliefNode:
        """Teach the agent a new belief.

        If an embedding function was provided at construction (or via
        :meth:`set_embed_fn`) and ``embedding`` is ``None``, the claim
        text is automatically embedded.

        Parameters
        ----------
        claim:
            The semantic content to learn.
        confidence:
            Initial confidence in ``[0.0, 1.0]``.  Default ``0.5``.
        embedding:
            Pre-computed embedding vector.  Overrides auto-embedding.
        source:
            Origin label (e.g. ``"user"``, ``"observation"``).
        namespace:
            Target namespace.  Defaults to the client's namespace.
        tags:
            Free-form labels for filtering.
        gamma:
            Temporal decay rate.  Defaults to engine config.
        auto_embed:
            If ``True`` and an embed function is set, automatically
            embed the claim when ``embedding`` is ``None``.

        Returns
        -------
        BeliefNode
            The persisted belief.
        """
        self._require_open()

        effective_embedding = embedding
        if effective_embedding is None and auto_embed and self._embed_fn is not None:
            effective_embedding = self._safe_embed(claim)

        return self._engine.learn(
            claim=claim,
            confidence=confidence,
            embedding=effective_embedding,
            source=source,
            namespace=namespace,
            tags=tags,
            gamma=gamma,
        )

    def recall(
        self,
        query: str | None = None,
        *,
        query_embedding: list[float] | None = None,
        n_results: int = 10,
        namespace: str | None = None,
        min_confidence: float = 0.0,
        apply_decay: bool = True,
        auto_embed: bool = True,
    ) -> list[RecallResult]:
        """Recall beliefs relevant to a query.

        At least one of ``query`` or ``query_embedding`` must be
        provided.  If an embed function is set and only ``query`` is
        given, it is automatically embedded for vector search.

        Parameters
        ----------
        query:
            Text query for semantic search.
        query_embedding:
            Pre-computed query embedding.  Overrides auto-embedding.
        n_results:
            Maximum number of results to return.
        namespace:
            Namespace to search.  Defaults to the client's namespace.
        min_confidence:
            Filter out results below this fused-score threshold.
        apply_decay:
            If ``True``, apply temporal decay before scoring.
        auto_embed:
            If ``True`` and an embed function is set, automatically embed
            ``query`` when ``query_embedding`` is ``None``.

        Returns
        -------
        list[RecallResult]
            Results sorted by fused score (descending).
        """
        self._require_open()

        effective_embedding = query_embedding
        if effective_embedding is None and query is not None and auto_embed and self._embed_fn is not None:
            effective_embedding = self._safe_embed(query)

        fused_results = self._engine.recall(
            query=query,
            query_embedding=effective_embedding,
            n_results=n_results,
            namespace=namespace,
            min_confidence=min_confidence,
            apply_decay=apply_decay,
        )

        return [RecallResult(fr) for fr in fused_results]

    def forget(
        self,
        belief_id: str,
        *,
        namespace: str | None = None,
    ) -> bool:
        """Forget (deprecate) a belief.

        This does not physically delete the belief — it marks it as
        ``DEPRECATED`` with zero confidence, preserving the audit trail.

        Parameters
        ----------
        belief_id:
            ID of the belief to forget.
        namespace:
            Namespace of the belief.

        Returns
        -------
        bool
            ``True`` if the belief was found and deprecated.
        """
        self._require_open()
        ns = namespace or self._engine.namespace

        belief = self._engine._load_belief(belief_id, ns)
        if belief is None:
            return False

        belief.deprecate(reason="forgotten")
        self._engine._persist_belief(belief, trace=None, is_update=True)

        logger.info("Forgot belief {}", belief_id)
        return True

    # ══════════════════════════════════════════════════════════════════
    # Contradiction
    # ══════════════════════════════════════════════════════════════════

    def check_contradiction(
        self,
        belief_id: str,
        incoming_claim: str,
        *,
        incoming_embedding: list[float] | None = None,
        namespace: str | None = None,
        auto_embed: bool = True,
    ) -> ContradictionEvent | None:
        """Check if a claim contradicts an existing belief.

        Parameters
        ----------
        belief_id:
            ID of the existing belief to check against.
        incoming_claim:
            The new claim that may conflict.
        incoming_embedding:
            Pre-computed embedding of the incoming claim.
        namespace:
            Namespace of the existing belief.
        auto_embed:
            Auto-embed the incoming claim if an embed function is set.

        Returns
        -------
        ContradictionEvent | None
            A contradiction event if detected, otherwise ``None``.
        """
        self._require_open()
        ns = namespace or self._engine.namespace

        belief = self._engine._load_belief(belief_id, ns)
        if belief is None:
            return None

        effective_embedding = incoming_embedding
        if effective_embedding is None and auto_embed and self._embed_fn is not None:
            effective_embedding = self._safe_embed(incoming_claim)

        return self._engine.check_contradiction(
            belief, incoming_claim, effective_embedding, namespace=ns,
        )

    # ══════════════════════════════════════════════════════════════════
    # Negative memory / guardrails
    # ══════════════════════════════════════════════════════════════════

    def guard(
        self,
        pattern: str,
        *,
        severity: NegativeMemorySeverity | str = NegativeMemorySeverity.WARNING,
        block_threshold: int = 1,
        context: dict[str, Any] | None = None,
        related_belief_id: str | None = None,
        namespace: str | None = None,
    ) -> NegativeMemory:
        """Record a negative-memory guardrail to prevent repeating failures.

        Parameters
        ----------
        pattern:
            Description of the failed path or rejected logic.
        severity:
            Severity level (enum or string).
        block_threshold:
            Occurrences before this becomes a hard block.
        context:
            Structured context about the failure.
        related_belief_id:
            ID of an associated belief (if any).
        namespace:
            Target namespace.

        Returns
        -------
        NegativeMemory
            The recorded (or incremented) negative memory.
        """
        self._require_open()

        # Accept both enum and string for severity
        if isinstance(severity, str):
            try:
                severity = NegativeMemorySeverity(severity)
            except ValueError:
                severity = NegativeMemorySeverity.WARNING

        return self._engine.record_negative(
            pattern=pattern,
            context=context,
            severity=severity,
            block_threshold=block_threshold,
            related_belief_id=related_belief_id,
            namespace=namespace,
        )

    def is_blocked(
        self,
        pattern: str,
        *,
        namespace: str | None = None,
    ) -> bool:
        """Check if a pattern is currently blocked by a guardrail."""
        self._require_open()
        return self._engine.is_blocked(pattern, namespace)

    # ══════════════════════════════════════════════════════════════════
    # Propagation
    # ══════════════════════════════════════════════════════════════════

    def propagate(
        self,
        belief_id: str,
        target_namespace: str,
        *,
        trust_factor: float | None = None,
        namespace: str | None = None,
    ) -> PropagationRecord:
        """Share a belief with another agent namespace.

        Parameters
        ----------
        belief_id:
            ID of the belief to propagate.
        target_namespace:
            Receiving namespace.
        trust_factor:
            Confidence multiplier in ``[0, 1]``.  Defaults to engine config.
        namespace:
            Source namespace.  Defaults to the client's namespace.

        Returns
        -------
        PropagationRecord
            Record of the propagation attempt.
        """
        self._require_open()
        return self._engine.propagate(
            belief_id=belief_id,
            target_namespace=target_namespace,
            trust_factor=trust_factor,
            namespace=namespace,
        )

    # ══════════════════════════════════════════════════════════════════
    # Decay management
    # ══════════════════════════════════════════════════════════════════

    def decay(
        self,
        *,
        namespace: str | None = None,
        advance_ticks: int = 1,
    ) -> int:
        """Advance the logical tick and apply temporal decay.

        Parameters
        ----------
        namespace:
            Namespace to decay.  Defaults to the client's namespace.
        advance_ticks:
            Number of ticks to advance before decaying.

        Returns
        -------
        int
            Number of beliefs that fell below the decay floor.
        """
        self._require_open()
        if advance_ticks > 0:
            self._engine.advance_tick(advance_ticks)
        return self._engine.apply_global_decay(namespace)

    # ══════════════════════════════════════════════════════════════════
    # Inspection helpers
    # ══════════════════════════════════════════════════════════════════

    def get_belief(
        self,
        belief_id: str,
        *,
        namespace: str | None = None,
    ) -> BeliefNode | None:
        """Fetch a single belief by ID.  Returns ``None`` if not found."""
        self._require_open()
        ns = namespace or self._engine.namespace
        return self._engine._load_belief(belief_id, ns)

    def list_beliefs(
        self,
        *,
        namespace: str | None = None,
        status: BeliefStatus | None = None,
    ) -> list[BeliefNode]:
        """List all beliefs in a namespace, optionally filtered by status.

        Parameters
        ----------
        namespace:
            Namespace to scan.  Defaults to the client's namespace.
        status:
            Optional status filter (e.g. ``BeliefStatus.ACTIVE``).

        Returns
        -------
        list[BeliefNode]
            All matching beliefs.
        """
        self._require_open()
        ns = namespace or self._engine.namespace
        beliefs = self._engine._scan_beliefs(ns)
        if status is not None:
            beliefs = [b for b in beliefs if b.status == status]
        return beliefs

    def count_beliefs(
        self,
        *,
        namespace: str | None = None,
        active_only: bool = False,
    ) -> int:
        """Count beliefs in a namespace.

        Parameters
        ----------
        namespace:
            Namespace to count.  Defaults to the client's namespace.
        active_only:
            If ``True``, count only ACTIVE beliefs.

        Returns
        -------
        int
            Belief count.
        """
        self._require_open()
        ns = namespace or self._engine.namespace
        beliefs = self._engine._scan_beliefs(ns)
        if active_only:
            return sum(1 for b in beliefs if b.status == BeliefStatus.ACTIVE)
        return len(beliefs)

    # ══════════════════════════════════════════════════════════════════
    # Embedding function management
    # ══════════════════════════════════════════════════════════════════

    def set_embed_fn(self, embed_fn: EmbedFn | None) -> None:
        """Set or replace the embedding function.

        Pass ``None`` to disable auto-embedding.
        """
        self._embed_fn = embed_fn
        logger.debug("Embedding function {}", "set" if embed_fn else "cleared")

    @property
    def has_embed_fn(self) -> bool:
        """``True`` if an embedding function is configured."""
        return self._embed_fn is not None

    # ══════════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════════

    @property
    def namespace(self) -> str:
        """The client's default namespace."""
        return self._engine.namespace

    @property
    def current_tick(self) -> int:
        """The current logical tick counter."""
        return self._engine.current_tick

    @property
    def engine(self) -> NeuroMemEngine:
        """Direct access to the underlying engine (advanced use)."""
        return self._engine

    @property
    def config(self) -> EngineConfig:
        """The engine's cognitive configuration (mutable at runtime)."""
        return self._engine.config

    @property
    def is_closed(self) -> bool:
        """``True`` if the client has been closed."""
        return self._closed

    # ══════════════════════════════════════════════════════════════════
    # Lifecycle
    # ══════════════════════════════════════════════════════════════════

    def close(self) -> None:
        """Close the client and release all storage resources.

        After closing, no further operations are permitted.
        """
        if self._closed:
            return
        logger.info("Closing NeuroMemClient")
        try:
            self._engine.graph.close()
        except Exception as exc:
            logger.warning("Error closing graph engine: {}", exc)
        try:
            self._engine.vector.close()
        except Exception as exc:
            logger.warning("Error closing vector engine: {}", exc)
        self._closed = True

    # ══════════════════════════════════════════════════════════════════
    # Context manager support
    # ══════════════════════════════════════════════════════════════════

    def __enter__(self) -> "NeuroMemClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    # ══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════

    def _require_open(self) -> None:
        """Raise if the client has been closed."""
        if self._closed:
            raise ConfigurationError(
                "client",
                "NeuroMemClient has been closed. Create a new client to "
                "perform further operations.",
            )

    def _safe_embed(self, text: str) -> list[float] | None:
        """Call the embedding function with error handling.

        Returns ``None`` if embedding fails (non-fatal — operations
        proceed without vector features).
        """
        if self._embed_fn is None:
            return None
        try:
            embedding = self._embed_fn(text)
            if not isinstance(embedding, list) or not embedding:
                logger.warning("Embedding function returned invalid result for text")
                return None
            return [float(x) for x in embedding]
        except Exception as exc:
            logger.warning("Embedding function failed (non-fatal): {}", exc)
            return None


# ═══════════════════════════════════════════════════════════════════════
# Public re-exports
# ═══════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    "NeuroMemClient",
    "RecallResult",
    "EmbedFn",
]
