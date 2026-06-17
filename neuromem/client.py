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
- **Compression integration** — context compression, reversible storage,
  anomaly-aware learning, cross-namespace memory sharing, and unified stats.

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

        # Compress a long conversation into a compact memory snapshot
        snapshot = client.compress_history(messages)
        print(snapshot.summary)

        # Store the snapshot as a belief with anomaly detection
        client.learn_compressed(snapshot.summary)

        # Retrieve the original uncompressed text later
        original = client.retrieve_original(snapshot.raw_reference)

        # Share memory across agent boundaries
        client.share_memory(snapshot.id, "agent_c")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Union

from loguru import logger

from neuromem.compression.compressor import CompressionEngine, estimate_tokens
from neuromem.compression.models import MemorySnapshot
from neuromem.compression.reversible_store import (
    MemoryNotFoundError,
    ReversibleStore,
    ReversibleStoreError,
)
from neuromem.compression.summarizer import BaseLLMProvider, ContextCompressor
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
    ReasoningStep,
    ReasoningTrace,
    TraceStepType,
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
# SharedMemoryRecord — result of cross-agent memory sharing
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SharedMemoryRecord:
    """Record returned by :meth:`NeuroMemClient.share_memory`.

    Attributes
    ----------
    memory_id:
        The snapshot ID that was shared.
    target_namespace:
        The receiving namespace.
    trust_score:
        Computed trust score (confidence × decay factor) in ``[0, 1]``.
    created_at:
        UTC timestamp when the share was recorded.
    """
    memory_id: str
    target_namespace: str
    trust_score: float
    created_at: datetime = field(default_factory=lambda: datetime.now())

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "memory_id": self.memory_id,
            "target_namespace": self.target_namespace,
            "trust_score": self.trust_score,
            "created_at": self.created_at.isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════
# NeuroMemClient
# ═══════════════════════════════════════════════════════════════════════

class NeuroMemClient:
    """The primary user-facing API for NeuroMem.

    This client wraps :class:`NeuroMemEngine` with ergonomic defaults,
    optional embedding-function injection, and context-manager lifecycle
    management.  It also integrates the compression layer (via
    :class:`CompressionEngine` and :class:`ReversibleStore`) so that
    callers can compress, learn, retrieve, share, and inspect memories
    through a single interface.

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

        # Compression subsystem (lazy-initialised on first use).
        self._compression_engine: CompressionEngine | None = None
        self._reversible_store: ReversibleStore | None = None

        # Track shared memory references for decay/cross-namespace awareness.
        self._shared_memory_registry: dict[str, dict[str, Any]] = {}

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
        6. Creates a :class:`ReversibleStore` in ``<storage_dir>/raw_archive``.
        7. Returns a :class:`NeuroMemClient` wrapping the engine.

        Parameters
        ----------
        storage_dir:
            Root directory for all NeuroMem data.  Subdirectories
            ``graph/``, ``vectors/``, and ``raw_archive/`` are created
            within it.
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
        archive_path = storage_path / "raw_archive"

        # Ensure the storage root exists, but do NOT pre-create the graph /
        # vectors sub-directories — Kuzu rejects a pre-existing empty
        # directory and ChromaDB prefers to manage its own folder.  Each
        # engine's ``initialize()`` creates its subdir as needed.
        storage_path.mkdir(parents=True, exist_ok=True)

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

            client = cls(engine, embed_fn=embed_fn)

            # Pre-initialise the reversible store so that compress()
            # works out of the box.
            store = ReversibleStore(str(archive_path))
            store.initialize()
            client._reversible_store = store
            client._compression_engine = CompressionEngine(
                reversible_store=store,
            )

            return client

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

        Note
        ----
        Compression features (``compress``, ``learn_compressed``, etc.)
        require a :class:`ReversibleStore` to be attached.  Either pass
        one via :meth:`attach_compression` after construction, or use
        :meth:`create` which wires everything automatically.
        """
        return cls(engine, embed_fn=embed_fn)

    # ══════════════════════════════════════════════════════════════════
    # Compression lifecycle
    # ══════════════════════════════════════════════════════════════════

    def attach_compression(
        self,
        reversible_store: ReversibleStore,
        *,
        llm: BaseLLMProvider | None = None,
    ) -> None:
        """Attach (or replace) the compression subsystem.

        Call this when constructing the client via :meth:`from_engine`
        or when you want to swap in a different LLM provider.

        Parameters
        ----------
        reversible_store:
            An already-initialised :class:`ReversibleStore`.
        llm:
            Optional LLM provider for semantic extraction.
        """
        self._require_open()
        if not reversible_store.is_ready:
            raise ConfigurationError(
                "reversible_store",
                "Store must be initialised before attaching; "
                "call initialize() or enter it as a context manager.",
            )
        self._reversible_store = reversible_store
        self._compression_engine = CompressionEngine(
            reversible_store=reversible_store,
            llm=llm,
        )
        logger.debug("Compression subsystem attached")

    def _ensure_compression(self) -> CompressionEngine:
        """Lazily initialise or return the compression engine.

        Falls back to an in-memory reversible store when none has been
        attached, so that ``compress()`` never crashes even in minimal
        setups.
        """
        if self._compression_engine is not None:
            return self._compression_engine

        # Best-effort: create a temporary in-memory store under a temp
        # directory.  This is a safety net, not a recommended path.
        import tempfile
        tmp = tempfile.mkdtemp(prefix="neuromem_compress_")
        store = ReversibleStore(tmp)
        store.initialize()
        self._reversible_store = store
        self._compression_engine = CompressionEngine(reversible_store=store)
        logger.warning(
            "Compression engine auto-created with ephemeral store at {}; "
            "call attach_compression() for persistent storage",
            tmp,
        )
        return self._compression_engine

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
    # Compression API
    # ══════════════════════════════════════════════════════════════════

    def compress(
        self,
        text: str,
        *,
        importance: float | None = None,
    ) -> MemorySnapshot:
        """Compress arbitrary text into a :class:`MemorySnapshot`.

        The compression engine auto-detects the content type (logs,
        conversation, code, RAG, markdown, plain text), picks the
        optimal strategy, and stores the original for later retrieval.

        Parameters
        ----------
        text:
            Raw text to compress.  Must be a non-empty string.
        importance:
            Optional importance override in ``[0.0, 1.0]``.  When
            omitted, importance is derived from content analysis
            (e.g. log severity, entity density).

        Returns
        -------
        MemorySnapshot
            The compressed snapshot with a ``raw_reference`` for
            later decompression via :meth:`retrieve_original`.
        """
        self._require_open()
        comp = self._ensure_compression()
        return comp.compress(text, importance=importance)

    def learn_compressed(
        self,
        text: str,
        *,
        confidence: float = 0.5,
        importance: float | None = None,
        source: str = "compressed",
        namespace: str | None = None,
        tags: list[str] | None = None,
    ) -> tuple[BeliefNode, MemorySnapshot, list[ContradictionEvent | NegativeMemory]]:
        """Compress text and store the snapshot into ChromaDB + Kuzu,
        then run anomaly detection.

        This is the full cognitive-compression pipeline:

        1. **Compress** the text via :meth:`compress` to produce a
           :class:`MemorySnapshot`.
        2. **Learn** the summary as a new :class:`BeliefNode` (stored
           in both Kuzu graph and ChromaDB vector store).  If an embed
           function is available the summary is embedded for vector
           search.
        3. **Anomaly detection** — the engine's internal contradiction
           check runs during ``learn``.  This method additionally
           post-checks the learned belief against all other active
           beliefs in the namespace, recording :class:`ContradictionEvent`
           and/or :class:`NegativeMemory` entries when anomalies are
           found.
        4. All steps are recorded in a :class:`ReasoningTrace` for
           full auditability.

        Parameters
        ----------
        text:
            Raw text to compress and learn.
        confidence:
            Initial confidence for the learned belief.
        importance:
            Importance override for the snapshot.  ``None`` → auto.
        source:
            Origin label for the belief node.  Defaults to
            ``"compressed"``.
        namespace:
            Target namespace.
        tags:
            Free-form labels for the belief.

        Returns
        -------
        tuple[BeliefNode, MemorySnapshot, list[ContradictionEvent | NegativeMemory]]
            A 3-tuple of:

            - The persisted :class:`BeliefNode`.
            - The :class:`MemorySnapshot`.
            - A list of anomaly artefacts (contradiction events and/or
              negative memories detected during learning).
        """
        self._require_open()

        # ── Step 1: Compress ──────────────────────────────────────────
        snapshot = self.compress(text, importance=importance)
        ns = namespace or self._engine.namespace

        # ── Step 2: Learn the summary as a belief ────────────────────
        embedding = self._safe_embed(snapshot.summary) if self._embed_fn is not None else None

        trace = ReasoningTrace(
            namespace=ns,
            trigger="learn_compressed",
            trigger_metadata={
                "snapshot_id": snapshot.id,
                "importance": snapshot.importance,
                "compression_ratio": snapshot.compression_ratio,
            },
        )
        trace.add_step(ReasoningStep(
            step_type=TraceStepType.CUSTOM,
            description=f"Compressed raw text → snapshot {snapshot.id}",
            metadata={
                "snapshot_id": snapshot.id,
                "importance": snapshot.importance,
                "compression_ratio": snapshot.compression_ratio,
                "keywords": snapshot.keywords[:10],
            },
        ))

        belief = BeliefNode(
            claim=snapshot.summary,
            confidence=confidence,
            gamma=self._engine.config.default_gamma,
            embedding=embedding,
            source=source,
            namespace=ns,
            tags=list(tags or []) + ["compressed", f"snap:{snapshot.id}"],
            last_decay_tick=self._engine.current_tick,
        )

        # Run through the engine's anomaly-aware learn path.
        # check_contradiction will fire during the internal learn, but
        # we explicitly scan existing beliefs afterwards to capture
        # events the engine may have produced internally.
        self._engine.learn(
            claim=snapshot.summary,
            confidence=confidence,
            embedding=embedding,
            source=source,
            namespace=ns,
            tags=list(tags or []) + ["compressed", f"snap:{snapshot.id}"],
            trace=trace,
        )

        # ── Step 3: Post-learn anomaly sweep ─────────────────────────
        anomalies: list[ContradictionEvent | NegativeMemory] = []

        # Scan existing beliefs for contradictions with the new one.
        existing_beliefs = self._engine._scan_beliefs(ns)
        for existing in existing_beliefs:
            if existing.id == belief.id:
                continue
            event = self._engine.check_contradiction(
                existing, snapshot.summary, embedding, namespace=ns,
            )
            if event is not None:
                anomalies.append(event)
                trace.add_step(ReasoningStep(
                    step_type=TraceStepType.CONTRADICTION_DETECT,
                    description=(
                        f"Anomaly: snapshot contradicts belief {existing.id} "
                        f"(sim={event.similarity_score:.3f})"
                    ),
                    belief_ids=[existing.id, belief.id],
                    contradiction_ids=[event.id],
                ))

                # If the contradiction is severe enough, record a
                # negative memory so future compressions of similar
                # content surface the conflict.
                if event.conflict_severity > 0.5:
                    neg = self._engine.record_negative(
                        pattern=f"contradicts:{existing.id[:24]}",
                        context={
                            "snapshot_id": snapshot.id,
                            "conflict_severity": event.conflict_severity,
                        },
                        severity=NegativeMemorySeverity.WARNING,
                        related_belief_id=existing.id,
                        namespace=ns,
                        trace=trace,
                    )
                    anomalies.append(neg)
                    trace.add_step(ReasoningStep(
                        step_type=TraceStepType.NEGATIVE_RECORD,
                        description=f"Negative memory {neg.id} created for high-severity conflict",
                        negative_ids=[neg.id],
                        belief_ids=[existing.id],
                    ))

        self._engine._store_trace(trace)

        logger.info(
            "learn_compressed: belief={} snapshot={} anomalies={}",
            belief.id, snapshot.id, len(anomalies),
        )
        return belief, snapshot, anomalies

    def compress_history(
        self,
        messages: list[dict[str, str]] | str,
        *,
        importance: float | None = None,
    ) -> MemorySnapshot:
        """Compress a conversation history into a :class:`MemorySnapshot`.

        Accepts either a list of message dicts (``{"role": "...", "content": "..."}``)
        or a raw conversation string with role markers (``User:``, ``Assistant:``, etc.).

        Parameters
        ----------
        messages:
            Conversation messages.  Either a list of dicts with ``role``
            and ``content`` keys, or a pre-formatted string transcript.
        importance:
            Optional importance override.

        Returns
        -------
        MemorySnapshot
            The compressed snapshot.
        """
        self._require_open()

        # Normalise list-of-dicts into a single string for the router.
        if isinstance(messages, list):
            parts: list[str] = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            text = "\n".join(parts)
        else:
            text = messages

        comp = self._ensure_compression()
        return comp.compress(text, importance=importance)

    def compress_logs(
        self,
        logs: str | list[str],
        *,
        importance: float | None = None,
    ) -> MemorySnapshot:
        """Compress log entries into a :class:`MemorySnapshot`.

        Accepts either a pre-joined log string or a list of individual
        log lines.  The logs compression strategy extracts errors,
        severity, and key milestones.

        Parameters
        ----------
        logs:
            Raw log text or a list of log-line strings.
        importance:
            Optional importance override.

        Returns
        -------
        MemorySnapshot
            The compressed snapshot.
        """
        self._require_open()

        if isinstance(logs, list):
            text = "\n".join(logs)
        else:
            text = logs

        comp = self._ensure_compression()
        return comp.compress(text, importance=importance)

    def compress_rag(
        self,
        chunks: list[str],
        *,
        importance: float | None = None,
    ) -> MemorySnapshot:
        """Compress RAG retrieval chunks into a :class:`MemorySnapshot`.

        Deduplicates overlapping chunks and merges them into a single
        coherent passage.  Source citations (``[source: …]`` /
        ``[doc: …]``) are preserved.

        Parameters
        ----------
        chunks:
            List of retrieved text passages, possibly overlapping.
        importance:
            Optional importance override.

        Returns
        -------
        MemorySnapshot
            The compressed snapshot.
        """
        self._require_open()

        # Join chunks with separators so the router can still classify
        # the combined text, but the compress_rag strategy inside the
        # engine will split and deduplicate.
        text = "\n\n".join(chunks) if chunks else ""

        comp = self._ensure_compression()
        return comp.compress(text, importance=importance)

    def retrieve_original(self, memory_id: str) -> str:
        """Retrieve the original, uncompressed text for a memory snapshot.

        Parameters
        ----------
        memory_id:
            The ``raw_reference`` (snapshot ID) to look up.

        Returns
        -------
        str
            The exact original text that was compressed.

        Raises
        ------
        MemoryNotFoundError
            If no original is stored under *memory_id*.
        NeuroMemError
            If the reversible store is not available.
        """
        self._require_open()
        store = self._ensure_compression().reversible_store

        try:
            return store.retrieve_original(memory_id)
        except MemoryNotFoundError:
            raise
        except ReversibleStoreError as exc:
            raise NeuroMemError(
                f"Failed to retrieve original for memory_id={memory_id!r}: {exc}",
                context={"memory_id": memory_id},
            ) from exc

    # ══════════════════════════════════════════════════════════════════
    # Cross-agent memory sharing
    # ══════════════════════════════════════════════════════════════════

    def share_memory(
        self,
        memory_id: str,
        target_namespace: str,
        *,
        trust_factor: float | None = None,
    ) -> SharedMemoryRecord:
        """Share a compressed memory snapshot across agent boundaries.

        Copies metadata references (summary, keywords, importance) into
        the target namespace as a new BeliefNode, appends a trust score
        and decay tracker.  The original uncompressed content remains
        accessible via the same ``memory_id`` — no data duplication.

        Parameters
        ----------
        memory_id:
            The snapshot ID (or any ``raw_reference``) to share.
        target_namespace:
            Receiving namespace for the shared memory.
        trust_factor:
            Confidence multiplier in ``[0, 1]``.  Defaults to the
            engine's ``default_trust_factor``.

        Returns
        -------
        SharedMemoryRecord
            Record of the sharing operation with trust score and timestamp.

        Raises
        ------
        NeuroMemError
            If the original cannot be retrieved or the engine rejects
            the propagation.
        """
        self._require_open()

        # Resolve the original content to produce a shareable belief.
        try:
            original = self.retrieve_original(memory_id)
        except MemoryNotFoundError:
            # Fall back: the memory_id might refer to a BeliefNode, not
            # a compression snapshot.  Attempt to load it from the graph.
            source_ns = self._engine.namespace
            belief = self._engine._load_belief(memory_id, source_ns)
            if belief is None:
                raise NeuroMemError(
                    f"Cannot share memory_id={memory_id!r}: not found in "
                    "reversible store or belief graph",
                    context={"memory_id": memory_id},
                )
            # Propagate the existing belief directly.
            record = self.propagate(
                memory_id, target_namespace,
                trust_factor=trust_factor,
            )
            return SharedMemoryRecord(
                memory_id=memory_id,
                target_namespace=target_namespace,
                trust_score=record.propagated_confidence,
            )

        # Compress the original to get a summary, then learn it in the
        # target namespace with reduced confidence.
        comp = self._ensure_compression()
        snapshot = comp.compress(original, memory_id=memory_id)

        effective_trust = (
            trust_factor
            if trust_factor is not None
            else self._engine.config.default_trust_factor
        )
        shared_confidence = snapshot.importance * effective_trust

        # Learn in the target namespace.
        embedding = self._safe_embed(snapshot.summary) if self._embed_fn is not None else None
        belief = self._engine.learn(
            claim=snapshot.summary,
            confidence=shared_confidence,
            embedding=embedding,
            source=f"shared:{self._engine.namespace}",
            namespace=target_namespace,
            tags=["shared", f"from:{self._engine.namespace}", f"snap:{memory_id}"],
        )

        # Register the share for decay tracking.
        self._shared_memory_registry[memory_id] = {
            "target_namespace": target_namespace,
            "trust_score": shared_confidence,
            "belief_id": belief.id,
            "shared_at": datetime.now(),
            "decay_tick": self._engine.current_tick,
        }

        logger.info(
            "Shared memory {} → {} (trust={:.3f}, belief={})",
            memory_id, target_namespace, shared_confidence, belief.id,
        )

        return SharedMemoryRecord(
            memory_id=memory_id,
            target_namespace=target_namespace,
            trust_score=shared_confidence,
        )

    # ══════════════════════════════════════════════════════════════════
    # Unified statistics
    # ══════════════════════════════════════════════════════════════════

    def stats(self) -> dict[str, Any]:
        """Return unified statistics across all subsystems.

        The returned dict contains:

        - ``engine`` — namespace, current tick, belief count (total /
          active-only).
        - ``compression`` — tokens saved, compression ratio, stored
          memories count (from :class:`CompressionEngine`).
        - ``shared_memory`` — number of cross-namespace shares, plus
          a registry summary.
        - ``storage`` — graph node/edge counts.

        Returns
        -------
        dict
            A fresh dictionary; safe to mutate.
        """
        self._require_open()

        ns = self._engine.namespace
        beliefs = self._engine._scan_beliefs(ns)
        active_count = sum(1 for b in beliefs if b.status == BeliefStatus.ACTIVE)

        result: dict[str, Any] = {
            "engine": {
                "namespace": ns,
                "current_tick": self._engine.current_tick,
                "total_beliefs": len(beliefs),
                "active_beliefs": active_count,
            },
            "compression": {
                "tokens_saved": 0,
                "compression_ratio": 0.0,
                "stored_memories_count": 0,
            },
            "shared_memory": {
                "total_shares": len(self._shared_memory_registry),
            },
        }

        # Compression stats (best-effort).
        if self._compression_engine is not None:
            result["compression"] = self._compression_engine.get_stats()

        # Storage counts (best-effort).
        try:
            result["storage"] = {
                "graph_nodes": self._engine.graph.count_nodes(),
                "graph_edges": self._engine.graph.count_edges(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Graph count failed: {}", exc)
            result["storage"] = {"graph_nodes": -1, "graph_edges": -1}

        return result

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

    @property
    def compression_engine(self) -> CompressionEngine | None:
        """The compression engine, or ``None`` if not attached."""
        return self._compression_engine

    @property
    def reversible_store(self) -> ReversibleStore | None:
        """The reversible store, or ``None`` if not attached."""
        return self._reversible_store

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
        try:
            if self._reversible_store is not None:
                self._reversible_store.close()
        except Exception as exc:
            logger.warning("Error closing reversible store: {}", exc)
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
    "SharedMemoryRecord",
    "EmbedFn",
]
