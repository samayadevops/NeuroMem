"""The NeuroMem cognitive engine — the brain that fuses graph + vector storage
with cognitive state logic.

The :class:`NeuroMemEngine` orchestrates five responsibilities:

1. **Confidence decay** — applies temporal gamma decay to beliefs based on
   a logical tick counter.
2. **Contradiction detection & resolution** — intercepts incoming claims
   that clash with existing beliefs and forces a state split or deprecation.
3. **Negative memory guardrails** — logs failed paths to prevent infinite
   LLM loops.
4. **Reasoning traces** — produces auditable step-by-step records of every
   cognitive decision.
5. **Query fusion** — merges graph (structural) and vector (semantic)
   results into a unified, confidence-weighted answer.
6. **Propagation** — shares beliefs across namespaces with a controlled
   trust-reduction factor.

The engine is storage-agnostic — it accepts any :class:`BaseGraphEngine`
and :class:`BaseVectorEngine` implementations.  This makes it fully
testable with in-memory fakes and swappable for production backends.
"""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from loguru import logger

from neuromem.core.exceptions import (
    BeliefConflictError,
    ConfidenceDecayError,
    ContradictionError,
    NamespaceIsolationError,
    NeuroMemError,
    NodeNotFoundError,
    PropagationError,
    TrustThresholdError,
    UnresolvableContradictionError,
)
from neuromem.core.models import (
    BeliefNode,
    BeliefStatus,
    ContradictionEvent,
    ContradictionResolution,
    NegativeMemory,
    NegativeMemoryPatternType,
    NegativeMemorySeverity,
    PropagationRecord,
    PropagationStatus,
    ReasoningStep,
    ReasoningTrace,
    TraceStepType,
)
from neuromem.storage.base import (
    BaseGraphEngine,
    BaseVectorEngine,
    VectorRecord,
)

# ═══════════════════════════════════════════════════════════════════════
# Constants & schema definitions
# ═══════════════════════════════════════════════════════════════════════

# Graph node / edge labels used throughout the engine.
BELIEF_LABEL = "BeliefNode"
NEGATIVE_LABEL = "NegativeMemory"
TRACE_LABEL = "ReasoningTrace"
CONTRADICTION_LABEL = "ContradictionEvent"
PROPAGATION_LABEL = "PropagationRecord"

CONTRADICTS_EDGE = "CONTRADICTS"
SUPPORTS_EDGE = "SUPPORTS"
PROPAGATED_FROM_EDGE = "PROPAGATED_FROM"
TRIGGERED_BY_EDGE = "TRIGGERED_BY"
OCCURRED_IN_EDGE = "OCCURRED_IN"

# Default cognitive thresholds.
DEFAULT_CONTRADICTION_THRESHOLD = 0.65  # similarity above this = contradiction
DEFAULT_DECAY_FLOOR = 0.05              # beliefs below this are marked DECAYED
DEFAULT_TRUST_THRESHOLD = 0.3           # propagation refused below this trust
DEFAULT_MAX_PROPAGATION_RETRIES = 3
DEFAULT_FUSION_VECTOR_WEIGHT = 0.5      # 0 = graph only, 1 = vector only


def _utcnow() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _generate_id(prefix: str) -> str:
    """Generate a deterministic-format ID with a semantic prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _matches_negative_pattern(candidate: str, neg: "NegativeMemory") -> bool:
    """Return ``True`` if *candidate* matches the guardrail *neg*.

    Three strategies are supported, controlled by ``neg.pattern_type``:

    * ``exact``  — full-string equality.
    * ``regex``  — ``re.search(neg.pattern, candidate)``.  An invalid
                   regex never raises; it simply returns ``False``.
    * ``fuzzy``  — Jaccard token-overlap ratio.  Blocks when the ratio
                   meets or exceeds ``neg.fuzzy_threshold``.
    """
    import re as _re  # noqa: PLC0415 — lazy to avoid circular at module level

    from neuromem.core.models import NegativeMemoryPatternType  # noqa: PLC0415

    pt = neg.pattern_type
    if pt == NegativeMemoryPatternType.REGEX:
        try:
            return bool(_re.search(neg.pattern, candidate))
        except _re.error:
            return False

    if pt == NegativeMemoryPatternType.FUZZY:
        a = set(candidate.lower().split())
        b = set(neg.pattern.lower().split())
        if not a and not b:
            return True
        union = a | b
        if not union:
            return False
        ratio = len(a & b) / len(union)
        return ratio >= neg.fuzzy_threshold

    # Default: EXACT
    return candidate == neg.pattern


# ═══════════════════════════════════════════════════════════════════════
# Engine configuration
# ═══════════════════════════════════════════════════════════════════════

class EngineConfig:
    """Tunable cognitive parameters for :class:`NeuroMemEngine`.

    All values are mutable at runtime so the engine can be re-tuned
    without reconstruction.
    """

    def __init__(
        self,
        *,
        contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
        decay_floor: float = DEFAULT_DECAY_FLOOR,
        trust_threshold: float = DEFAULT_TRUST_THRESHOLD,
        max_propagation_retries: int = DEFAULT_MAX_PROPAGATION_RETRIES,
        fusion_vector_weight: float = DEFAULT_FUSION_VECTOR_WEIGHT,
        default_gamma: float = 0.99,
        default_trust_factor: float = 0.8,
        reinforce_on_duplicate: bool = True,
    ) -> None:
        self.contradiction_threshold: float = contradiction_threshold
        self.decay_floor: float = decay_floor
        self.trust_threshold: float = trust_threshold
        self.max_propagation_retries: int = max_propagation_retries
        self.fusion_vector_weight: float = fusion_vector_weight
        self.default_gamma: float = default_gamma
        self.default_trust_factor: float = default_trust_factor
        self.reinforce_on_duplicate: bool = reinforce_on_duplicate

    def validate(self) -> None:
        """Validate all config values are within legal ranges."""
        if not 0.0 <= self.contradiction_threshold <= 1.0:
            raise ValueError("contradiction_threshold must be in [0.0, 1.0]")
        if not 0.0 <= self.decay_floor <= 1.0:
            raise ValueError("decay_floor must be in [0.0, 1.0]")
        if not 0.0 <= self.trust_threshold <= 1.0:
            raise ValueError("trust_threshold must be in [0.0, 1.0]")
        if self.max_propagation_retries < 0:
            raise ValueError("max_propagation_retries must be >= 0")
        if not 0.0 <= self.fusion_vector_weight <= 1.0:
            raise ValueError("fusion_vector_weight must be in [0.0, 1.0]")
        if not 0.0 <= self.default_gamma <= 1.0:
            raise ValueError("default_gamma must be in [0.0, 1.0]")
        if not 0.0 <= self.default_trust_factor <= 1.0:
            raise ValueError("default_trust_factor must be in [0.0, 1.0]")


# ═══════════════════════════════════════════════════════════════════════
# Fusion result container
# ═══════════════════════════════════════════════════════════════════════

class FusedResult:
    """A unified result from graph + vector query fusion.

    Combines a belief (from the graph) with its vector-context
    similarity, producing a single confidence-weighted score.
    """

    __slots__ = ("belief", "vector_distance", "graph_confidence", "fused_score")

    def __init__(
        self,
        belief: BeliefNode,
        vector_distance: float | None,
        graph_confidence: float,
        fused_score: float,
    ) -> None:
        self.belief: BeliefNode = belief
        self.vector_distance: float | None = vector_distance
        self.graph_confidence: float = graph_confidence
        self.fused_score: float = fused_score

    def __repr__(self) -> str:
        return (
            f"FusedResult(belief={self.belief.id!r}, "
            f"fused={self.fused_score:.4f}, "
            f"graph={self.graph_confidence:.4f}, "
            f"vec_dist={self.vector_distance})"
        )


# ═══════════════════════════════════════════════════════════════════════
# NeuroMemEngine
# ═══════════════════════════════════════════════════════════════════════

class NeuroMemEngine:
    """The cognitive brain that orchestrates storage + cognitive logic.

    Parameters
    ----------
    graph_engine:
        Any :class:`BaseGraphEngine` implementation (e.g. KuzuGraphEngine).
    vector_engine:
        Any :class:`BaseVectorEngine` implementation (e.g. ChromaVectorEngine).
    config:
        Optional :class:`EngineConfig` for tunable parameters.  If
        ``None``, defaults are used.
    namespace:
        The default namespace for this engine instance.  Used when
        operations do not specify an explicit namespace.
    auto_bootstrap:
        If ``True`` (default), automatically register graph schemas
        (node labels, edge types) on first use.
    """

    def __init__(
        self,
        graph_engine: BaseGraphEngine,
        vector_engine: BaseVectorEngine,
        *,
        config: EngineConfig | None = None,
        namespace: str = "default",
        auto_bootstrap: bool = True,
    ) -> None:
        self.graph: BaseGraphEngine = graph_engine
        self.vector: BaseVectorEngine = vector_engine
        self.config: EngineConfig = config or EngineConfig()
        self.config.validate()
        self.namespace: str = namespace
        self.auto_bootstrap: bool = auto_bootstrap
        self._current_tick: int = 0
        self._bootstrapped: bool = False

        if self.auto_bootstrap:
            self._bootstrap_schemas()

    # ── Tick management ───────────────────────────────────────────────

    @property
    def current_tick(self) -> int:
        """Return the current logical tick counter."""
        return self._current_tick

    def advance_tick(self, steps: int = 1) -> int:
        """Advance the logical tick counter and return the new value."""
        if steps < 0:
            raise ValueError("steps must be >= 0")
        self._current_tick += steps
        return self._current_tick

    # ── Schema bootstrap ──────────────────────────────────────────────

    def _bootstrap_schemas(self) -> None:
        """Register all graph node labels and edge types."""
        if self._bootstrapped:
            return
        logger.info("Bootstrapping NeuroMem graph schemas")

        try:
            # Node labels
            self.graph.create_node_label(
                BELIEF_LABEL,
                {
                    "claim": "STRING",
                    "confidence": "DOUBLE",
                    "gamma": "DOUBLE",
                    "evidence_count": "INT64",
                    "source": "STRING",
                    "status": "STRING",
                    "namespace": "STRING",
                    "last_decay_tick": "INT64",
                    "tags": "STRING",
                },
                primary_key="id",
            )
            self.graph.create_node_label(
                NEGATIVE_LABEL,
                {
                    "pattern": "STRING",
                    "severity": "STRING",
                    "block_threshold": "INT64",
                    "occurrence_count": "INT64",
                    "namespace": "STRING",
                    "pattern_type": "STRING",
                    "fuzzy_threshold": "DOUBLE",
                },
                primary_key="id",
            )
            self.graph.create_node_label(
                TRACE_LABEL,
                {
                    "trigger": "STRING",
                    "namespace": "STRING",
                },
                primary_key="id",
            )
            self.graph.create_node_label(
                CONTRADICTION_LABEL,
                {
                    "belief_id": "STRING",
                    "incoming_claim": "STRING",
                    "resolution": "STRING",
                    "namespace": "STRING",
                },
                primary_key="id",
            )
            self.graph.create_node_label(
                PROPAGATION_LABEL,
                {
                    "source_namespace": "STRING",
                    "target_namespace": "STRING",
                    "belief_id": "STRING",
                    "status": "STRING",
                },
                primary_key="id",
            )

            # Edge types
            self.graph.create_edge_type(
                CONTRADICTS_EDGE, BELIEF_LABEL, BELIEF_LABEL,
                {"similarity": "DOUBLE", "severity": "DOUBLE"},
            )
            self.graph.create_edge_type(
                SUPPORTS_EDGE, BELIEF_LABEL, BELIEF_LABEL,
                {"weight": "DOUBLE"},
            )
            self.graph.create_edge_type(
                PROPAGATED_FROM_EDGE, BELIEF_LABEL, PROPAGATION_LABEL,
                {},
            )
            self.graph.create_edge_type(
                TRIGGERED_BY_EDGE, CONTRADICTION_LABEL, BELIEF_LABEL,
                {},
            )
            self.graph.create_edge_type(
                OCCURRED_IN_EDGE, NEGATIVE_LABEL, TRACE_LABEL,
                {},
            )

            self._bootstrapped = True
            logger.info("Graph schemas bootstrapped successfully")

        except NeuroMemError as exc:
            logger.error("Schema bootstrap failed: {}", exc)
            raise

    # ── Belief management ─────────────────────────────────────────────

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
        trace: ReasoningTrace | None = None,
    ) -> BeliefNode:
        """Learn a new belief, detecting contradictions on the way in.

        This is the primary write path.  The engine:

        1. Searches the vector store for semantically similar existing
           beliefs.
        2. If a high-similarity match is found, fires a contradiction
           check.
        3. If no contradiction (or resolved), persists the new belief to
           both the graph and vector stores.
        4. Records every step in the reasoning trace.

        Parameters
        ----------
        claim:
            The semantic content of the belief.
        confidence:
            Initial confidence in ``[0.0, 1.0]``.
        embedding:
            Optional dense vector for similarity search.  Required for
            contradiction detection.
        source:
            Origin of the belief.
        namespace:
            Target namespace.  Defaults to the engine's namespace.
        tags:
            Free-form labels.
        gamma:
            Temporal decay rate.  Defaults to ``config.default_gamma``.
        trace:
            Optional reasoning trace to append steps to.  A new trace is
            created if ``None``.

        Returns
        -------
        BeliefNode
            The persisted belief.
        """
        ns = namespace or self.namespace
        effective_gamma = gamma if gamma is not None else self.config.default_gamma
        owns_trace = trace is None
        if owns_trace:
            trace = ReasoningTrace(
                namespace=ns,
                trigger="learn",
                trigger_metadata={"claim": claim[:200]},
            )

        # Step 1: Check for existing beliefs via the graph
        existing = self._find_existing_belief(claim, ns, trace)

        # Step 2: Check vector similarity for contradiction detection
        contradiction_event: ContradictionEvent | None = None
        if embedding is not None and existing is not None:
            contradiction_event = self._detect_contradiction(
                existing, claim, embedding, ns, trace,
            )

        # Step 2b: Reinforce existing belief if no contradiction was found
        if (
            self.config.reinforce_on_duplicate
            and existing is not None
            and contradiction_event is None
        ):
            reinforce_amount = max(0.01, confidence * 0.1)
            new_conf = existing.reinforce(amount=reinforce_amount)
            self._persist_belief(existing, trace, is_update=True)
            if trace is not None:
                trace.add_step(ReasoningStep(
                    step_type=TraceStepType.BELIEF_UPDATE,
                    description=(
                        f"Reinforced existing belief {existing.id} "
                        f"(evidence_count={existing.evidence_count}, "
                        f"confidence={new_conf:.3f})"
                    ),
                    belief_ids=[existing.id],
                    confidence_after=new_conf,
                ))
            if owns_trace:
                self._store_trace(trace)
            logger.info(
                "Reinforced belief {} (confidence={:.3f}, evidence_count={})",
                existing.id, new_conf, existing.evidence_count,
            )
            return existing

        # Step 3: Create the new belief
        belief = BeliefNode(
            claim=claim,
            confidence=confidence,
            gamma=effective_gamma,
            embedding=embedding,
            source=source,
            namespace=ns,
            tags=tags or [],
            last_decay_tick=self._current_tick,
        )

        self._persist_belief(belief, trace)

        if owns_trace:
            self._store_trace(trace)

        logger.info(
            "Learned belief {} (confidence={:.3f}, contradictions={})",
            belief.id, belief.confidence, 1 if contradiction_event else 0,
        )
        return belief

    def recall(
        self,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        *,
        n_results: int = 10,
        namespace: str | None = None,
        min_confidence: float = 0.0,
        apply_decay: bool = True,
        trace: ReasoningTrace | None = None,
    ) -> list[FusedResult]:
        """Recall beliefs by semantic similarity, fusing graph + vector.

        The recall pipeline:

        1. If ``query_embedding`` is provided, run a vector similarity
           search to find semantically close records.
        2. Map each vector hit back to its graph :class:`BeliefNode`.
        3. Apply temporal decay to each belief (if ``apply_decay``).
        4. Fuse graph confidence with vector similarity using
           ``config.fusion_vector_weight``.
        5. Filter by ``min_confidence`` and return sorted results.

        At least one of ``query`` or ``query_embedding`` must be provided.
        """
        if query is None and query_embedding is None:
            raise ValueError("recall() requires at least one of query or query_embedding")

        ns = namespace or self.namespace
        owns_trace = trace is None
        if owns_trace:
            trace = ReasoningTrace(
                namespace=ns,
                trigger="recall",
                trigger_metadata={"query": (query or "")[:200]},
            )

        # Vector search
        vector_hits: list[VectorRecord] = []
        if query_embedding is not None:
            t0 = time.perf_counter()
            try:
                vector_hits = self.vector.similarity_search(
                    self._vector_collection(ns),
                    query_embedding,
                    n_results=n_results,
                )
                elapsed = (time.perf_counter() - t0) * 1000.0
                trace.add_step(ReasoningStep(
                    step_type=TraceStepType.VECTOR_SEARCH,
                    description=f"Vector search returned {len(vector_hits)} hits",
                    vector_ids=[r.id for r in vector_hits],
                    metadata={"n_results": n_results},
                    duration_ms=round(elapsed, 2),
                ))
            except NeuroMemError as exc:
                logger.warning("Vector search failed during recall: {}", exc)

        # Map vector hits → beliefs and fuse
        fused: list[FusedResult] = []
        belief_ids_seen: set[str] = set()

        for hit in vector_hits:
            belief_id = hit.metadata.get("belief_id") or hit.id
            if belief_id in belief_ids_seen:
                continue
            belief_ids_seen.add(belief_id)

            belief = self._load_belief(belief_id, ns)
            if belief is None:
                continue

            if apply_decay:
                self._apply_decay_to_belief(belief, trace)

            graph_conf = belief.effective_confidence(self._current_tick)
            vec_sim = self._distance_to_similarity(hit.distance)

            fused_score = self._fuse_scores(graph_conf, vec_sim)

            if fused_score >= min_confidence:
                fused.append(FusedResult(
                    belief=belief,
                    vector_distance=hit.distance,
                    graph_confidence=graph_conf,
                    fused_score=fused_score,
                ))

        # If no embedding, fall back to scanning all beliefs in namespace
        if query_embedding is None and query is not None:
            all_beliefs = self._scan_beliefs(ns, trace)
            for belief in all_beliefs:
                if belief.id in belief_ids_seen:
                    continue
                if apply_decay:
                    self._apply_decay_to_belief(belief, trace)
                graph_conf = belief.effective_confidence(self._current_tick)
                if graph_conf >= min_confidence:
                    fused.append(FusedResult(
                        belief=belief,
                        vector_distance=None,
                        graph_confidence=graph_conf,
                        fused_score=graph_conf,
                    ))

        # Sort by fused score descending
        fused.sort(key=lambda r: r.fused_score, reverse=True)

        # Record fusion step
        trace.add_step(ReasoningStep(
            step_type=TraceStepType.FUSION,
            description=f"Fused {len(fused)} results (vector_weight={self.config.fusion_vector_weight})",
            belief_ids=[r.belief.id for r in fused[:n_results]],
            metadata={"returned_count": len(fused)},
        ))

        if owns_trace:
            self._store_trace(trace)

        logger.info("Recall returned {} fused results", len(fused))
        return fused[:n_results]

    # ── Confidence decay ──────────────────────────────────────────────

    def apply_global_decay(self, namespace: str | None = None) -> int:
        """Apply temporal decay to all active beliefs.

        Returns the number of beliefs that transitioned to DECAYED status.
        """
        ns = namespace or self.namespace
        trace = ReasoningTrace(
            namespace=ns,
            trigger="decay",
            trigger_metadata={"tick": self._current_tick},
        )

        beliefs = self._scan_beliefs(ns, trace)
        decayed_count = 0

        for belief in beliefs:
            if belief.status not in (BeliefStatus.ACTIVE,):
                continue

            conf_before = belief.confidence
            belief.apply_decay(self._current_tick)

            if belief.confidence <= self.config.decay_floor:
                belief.status = BeliefStatus.DECAYED
                decayed_count += 1
                logger.debug(
                    "Belief {} decayed below floor ({:.4f} <= {:.4f})",
                    belief.id, belief.confidence, self.config.decay_floor,
                )

            self._persist_belief(belief, trace, is_update=True)

            trace.add_step(ReasoningStep(
                step_type=TraceStepType.BELIEF_DECAY,
                description=f"Decayed belief {belief.id}",
                belief_ids=[belief.id],
                confidence_before=conf_before,
                confidence_after=belief.confidence,
            ))

        self._store_trace(trace)
        logger.info(
            "Global decay applied: {} beliefs decayed below floor",
            decayed_count,
        )
        return decayed_count

    # ── Contradiction detection ───────────────────────────────────────

    def check_contradiction(
        self,
        existing_belief: BeliefNode,
        incoming_claim: str,
        incoming_embedding: list[float] | None = None,
        *,
        namespace: str | None = None,
    ) -> ContradictionEvent | None:
        """Check if *incoming_claim* contradicts *existing_belief*.

        Returns a :class:`ContradictionEvent` if a contradiction is
        detected, or ``None`` if the claims are compatible.

        The contradiction is detected when:

        - The incoming claim is semantically similar to the existing
          belief (vector cosine similarity above
          ``config.contradiction_threshold``), **AND**
        - The two claims represent opposing assertions.

        Without embeddings, only a coarse textual check is performed.
        """
        ns = namespace or existing_belief.namespace
        similarity = 0.0

        # Compute similarity
        if incoming_embedding is not None and existing_belief.embedding is not None:
            similarity = self._cosine_similarity(
                existing_belief.embedding, incoming_embedding,
            )
        else:
            # Fallback: token overlap heuristic
            similarity = self._text_overlap(existing_belief.claim, incoming_claim)

        if similarity < self.config.contradiction_threshold:
            return None

        # We have a potential contradiction — resolve it
        conf_before = existing_belief.confidence
        resolution, conf_after, reasoning = self._resolve_contradiction(
            existing_belief, incoming_claim, similarity,
        )

        # Compute severity: how much does the incoming claim undermine?
        severity = similarity * (1.0 - conf_before)

        event = ContradictionEvent(
            namespace=ns,
            belief_id=existing_belief.id,
            incoming_claim=incoming_claim,
            similarity_score=round(similarity, 6),
            conflict_severity=round(severity, 6),
            confidence_before=conf_before,
            confidence_after=conf_after,
            resolution=resolution,
            reasoning=reasoning,
        )

        # Persist the contradiction event to the graph
        self._persist_contradiction(event, existing_belief)

        # Apply the resolution
        if resolution == ContradictionResolution.DEPRECATE_OLD:
            existing_belief.deprecate(reason="contradicted")
            self._persist_belief(existing_belief, None, is_update=True)
        elif resolution == ContradictionResolution.SPLIT:
            existing_belief.confidence = conf_after
            existing_belief.touch()
            self._persist_belief(existing_belief, None, is_update=True)

        logger.info(
            "Contradiction detected: belief {} vs '{}' "
            "(sim={:.3f}, resolution={})",
            existing_belief.id, incoming_claim[:50],
            similarity, resolution.value,
        )
        return event

    # ── Negative memory ───────────────────────────────────────────────

    def record_negative(
        self,
        pattern: str,
        *,
        context: dict[str, Any] | None = None,
        severity: NegativeMemorySeverity = NegativeMemorySeverity.WARNING,
        block_threshold: int = 1,
        related_belief_id: str | None = None,
        namespace: str | None = None,
        trace: ReasoningTrace | None = None,
        pattern_type: NegativeMemoryPatternType = NegativeMemoryPatternType.EXACT,
        fuzzy_threshold: float = 0.8,
    ) -> NegativeMemory:
        """Record a negative memory guardrail.

        If a negative memory with the same ``pattern`` already exists in
        the namespace, its occurrence count is incremented instead of
        creating a duplicate.
        """
        ns = namespace or self.namespace
        owns_trace = trace is None
        if owns_trace:
            trace = ReasoningTrace(
                namespace=ns,
                trigger="negative_record",
                trigger_metadata={"pattern": pattern[:200]},
            )

        # Check for existing negative memory with the same pattern
        existing_neg = self._find_negative_memory(pattern, ns)
        if existing_neg is not None:
            existing_neg.record_occurrence()
            self._persist_negative(existing_neg, trace, is_update=True)
            trace.add_step(ReasoningStep(
                step_type=TraceStepType.NEGATIVE_RECORD,
                description=f"Incremented negative memory {existing_neg.id} (count={existing_neg.occurrence_count})",
                negative_ids=[existing_neg.id],
            ))
            if owns_trace:
                self._store_trace(trace)
            return existing_neg

        neg = NegativeMemory(
            namespace=ns,
            pattern=pattern,
            context=context or {},
            severity=severity,
            block_threshold=block_threshold,
            related_belief_id=related_belief_id,
            pattern_type=pattern_type,
            fuzzy_threshold=fuzzy_threshold,
        )
        self._persist_negative(neg, trace)
        trace.add_step(ReasoningStep(
            step_type=TraceStepType.NEGATIVE_RECORD,
            description=f"Recorded negative memory {neg.id}",
            negative_ids=[neg.id],
        ))

        if owns_trace:
            self._store_trace(trace)

        logger.info(
            "Recorded negative memory {} (severity={}, block_threshold={}, pattern_type={})",
            neg.id, severity.value, block_threshold, pattern_type.value,
        )
        return neg

    def is_blocked(self, pattern: str, namespace: str | None = None) -> bool:
        """Check if a pattern is blocked by a negative memory guardrail.

        Iterates all negative memories in the namespace and applies the
        appropriate matching strategy (``exact``, ``regex``, or ``fuzzy``)
        for each one.  Returns ``True`` as soon as a match is found whose
        ``should_block`` flag is set.
        """
        ns = namespace or self.namespace
        negatives = self._scan_negatives(ns)
        for neg in negatives:
            if _matches_negative_pattern(pattern, neg) and neg.should_block:
                return True
        return False

    # ── Propagation ───────────────────────────────────────────────────

    def propagate(
        self,
        belief_id: str,
        target_namespace: str,
        *,
        trust_factor: float | None = None,
        namespace: str | None = None,
        trace: ReasoningTrace | None = None,
    ) -> PropagationRecord:
        """Propagate a belief to another namespace with trust reduction.

        The belief's confidence is multiplied by ``trust_factor`` before
        being delivered to the target namespace.  If the resulting
        confidence falls below ``config.trust_threshold``, the
        propagation is rejected.
        """
        ns = namespace or self.namespace
        effective_trust = trust_factor if trust_factor is not None else self.config.default_trust_factor

        if ns == target_namespace:
            raise NamespaceIsolationError(
                ns, target_namespace,
                "Cannot propagate within the same namespace.",
            )

        owns_trace = trace is None
        if owns_trace:
            trace = ReasoningTrace(
                namespace=ns,
                trigger="propagation",
                trigger_metadata={
                    "belief_id": belief_id,
                    "target_namespace": target_namespace,
                },
            )

        # Load the source belief
        belief = self._load_belief(belief_id, ns)
        if belief is None:
            raise NodeNotFoundError(belief_id, label=BELIEF_LABEL)

        propagated_conf = belief.confidence * effective_trust

        record = PropagationRecord(
            namespace=ns,
            source_namespace=ns,
            target_namespace=target_namespace,
            belief_id=belief_id,
            belief_claim=belief.claim,
            original_confidence=belief.confidence,
            trust_factor=effective_trust,
            attempted_at=_utcnow(),
        )

        # Check trust threshold
        if propagated_conf < self.config.trust_threshold:
            record.mark_rejected(
                reason=f"propagated_confidence {propagated_conf:.4f} < "
                       f"threshold {self.config.trust_threshold:.4f}",
            )
            self._persist_propagation(record, trace)
            trace.add_step(ReasoningStep(
                step_type=TraceStepType.PROPAGATION,
                description=f"Propagation rejected (trust too low: {propagated_conf:.4f})",
                belief_ids=[belief_id],
            ))
            if owns_trace:
                self._store_trace(trace)
            raise TrustThresholdError(
                ns, target_namespace, propagated_conf, self.config.trust_threshold,
            )

        # Deliver: create a belief in the target namespace
        target_belief = BeliefNode(
            namespace=target_namespace,
            claim=belief.claim,
            confidence=propagated_conf,
            gamma=belief.gamma,
            embedding=belief.embedding,
            source=f"propagated:{ns}",
            tags=list(belief.tags) + ["propagated"],
            last_decay_tick=self._current_tick,
        )
        self._persist_belief(target_belief, trace)

        record.mark_delivered()
        self._persist_propagation(record, trace)

        trace.add_step(ReasoningStep(
            step_type=TraceStepType.PROPAGATION,
            description=(
                f"Propagated belief {belief_id} to {target_namespace} "
                f"(conf {belief.confidence:.3f} -> {propagated_conf:.3f})"
            ),
            belief_ids=[belief_id, target_belief.id],
        ))

        if owns_trace:
            self._store_trace(trace)

        logger.info(
            "Propagated belief {} -> {} (trust={:.3f}, conf={:.3f}->{:.3f})",
            belief_id, target_namespace, effective_trust,
            belief.confidence, propagated_conf,
        )
        return record

    # ── Internal: belief persistence ──────────────────────────────────

    def _persist_belief(
        self,
        belief: BeliefNode,
        trace: ReasoningTrace | None,
        *,
        is_update: bool = False,
    ) -> None:
        """Persist a belief to both graph and vector stores."""
        # Graph
        props: dict[str, Any] = {
            "claim": belief.claim,
            "confidence": belief.confidence,
            "gamma": belief.gamma,
            "evidence_count": belief.evidence_count,
            "source": belief.source,
            "status": belief.status.value if isinstance(belief.status, BeliefStatus) else str(belief.status),
            "namespace": belief.namespace,
            "last_decay_tick": belief.last_decay_tick,
        }
        # Merge tags as a pipe-delimited string (Kuzu has no native LIST)
        props["tags"] = "|".join(belief.tags) if belief.tags else ""

        self.graph.add_node(BELIEF_LABEL, belief.id, props)

        # Vector
        if belief.embedding is not None:
            collection = self._vector_collection(belief.namespace)
            if not self.vector.collection_exists(collection):
                self.vector.create_collection(
                    collection,
                    dimension=len(belief.embedding),
                    distance_metric="cosine",
                )
            self.vector.upsert(
                collection,
                ids=[belief.id],
                embeddings=[belief.embedding],
                documents=[belief.claim],
                metadatas=[{
                    "belief_id": belief.id,
                    "namespace": belief.namespace,
                    "confidence": belief.confidence,
                    "status": props["status"],
                }],
            )

        if trace is not None:
            step_type = TraceStepType.BELIEF_UPDATE if is_update else TraceStepType.BELIEF_CREATE
            trace.add_step(ReasoningStep(
                step_type=step_type,
                description=f"{'Updated' if is_update else 'Created'} belief {belief.id}",
                belief_ids=[belief.id],
                confidence_after=belief.confidence,
            ))

    def _load_belief(self, belief_id: str, namespace: str) -> BeliefNode | None:
        """Load a belief from the graph by ID."""
        node = self.graph.get_node(BELIEF_LABEL, belief_id)
        if node is None:
            return None
        return self._node_to_belief(node, namespace)

    def _scan_beliefs(
        self,
        namespace: str,
        trace: ReasoningTrace | None = None,
    ) -> list[BeliefNode]:
        """Scan all beliefs in a namespace via Cypher."""
        beliefs: list[BeliefNode] = []
        try:
            result = self.graph.execute_query(
                "MATCH (n:BeliefNode) WHERE n.namespace = $ns RETURN n",
                {"ns": namespace},
            )
            for record in result.records:
                node_dict = record.get("n", record) if isinstance(record, dict) else record
                belief = self._node_dict_to_belief(node_dict, namespace)
                if belief is not None:
                    beliefs.append(belief)
        except NeuroMemError as exc:
            logger.warning("Belief scan failed for namespace {}: {}", namespace, exc)

        if trace is not None:
            trace.add_step(ReasoningStep(
                step_type=TraceStepType.BELIEF_QUERY,
                description=f"Scanned {len(beliefs)} beliefs in namespace {namespace}",
                belief_ids=[b.id for b in beliefs],
            ))
        return beliefs

    def _find_existing_belief(
        self,
        claim: str,
        namespace: str,
        trace: ReasoningTrace | None,
    ) -> BeliefNode | None:
        """Find an existing belief with the same claim in the namespace."""
        try:
            result = self.graph.execute_query(
                "MATCH (n:BeliefNode {claim: $claim, namespace: $ns}) RETURN n",
                {"claim": claim, "ns": namespace},
            )
            if result.records:
                node_dict = result.records[0].get("n", result.records[0])
                return self._node_dict_to_belief(node_dict, namespace)
        except NeuroMemError as exc:
            logger.debug("Find existing belief failed: {}", exc)

        if trace is not None:
            trace.add_step(ReasoningStep(
                step_type=TraceStepType.BELIEF_QUERY,
                description=f"Checked for existing belief with claim '{claim[:50]}'",
            ))
        return None

    def _node_to_belief(self, node: Any, namespace: str) -> BeliefNode | None:
        """Convert a graph NodeRecord into a BeliefNode."""
        props = dict(node.properties)
        return self._props_to_belief(props, node.node_id, namespace)

    def _node_dict_to_belief(
        self,
        node_dict: dict[str, Any],
        namespace: str,
    ) -> BeliefNode | None:
        """Convert a raw node dict (from Cypher RETURN n) into a BeliefNode."""
        # Strip internal keys
        clean = {k: v for k, v in node_dict.items() if not k.startswith("_")}
        return self._props_to_belief(clean, clean.get("id", ""), namespace)

    def _props_to_belief(
        self,
        props: dict[str, Any],
        belief_id: str,
        namespace: str,
    ) -> BeliefNode | None:
        """Build a BeliefNode from a property dict."""
        if not belief_id:
            return None
        try:
            status_str = props.get("status", "active")
            try:
                status = BeliefStatus(status_str)
            except ValueError:
                status = BeliefStatus.ACTIVE

            tags_str = props.get("tags", "")
            tags = tags_str.split("|") if tags_str else []

            return BeliefNode(
                id=belief_id,
                namespace=props.get("namespace", namespace),
                claim=props.get("claim", ""),
                confidence=float(props.get("confidence", 0.5)),
                gamma=float(props.get("gamma", self.config.default_gamma)),
                evidence_count=int(props.get("evidence_count", 1)),
                source=props.get("source", "unknown"),
                status=status,
                tags=tags,
                last_decay_tick=int(props.get("last_decay_tick", 0)),
            )
        except Exception as exc:
            logger.debug("Failed to convert props to BeliefNode: {}", exc)
            return None

    # ── Internal: contradiction logic ─────────────────────────────────

    def _detect_contradiction(
        self,
        existing: BeliefNode,
        incoming_claim: str,
        incoming_embedding: list[float],
        namespace: str,
        trace: ReasoningTrace | None,
    ) -> ContradictionEvent | None:
        """Run contradiction detection and record in trace."""
        event = self.check_contradiction(
            existing, incoming_claim, incoming_embedding, namespace=namespace,
        )
        if event is not None and trace is not None:
            trace.add_step(ReasoningStep(
                step_type=TraceStepType.CONTRADICTION_DETECT,
                description=(
                    f"Detected contradiction: belief {existing.id} vs "
                    f"'{incoming_claim[:50]}' (sim={event.similarity_score:.3f})"
                ),
                belief_ids=[existing.id],
                contradiction_ids=[event.id],
                confidence_before=event.confidence_before,
                confidence_after=event.confidence_after,
            ))
        return event

    def _resolve_contradiction(
        self,
        existing: BeliefNode,
        incoming_claim: str,
        similarity: float,
    ) -> tuple[ContradictionResolution, float, str]:
        """Decide how to resolve a contradiction.

        Returns ``(resolution, new_confidence, reasoning)``.

        Strategy:
        - If the existing belief has very high confidence (>0.8) and the
          incoming claim is not overwhelmingly similar (<0.9), keep the
          existing belief (DEPRECATE_NEW).
        - If the existing belief has low confidence (<0.4), accept the
          new claim (DEPRECATE_OLD).
        - Otherwise, split into parallel branches (SPLIT) with reduced
          confidence on both.
        """
        conf = existing.confidence

        if conf > 0.8 and similarity < 0.9:
            return (
                ContradictionResolution.DEPRECATE_NEW,
                conf,
                f"Existing belief confidence {conf:.3f} is high; "
                f"incoming claim rejected.",
            )

        if conf < 0.4:
            new_conf = max(0.0, conf - 0.2)
            return (
                ContradictionResolution.DEPRECATE_OLD,
                new_conf,
                f"Existing belief confidence {conf:.3f} is low; "
                f"deprecating in favour of incoming claim.",
            )

        # Split: reduce both by a factor proportional to similarity
        reduction = similarity * 0.3
        new_conf = max(0.0, conf - reduction)
        return (
            ContradictionResolution.SPLIT,
            new_conf,
            f"Conflicting claims split into parallel branches; "
            f"existing confidence reduced from {conf:.3f} to {new_conf:.3f}.",
        )

    def _persist_contradiction(
        self,
        event: ContradictionEvent,
        existing_belief: BeliefNode,
    ) -> None:
        """Persist a contradiction event to the graph."""
        props = {
            "belief_id": event.belief_id,
            "incoming_claim": event.incoming_claim,
            "resolution": event.resolution.value,
            "namespace": event.namespace,
        }
        self.graph.add_node(CONTRADICTION_LABEL, event.id, props)

        # Create CONTRADICTS edge from the existing belief to itself
        # (representing the internal conflict)
        try:
            self.graph.add_edge(
                BELIEF_LABEL, existing_belief.id,
                BELIEF_LABEL, existing_belief.id,
                CONTRADICTS_EDGE,
                {"similarity": event.similarity_score, "severity": event.conflict_severity},
            )
        except NeuroMemError as exc:
            logger.debug("Contradiction edge creation skipped: {}", exc)

        # Link contradiction event to the belief
        try:
            self.graph.add_edge(
                CONTRADICTION_LABEL, event.id,
                BELIEF_LABEL, existing_belief.id,
                TRIGGERED_BY_EDGE,
            )
        except NeuroMemError as exc:
            logger.debug("Triggered-by edge skipped: {}", exc)

    # ── Internal: negative memory persistence ─────────────────────────

    def _persist_negative(
        self,
        neg: NegativeMemory,
        trace: ReasoningTrace | None,
        *,
        is_update: bool = False,
    ) -> None:
        """Persist a negative memory to the graph."""
        props = {
            "pattern": neg.pattern,
            "severity": neg.severity.value,
            "block_threshold": neg.block_threshold,
            "occurrence_count": neg.occurrence_count,
            "namespace": neg.namespace,
            "pattern_type": neg.pattern_type.value if hasattr(neg.pattern_type, "value") else str(neg.pattern_type),
            "fuzzy_threshold": neg.fuzzy_threshold,
        }
        self.graph.add_node(NEGATIVE_LABEL, neg.id, props)

    def _scan_negatives(self, namespace: str) -> list[NegativeMemory]:
        """Load all NegativeMemory nodes for a namespace."""
        results: list[NegativeMemory] = []
        try:
            result = self.graph.execute_query(
                "MATCH (n:NegativeMemory) WHERE n.namespace = $ns RETURN n",
                {"ns": namespace},
            )
            for record in result.records:
                node_dict = record.get("n", record) if isinstance(record, dict) else record
                clean = {k: v for k, v in node_dict.items() if not k.startswith("_")}
                neg_id = clean.get("id", "")
                if not neg_id:
                    continue
                try:
                    sev_str = clean.get("severity", "warning")
                    try:
                        severity = NegativeMemorySeverity(sev_str)
                    except ValueError:
                        severity = NegativeMemorySeverity.WARNING
                    pt_str = clean.get("pattern_type", "exact")
                    try:
                        pt = NegativeMemoryPatternType(pt_str)
                    except ValueError:
                        pt = NegativeMemoryPatternType.EXACT
                    results.append(NegativeMemory(
                        id=neg_id,
                        namespace=clean.get("namespace", namespace),
                        pattern=clean.get("pattern", ""),
                        severity=severity,
                        block_threshold=int(clean.get("block_threshold", 1)),
                        occurrence_count=int(clean.get("occurrence_count", 1)),
                        pattern_type=pt,
                        fuzzy_threshold=float(clean.get("fuzzy_threshold", 0.8)),
                    ))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to reconstruct NegativeMemory from scan: {}", exc)
        except NeuroMemError as exc:
            logger.debug("Scan negatives failed for namespace {}: {}", namespace, exc)
        return results

    def _find_negative_memory(
        self,
        pattern: str,
        namespace: str,
    ) -> NegativeMemory | None:
        """Find an existing negative memory by exact pattern + namespace."""
        try:
            result = self.graph.execute_query(
                "MATCH (n:NegativeMemory {pattern: $pattern, namespace: $ns}) RETURN n",
                {"pattern": pattern, "ns": namespace},
            )
            if result.records:
                node_dict = result.records[0].get("n", result.records[0])
                clean = {k: v for k, v in node_dict.items() if not k.startswith("_")}
                neg_id = clean.get("id", "")
                if not neg_id:
                    return None
                try:
                    sev_str = clean.get("severity", "warning")
                    try:
                        severity = NegativeMemorySeverity(sev_str)
                    except ValueError:
                        severity = NegativeMemorySeverity.WARNING
                    pt_str = clean.get("pattern_type", "exact")
                    try:
                        pt = NegativeMemoryPatternType(pt_str)
                    except ValueError:
                        pt = NegativeMemoryPatternType.EXACT
                    return NegativeMemory(
                        id=neg_id,
                        namespace=clean.get("namespace", namespace),
                        pattern=clean.get("pattern", pattern),
                        severity=severity,
                        block_threshold=int(clean.get("block_threshold", 1)),
                        occurrence_count=int(clean.get("occurrence_count", 1)),
                        pattern_type=pt,
                        fuzzy_threshold=float(clean.get("fuzzy_threshold", 0.8)),
                    )
                except Exception as exc:
                    logger.debug("Failed to reconstruct NegativeMemory: {}", exc)
                    return None
        except NeuroMemError as exc:
            logger.debug("Find negative memory failed: {}", exc)
        return None

    # ── Internal: propagation persistence ─────────────────────────────

    def _persist_propagation(
        self,
        record: PropagationRecord,
        trace: ReasoningTrace | None,
    ) -> None:
        """Persist a propagation record to the graph."""
        props = {
            "source_namespace": record.source_namespace,
            "target_namespace": record.target_namespace,
            "belief_id": record.belief_id,
            "status": record.status.value,
        }
        self.graph.add_node(PROPAGATION_LABEL, record.id, props)

    # ── Internal: trace persistence ───────────────────────────────────

    def _store_trace(self, trace: ReasoningTrace) -> None:
        """Persist a reasoning trace to the graph."""
        try:
            props = {
                "trigger": trace.trigger,
                "namespace": trace.namespace,
            }
            self.graph.add_node(TRACE_LABEL, trace.id, props)
            logger.debug(
                "Stored trace {} ({} steps, {:.1f}ms total)",
                trace.id, trace.step_count, trace.total_duration_ms,
            )
        except NeuroMemError as exc:
            logger.warning("Failed to store trace {}: {}", trace.id, exc)

    # ── Internal: decay helpers ───────────────────────────────────────

    def _apply_decay_to_belief(
        self,
        belief: BeliefNode,
        trace: ReasoningTrace | None,
    ) -> None:
        """Apply temporal decay to a single belief and persist."""
        conf_before = belief.confidence
        belief.apply_decay(self._current_tick)

        if belief.confidence <= self.config.decay_floor and belief.status == BeliefStatus.ACTIVE:
            belief.status = BeliefStatus.DECAYED

        self._persist_belief(belief, trace, is_update=True)

        if trace is not None:
            trace.add_step(ReasoningStep(
                step_type=TraceStepType.BELIEF_DECAY,
                description=f"Decayed belief {belief.id}",
                belief_ids=[belief.id],
                confidence_before=conf_before,
                confidence_after=belief.confidence,
            ))

    # ── Internal: math helpers ────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns a value in ``[0.0, 1.0]`` (clamped; negatives treated as 0).
        """
        if not a or not b or len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        sim = dot / (norm_a * norm_b)
        return max(0.0, min(1.0, sim))

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        """Compute a coarse token-overlap similarity between two strings.

        Returns a value in ``[0.0, 1.0]``.
        """
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _distance_to_similarity(distance: float | None) -> float:
        """Convert a ChromaDB distance to a similarity score in ``[0, 1]``.

        ChromaDB cosine distance = ``1 - cosine_similarity``, so we invert it.
        For L2 we apply ``1 / (1 + d)``.
        """
        if distance is None:
            return 0.0
        if distance < 0:
            return 1.0
        # Cosine distance is in [0, 2]; similarity = 1 - distance
        sim = 1.0 - distance
        return max(0.0, min(1.0, sim))

    def _fuse_scores(self, graph_confidence: float, vector_similarity: float) -> float:
        """Fuse graph confidence with vector similarity.

        Uses a weighted average controlled by
        ``config.fusion_vector_weight``::

            fused = (1 - w) * graph_confidence + w * vector_similarity

        Returns a value clamped to ``[0.0, 1.0]``.
        """
        w = self.config.fusion_vector_weight
        fused = (1.0 - w) * graph_confidence + w * vector_similarity
        return max(0.0, min(1.0, fused))

    # ── Internal: namespace helpers ───────────────────────────────────

    @staticmethod
    def _vector_collection(namespace: str) -> str:
        """Map a namespace to its ChromaDB collection name."""
        # Sanitise: lowercase, replace non-alphanumeric with underscore
        safe = "".join(c.lower() if c.isalnum() else "_" for c in namespace)
        return f"ns_{safe}"


# ═══════════════════════════════════════════════════════════════════════
# Public re-exports
# ═══════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    "NeuroMemEngine",
    "EngineConfig",
    "FusedResult",
]
