"""Custom exception hierarchy for NeuroMem.

Design rationale
----------------
Every exception derives from :class:`NeuroMemError` so callers can always
`except NeuroMemError` for blanket handling while retaining the ability to
target specific failure domains (storage, schema, confidence, contradiction,
propagation, etc.).

Each leaf exception carries structured context attributes so that upstream
consumers (loguru sinks, Sentry integrations, CLI formatters) can render
rich diagnostics without parsing the message string.

Hierarchy map::

    NeuroMemError
    ├── StorageError
    │   ├── GraphEngineError
    │   │   ├── NodeNotFoundError
    │   │   ├── EdgeNotFoundError
    │   │   ├── SchemaViolationError
    │   │   └── GraphQueryError
    │   └── VectorEngineError
    │       ├── CollectionNotFoundError
    │       ├── EmbeddingDimensionError
    │       └── VectorQueryError
    ├── ModelError
    │   ├── ValidationError          (Pydantic re-export convenience)
    │   ├── ConfidenceDecayError
    │   └── StateTransitionError
    ├── ContradictionError
    │   ├── BeliefConflictError
    │   └── UnresolvableContradictionError
    ├── PropagationError
    │   ├── TrustThresholdError
    │   ├── NamespaceIsolationError
    │   └── CrossAgentSyncError
    └── ConfigurationError
"""

from __future__ import annotations

from typing import Any


# ═══════════════════════════════════════════════════════════════════════
# Root
# ═══════════════════════════════════════════════════════════════════════

class NeuroMemError(Exception):
    """Base exception for all NeuroMem failures.

    Parameters
    ----------
    message:
        Human-readable description of the failure.
    context:
        Optional free-form dictionary carrying structured diagnostics
        (e.g. ``{"node_id": "...", "operation": "..."}``).
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        self.context: dict[str, Any] = context or {}
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            details = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base}  [{details}]"
        return base

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__str__()!r}, context={self.context!r})"


# ═══════════════════════════════════════════════════════════════════════
# Storage domain
# ═══════════════════════════════════════════════════════════════════════

class StorageError(NeuroMemError):
    """Any failure originating from a storage backend (graph or vector).

    Parameters
    ----------
    message:
        Description of the storage failure.
    backend:
        Identifier of the offending backend (e.g. ``"kuzu"``, ``"chromadb"``).
    context:
        Additional structured context.
    """

    def __init__(
        self,
        message: str,
        *,
        backend: str = "unknown",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.backend: str = backend
        merged = {"backend": backend, **(context or {})}
        super().__init__(message, context=merged)


# ── Graph sub-tree ────────────────────────────────────────────────────

class GraphEngineError(StorageError):
    """Base for graph-engine-specific errors."""

    def __init__(
        self,
        message: str,
        *,
        backend: str = "kuzu",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, backend=backend, context=context)


class NodeNotFoundError(GraphEngineError):
    """A requested node does not exist in the graph.

    Parameters
    ----------
    node_id:
        The primary-key identifier that could not be resolved.
    label:
        Optional node label/type that was queried.
    """

    def __init__(self, node_id: str, *, label: str | None = None) -> None:
        self.node_id: str = node_id
        self.label: str | None = label
        super().__init__(
            f"Node {node_id!r} not found",
            context={"node_id": node_id, "label": label},
        )


class EdgeNotFoundError(GraphEngineError):
    """A requested edge does not exist in the graph.

    Parameters
    ----------
    src_id:
        Source node identifier.
    dst_id:
        Destination node identifier.
    edge_type:
        Relationship type label.
    """

    def __init__(self, src_id: str, dst_id: str, edge_type: str) -> None:
        self.src_id: str = src_id
        self.dst_id: str = dst_id
        self.edge_type: str = edge_type
        super().__init__(
            f"Edge ({src_id!r})-[{edge_type!r}]->({dst_id!r}) not found",
            context={"src_id": src_id, "dst_id": dst_id, "edge_type": edge_type},
        )


class SchemaViolationError(GraphEngineError):
    """An operation violated the graph schema (wrong label, missing property, etc.).

    Parameters
    ----------
    operation:
        The Cypher/DML operation that triggered the violation.
    detail:
        Underlying engine error message.
    """

    def __init__(self, operation: str, detail: str) -> None:
        self.operation: str = operation
        self.detail: str = detail
        super().__init__(
            f"Schema violation during '{operation}': {detail}",
            context={"operation": operation, "detail": detail},
        )


class GraphQueryError(GraphEngineError):
    """A Cypher query failed during execution.

    Parameters
    ----------
    query:
        The offending Cypher string (truncated in ``__str__`` for safety).
    detail:
        Engine-level error text.
    """

    MAX_QUERY_DISPLAY = 300  # characters shown in string representation

    def __init__(self, query: str, detail: str) -> None:
        self.query: str = query
        self.detail: str = detail
        display_query = (
            query[: self.MAX_QUERY_DISPLAY] + "..." if len(query) > self.MAX_QUERY_DISPLAY else query
        )
        super().__init__(
            f"Graph query failed: {detail}",
            context={"query": display_query},
        )


# ── Vector sub-tree ───────────────────────────────────────────────────

class VectorEngineError(StorageError):
    """Base for vector-engine-specific errors."""

    def __init__(
        self,
        message: str,
        *,
        backend: str = "chromadb",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, backend=backend, context=context)


class CollectionNotFoundError(VectorEngineError):
    """The requested ChromaDB collection does not exist.

    Parameters
    ----------
    collection_name:
        Name of the missing collection.
    """

    def __init__(self, collection_name: str) -> None:
        self.collection_name: str = collection_name
        super().__init__(
            f"Collection {collection_name!r} not found",
            context={"collection_name": collection_name},
        )


class EmbeddingDimensionError(VectorEngineError):
    """The embedding vector has an unexpected dimensionality.

    Parameters
    ----------
    expected:
        Expected dimension count.
    actual:
        Actual dimension count received.
    """

    def __init__(self, expected: int, actual: int) -> None:
        self.expected: int = expected
        self.actual: int = actual
        super().__init__(
            f"Embedding dimension mismatch: expected {expected}, got {actual}",
            context={"expected": expected, "actual": actual},
        )


class VectorQueryError(VectorEngineError):
    """A similarity search or vector operation failed.

    Parameters
    ----------
    operation:
        Human-readable label (e.g. ``"similarity_search"``).
    detail:
        Underlying engine error text.
    """

    def __init__(self, operation: str, detail: str) -> None:
        self.operation: str = operation
        self.detail: str = detail
        super().__init__(
            f"Vector operation '{operation}' failed: {detail}",
            context={"operation": operation, "detail": detail},
        )


# ═══════════════════════════════════════════════════════════════════════
# Model / cognitive domain
# ═══════════════════════════════════════════════════════════════════════

class ModelError(NeuroMemError):
    """Base for errors arising from cognitive model validation or logic."""


class ValidationError(ModelError):
    """Convenience wrapper re-exporting Pydantic validation failures.

    We capture the original ``pydantic.ValidationError`` so that callers
    who need the full ``error_count`` / ``errors()`` list can access it.

    Parameters
    ----------
    message:
        Short summary of what failed to validate.
    original_error:
        The ``pydantic.ValidationError`` instance.
    """

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        self.original_error: Exception | None = original_error
        super().__init__(
            message,
            context={"original_error_type": type(original_error).__name__ if original_error else None},
        )


class ConfidenceDecayError(ModelError):
    """An operation on a belief's confidence fell outside acceptable bounds.

    Parameters
    ----------
    belief_id:
        Identifier of the affected belief.
    current_confidence:
        The confidence value that triggered the error.
    reason:
        Why the value is invalid.
    """

    def __init__(self, belief_id: str, current_confidence: float, reason: str) -> None:
        self.belief_id: str = belief_id
        self.current_confidence: float = current_confidence
        self.reason: str = reason
        super().__init__(
            f"Confidence decay error on belief {belief_id!r}: {reason} "
            f"(confidence={current_confidence})",
            context={
                "belief_id": belief_id,
                "current_confidence": current_confidence,
                "reason": reason,
            },
        )


class StateTransitionError(ModelError):
    """A belief or memory node attempted an illegal state transition.

    Parameters
    ----------
    node_id:
        Affected node identifier.
    from_state:
        Current state label.
    to_state:
        Target state label.
    reason:
        Why the transition is disallowed.
    """

    def __init__(self, node_id: str, from_state: str, to_state: str, reason: str) -> None:
        self.node_id: str = node_id
        self.from_state: str = from_state
        self.to_state: str = to_state
        self.reason: str = reason
        super().__init__(
            f"Invalid state transition on {node_id!r}: {from_state!r} -> {to_state!r} — {reason}",
            context={
                "node_id": node_id,
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason,
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# Contradiction domain
# ═══════════════════════════════════════════════════════════════════════

class ContradictionError(NeuroMemError):
    """Base for all contradiction-detection errors."""


class BeliefConflictError(ContradictionError):
    """An incoming observation conflicts with one or more existing beliefs.

    This is a *resolvable* contradiction — the engine can split or deprecate.

    Parameters
    ----------
    existing_belief_id:
        The belief that is being challenged.
    incoming_claim:
        The new claim that triggered the conflict.
    similarity_score:
        Numeric similarity between the existing belief and the incoming
        claim (higher = more contradictory).
    """

    def __init__(
        self,
        existing_belief_id: str,
        incoming_claim: str,
        similarity_score: float,
    ) -> None:
        self.existing_belief_id: str = existing_belief_id
        self.incoming_claim: str = incoming_claim
        self.similarity_score: float = similarity_score
        super().__init__(
            f"Belief conflict: {existing_belief_id!r} contradicts incoming claim "
            f"(similarity={similarity_score:.4f})",
            context={
                "existing_belief_id": existing_belief_id,
                "incoming_claim": incoming_claim,
                "similarity_score": similarity_score,
            },
        )


class UnresolvableContradictionError(ContradictionError):
    """The engine could not reconcile a contradiction after all strategies.

    This typically means the confidence values of conflicting beliefs are
    too close to automatically deprecate one.

    Parameters
    ----------
    belief_ids:
        The set of belief identifiers that are irreconcilable.
    reason:
        Description of why automatic resolution failed.
    """

    def __init__(self, belief_ids: list[str], reason: str) -> None:
        self.belief_ids: list[str] = belief_ids
        self.reason: str = reason
        super().__init__(
            f"Unresolvable contradiction among beliefs {belief_ids}: {reason}",
            context={"belief_ids": belief_ids, "reason": reason},
        )


# ═══════════════════════════════════════════════════════════════════════
# Propagation / multi-agent domain
# ═══════════════════════════════════════════════════════════════════════

class PropagationError(NeuroMemError):
    """Base for errors in cross-agent knowledge propagation."""


class TrustThresholdError(PropagationError):
    """A propagation was rejected because the trust score fell below the
    configured minimum.

    Parameters
    ----------
    source_namespace:
        Namespace of the sending agent.
    target_namespace:
        Namespace of the receiving agent.
    trust_score:
        The computed trust value (0.0 – 1.0).
    minimum_threshold:
        Configured floor for propagation.
    """

    def __init__(
        self,
        source_namespace: str,
        target_namespace: str,
        trust_score: float,
        minimum_threshold: float,
    ) -> None:
        self.source_namespace: str = source_namespace
        self.target_namespace: str = target_namespace
        self.trust_score: float = trust_score
        self.minimum_threshold: float = minimum_threshold
        super().__init__(
            f"Trust threshold not met for propagation from {source_namespace!r} "
            f"to {target_namespace!r}: trust={trust_score:.4f} < min={minimum_threshold:.4f}",
            context={
                "source_namespace": source_namespace,
                "target_namespace": target_namespace,
                "trust_score": trust_score,
                "minimum_threshold": minimum_threshold,
            },
        )


class NamespaceIsolationError(PropagationError):
    """An operation attempted to violate namespace isolation boundaries.

    Parameters
    ----------
    source_namespace:
        Namespace making the request.
    target_namespace:
        Namespace that was targeted.
    reason:
        Why the operation was blocked.
    """

    def __init__(self, source_namespace: str, target_namespace: str, reason: str) -> None:
        self.source_namespace: str = source_namespace
        self.target_namespace: str = target_namespace
        self.reason: str = reason
        super().__init__(
            f"Namespace isolation violation: {source_namespace!r} -> {target_namespace!r} — {reason}",
            context={
                "source_namespace": source_namespace,
                "target_namespace": target_namespace,
                "reason": reason,
            },
        )


class CrossAgentSyncError(PropagationError):
    """Synchronisation of beliefs across agents failed after all retries.

    Parameters
    ----------
    involved_namespaces:
        Namespaces that were being synchronised.
    reason:
        Underlying cause.
    retry_count:
        Number of retry attempts made before giving up.
    """

    def __init__(
        self,
        involved_namespaces: list[str],
        reason: str,
        retry_count: int,
    ) -> None:
        self.involved_namespaces: list[str] = involved_namespaces
        self.reason: str = reason
        self.retry_count: int = retry_count
        super().__init__(
            f"Cross-agent sync failed after {retry_count} retries among "
            f"{involved_namespaces}: {reason}",
            context={
                "involved_namespaces": involved_namespaces,
                "reason": reason,
                "retry_count": retry_count,
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# Configuration domain
# ═══════════════════════════════════════════════════════════════════════

class ConfigurationError(NeuroMemError):
    """The framework was initialised with invalid or missing configuration.

    Parameters
    ----------
    parameter:
        The configuration key that is problematic.
    reason:
        What is wrong and how to fix it.
    """

    def __init__(self, parameter: str, reason: str) -> None:
        self.parameter: str = parameter
        self.reason: str = reason
        super().__init__(
            f"Configuration error on parameter {parameter!r}: {reason}",
            context={"parameter": parameter, "reason": reason},
        )


# ═══════════════════════════════════════════════════════════════════════
# Public re-export helper
# ═══════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    # Root
    "NeuroMemError",
    # Storage — Graph
    "StorageError",
    "GraphEngineError",
    "NodeNotFoundError",
    "EdgeNotFoundError",
    "SchemaViolationError",
    "GraphQueryError",
    # Storage — Vector
    "VectorEngineError",
    "CollectionNotFoundError",
    "EmbeddingDimensionError",
    "VectorQueryError",
    # Model / cognitive
    "ModelError",
    "ValidationError",
    "ConfidenceDecayError",
    "StateTransitionError",
    # Contradiction
    "ContradictionError",
    "BeliefConflictError",
    "UnresolvableContradictionError",
    # Propagation
    "PropagationError",
    "TrustThresholdError",
    "NamespaceIsolationError",
    "CrossAgentSyncError",
    # Config
    "ConfigurationError",
]
