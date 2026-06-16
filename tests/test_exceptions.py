"""Tests for the full NeuroMem exception hierarchy.

Every exception class is constructed, its structured attributes are asserted,
and both ``str()`` and ``repr()`` are verified.  This brings
``neuromem/core/exceptions.py`` to 100 % line + branch coverage.
"""

from __future__ import annotations

import pytest

from neuromem.core.exceptions import (
    BeliefConflictError,
    CollectionNotFoundError,
    ConfigurationError,
    ConfidenceDecayError,
    ContradictionError,
    CrossAgentSyncError,
    EmbeddingDimensionError,
    GraphEngineError,
    GraphQueryError,
    ModelError,
    NamespaceIsolationError,
    NeuroMemError,
    NodeNotFoundError,
    SchemaViolationError,
    StateTransitionError,
    StorageError,
    TrustThresholdError,
    UnresolvableContradictionError,
    ValidationError,
    VectorEngineError,
    VectorQueryError,
    EdgeNotFoundError,
    PropagationError,
)


# ═══════════════════════════════════════════════════════════════════════
# Root — NeuroMemError
# ═══════════════════════════════════════════════════════════════════════


class TestNeuroMemError:
    """NeuroMemError: base class with context dict."""

    def test_message_only(self) -> None:
        err = NeuroMemError("something went wrong")
        assert str(err) == "something went wrong"
        assert err.context == {}
        assert isinstance(err, Exception)

    def test_with_context(self) -> None:
        err = NeuroMemError("fail", context={"key": 42, "node": "abc"})
        s = str(err)
        assert "fail" in s
        assert "key=42" in s
        assert "node='abc'" in s

    def test_repr(self) -> None:
        err = NeuroMemError("oops", context={"x": 1})
        r = repr(err)
        assert "NeuroMemError" in r
        assert "'oops'" in r
        assert "context=" in r

    def test_empty_context_produces_no_brackets(self) -> None:
        err = NeuroMemError("plain")
        assert "[" not in str(err)


# ═══════════════════════════════════════════════════════════════════════
# Storage domain — Graph subtree
# ═══════════════════════════════════════════════════════════════════════


class TestStorageError:
    """StorageError base — carries ``backend`` in context."""

    def test_default_backend(self) -> None:
        err = StorageError("disk full")
        assert err.backend == "unknown"
        assert err.context["backend"] == "unknown"
        assert "disk full" in str(err)

    def test_custom_backend(self) -> None:
        err = StorageError("timeout", backend="kuzu")
        assert err.backend == "kuzu"

    def test_context_merged(self) -> None:
        err = StorageError("err", backend="chroma", context={"extra": True})
        assert err.context["extra"] is True
        assert err.context["backend"] == "chroma"


class TestGraphEngineError:
    """GraphEngineError — defaults backend to ``kuzu``."""

    def test_default_backend(self) -> None:
        err = GraphEngineError("write failed")
        assert err.backend == "kuzu"

    def test_custom_backend(self) -> None:
        err = GraphEngineError("fail", backend="neo4j")
        assert err.backend == "neo4j"
        assert isinstance(err, StorageError)
        assert isinstance(err, NeuroMemError)


class TestNodeNotFoundError:
    """NodeNotFoundError — structured node_id + optional label."""

    def test_with_label(self) -> None:
        err = NodeNotFoundError("n1", label="BeliefNode")
        assert err.node_id == "n1"
        assert err.label == "BeliefNode"
        s = str(err)
        assert "n1" in s
        assert "not found" in s.lower()

    def test_without_label(self) -> None:
        err = NodeNotFoundError("n2")
        assert err.label is None


class TestEdgeNotFoundError:
    """EdgeNotFoundError — src, dst, edge_type."""

    def test_construction(self) -> None:
        err = EdgeNotFoundError("src1", "dst1", "SUPPORTS")
        assert err.src_id == "src1"
        assert err.dst_id == "dst1"
        assert err.edge_type == "SUPPORTS"
        s = str(err)
        assert "src1" in s
        assert "SUPPORTS" in s
        assert "dst1" in s


class TestSchemaViolationError:
    """SchemaViolationError — operation + detail."""

    def test_construction(self) -> None:
        err = SchemaViolationError("CREATE", "unknown property 'foo'")
        assert err.operation == "CREATE"
        assert err.detail == "unknown property 'foo'"
        s = str(err)
        assert "CREATE" in s
        assert "unknown property 'foo'" in s


class TestGraphQueryError:
    """GraphQueryError — query truncation at MAX_QUERY_DISPLAY."""

    def test_short_query(self) -> None:
        q = "MATCH (n) RETURN n"
        err = GraphQueryError(q, "syntax error")
        assert err.query == q
        assert "syntax error" in str(err)

    def test_long_query_truncation(self) -> None:
        long_q = "MATCH (n) WHERE " + "x AND " * 200 + "RETURN n"
        err = GraphQueryError(long_q, "timeout")
        # The stored context query is truncated
        display = err.context["query"]
        assert display.endswith("...")
        assert len(display) <= 310  # MAX_QUERY_DISPLAY + "..."
        # But the raw query attribute is preserved
        assert err.query == long_q

    def test_max_query_display_constant(self) -> None:
        assert GraphQueryError.MAX_QUERY_DISPLAY == 300


# ═══════════════════════════════════════════════════════════════════════
# Storage domain — Vector subtree
# ═══════════════════════════════════════════════════════════════════════


class TestVectorEngineError:
    """VectorEngineError — defaults backend to ``chromadb``."""

    def test_default_backend(self) -> None:
        err = VectorEngineError("index corrupt")
        assert err.backend == "chromadb"

    def test_custom_backend(self) -> None:
        err = VectorEngineError("fail", backend="qdrant")
        assert err.backend == "qdrant"
        assert isinstance(err, StorageError)


class TestCollectionNotFoundError:
    """CollectionNotFoundError — collection_name."""

    def test_construction(self) -> None:
        err = CollectionNotFoundError("ns_test")
        assert err.collection_name == "ns_test"
        s = str(err)
        assert "ns_test" in s
        assert "not found" in s.lower()


class TestEmbeddingDimensionError:
    """EmbeddingDimensionError — expected vs actual dimensions."""

    def test_construction(self) -> None:
        err = EmbeddingDimensionError(128, 64)
        assert err.expected == 128
        assert err.actual == 64
        s = str(err)
        assert "128" in s
        assert "64" in s
        assert "mismatch" in s.lower()


class TestVectorQueryError:
    """VectorQueryError — operation + detail."""

    def test_construction(self) -> None:
        err = VectorQueryError("similarity_search", "HNSW index corrupted")
        assert err.operation == "similarity_search"
        assert err.detail == "HNSW index corrupted"
        s = str(err)
        assert "similarity_search" in s
        assert "HNSW" in s


# ═══════════════════════════════════════════════════════════════════════
# Model / cognitive domain
# ═══════════════════════════════════════════════════════════════════════


class TestModelError:
    """ModelError base — no extra attributes."""

    def test_construction(self) -> None:
        err = ModelError("invalid model state")
        assert isinstance(err, NeuroMemError)
        assert "invalid model state" in str(err)


class TestValidationError:
    """ValidationError — wraps an optional original_error."""

    def test_with_original_error(self) -> None:
        original = ValueError("field required")
        err = ValidationError("validation failed", original_error=original)
        assert err.original_error is original
        assert err.context["original_error_type"] == "ValueError"

    def test_without_original_error(self) -> None:
        err = ValidationError("generic validation issue")
        assert err.original_error is None
        assert err.context["original_error_type"] is None


class TestConfidenceDecayError:
    """ConfidenceDecayError — belief_id, current_confidence, reason."""

    def test_construction(self) -> None:
        err = ConfidenceDecayError("b_001", -0.05, "confidence went negative")
        assert err.belief_id == "b_001"
        assert err.current_confidence == -0.05
        assert err.reason == "confidence went negative"
        s = str(err)
        assert "b_001" in s
        assert "-0.05" in s


class TestStateTransitionError:
    """StateTransitionError — node_id, from_state, to_state, reason."""

    def test_construction(self) -> None:
        err = StateTransitionError("n1", "ACTIVE", "CONTRADICTED", "missing evidence")
        assert err.node_id == "n1"
        assert err.from_state == "ACTIVE"
        assert err.to_state == "CONTRADICTED"
        assert err.reason == "missing evidence"
        s = str(err)
        assert "ACTIVE" in s
        assert "CONTRADICTED" in s
        assert "missing evidence" in s


# ═══════════════════════════════════════════════════════════════════════
# Contradiction domain
# ═══════════════════════════════════════════════════════════════════════


class TestContradictionError:
    """ContradictionError base — no extra attributes."""

    def test_construction(self) -> None:
        err = ContradictionError("conflict detected")
        assert isinstance(err, NeuroMemError)
        assert "conflict" in str(err)


class TestBeliefConflictError:
    """BeliefConflictError — existing_belief_id, incoming_claim, similarity."""

    def test_construction(self) -> None:
        err = BeliefConflictError("b_old", "sky is green", 0.92)
        assert err.existing_belief_id == "b_old"
        assert err.incoming_claim == "sky is green"
        assert err.similarity_score == 0.92
        s = str(err)
        assert "b_old" in s
        assert "0.9200" in s


class TestUnresolvableContradictionError:
    """UnresolvableContradictionError — belief_ids list + reason."""

    def test_construction(self) -> None:
        err = UnresolvableContradictionError(["b1", "b2", "b3"], "confidence too close")
        assert err.belief_ids == ["b1", "b2", "b3"]
        assert err.reason == "confidence too close"
        s = str(err)
        assert "b1" in s
        assert "confidence too close" in s


# ═══════════════════════════════════════════════════════════════════════
# Propagation domain
# ═══════════════════════════════════════════════════════════════════════


class TestPropagationError:
    """PropagationError base — no extra attributes."""

    def test_construction(self) -> None:
        err = PropagationError("sync failed")
        assert isinstance(err, NeuroMemError)


class TestTrustThresholdError:
    """TrustThresholdError — source/target namespace, trust_score, minimum."""

    def test_construction(self) -> None:
        err = TrustThresholdError("agent_a", "agent_b", 0.15, 0.30)
        assert err.source_namespace == "agent_a"
        assert err.target_namespace == "agent_b"
        assert err.trust_score == 0.15
        assert err.minimum_threshold == 0.30
        s = str(err)
        assert "0.1500" in s
        assert "0.3000" in s


class TestNamespaceIsolationError:
    """NamespaceIsolationError — source/target namespace + reason."""

    def test_construction(self) -> None:
        err = NamespaceIsolationError("ns1", "ns2", "cannot write across agents")
        assert err.source_namespace == "ns1"
        assert err.target_namespace == "ns2"
        assert err.reason == "cannot write across agents"
        s = str(err)
        assert "ns1" in s
        assert "ns2" in s


class TestCrossAgentSyncError:
    """CrossAgentSyncError — involved_namespaces, reason, retry_count."""

    def test_construction(self) -> None:
        err = CrossAgentSyncError(["ns_a", "ns_b"], "network timeout", 5)
        assert err.involved_namespaces == ["ns_a", "ns_b"]
        assert err.reason == "network timeout"
        assert err.retry_count == 5
        s = str(err)
        assert "5 retries" in s
        assert "network timeout" in s


# ═══════════════════════════════════════════════════════════════════════
# Configuration domain
# ═══════════════════════════════════════════════════════════════════════


class TestConfigurationError:
    """ConfigurationError — parameter + reason."""

    def test_construction(self) -> None:
        err = ConfigurationError("trust_threshold", "must be in [0, 1]")
        assert err.parameter == "trust_threshold"
        assert err.reason == "must be in [0, 1]"
        s = str(err)
        assert "trust_threshold" in s
        assert "must be in [0, 1]" in s


# ═══════════════════════════════════════════════════════════════════════
# Hierarchy invariant — every leaf is catchable as NeuroMemError
# ═══════════════════════════════════════════════════════════════════════


class TestHierarchyInvariants:
    """Every exception class must be catchable as ``NeuroMemError``."""

    @pytest.mark.parametrize(
        "exc_cls,kwargs",
        [
            (NeuroMemError, {"message": "x"}),
            (StorageError, {"message": "x"}),
            (GraphEngineError, {"message": "x"}),
            (VectorEngineError, {"message": "x"}),
            (NodeNotFoundError, {"node_id": "n1"}),
            (EdgeNotFoundError, {"src_id": "a", "dst_id": "b", "edge_type": "T"}),
            (SchemaViolationError, {"operation": "op", "detail": "d"}),
            (GraphQueryError, {"query": "q", "detail": "d"}),
            (CollectionNotFoundError, {"collection_name": "c"}),
            (EmbeddingDimensionError, {"expected": 8, "actual": 4}),
            (VectorQueryError, {"operation": "op", "detail": "d"}),
            (ModelError, {"message": "x"}),
            (ValidationError, {"message": "x"}),
            (ConfidenceDecayError, {"belief_id": "b", "current_confidence": 0.0, "reason": "r"}),
            (StateTransitionError, {"node_id": "n", "from_state": "a", "to_state": "b", "reason": "r"}),
            (ContradictionError, {"message": "x"}),
            (BeliefConflictError, {"existing_belief_id": "b", "incoming_claim": "c", "similarity_score": 0.5}),
            (UnresolvableContradictionError, {"belief_ids": ["b1"], "reason": "r"}),
            (PropagationError, {"message": "x"}),
            (TrustThresholdError, {"source_namespace": "a", "target_namespace": "b", "trust_score": 0.1, "minimum_threshold": 0.5}),
            (NamespaceIsolationError, {"source_namespace": "a", "target_namespace": "b", "reason": "r"}),
            (CrossAgentSyncError, {"involved_namespaces": ["a"], "reason": "r", "retry_count": 1}),
            (ConfigurationError, {"parameter": "p", "reason": "r"}),
        ],
    )
    def test_catchable_as_neuromem_error(self, exc_cls: type[NeuroMemError], kwargs: dict) -> None:
        err = exc_cls(**kwargs)
        assert isinstance(err, NeuroMemError)

    @pytest.mark.parametrize(
        "exc_cls,kwargs",
        [
            (StorageError, {"message": "x"}),
            (GraphEngineError, {"message": "x"}),
            (VectorEngineError, {"message": "x"}),
            (NodeNotFoundError, {"node_id": "n"}),
            (EdgeNotFoundError, {"src_id": "a", "dst_id": "b", "edge_type": "T"}),
            (SchemaViolationError, {"operation": "o", "detail": "d"}),
            (GraphQueryError, {"query": "q", "detail": "d"}),
            (CollectionNotFoundError, {"collection_name": "c"}),
            (EmbeddingDimensionError, {"expected": 8, "actual": 4}),
            (VectorQueryError, {"operation": "o", "detail": "d"}),
            (ModelError, {"message": "x"}),
            (ValidationError, {"message": "x"}),
            (ConfidenceDecayError, {"belief_id": "b", "current_confidence": 0.0, "reason": "r"}),
            (StateTransitionError, {"node_id": "n", "from_state": "a", "to_state": "b", "reason": "r"}),
            (ContradictionError, {"message": "x"}),
            (BeliefConflictError, {"existing_belief_id": "b", "incoming_claim": "c", "similarity_score": 0.5}),
            (UnresolvableContradictionError, {"belief_ids": [], "reason": "r"}),
            (PropagationError, {"message": "x"}),
            (TrustThresholdError, {"source_namespace": "a", "target_namespace": "b", "trust_score": 0.1, "minimum_threshold": 0.5}),
            (NamespaceIsolationError, {"source_namespace": "a", "target_namespace": "b", "reason": "r"}),
            (CrossAgentSyncError, {"involved_namespaces": ["a"], "reason": "r", "retry_count": 1}),
            (ConfigurationError, {"parameter": "p", "reason": "r"}),
        ],
    )
    def test_has_str_and_repr(self, exc_cls: type[NeuroMemError], kwargs: dict) -> None:
        err = exc_cls(**kwargs)
        s = str(err)
        r = repr(err)
        assert isinstance(s, str) and len(s) > 0
        assert isinstance(r, str) and len(r) > 0
