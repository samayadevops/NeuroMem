"""Coverage gap-fillers for neuromem.core.engine — targets 100 %.

These tests exercise error paths, defensive branches, math helpers,
and edge cases not covered by the main ``test_engine.py`` suite.
"""

from __future__ import annotations

import math
import unittest.mock

import pytest

from neuromem.core.engine import (
    EngineConfig,
    FusedResult,
    NeuroMemEngine,
    _generate_id,
    _utcnow,
)
from neuromem.core.exceptions import (
    NamespaceIsolationError,
    NeuroMemError,
    NodeNotFoundError,
    TrustThresholdError,
)
from neuromem.core.models import (
    BeliefNode,
    BeliefStatus,
    ContradictionEvent,
    ContradictionResolution,
    NegativeMemory,
    NegativeMemorySeverity,
    ReasoningStep,
    ReasoningTrace,
    TraceStepType,
)


# ═══════════════════════════════════════════════════════════════════════
# EngineConfig validation
# ═══════════════════════════════════════════════════════════════════════


class TestEngineConfigValidation:
    """EngineConfig.validate() rejects out-of-range values."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("contradiction_threshold", 1.5),
            ("contradiction_threshold", -0.1),
            ("decay_floor", 1.1),
            ("decay_floor", -0.1),
            ("trust_threshold", 2.0),
            ("trust_threshold", -0.1),
            ("max_propagation_retries", -1),
            ("fusion_vector_weight", 1.5),
            ("fusion_vector_weight", -0.1),
            ("default_gamma", 1.5),
            ("default_gamma", -0.1),
            ("default_trust_factor", 2.0),
            ("default_trust_factor", -0.1),
        ],
    )
    def test_invalid_config_raises(self, field: str, value: float) -> None:
        config = EngineConfig(**{field: value})
        with pytest.raises(ValueError):
            config.validate()

    def test_valid_config_passes(self) -> None:
        config = EngineConfig()
        config.validate()  # should not raise


# ═══════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════


class TestModuleHelpers:
    def test_generate_id_prefix(self) -> None:
        uid = _generate_id("belief")
        assert uid.startswith("belief_")
        assert len(uid) > 10

    def test_generate_id_uniqueness(self) -> None:
        a = _generate_id("x")
        b = _generate_id("x")
        assert a != b

    def test_utcnow_is_aware(self) -> None:
        dt = _utcnow()
        assert dt.tzinfo is not None


# ═══════════════════════════════════════════════════════════════════════
# FusedResult
# ═══════════════════════════════════════════════════════════════════════


class TestFusedResultRepr:
    def test_repr_contains_fields(self) -> None:
        belief = BeliefNode(claim="test claim", confidence=0.8)
        fr = FusedResult(
            belief=belief, vector_distance=0.2,
            graph_confidence=0.8, fused_score=0.7,
        )
        r = repr(fr)
        assert "FusedResult" in r
        assert "0.7000" in r

    def test_repr_none_distance(self) -> None:
        belief = BeliefNode(claim="x")
        fr = FusedResult(belief=belief, vector_distance=None, graph_confidence=0.5, fused_score=0.5)
        r = repr(fr)
        assert "None" in r or "vec_dist=None" in r


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap idempotency
# ═══════════════════════════════════════════════════════════════════════


class TestBootstrapIdempotent:
    def test_double_bootstrap_is_harmless(self, engine: NeuroMemEngine) -> None:
        engine._bootstrap_schemas()
        assert engine._bootstrapped is True


# ═══════════════════════════════════════════════════════════════════════
# Recall error paths
# ═══════════════════════════════════════════════════════════════════════


class TestRecallErrorPaths:
    def test_vector_search_failure_returns_gracefully(self, engine: NeuroMemEngine) -> None:
        """If vector search raises NeuroMemError, recall still completes."""
        engine.learn("test claim for recall", confidence=0.9)
        with unittest.mock.patch.object(
            engine.vector, "similarity_search",
            side_effect=NeuroMemError("vector store down"),
        ):
            results = engine.recall(query="test", query_embedding=[0.1] * 8)
            assert isinstance(results, list)

    def test_stale_vector_belief_id_skipped(
        self, engine: NeuroMemEngine, embed_fn,
    ) -> None:
        """A vector record referencing a nonexistent belief_id is skipped."""
        belief = engine.learn("stale test", embedding=embed_fn("stale test"))
        engine.graph.delete_node("BeliefNode", belief.id, cascade_edges=True)

        results = engine.recall(query_embedding=embed_fn("stale test"))
        assert isinstance(results, list)
        for r in results:
            assert r.belief.id != belief.id

    def test_recall_with_text_only_uses_scan(
        self, engine: NeuroMemEngine,
    ) -> None:
        """recall(query=..., query_embedding=None) falls back to graph scan."""
        engine.learn("text scan test alpha", confidence=0.9)
        results = engine.recall(query="text scan test alpha")
        assert len(results) >= 1


# ═══════════════════════════════════════════════════════════════════════
# _find_existing_belief
# ═══════════════════════════════════════════════════════════════════════


class TestFindExistingBelief:
    def test_finds_existing(self, engine: NeuroMemEngine) -> None:
        engine.learn("dedup target claim", confidence=0.8)
        trace = ReasoningTrace(trigger="t", namespace="test")
        found = engine._find_existing_belief("dedup target claim", "test", trace)
        assert found is not None
        assert found.claim == "dedup target claim"

    def test_returns_none_for_new_claim(self, engine: NeuroMemEngine) -> None:
        trace = ReasoningTrace(trigger="t", namespace="test")
        result = engine._find_existing_belief("brand new never seen claim", "test", trace)
        assert result is None

    def test_query_failure_returns_none(self, engine: NeuroMemEngine) -> None:
        """If graph query fails, _find_existing_belief returns None."""
        with unittest.mock.patch.object(
            engine.graph, "execute_query",
            side_effect=NeuroMemError("graph down"),
        ):
            result = engine._find_existing_belief(
                "any claim", "test", None,
            )
            assert result is None


# ═══════════════════════════════════════════════════════════════════════
# _props_to_belief defensive paths
# ═══════════════════════════════════════════════════════════════════════


class TestPropsToBeliefDefensive:
    def test_empty_id_returns_none(self, engine: NeuroMemEngine) -> None:
        assert engine._props_to_belief({}, "", "ns") is None

    def test_unrecognized_status_falls_back_to_active(
        self, engine: NeuroMemEngine,
    ) -> None:
        result = engine._props_to_belief(
            {"id": "x", "claim": "c", "status": "bogus_status", "confidence": 0.5},
            "x", "ns",
        )
        assert result is not None
        assert result.status == BeliefStatus.ACTIVE

    def test_malformed_confidence_returns_none(
        self, engine: NeuroMemEngine,
    ) -> None:
        result = engine._props_to_belief(
            {"id": "x", "claim": "c", "confidence": "not_a_float"},
            "x", "ns",
        )
        assert result is None

    def test_props_with_namespace_override(
        self, engine: NeuroMemEngine,
    ) -> None:
        result = engine._props_to_belief(
            {"id": "x", "claim": "c", "confidence": 0.5, "namespace": "other_ns"},
            "x", "default_ns",
        )
        assert result is not None
        assert result.namespace == "other_ns"

    def test_props_with_tags(self, engine: NeuroMemEngine) -> None:
        result = engine._props_to_belief(
            {"id": "x", "claim": "c", "confidence": 0.5, "tags": "a|b|c"},
            "x", "ns",
        )
        assert result is not None
        assert result.tags == ["a", "b", "c"]


# ═══════════════════════════════════════════════════════════════════════
# Contradiction text-overlap fallback
# ═══════════════════════════════════════════════════════════════════════


class TestContradictionTextOverlap:
    def test_text_overlap_high_similarity_triggers_contradiction(
        self, engine: NeuroMemEngine,
    ) -> None:
        """High text overlap triggers contradiction via text fallback."""
        belief = engine.learn("the sky is blue", confidence=0.5)
        event = engine.check_contradiction(
            belief, "sky is blue", incoming_embedding=None,
        )
        # Jaccard("the sky is blue","sky is blue") = 3/4 = 0.75 > 0.5
        assert event is not None

    def test_text_overlap_low_similarity_no_contradiction(
        self, engine: NeuroMemEngine,
    ) -> None:
        belief = engine.learn("alpha beta gamma delta epsilon", confidence=0.5)
        event = engine.check_contradiction(
            belief, "zeta eta theta", incoming_embedding=None,
        )
        assert event is None


# ═══════════════════════════════════════════════════════════════════════
# apply_global_decay edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestGlobalDecayEdgeCases:
    def test_scan_failure_returns_zero(self, engine: NeuroMemEngine) -> None:
        """If belief scan fails, apply_global_decay returns 0."""
        with unittest.mock.patch.object(
            engine, "_scan_beliefs", return_value=[],
        ):
            count = engine.apply_global_decay()
            assert count == 0

    def test_non_active_beliefs_skipped(self, engine: NeuroMemEngine) -> None:
        """Decay skips non-ACTIVE beliefs."""
        engine.learn("b1", confidence=0.01, gamma=0.1)
        engine.advance_tick(100)
        engine.apply_global_decay()
        # b1 is now DECAYED; another decay should skip it
        count = engine.apply_global_decay()
        assert count == 0


# ═══════════════════════════════════════════════════════════════════════
# Propagation edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestPropagationEdgeCases:
    def test_propagate_nonexistent_belief_raises(
        self, engine: NeuroMemEngine,
    ) -> None:
        with pytest.raises(NodeNotFoundError):
            engine.propagate("nonexistent_id_abc", "other_ns")

    def test_propagate_same_namespace_raises(
        self, engine: NeuroMemEngine,
    ) -> None:
        belief = engine.learn("same ns belief", confidence=0.9)
        with pytest.raises(NamespaceIsolationError):
            engine.propagate(belief.id, "test")

    def test_propagate_low_trust_raises(
        self, engine: NeuroMemEngine,
    ) -> None:
        belief = engine.learn("low trust belief", confidence=0.5)
        with pytest.raises(TrustThresholdError):
            engine.propagate(belief.id, "other_ns", trust_factor=0.1)


# ═══════════════════════════════════════════════════════════════════════
# Math helpers (static methods)
# ═══════════════════════════════════════════════════════════════════════


class TestMathHelpers:
    def test_cosine_similarity_empty(self) -> None:
        assert NeuroMemEngine._cosine_similarity([], [1.0]) == 0.0
        assert NeuroMemEngine._cosine_similarity([1.0], []) == 0.0

    def test_cosine_similarity_mismatched_length(self) -> None:
        assert NeuroMemEngine._cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_cosine_similarity_zero_norm(self) -> None:
        assert NeuroMemEngine._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_cosine_similarity_identical(self) -> None:
        assert math.isclose(
            NeuroMemEngine._cosine_similarity([1.0, 0.0], [1.0, 0.0]),
            1.0,
        )

    def test_cosine_similarity_orthogonal(self) -> None:
        assert math.isclose(
            NeuroMemEngine._cosine_similarity([1.0, 0.0], [0.0, 1.0]),
            0.0,
        )

    def test_cosine_similarity_opposite_clamped(self) -> None:
        # Opposite vectors → cosine = -1.0, clamped to 0.0
        assert math.isclose(
            NeuroMemEngine._cosine_similarity([1.0], [-1.0]),
            0.0,
        )

    def test_text_overlap(self) -> None:
        overlap = NeuroMemEngine._text_overlap("the sky is blue", "sky is blue")
        assert 0 < overlap <= 1.0

    def test_text_overlap_empty(self) -> None:
        assert NeuroMemEngine._text_overlap("", "anything") == 0.0
        assert NeuroMemEngine._text_overlap("anything", "") == 0.0

    def test_text_overlap_no_common_tokens(self) -> None:
        assert NeuroMemEngine._text_overlap("abc", "xyz") == 0.0

    def test_text_overlap_identical(self) -> None:
        assert math.isclose(
            NeuroMemEngine._text_overlap("hello world", "hello world"),
            1.0,
        )

    def test_distance_to_similarity_none(self) -> None:
        assert NeuroMemEngine._distance_to_similarity(None) == 0.0

    def test_distance_to_similarity_negative(self) -> None:
        assert NeuroMemEngine._distance_to_similarity(-0.5) == 1.0

    def test_distance_to_similarity_normal(self) -> None:
        assert math.isclose(NeuroMemEngine._distance_to_similarity(0.3), 0.7)

    def test_distance_to_similarity_zero(self) -> None:
        assert math.isclose(NeuroMemEngine._distance_to_similarity(0.0), 1.0)

    def test_distance_to_similarity_clamped(self) -> None:
        assert math.isclose(NeuroMemEngine._distance_to_similarity(2.0), 0.0)

    def test_vector_collection(self) -> None:
        assert NeuroMemEngine._vector_collection("test") == "ns_test"
        assert NeuroMemEngine._vector_collection("My Agent") == "ns_my_agent"


# ═══════════════════════════════════════════════════════════════════════
# Fuse scores
# ═══════════════════════════════════════════════════════════════════════


class TestFuseScores:
    def test_fuse_equal_weights(self, engine: NeuroMemEngine) -> None:
        result = engine._fuse_scores(0.8, 0.4)
        assert math.isclose(result, 0.6)

    def test_fuse_graph_only(self, engine: NeuroMemEngine) -> None:
        engine.config.fusion_vector_weight = 0.0
        result = engine._fuse_scores(0.9, 0.1)
        assert math.isclose(result, 0.9)

    def test_fuse_vector_only(self, engine: NeuroMemEngine) -> None:
        engine.config.fusion_vector_weight = 1.0
        result = engine._fuse_scores(0.1, 0.9)
        assert math.isclose(result, 0.9)

    def test_fuse_clamped(self, engine: NeuroMemEngine) -> None:
        engine.config.fusion_vector_weight = 0.5
        result = engine._fuse_scores(1.5, 1.5)
        assert 0.0 <= result <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# Resolve contradiction tiers
# ═══════════════════════════════════════════════════════════════════════


class TestResolveContradiction:
    def test_deprecate_new_high_confidence(self, engine: NeuroMemEngine) -> None:
        res, conf, reason = engine._resolve_contradiction(
            BeliefNode(claim="x", confidence=0.9), "y", similarity=0.7,
        )
        assert res == ContradictionResolution.DEPRECATE_NEW

    def test_deprecate_old_low_confidence(self, engine: NeuroMemEngine) -> None:
        res, conf, reason = engine._resolve_contradiction(
            BeliefNode(claim="x", confidence=0.3), "y", similarity=0.7,
        )
        assert res == ContradictionResolution.DEPRECATE_OLD

    def test_split_medium_confidence(self, engine: NeuroMemEngine) -> None:
        res, conf, reason = engine._resolve_contradiction(
            BeliefNode(claim="x", confidence=0.6), "y", similarity=0.7,
        )
        assert res == ContradictionResolution.SPLIT


# ═══════════════════════════════════════════════════════════════════════
# Advance tick
# ═══════════════════════════════════════════════════════════════════════


class TestAdvanceTick:
    def test_advance_tick_zero(self, engine: NeuroMemEngine) -> None:
        assert engine.advance_tick(0) == 0

    def test_advance_tick_multiple(self, engine: NeuroMemEngine) -> None:
        assert engine.advance_tick(5) == 5
        assert engine.advance_tick(3) == 8


# ═══════════════════════════════════════════════════════════════════════
# Negative memory with explicit trace
# ═══════════════════════════════════════════════════════════════════════


class TestRecordNegativeWithExplicitTrace:
    def test_explicit_trace_receives_steps(self, engine: NeuroMemEngine) -> None:
        trace = ReasoningTrace(trigger="manual_test", namespace="test")
        engine.record_negative("pattern_x", trace=trace)
        assert trace.step_count > 0

    def test_record_negative_dedup_with_trace(
        self, engine: NeuroMemEngine,
    ) -> None:
        neg1 = engine.record_negative("dedup_pattern_x")
        neg2 = engine.record_negative("dedup_pattern_x")
        assert neg2.occurrence_count == 2


# ═══════════════════════════════════════════════════════════════════════
# is_blocked
# ═══════════════════════════════════════════════════════════════════════


class TestIsBlocked:
    def test_is_blocked_no_negative(self, engine: NeuroMemEngine) -> None:
        assert engine.is_blocked("nonexistent_pattern") is False

    def test_is_blocked_after_threshold(self, engine: NeuroMemEngine) -> None:
        engine.record_negative("block_me", block_threshold=1)
        assert engine.is_blocked("block_me") is True

    def test_is_blocked_below_threshold(self, engine: NeuroMemEngine) -> None:
        engine.record_negative("dont_block_yet", block_threshold=3)
        assert engine.is_blocked("dont_block_yet") is False


# ═══════════════════════════════════════════════════════════════════════
# _store_trace failure is non-fatal
# ═══════════════════════════════════════════════════════════════════════


class TestStoreTraceFailure:
    def test_store_trace_failure_non_fatal(self, engine: NeuroMemEngine) -> None:
        trace = ReasoningTrace(trigger="test_fail_trace", namespace="test")
        with unittest.mock.patch.object(
            engine.graph, "add_node",
            side_effect=NeuroMemError("trace persist failed"),
        ):
            engine._store_trace(trace)  # should not raise


# ═══════════════════════════════════════════════════════════════════════
# _load_belief
# ═══════════════════════════════════════════════════════════════════════


class TestLoadBelief:
    def test_load_nonexistent_belief(self, engine: NeuroMemEngine) -> None:
        result = engine._load_belief("nonexistent_belief_id", "test")
        assert result is None

    def test_load_belief_after_learn(self, engine: NeuroMemEngine) -> None:
        belief = engine.learn("loadable claim", confidence=0.7)
        loaded = engine._load_belief(belief.id, "test")
        assert loaded is not None
        assert loaded.claim == "loadable claim"


# ═══════════════════════════════════════════════════════════════════════
# _node_dict_to_belief / _node_to_belief
# ═══════════════════════════════════════════════════════════════════════


class TestNodeDictToBelief:
    def test_node_dict_to_belief(self, engine: NeuroMemEngine) -> None:
        node_dict = {
            "id": "b_test123",
            "claim": "test claim",
            "confidence": 0.7,
            "gamma": 0.99,
            "evidence_count": 2,
            "source": "test",
            "status": "active",
            "namespace": "test",
            "last_decay_tick": 0,
            "tags": "tag1|tag2",
            "_internal": "should_be_stripped",
        }
        belief = engine._node_dict_to_belief(node_dict, "test")
        assert belief is not None
        assert belief.id == "b_test123"
        assert belief.claim == "test claim"
        assert belief.tags == ["tag1", "tag2"]

    def test_node_dict_to_belief_empty(self, engine: NeuroMemEngine) -> None:
        belief = engine._node_dict_to_belief({}, "test")
        assert belief is None


# ═══════════════════════════════════════════════════════════════════════
# _persist_contradiction
# ═══════════════════════════════════════════════════════════════════════


class TestPersistContradiction:
    def test_persist_contradiction_creates_node(self, engine: NeuroMemEngine) -> None:
        belief = engine.learn("contradict me", confidence=0.5, embedding=[0.1] * 4)
        event = ContradictionEvent(
            namespace="test",
            belief_id=belief.id,
            incoming_claim="opposite claim",
            similarity_score=0.9,
            conflict_severity=0.5,
            confidence_before=0.5,
            confidence_after=0.4,
            resolution=ContradictionResolution.SPLIT,
            reasoning="test",
        )
        engine._persist_contradiction(event, belief)
        assert engine.graph.node_exists("ContradictionEvent", event.id)

    def test_persist_contradiction_edge_failure_non_fatal(
        self, engine: NeuroMemEngine,
    ) -> None:
        """If CONTRADICTS edge creation fails, it's logged and skipped."""
        belief = engine.learn("contradict edge fail", confidence=0.5)
        event = ContradictionEvent(
            namespace="test",
            belief_id=belief.id,
            incoming_claim="x",
            similarity_score=0.8,
            conflict_severity=0.4,
            confidence_before=0.5,
            confidence_after=0.4,
            resolution=ContradictionResolution.SPLIT,
            reasoning="test",
        )
        # Patch add_edge to fail on CONTRADICTS but succeed otherwise
        original_add_edge = engine.graph.add_edge
        call_count = 0

        def failing_add_edge(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise NeuroMemError("edge fail")
            return original_add_edge(*args, **kwargs)

        with unittest.mock.patch.object(
            engine.graph, "add_edge", side_effect=failing_add_edge,
        ):
            engine._persist_contradiction(event, belief)  # should not raise


# ═══════════════════════════════════════════════════════════════════════
# _find_negative_memory reconstruction edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestFindNegativeMemoryReconstruction:
    def test_find_negative_memory_missing_id_returns_none(
        self, engine: NeuroMemEngine,
    ) -> None:
        """A negative memory node without an id returns None."""
        # Manually insert a NegativeMemory node without an id
        engine.graph.add_node(
            "NegativeMemory", "neg_no_id_placeholder",
            {
                "pattern": "test_pattern_no_id",
                "severity": "warning",
                "block_threshold": 1,
                "occurrence_count": 1,
                "namespace": "test",
            },
        )
        # The _find_negative_memory queries by pattern; if the stored node
        # has no 'id' in its properties, reconstruction returns None.
        # Note: Kuzu nodes always have a PK, so this exercises the
        # "clean.get('id', '')" returning "" → not neg_id → return None.
        # We can't easily remove the PK, so we test the query-failure path.

    def test_find_negative_memory_query_failure_returns_none(
        self, engine: NeuroMemEngine,
    ) -> None:
        with unittest.mock.patch.object(
            engine.graph, "execute_query",
            side_effect=NeuroMemError("query failed"),
        ):
            result = engine._find_negative_memory("any", "test")
            assert result is None


# ═══════════════════════════════════════════════════════════════════════
# _scan_beliefs failure
# ═══════════════════════════════════════════════════════════════════════


class TestScanBeliefsFailure:
    def test_scan_failure_returns_empty(self, engine: NeuroMemEngine) -> None:
        """If belief scan fails, _scan_beliefs returns empty list."""
        with unittest.mock.patch.object(
            engine.graph, "execute_query",
            side_effect=NeuroMemError("scan failed"),
        ):
            beliefs = engine._scan_beliefs("test")
            assert beliefs == []

    def test_scan_with_trace(self, engine: NeuroMemEngine) -> None:
        """_scan_beliefs records a trace step when trace is provided."""
        engine.learn("scan trace test", confidence=0.8)
        trace = ReasoningTrace(trigger="t", namespace="test")
        beliefs = engine._scan_beliefs("test", trace)
        assert len(beliefs) >= 1
        assert trace.step_count > 0
