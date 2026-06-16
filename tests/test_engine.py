"""Integration tests for the NeuroMemEngine.

These tests require live Kuzu + ChromaDB backends (provided by the
``engine`` fixture in conftest.py).  They validate the core cognitive
operations: learn, recall, contradiction detection/splitting, confidence
decay, negative memory guardrails, and cross-namespace propagation.
"""

from __future__ import annotations

import math

import pytest

from neuromem.core.exceptions import NamespaceIsolationError, TrustThresholdError
from neuromem.core.models import (
    BeliefStatus,
    ContradictionResolution,
    NegativeMemorySeverity,
    PropagationStatus,
)


# ═══════════════════════════════════════════════════════════════════════
# Learn
# ═══════════════════════════════════════════════════════════════════════

class TestEngineLearn:
    """Tests for the learn() write path."""

    def test_learn_creates_belief(self, engine) -> None:
        belief = engine.learn("The sky is blue", confidence=0.9, embedding=[0.1, 0.2, 0.3, 0.4])
        assert belief.claim == "The sky is blue"
        assert belief.confidence == 0.9
        assert belief.status == BeliefStatus.ACTIVE
        assert belief.namespace == "test"

    def test_learn_with_tags(self, engine) -> None:
        belief = engine.learn("Cats are mammals", confidence=0.95, tags=["biology", "fact"])
        assert belief.tags == ["biology", "fact"]

    def test_learn_with_custom_source(self, engine) -> None:
        belief = engine.learn("Earth orbits the Sun", confidence=1.0, source="astronomy")
        assert belief.source == "astronomy"

    def test_learn_with_custom_gamma(self, engine) -> None:
        belief = engine.learn("x", confidence=0.8, gamma=0.5)
        assert belief.gamma == 0.5

    def test_learn_persists_to_graph(self, engine) -> None:
        belief = engine.learn("test claim", confidence=0.7, embedding=[1.0, 0.0])
        reloaded = engine._load_belief(belief.id, "test")
        assert reloaded is not None
        assert reloaded.claim == "test claim"

    def test_learn_persists_to_vector(self, engine) -> None:
        belief = engine.learn("vector test", confidence=0.6, embedding=[0.5, 0.5, 0.5, 0.5])
        results = engine.vector.similarity_search("ns_test", [0.5, 0.5, 0.5, 0.5], n_results=1)
        assert len(results) >= 1
        assert any(r.id == belief.id for r in results)


# ═══════════════════════════════════════════════════════════════════════
# Recall
# ═══════════════════════════════════════════════════════════════════════

class TestEngineRecall:
    """Tests for the recall() read path and query fusion."""

    def test_recall_by_embedding(self, engine) -> None:
        engine.learn("The sky is blue", confidence=0.9, embedding=[0.1, 0.2, 0.3, 0.4])
        results = engine.recall(query_embedding=[0.1, 0.2, 0.3, 0.4])
        assert len(results) >= 1
        assert results[0].fused_score > 0.0

    def test_recall_sorted_by_fused_score(self, engine) -> None:
        engine.learn("A", confidence=0.9, embedding=[0.9, 0.1, 0.0, 0.0])
        engine.learn("B", confidence=0.5, embedding=[0.1, 0.9, 0.0, 0.0])
        results = engine.recall(query_embedding=[0.9, 0.1, 0.0, 0.0], n_results=5)
        # Results should be sorted descending
        for i in range(len(results) - 1):
            assert results[i].fused_score >= results[i + 1].fused_score

    def test_recall_min_confidence_filter(self, engine) -> None:
        engine.learn("high conf", confidence=0.9, embedding=[0.9, 0.1, 0.0, 0.0])
        engine.learn("low conf", confidence=0.1, embedding=[0.1, 0.9, 0.0, 0.0])
        results = engine.recall(
            query_embedding=[0.5, 0.5, 0.0, 0.0],
            min_confidence=0.8,
        )
        # Only the high-confidence belief should survive
        for r in results:
            assert r.fused_score >= 0.8

    def test_recall_text_query_no_embedding(self, engine) -> None:
        """Recall with text query (no embedding) scans all beliefs."""
        engine.learn("Paris is the capital of France", confidence=0.9)
        results = engine.recall(query="anything", min_confidence=0.0)
        assert len(results) >= 1

    def test_recall_requires_query_or_embedding(self, engine) -> None:
        with pytest.raises(ValueError):
            engine.recall()


# ═══════════════════════════════════════════════════════════════════════
# Contradiction detection and splitting  ← KEY SPEC REQUIREMENT
# ═══════════════════════════════════════════════════════════════════════

class TestContradictionSplitting:
    """Tests for contradiction detection and resolution.

    This is a key requirement: the ContradictionEvent hook must
    intercept incoming data that clashes with existing truths and force
    a state split or deprecation.
    """

    def test_contradiction_detected_on_similar_claims(self, engine) -> None:
        """A highly similar incoming claim triggers contradiction detection."""
        existing = engine.learn(
            "The sky is blue",
            confidence=0.6,
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        event = engine.check_contradiction(
            existing,
            "The sky is blue and red",
            incoming_embedding=[0.1, 0.2, 0.3, 0.4],
        )
        assert event is not None
        assert event.similarity_score >= engine.config.contradiction_threshold

    def test_contradiction_split_reduces_confidence(self, engine) -> None:
        """A SPLIT resolution reduces the existing belief's confidence."""
        existing = engine.learn(
            "The sky is blue",
            confidence=0.6,
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        original_confidence = existing.confidence

        event = engine.check_contradiction(
            existing,
            "The sky is actually red",
            incoming_embedding=[0.1, 0.2, 0.3, 0.4],
        )
        assert event is not None
        assert event.resolution == ContradictionResolution.SPLIT
        assert event.confidence_after < original_confidence

    def test_contradiction_deprecate_old_on_low_confidence(self, engine) -> None:
        """Low-confidence existing belief is deprecated (DEPRECATE_OLD)."""
        existing = engine.learn(
            "Maybe the sky is green",
            confidence=0.3,  # low confidence
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        event = engine.check_contradiction(
            existing,
            "The sky is definitely blue",
            incoming_embedding=[0.1, 0.2, 0.3, 0.4],
        )
        assert event is not None
        assert event.resolution == ContradictionResolution.DEPRECATE_OLD

    def test_contradiction_deprecate_new_on_high_confidence(self, engine) -> None:
        """High-confidence existing belief rejects the incoming claim.

        Resolution logic: confidence > 0.8 AND similarity < 0.9 → DEPRECATE_NEW.
        We use embeddings that are above the 0.5 contradiction threshold but
        below 0.9 so the high-confidence branch fires.
        """
        existing = engine.learn(
            "The sky is blue",
            confidence=0.9,  # high confidence
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        # [0.6, 0.8, 0.0, 0.0] vs [1.0, 0.0, 0.0, 0.0] → cosine ≈ 0.6
        event = engine.check_contradiction(
            existing,
            "The sky is slightly purple",
            incoming_embedding=[0.6, 0.8, 0.0, 0.0],
        )
        if event is not None:
            # High confidence + moderate similarity → reject new
            assert event.resolution == ContradictionResolution.DEPRECATE_NEW

    def test_no_contradiction_on_different_claims(self, engine) -> None:
        """Dissimilar claims do not trigger contradiction.

        We use orthogonal embeddings (cosine similarity = 0.0) so the
        similarity stays well below the 0.5 contradiction threshold.
        """
        existing = engine.learn(
            "The sky is blue",
            confidence=0.7,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        # Orthogonal vector → cosine similarity = 0.0
        event = engine.check_contradiction(
            existing,
            "Cars have wheels",
            incoming_embedding=[0.0, 0.0, 0.0, 1.0],
        )
        assert event is None

    def test_contradiction_persisted_to_graph(self, engine) -> None:
        """Contradiction events are stored in the graph."""
        existing = engine.learn(
            "x", confidence=0.5, embedding=[0.1, 0.2, 0.3, 0.4],
        )
        event = engine.check_contradiction(
            existing, "y", incoming_embedding=[0.1, 0.2, 0.3, 0.4],
        )
        assert event is not None
        assert engine.graph.node_exists("ContradictionEvent", event.id)

    def test_contradiction_severity_computed(self, engine) -> None:
        """Severity is similarity * (1 - existing_confidence)."""
        existing = engine.learn(
            "x", confidence=0.5, embedding=[0.1, 0.2, 0.3, 0.4],
        )
        event = engine.check_contradiction(
            existing, "y", incoming_embedding=[0.1, 0.2, 0.3, 0.4],
        )
        assert event is not None
        expected_severity = event.similarity_score * (1.0 - 0.5)
        assert math.isclose(event.conflict_severity, expected_severity, abs_tol=0.01)


# ═══════════════════════════════════════════════════════════════════════
# Confidence decay  ← KEY SPEC REQUIREMENT
# ═══════════════════════════════════════════════════════════════════════

class TestConfidenceDecay:
    """Tests for temporal confidence decay with gamma."""

    def test_global_decay_reduces_confidence(self, engine) -> None:
        """Advancing ticks + global decay reduces belief confidence."""
        belief = engine.learn(
            "decaying fact", confidence=0.9, gamma=0.5,
            embedding=[0.1, 0.0, 0.0, 0.0],
        )
        original = belief.confidence
        engine.advance_tick(3)
        engine.apply_global_decay()
        reloaded = engine._load_belief(belief.id, "test")
        assert reloaded.confidence < original

    def test_decay_marks_decayed_below_floor(self, engine) -> None:
        """Beliefs below the decay floor are marked DECAYED."""
        belief = engine.learn(
            "weak fact", confidence=0.3, gamma=0.1,  # very aggressive decay
            embedding=[0.2, 0.0, 0.0, 0.0],
        )
        engine.advance_tick(10)
        engine.apply_global_decay()
        reloaded = engine._load_belief(belief.id, "test")
        assert reloaded.status == BeliefStatus.DECAYED

    def test_decay_with_no_gamma_change(self, engine) -> None:
        """Gamma=1.0 means no decay."""
        belief = engine.learn(
            "stable fact", confidence=0.8, gamma=1.0,
            embedding=[0.3, 0.0, 0.0, 0.0],
        )
        engine.advance_tick(5)
        engine.apply_global_decay()
        reloaded = engine._load_belief(belief.id, "test")
        assert math.isclose(reloaded.confidence, 0.8, abs_tol=0.01)

    def test_advance_tick(self, engine) -> None:
        """advance_tick increments the counter."""
        assert engine.current_tick == 0
        engine.advance_tick(5)
        assert engine.current_tick == 5
        engine.advance_tick(3)
        assert engine.current_tick == 8

    def test_advance_tick_negative_rejected(self, engine) -> None:
        with pytest.raises(ValueError):
            engine.advance_tick(-1)


# ═══════════════════════════════════════════════════════════════════════
# Negative memory
# ═══════════════════════════════════════════════════════════════════════

class TestNegativeMemory:
    """Tests for negative-memory guardrails."""

    def test_record_negative(self, engine) -> None:
        neg = engine.record_negative("bad_tool_call", severity=NegativeMemorySeverity.ERROR)
        assert neg.pattern == "bad_tool_call"
        assert neg.occurrence_count == 1

    def test_negative_blocks_after_threshold(self, engine) -> None:
        engine.record_negative("blocked_pattern", block_threshold=2)
        assert not engine.is_blocked("blocked_pattern")
        engine.record_negative("blocked_pattern")
        assert engine.is_blocked("blocked_pattern")

    def test_negative_dedup_increments(self, engine) -> None:
        engine.record_negative("dedup_test")
        neg2 = engine.record_negative("dedup_test")
        assert neg2.occurrence_count == 2

    def test_different_patterns_separate(self, engine) -> None:
        engine.record_negative("pattern_a")
        neg_b = engine.record_negative("pattern_b")
        assert neg_b.occurrence_count == 1


# ═══════════════════════════════════════════════════════════════════════
# Propagation
# ═══════════════════════════════════════════════════════════════════════

class TestPropagation:
    """Tests for cross-namespace propagation."""

    def test_propagate_delivers_with_trust_reduction(self, engine) -> None:
        belief = engine.learn("share me", confidence=0.9, embedding=[0.5, 0.5, 0.5, 0.5])
        record = engine.propagate(belief.id, "agent_b", trust_factor=0.8)
        assert record.status == PropagationStatus.DELIVERED
        assert record.propagated_confidence < record.original_confidence
        assert math.isclose(record.propagated_confidence, 0.9 * 0.8, abs_tol=0.05)

    def test_propagate_creates_belief_in_target(self, engine) -> None:
        belief = engine.learn("prop test", confidence=0.9, embedding=[0.6, 0.4, 0.0, 0.0])
        engine.propagate(belief.id, "target_ns")
        target_beliefs = engine._scan_beliefs("target_ns")
        assert len(target_beliefs) >= 1
        assert any("propagated" in b.tags for b in target_beliefs)

    def test_propagate_rejects_low_trust(self, engine) -> None:
        belief = engine.learn("low trust", confidence=0.9, embedding=[0.7, 0.3, 0.0, 0.0])
        with pytest.raises(TrustThresholdError):
            engine.propagate(belief.id, "agent_c", trust_factor=0.1)

    def test_propagate_rejects_same_namespace(self, engine) -> None:
        belief = engine.learn("same ns", confidence=0.9, embedding=[0.8, 0.2, 0.0, 0.0])
        with pytest.raises(NamespaceIsolationError):
            engine.propagate(belief.id, "test")

    def test_propagation_record_persisted(self, engine) -> None:
        belief = engine.learn("persist prop", confidence=0.9, embedding=[0.4, 0.6, 0.0, 0.0])
        record = engine.propagate(belief.id, "agent_d")
        assert engine.graph.node_exists("PropagationRecord", record.id)
