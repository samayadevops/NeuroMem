"""Unit tests for the Pydantic cognitive models.

These tests exercise model validation, decay math, and lifecycle
transitions without touching any storage backend.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from neuromem.core.models import (
    BeliefNode,
    BeliefStatus,
    ContradictionEvent,
    ContradictionResolution,
    NegativeMemory,
    NegativeMemorySeverity,
    PropagationRecord,
    PropagationStatus,
    ReasoningStep,
    ReasoningTrace,
    TraceStepType,
    _ensure_utc,
    _generate_id,
)


# ═══════════════════════════════════════════════════════════════════════
# BeliefNode
# ═══════════════════════════════════════════════════════════════════════

class TestBeliefNode:
    """Tests for BeliefNode validation and decay logic."""

    def test_default_creation(self) -> None:
        """A belief created with defaults has expected values."""
        belief = BeliefNode(claim="The sky is blue")
        assert belief.claim == "The sky is blue"
        assert belief.confidence == 0.5
        assert belief.gamma == 0.99
        assert belief.status == BeliefStatus.ACTIVE
        assert belief.evidence_count == 1
        assert belief.source == "unknown"
        assert belief.is_active is True
        assert belief.id.startswith("belief_")

    def test_confidence_bounds(self) -> None:
        """Confidence must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            BeliefNode(claim="x", confidence=1.5)
        with pytest.raises(ValidationError):
            BeliefNode(claim="x", confidence=-0.1)
        # Boundary values are valid
        BeliefNode(claim="x", confidence=0.0)
        BeliefNode(claim="x", confidence=1.0)

    def test_gamma_bounds(self) -> None:
        """Gamma must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            BeliefNode(claim="x", gamma=1.5)
        with pytest.raises(ValidationError):
            BeliefNode(claim="x", gamma=-0.1)

    def test_claim_empty_rejected(self) -> None:
        """Empty claims are rejected."""
        with pytest.raises(ValidationError):
            BeliefNode(claim="")

    def test_tags_dedup_and_strip(self) -> None:
        """Tags are stripped of whitespace and deduplicated."""
        belief = BeliefNode(
            claim="x",
            tags=["  science  ", "science", "fact", ""],
        )
        assert belief.tags == ["science", "fact"]

    def test_embedding_normalised_to_float(self) -> None:
        """Embeddings are converted to plain floats."""
        belief = BeliefNode(claim="x", embedding=[1, 2, 3])
        assert belief.embedding == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in belief.embedding)

    def test_embedding_empty_becomes_none(self) -> None:
        """Empty embeddings become None."""
        belief = BeliefNode(claim="x", embedding=[])
        assert belief.embedding is None

    def test_effective_confidence_no_decay(self) -> None:
        """With gamma=1.0, confidence does not decay."""
        belief = BeliefNode(claim="x", confidence=0.8, gamma=1.0)
        assert belief.effective_confidence(0) == 0.8
        assert belief.effective_confidence(100) == 0.8

    def test_effective_confidence_exponential_decay(self) -> None:
        """Confidence decays exponentially over ticks."""
        belief = BeliefNode(claim="x", confidence=1.0, gamma=0.5)
        # After 1 tick: 1.0 * 0.5^1 = 0.5
        assert math.isclose(belief.effective_confidence(1), 0.5, abs_tol=1e-9)
        # After 2 ticks: 1.0 * 0.5^2 = 0.25
        assert math.isclose(belief.effective_confidence(2), 0.25, abs_tol=1e-9)
        # After 10 ticks: 1.0 * 0.5^10 ≈ 0.000977
        assert math.isclose(belief.effective_confidence(10), 0.5**10, abs_tol=1e-9)

    def test_effective_confidence_half_life(self) -> None:
        """Half-life overrides gamma for decay calculation."""
        belief = BeliefNode(
            claim="x", confidence=1.0, gamma=0.99, half_life_ticks=5,
        )
        # After 5 ticks (one half-life), confidence should be ~0.5
        decayed = belief.effective_confidence(5)
        assert math.isclose(decayed, 0.5, abs_tol=0.01)

    def test_effective_confidence_zero_elapsed(self) -> None:
        """If current_tick equals last_decay_tick, no decay."""
        belief = BeliefNode(claim="x", confidence=0.7, gamma=0.5, last_decay_tick=5)
        assert belief.effective_confidence(5) == 0.7

    def test_effective_confidence_deprecated_belief(self) -> None:
        """Deprecated beliefs always have 0 effective confidence."""
        belief = BeliefNode(
            claim="x", confidence=0.9, status=BeliefStatus.DEPRECATED,
        )
        assert belief.effective_confidence(0) == 0.0
        assert belief.effective_confidence(100) == 0.0

    def test_effective_confidence_contradicted_belief(self) -> None:
        """Contradicted beliefs always have 0 effective confidence."""
        belief = BeliefNode(
            claim="x", confidence=0.9, status=BeliefStatus.CONTRADICTED,
        )
        assert belief.effective_confidence(0) == 0.0

    def test_apply_decay_mutates_state(self) -> None:
        """apply_decay updates confidence and last_decay_tick."""
        belief = BeliefNode(claim="x", confidence=1.0, gamma=0.5)
        new_conf = belief.apply_decay(3)
        assert math.isclose(new_conf, 0.125, abs_tol=1e-9)
        assert math.isclose(belief.confidence, 0.125, abs_tol=1e-9)
        assert belief.last_decay_tick == 3

    def test_reinforce_increases_confidence(self) -> None:
        """Reinforce adds to confidence and increments evidence count."""
        belief = BeliefNode(claim="x", confidence=0.5)
        new_conf = belief.reinforce(amount=0.3)
        assert math.isclose(new_conf, 0.8, abs_tol=1e-9)
        assert belief.evidence_count == 2

    def test_reinforce_clamped_to_max(self) -> None:
        """Reinforce does not exceed max_confidence."""
        belief = BeliefNode(claim="x", confidence=0.9)
        new_conf = belief.reinforce(amount=0.3)
        assert new_conf == 1.0

    def test_reinforce_revives_deprecated(self) -> None:
        """Reinforcing a deprecated belief revives it to ACTIVE."""
        belief = BeliefNode(
            claim="x", confidence=0.0, status=BeliefStatus.DEPRECATED,
        )
        belief.reinforce(amount=0.5)
        assert belief.status == BeliefStatus.ACTIVE
        assert belief.confidence == 0.5

    def test_deprecate_zeros_confidence(self) -> None:
        """Deprecate sets confidence to 0 and status to DEPRECATED."""
        belief = BeliefNode(claim="x", confidence=0.9)
        belief.deprecate()
        assert belief.status == BeliefStatus.DEPRECATED
        assert belief.confidence == 0.0

    def test_dimension_property(self) -> None:
        """Dimension returns embedding length or None."""
        assert BeliefNode(claim="x", embedding=[1, 2, 3]).dimension == 3
        assert BeliefNode(claim="x").dimension is None

    def test_created_at_is_utc(self) -> None:
        """created_at is timezone-aware UTC."""
        belief = BeliefNode(claim="x")
        assert belief.created_at.tzinfo is not None

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError):
            BeliefNode(claim="x", nonexistent_field=42)


# ═══════════════════════════════════════════════════════════════════════
# ContradictionEvent
# ═══════════════════════════════════════════════════════════════════════

class TestContradictionEvent:
    """Tests for ContradictionEvent."""

    def test_default_creation(self) -> None:
        event = ContradictionEvent(
            belief_id="b1",
            incoming_claim="The sky is red",
        )
        assert event.belief_id == "b1"
        assert event.incoming_claim == "The sky is red"
        assert event.resolution == ContradictionResolution.ESCALATE
        assert event.id.startswith("contradiction_")

    def test_was_resolved(self) -> None:
        """was_resolved is True for any non-ESCALATE resolution."""
        assert not ContradictionEvent(
            belief_id="b1", incoming_claim="x",
            resolution=ContradictionResolution.ESCALATE,
        ).was_resolved
        assert ContradictionEvent(
            belief_id="b1", incoming_claim="x",
            resolution=ContradictionResolution.SPLIT,
        ).was_resolved

    def test_caused_split(self) -> None:
        split = ContradictionEvent(
            belief_id="b1", incoming_claim="x",
            resolution=ContradictionResolution.SPLIT,
        )
        assert split.caused_split is True

        deprecate = ContradictionEvent(
            belief_id="b1", incoming_claim="x",
            resolution=ContradictionResolution.DEPRECATE_OLD,
        )
        assert deprecate.caused_split is False

    def test_confidence_delta(self) -> None:
        event = ContradictionEvent(
            belief_id="b1", incoming_claim="x",
            confidence_before=0.9, confidence_after=0.6,
        )
        assert math.isclose(event.confidence_delta(), -0.3, abs_tol=1e-9)

    def test_similarity_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ContradictionEvent(belief_id="b1", incoming_claim="x", similarity_score=1.5)
        with pytest.raises(ValidationError):
            ContradictionEvent(belief_id="b1", incoming_claim="x", similarity_score=-0.1)


# ═══════════════════════════════════════════════════════════════════════
# NegativeMemory
# ═══════════════════════════════════════════════════════════════════════

class TestNegativeMemory:
    """Tests for NegativeMemory."""

    def test_default_creation(self) -> None:
        neg = NegativeMemory(pattern="bad_pattern")
        assert neg.pattern == "bad_pattern"
        assert neg.severity == NegativeMemorySeverity.WARNING
        assert neg.block_threshold == 1
        assert neg.occurrence_count == 1
        assert neg.id.startswith("negative_")

    def test_record_occurrence_increments(self) -> None:
        neg = NegativeMemory(pattern="x", occurrence_count=1)
        assert neg.record_occurrence() == 2
        assert neg.occurrence_count == 2
        assert neg.record_occurrence() == 3
        assert neg.occurrence_count == 3

    def test_should_block_at_threshold(self) -> None:
        """should_block is True when occurrence_count >= block_threshold."""
        neg = NegativeMemory(pattern="x", block_threshold=3, occurrence_count=2)
        assert neg.should_block is False
        neg.record_occurrence()
        assert neg.should_block is True

    def test_block_threshold_zero_disables_blocking(self) -> None:
        neg = NegativeMemory(pattern="x", block_threshold=0, occurrence_count=100)
        assert neg.should_block is False

    def test_is_fatal(self) -> None:
        assert NegativeMemory(pattern="x", severity=NegativeMemorySeverity.FATAL).is_fatal
        assert not NegativeMemory(pattern="x", severity=NegativeMemorySeverity.WARNING).is_fatal


# ═══════════════════════════════════════════════════════════════════════
# ReasoningTrace
# ═══════════════════════════════════════════════════════════════════════

class TestReasoningTrace:
    """Tests for ReasoningTrace and ReasoningStep."""

    def test_empty_trace(self) -> None:
        trace = ReasoningTrace(trigger="test")
        assert trace.step_count == 0
        assert trace.is_empty is True

    def test_add_step(self) -> None:
        trace = ReasoningTrace(trigger="test")
        step = ReasoningStep(
            step_type=TraceStepType.BELIEF_CREATE,
            description="Created a belief",
            belief_ids=["b1"],
        )
        trace.add_step(step)
        assert trace.step_count == 1
        assert not trace.is_empty

    def test_related_belief_ids_aggregated(self) -> None:
        trace = ReasoningTrace(trigger="test")
        trace.add_step(ReasoningStep(
            step_type=TraceStepType.BELIEF_QUERY,
            description="q1", belief_ids=["b1", "b2"],
        ))
        trace.add_step(ReasoningStep(
            step_type=TraceStepType.BELIEF_CREATE,
            description="c1", belief_ids=["b2", "b3"],
        ))
        assert trace.related_belief_ids == ["b1", "b2", "b3"]

    def test_related_contradiction_ids(self) -> None:
        trace = ReasoningTrace(trigger="test")
        trace.add_step(ReasoningStep(
            step_type=TraceStepType.CONTRADICTION_DETECT,
            description="d1", contradiction_ids=["c1"],
        ))
        trace.add_step(ReasoningStep(
            step_type=TraceStepType.CONTRADICTION_RESOLVE,
            description="r1", contradiction_ids=["c1", "c2"],
        ))
        assert trace.related_contradiction_ids == ["c1", "c2"]

    def test_total_duration_ms(self) -> None:
        trace = ReasoningTrace(trigger="test")
        trace.add_step(ReasoningStep(
            step_type=TraceStepType.BELIEF_QUERY, description="q", duration_ms=10.0,
        ))
        trace.add_step(ReasoningStep(
            step_type=TraceStepType.FUSION, description="f", duration_ms=5.5,
        ))
        assert math.isclose(trace.total_duration_ms, 15.5, abs_tol=1e-9)

    def test_steps_of_type_filter(self) -> None:
        trace = ReasoningTrace(trigger="test")
        trace.add_step(ReasoningStep(step_type=TraceStepType.BELIEF_QUERY, description="q1"))
        trace.add_step(ReasoningStep(step_type=TraceStepType.FUSION, description="f1"))
        trace.add_step(ReasoningStep(step_type=TraceStepType.BELIEF_QUERY, description="q2"))
        queries = trace.steps_of_type(TraceStepType.BELIEF_QUERY)
        assert len(queries) == 2

    def test_confidence_timeline(self) -> None:
        trace = ReasoningTrace(trigger="test")
        trace.add_step(ReasoningStep(
            step_type=TraceStepType.BELIEF_CREATE, description="c",
            confidence_after=0.5,
        ))
        trace.add_step(ReasoningStep(
            step_type=TraceStepType.BELIEF_UPDATE, description="u",
            confidence_after=0.8,
        ))
        timeline = trace.confidence_timeline()
        assert timeline == [(0, 0.5), (1, 0.8)]

    def test_step_confidence_delta(self) -> None:
        step = ReasoningStep(
            step_type=TraceStepType.BELIEF_UPDATE, description="x",
            confidence_before=0.6, confidence_after=0.9,
        )
        assert math.isclose(step.confidence_delta, 0.3, abs_tol=1e-9)

    def test_step_confidence_delta_none(self) -> None:
        step = ReasoningStep(
            step_type=TraceStepType.BELIEF_CREATE, description="x",
        )
        assert step.confidence_delta is None


# ═══════════════════════════════════════════════════════════════════════
# PropagationRecord
# ═══════════════════════════════════════════════════════════════════════

class TestPropagationRecord:
    """Tests for PropagationRecord."""

    def test_default_creation(self) -> None:
        record = PropagationRecord(
            source_namespace="agent_a",
            target_namespace="agent_b",
            belief_id="b1",
            belief_claim="The sky is blue",
            original_confidence=0.9,
        )
        assert record.id.startswith("propagation_")
        assert record.status == PropagationStatus.PENDING
        assert record.trust_factor == 0.8  # default

    def test_propagated_confidence_auto_computed(self) -> None:
        record = PropagationRecord(
            source_namespace="a", target_namespace="b",
            belief_id="b1", belief_claim="x",
            original_confidence=0.9, trust_factor=0.5,
        )
        assert math.isclose(record.propagated_confidence, 0.45, abs_tol=1e-9)

    def test_confidence_loss(self) -> None:
        record = PropagationRecord(
            source_namespace="a", target_namespace="b",
            belief_id="b1", belief_claim="x",
            original_confidence=0.9, trust_factor=0.5,
        )
        assert math.isclose(record.confidence_loss, 0.45, abs_tol=1e-9)

    def test_same_namespace_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PropagationRecord(
                source_namespace="a", target_namespace="a",
                belief_id="b1", belief_claim="x",
                original_confidence=0.5,
            )

    def test_mark_delivered(self) -> None:
        record = PropagationRecord(
            source_namespace="a", target_namespace="b",
            belief_id="b1", belief_claim="x",
            original_confidence=0.9,
        )
        record.mark_delivered()
        assert record.status == PropagationStatus.DELIVERED
        assert record.delivered_at is not None
        assert record.is_complete is True

    def test_mark_rejected(self) -> None:
        record = PropagationRecord(
            source_namespace="a", target_namespace="b",
            belief_id="b1", belief_claim="x",
            original_confidence=0.9,
        )
        record.mark_rejected("too_low")
        assert record.status == PropagationStatus.REJECTED
        assert record.failure_reason == "too_low"
        assert record.is_complete is True

    def test_mark_failed_retry_logic(self) -> None:
        record = PropagationRecord(
            source_namespace="a", target_namespace="b",
            belief_id="b1", belief_claim="x",
            original_confidence=0.9,
        )
        # First failure — should retry
        assert record.mark_failed("timeout", max_retries=3) is True
        assert record.status == PropagationStatus.PENDING
        assert record.retry_count == 1

        # Second failure — should retry
        assert record.mark_failed("timeout", max_retries=3) is True
        assert record.retry_count == 2

        # Third failure — terminal
        assert record.mark_failed("timeout", max_retries=3) is False
        assert record.status == PropagationStatus.FAILED
        assert record.is_complete is True

    def test_trust_factor_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PropagationRecord(
                source_namespace="a", target_namespace="b",
                belief_id="b1", belief_claim="x",
                original_confidence=0.5, trust_factor=1.5,
            )


# ═══════════════════════════════════════════════════════════════════════
# Helpers — _ensure_utc, _generate_id
# ═══════════════════════════════════════════════════════════════════════


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_ensure_utc_none_returns_none(self) -> None:
        assert _ensure_utc(None) is None

    def test_ensure_utc_naive_adds_utc(self) -> None:
        naive = datetime(2020, 1, 15, 12, 30, 0)
        result = _ensure_utc(naive)
        assert result.tzinfo is not None
        assert result.year == 2020
        assert result.month == 1
        assert result.day == 15

    def test_ensure_utc_aware_converts(self) -> None:
        aware = datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = _ensure_utc(aware)
        assert result.tzinfo is not None

    def test_generate_id_prefix(self) -> None:
        uid = _generate_id("belief")
        assert uid.startswith("belief_")
        assert len(uid) > len("belief_")


# ═══════════════════════════════════════════════════════════════════════
# BeliefNode — additional edge cases for 100 % coverage
# ═══════════════════════════════════════════════════════════════════════


class TestBeliefNodeEdgeCases:
    """Remaining edge-case coverage for BeliefNode."""

    def test_apply_decay_on_deprecated_returns_zero(self) -> None:
        belief = BeliefNode(
            claim="x", confidence=0.5, status=BeliefStatus.DEPRECATED,
        )
        result = belief.apply_decay(10)
        assert result == 0.0
        # Confidence should NOT change for deprecated beliefs
        assert belief.confidence == 0.5
        assert belief.last_decay_tick == 0  # unchanged

    def test_explicit_none_embedding(self) -> None:
        """Explicitly passing embedding=None should be accepted."""
        belief = BeliefNode(claim="x", embedding=None)
        assert belief.embedding is None

    def test_apply_decay_on_contradicted_returns_zero(self) -> None:
        belief = BeliefNode(
            claim="x", confidence=0.8, status=BeliefStatus.CONTRADICTED,
        )
        result = belief.apply_decay(10)
        assert result == 0.0


# ═══════════════════════════════════════════════════════════════════════
# ReasoningTrace — additional edge cases for 100 % coverage
# ═══════════════════════════════════════════════════════════════════════


class TestReasoningTraceEdgeCases:
    """Remaining edge-case coverage for ReasoningTrace."""

    def test_related_negative_ids_aggregated(self) -> None:
        trace = ReasoningTrace(trigger="test")
        trace.add_step(ReasoningStep(
            step_type=TraceStepType.NEGATIVE_RECORD,
            description="n1", negative_ids=["neg_1", "neg_2"],
        ))
        trace.add_step(ReasoningStep(
            step_type=TraceStepType.NEGATIVE_RECORD,
            description="n2", negative_ids=["neg_2", "neg_3"],
        ))
        assert trace.related_negative_ids == ["neg_1", "neg_2", "neg_3"]

    def test_related_negative_ids_empty(self) -> None:
        trace = ReasoningTrace(trigger="test")
        assert trace.related_negative_ids == []

    def test_confidence_delta_with_before_only(self) -> None:
        step = ReasoningStep(
            step_type=TraceStepType.BELIEF_QUERY, description="x",
            confidence_before=0.5,
        )
        assert step.confidence_delta is None

    def test_confidence_delta_with_after_only(self) -> None:
        step = ReasoningStep(
            step_type=TraceStepType.BELIEF_QUERY, description="x",
            confidence_after=0.7,
        )
        assert step.confidence_delta is None

    def test_step_timestamp_none_fills_utcnow(self) -> None:
        """Passing timestamp=None should be filled by the before-validator."""
        step = ReasoningStep(
            step_type=TraceStepType.BELIEF_CREATE, description="x",
            timestamp=None,
        )
        assert step.timestamp.tzinfo is not None
