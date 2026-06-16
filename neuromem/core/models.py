"""Pydantic v2 cognitive models for NeuroMem.

This module defines the five primary memory constructs used across the
framework:

1. :class:`BeliefNode` — a semantic fact with a confidence score subject
   to temporal decay (gamma).
2. :class:`ContradictionEvent` — a record of a detected clash between an
   incoming observation and an existing belief, capturing how the engine
   resolved it.
3. :class:`NegativeMemory` — an execution guardrail logging failed agent
   paths or explicit logic rejections.
4. :class:`ReasoningTrace` — a sequential audit trail mapping how beliefs,
   vector contexts, and confidence shifts interacted.
5. :class:`PropagationRecord` — a record of multi-agent knowledge sharing
   across namespaces using a controlled trust reduction factor.

All models derive from a shared :class:`_BaseCognitiveModel` that
enforces common metadata (id, namespace, timestamps) and serialisation
behaviour.  Every field is explicitly typed and validated; immutable
models use ``model_config = ConfigDict(frozen=True)``.

Time-handling convention
------------------------
All timestamps are timezone-aware UTC ``datetime`` objects produced via
``datetime.now(timezone.utc)``.  This guarantees reproducible decay math
and safe serialisation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _utcnow() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _generate_id(prefix: str) -> str:
    """Generate a deterministic-format ID with a semantic prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _ensure_utc(dt: datetime | None) -> datetime | None:
    """Coerce a naive datetime to UTC.  Returns ``None`` for ``None``."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════

class BeliefStatus(str, Enum):
    """Lifecycle status of a :class:`BeliefNode`."""

    ACTIVE = "active"
    """Belief is current and authoritative."""

    DECAYED = "decayed"
    """Confidence has fallen below the deprecation floor but the belief
    is retained for historical reference."""

    DEPRECATED = "deprecated"
    """Belief has been superseded by a newer one (split or replacement)."""

    CONTRADICTED = "contradicted"
    """Belief has been directly refuted and is no longer authoritative."""


class ContradictionResolution(str, Enum):
    """How a :class:`ContradictionEvent` was resolved by the engine."""

    SPLIT = "split"
    """The conflicting beliefs were split into separate parallel branches."""

    DEPRECATE_OLD = "deprecate_old"
    """The older belief was deprecated in favour of the incoming claim."""

    DEPRECATE_NEW = "deprecate_new"
    """The incoming claim was rejected in favour of the existing belief."""

    ESCALATE = "escalate"
    """The engine could not auto-resolve; escalated to the agent."""

    MERGE = "merge"
    """The two claims were merged into a single refined belief."""


class NegativeMemorySeverity(str, Enum):
    """Severity level of a :class:`NegativeMemory` guardrail entry."""

    INFO = "info"
    """A minor rejection worth recording but not blocking."""

    WARNING = "warning"
    """A path that should be avoided but was not catastrophic."""

    ERROR = "error"
    """A failed execution that must be blocked on recurrence."""

    FATAL = "fatal"
    """A failure that indicates a fundamental logic flaw."""


class PropagationStatus(str, Enum):
    """Outcome of a cross-agent :class:`PropagationRecord`."""

    PENDING = "pending"
    """Propagation has been queued but not yet delivered."""

    DELIVERED = "delivered"
    """Target namespace acknowledged receipt."""

    REJECTED = "rejected"
    """Trust threshold not met; propagation refused."""

    FAILED = "failed"
    """Delivery failed after all retries."""


# ═══════════════════════════════════════════════════════════════════════
# Base model
# ═══════════════════════════════════════════════════════════════════════

class _BaseCognitiveModel(BaseModel):
    """Shared configuration for all cognitive models.

    All cognitive models:
    - Use Pydantic v2 strict validation.
    - Populate ``created_at`` / ``updated_at`` automatically.
    - Forbid extra fields to catch typos at the boundary.
    - Provide a stable ``id`` with a configurable prefix.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        str_strip_whitespace=True,
        frozen=False,
    )

    id: str = Field(..., description="Globally unique identifier for this record.")
    namespace: str = Field(
        default="default",
        min_length=1,
        max_length=256,
        description="Agent / memory domain this record belongs to.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of record creation.",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="UTC timestamp of the most recent mutation.",
    )

    @model_validator(mode="after")
    def _normalise_timestamps(self) -> _BaseCognitiveModel:
        """Ensure all timestamps are timezone-aware UTC."""
        object.__setattr__(self, "created_at", _ensure_utc(self.created_at) or _utcnow())
        if self.updated_at is not None:
            object.__setattr__(self, "updated_at", _ensure_utc(self.updated_at))
        return self

    def touch(self) -> None:
        """Mark this record as updated by setting ``updated_at`` to now."""
        # frozen=False, so direct assignment works because validate_assignment=True
        self.updated_at = _utcnow()


# ═══════════════════════════════════════════════════════════════════════
# 1. BeliefNode
# ═══════════════════════════════════════════════════════════════════════

class BeliefNode(_BaseCognitiveModel):
    """A semantic fact with a confidence score subject to temporal decay.

    The belief's confidence decays over time according to an exponential
    factor governed by ``gamma`` (the decay rate).  The decayed
    confidence is computed on-demand via :meth:`effective_confidence`.

    Parameters
    ----------
    claim:
        The semantic content of the belief (e.g. ``"The sky is blue"``).
    confidence:
        Current raw confidence in ``[0.0, 1.0]``.
    gamma:
        Temporal decay rate in ``[0.0, 1.0]``.  ``gamma=1.0`` means no
        decay; ``gamma=0.0`` means instant decay.  A typical value is
        ``0.99`` (1% decay per tick).
    half_life_ticks:
        Optional explicit half-life in ticks (logical time steps).
        If provided, overrides the gamma-derived half-life.
    embedding:
        Optional dense vector representation for similarity search.
    evidence_count:
        Number of supporting observations that contributed to this belief.
    source:
        Origin of the belief (e.g. ``"agent:planner"``, ``"user"``, ``"inference"``).
    tags:
        Free-form labels for filtering.
    status:
        Lifecycle status.  See :class:`BeliefStatus`.
    """

    id: str = Field(
        default_factory=lambda: _generate_id("belief"),
        description="Unique belief identifier.",
    )

    claim: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="The semantic content of the belief.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Current raw confidence score in [0.0, 1.0].",
    )
    gamma: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Temporal decay rate per tick. 1.0 = no decay.",
    )
    half_life_ticks: int | None = Field(
        default=None,
        ge=1,
        description="Optional explicit half-life in ticks.",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Optional dense vector representation.",
    )
    evidence_count: int = Field(
        default=1,
        ge=0,
        description="Number of supporting observations.",
    )
    source: str = Field(
        default="unknown",
        max_length=256,
        description="Origin of the belief.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form labels for filtering.",
    )
    status: BeliefStatus = Field(
        default=BeliefStatus.ACTIVE,
        description="Lifecycle status of the belief.",
    )
    last_decay_tick: int = Field(
        default=0,
        ge=0,
        description="Logical tick at which decay was last applied.",
    )

    @field_validator("embedding")
    @classmethod
    def _validate_embedding(cls, v: list[float] | None) -> list[float] | None:
        """Ensure embeddings are plain floats with consistent dimensionality."""
        if v is None:
            return None
        if len(v) == 0:
            return None
        return [float(x) for x in v]

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, v: list[str]) -> list[str]:
        """Strip whitespace and drop empty tags."""
        cleaned: list[str] = []
        seen: set[str] = set()
        for tag in v:
            t = tag.strip()
            if t and t not in seen:
                seen.add(t)
                cleaned.append(t)
        return cleaned

    # ── Decay logic ───────────────────────────────────────────────────

    def effective_confidence(self, current_tick: int) -> float:
        """Compute the time-decayed confidence as of *current_tick*.

        Uses an exponential decay model::

            C_eff = C_raw * gamma ^ (current_tick - last_decay_tick)

        If ``half_life_ticks`` is set, gamma is derived so that the
        confidence halves every ``half_life_ticks`` ticks::

            gamma = 0.5 ^ (1 / half_life_ticks)

        Returns a value clamped to ``[0.0, 1.0]``.
        """
        if self.status in (BeliefStatus.DEPRECATED, BeliefStatus.CONTRADICTED):
            return 0.0

        elapsed = max(0, current_tick - self.last_decay_tick)
        if elapsed == 0:
            return self.confidence

        if self.half_life_ticks is not None and self.half_life_ticks > 0:
            effective_gamma = 0.5 ** (1.0 / self.half_life_ticks)
        else:
            effective_gamma = self.gamma

        decayed = self.confidence * (effective_gamma ** elapsed)
        return max(0.0, min(1.0, decayed))

    def apply_decay(self, current_tick: int) -> float:
        """Apply temporal decay up to *current_tick* and update internal state.

        Mutates ``confidence`` and ``last_decay_tick`` in place.  Returns
        the new confidence value.  Also transitions the belief to
        ``DECAYED`` status if confidence drops below ``deprecation_floor``.
        """
        if self.status in (BeliefStatus.DEPRECATED, BeliefStatus.CONTRADICTED):
            return 0.0

        new_conf = self.effective_confidence(current_tick)
        self.confidence = new_conf
        self.last_decay_tick = current_tick
        self.touch()
        return new_conf

    def reinforce(self, amount: float = 0.1, max_confidence: float = 1.0) -> float:
        """Increase confidence by *amount*, clamped to ``max_confidence``.

        Also increments ``evidence_count`` and refreshes ``updated_at``.
        """
        if self.status in (BeliefStatus.DEPRECATED, BeliefStatus.CONTRADICTED):
            # Reinforcing a dead belief revives it
            self.status = BeliefStatus.ACTIVE

        self.confidence = min(max_confidence, self.confidence + amount)
        self.evidence_count += 1
        self.touch()
        return self.confidence

    def deprecate(self, reason: str = "superseded") -> None:
        """Mark this belief as deprecated."""
        self.status = BeliefStatus.DEPRECATED
        self.confidence = 0.0
        self.touch()

    # ── Convenience ───────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """``True`` if the belief is currently authoritative."""
        return self.status == BeliefStatus.ACTIVE

    @property
    def dimension(self) -> int | None:
        """Return the embedding dimensionality, or ``None`` if unset."""
        return len(self.embedding) if self.embedding else None


# ═══════════════════════════════════════════════════════════════════════
# 2. ContradictionEvent
# ═══════════════════════════════════════════════════════════════════════

class ContradictionEvent(_BaseCognitiveModel):
    """A record of a detected clash between an incoming claim and an
    existing belief.

    The engine creates one of these whenever a new observation
    contradicts an established :class:`BeliefNode`.  The event captures
    the conflicting parties, the similarity/conflict metric, and how the
    engine resolved the contradiction.

    Parameters
    ----------
    belief_id:
        ID of the existing belief that was challenged.
    incoming_claim:
        The new claim that triggered the conflict.
    incoming_belief_id:
        ID of the belief created from the incoming claim (if any).
    similarity_score:
        Numeric similarity between the existing belief and the incoming
        claim.  Higher values indicate more direct contradiction.
    confidence_before:
        The existing belief's confidence *before* resolution.
    confidence_after:
        The existing belief's confidence *after* resolution.
    resolution:
        How the engine resolved the contradiction.
    reasoning:
        Human-readable explanation of the resolution decision.
    """

    id: str = Field(
        default_factory=lambda: _generate_id("contradiction"),
        description="Unique contradiction event identifier.",
    )

    belief_id: str = Field(
        ...,
        min_length=1,
        description="ID of the existing belief that was challenged.",
    )
    incoming_claim: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="The new claim that triggered the conflict.",
    )
    incoming_belief_id: str | None = Field(
        default=None,
        description="ID of the belief created from the incoming claim (if any).",
    )
    similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Similarity between the conflicting claims. Higher = more direct.",
    )
    conflict_severity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Computed severity of the conflict (0 = benign, 1 = total).",
    )
    confidence_before: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Existing belief confidence before resolution.",
    )
    confidence_after: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Existing belief confidence after resolution.",
    )
    resolution: ContradictionResolution = Field(
        default=ContradictionResolution.ESCALATE,
        description="How the engine resolved the contradiction.",
    )
    reasoning: str = Field(
        default="",
        description="Human-readable explanation of the resolution decision.",
    )
    related_trace_id: str | None = Field(
        default=None,
        description="ID of the ReasoningTrace that produced this event (if any).",
    )

    @property
    def was_resolved(self) -> bool:
        """``True`` if the engine auto-resolved the contradiction."""
        return self.resolution != ContradictionResolution.ESCALATE

    @property
    def caused_split(self) -> bool:
        """``True`` if the resolution was a state split."""
        return self.resolution == ContradictionResolution.SPLIT

    def confidence_delta(self) -> float:
        """Return the change in confidence caused by this event."""
        return self.confidence_after - self.confidence_before


# ═══════════════════════════════════════════════════════════════════════
# 3. NegativeMemory
# ═══════════════════════════════════════════════════════════════════════

class NegativeMemory(_BaseCognitiveModel):
    """An execution guardrail logging a failed agent path or explicit
    logic rejection.

    Negative memories prevent infinite LLM loops by recording paths that
    have already failed, so the agent can avoid re-attempting them.

    Parameters
    ----------
    pattern:
        A description of the failed path or rejected logic (e.g. a
        prompt prefix, a tool-call signature, or a regex).
    context:
        Structured context surrounding the failure (inputs, outputs,
        environment state).
    severity:
        How serious the failure is.  See :class:`NegativeMemorySeverity`.
    block_threshold:
        Number of occurrences before this memory becomes a hard block.
        ``1`` means the first occurrence blocks.  ``0`` disables blocking.
    occurrence_count:
        Number of times this negative pattern has been observed.
    related_belief_id:
        ID of a belief associated with this failure (if any).
    related_trace_id:
        ID of the reasoning trace that triggered this failure (if any).
    """

    id: str = Field(
        default_factory=lambda: _generate_id("negative"),
        description="Unique negative memory identifier.",
    )

    pattern: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="Description of the failed path or rejected logic.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured context surrounding the failure.",
    )
    severity: NegativeMemorySeverity = Field(
        default=NegativeMemorySeverity.WARNING,
        description="How serious the failure is.",
    )
    block_threshold: int = Field(
        default=1,
        ge=0,
        le=1000,
        description="Occurrences before this memory becomes a hard block.",
    )
    occurrence_count: int = Field(
        default=1,
        ge=1,
        description="Number of times this negative pattern has been observed.",
    )
    related_belief_id: str | None = Field(
        default=None,
        description="ID of an associated belief (if any).",
    )
    related_trace_id: str | None = Field(
        default=None,
        description="ID of the reasoning trace that triggered this (if any).",
    )

    def record_occurrence(self) -> int:
        """Increment the occurrence counter and return the new count."""
        self.occurrence_count += 1
        self.touch()
        return self.occurrence_count

    @property
    def should_block(self) -> bool:
        """``True`` if this negative memory has reached its block threshold.

        Returns ``False`` if ``block_threshold == 0`` (blocking disabled).
        """
        if self.block_threshold == 0:
            return False
        return self.occurrence_count >= self.block_threshold

    @property
    def is_fatal(self) -> bool:
        """``True`` if this memory has FATAL severity."""
        return self.severity == NegativeMemorySeverity.FATAL


# ═══════════════════════════════════════════════════════════════════════
# 4. ReasoningTrace
# ═══════════════════════════════════════════════════════════════════════

class TraceStepType(str, Enum):
    """The kind of action a single reasoning-trace step represents."""

    BELIEF_QUERY = "belief_query"
    """Queried the graph for existing beliefs."""

    BELIEF_CREATE = "belief_create"
    """Created a new belief."""

    BELIEF_UPDATE = "belief_update"
    """Updated an existing belief's properties."""

    BELIEF_DECAY = "belief_decay"
    """Applied temporal confidence decay."""

    VECTOR_SEARCH = "vector_search"
    """Performed a similarity search against the vector store."""

    CONTRADICTION_DETECT = "contradiction_detect"
    """Detected a contradiction between claims."""

    CONTRADICTION_RESOLVE = "contradiction_resolve"
    """Resolved a contradiction (split / deprecate / merge)."""

    NEGATIVE_RECORD = "negative_record"
    """Recorded a negative memory guardrail."""

    PROPAGATION = "propagation"
    """Propagated knowledge to another namespace."""

    FUSION = "fusion"
    """Fused graph + vector results into a single answer."""

    CUSTOM = "custom"
    """A custom step type supplied by the caller."""


class ReasoningStep(BaseModel):
    """A single atomic step within a :class:`ReasoningTrace`.

    Steps are append-only and immutable once added to a trace.

    Parameters
    ----------
    step_type:
        The category of action.  See :class:`TraceStepType`.
    description:
        Human-readable summary of what happened.
    belief_ids:
        IDs of beliefs touched in this step.
    contradiction_ids:
        IDs of contradiction events produced/touched.
    negative_ids:
        IDs of negative memories produced/touched.
    vector_ids:
        IDs of vector records returned by a search.
    metadata:
        Arbitrary structured payload for debugging / replay.
    confidence_before:
        Belief confidence before this step (if applicable).
    confidence_after:
        Belief confidence after this step (if applicable).
    duration_ms:
        Wall-clock duration of this step in milliseconds.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    step_type: TraceStepType = Field(
        ...,
        description="The category of action.",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Human-readable summary of what happened.",
    )
    belief_ids: list[str] = Field(
        default_factory=list,
        description="IDs of beliefs touched in this step.",
    )
    contradiction_ids: list[str] = Field(
        default_factory=list,
        description="IDs of contradiction events produced/touched.",
    )
    negative_ids: list[str] = Field(
        default_factory=list,
        description="IDs of negative memories produced/touched.",
    )
    vector_ids: list[str] = Field(
        default_factory=list,
        description="IDs of vector records returned by a search.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary structured payload for debugging / replay.",
    )
    confidence_before: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Belief confidence before this step (if applicable).",
    )
    confidence_after: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Belief confidence after this step (if applicable).",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock duration of this step in milliseconds.",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when this step executed.",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_utc_step(cls, v: datetime | None) -> datetime:
        return _ensure_utc(v) if v is not None else _utcnow()

    @property
    def confidence_delta(self) -> float | None:
        """Return the confidence change in this step, or ``None``."""
        if self.confidence_before is None or self.confidence_after is None:
            return None
        return self.confidence_after - self.confidence_before


class ReasoningTrace(_BaseCognitiveModel):
    """A sequential audit trail mapping how beliefs, vector contexts, and
    confidence shifts interacted during a single agent decision.

    Parameters
    ----------
    trigger:
        What initiated this trace (e.g. a user query, an internal tick).
    trigger_metadata:
        Structured context describing the trigger.
    steps:
        Ordered list of :class:`ReasoningStep` objects.
    conclusion:
        Optional final conclusion or answer produced by the trace.
    final_confidence:
        Aggregate confidence of the conclusion.
    related_belief_ids:
        All belief IDs touched across the entire trace (aggregated).
    related_contradiction_ids:
        All contradiction event IDs touched across the trace.
    """

    id: str = Field(
        default_factory=lambda: _generate_id("trace"),
        description="Unique trace identifier.",
    )

    trigger: str = Field(
        default="manual",
        max_length=256,
        description="What initiated this trace.",
    )
    trigger_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured context describing the trigger.",
    )
    steps: list[ReasoningStep] = Field(
        default_factory=list,
        description="Ordered list of reasoning steps.",
    )
    conclusion: str | None = Field(
        default=None,
        max_length=8192,
        description="Optional final conclusion produced by the trace.",
    )
    final_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Aggregate confidence of the conclusion.",
    )

    # Aggregated IDs are computed lazily via properties to avoid
    # desync between steps and these caches.

    def add_step(self, step: ReasoningStep) -> None:
        """Append a step to this trace and refresh ``updated_at``."""
        self.steps.append(step)
        self.touch()

    @property
    def step_count(self) -> int:
        """Return the number of steps in this trace."""
        return len(self.steps)

    @property
    def is_empty(self) -> bool:
        """``True`` if this trace has no steps."""
        return len(self.steps) == 0

    @property
    def related_belief_ids(self) -> list[str]:
        """Aggregate all belief IDs touched across the trace."""
        seen: set[str] = set()
        result: list[str] = []
        for step in self.steps:
            for bid in step.belief_ids:
                if bid not in seen:
                    seen.add(bid)
                    result.append(bid)
        return result

    @property
    def related_contradiction_ids(self) -> list[str]:
        """Aggregate all contradiction event IDs touched across the trace."""
        seen: set[str] = set()
        result: list[str] = []
        for step in self.steps:
            for cid in step.contradiction_ids:
                if cid not in seen:
                    seen.add(cid)
                    result.append(cid)
        return result

    @property
    def related_negative_ids(self) -> list[str]:
        """Aggregate all negative memory IDs touched across the trace."""
        seen: set[str] = set()
        result: list[str] = []
        for step in self.steps:
            for nid in step.negative_ids:
                if nid not in seen:
                    seen.add(nid)
                    result.append(nid)
        return result

    @property
    def total_duration_ms(self) -> float:
        """Sum of all step durations in milliseconds."""
        return sum(step.duration_ms for step in self.steps)

    def steps_of_type(self, step_type: TraceStepType) -> list[ReasoningStep]:
        """Return only the steps matching *step_type*."""
        return [s for s in self.steps if s.step_type == step_type]

    def confidence_timeline(self) -> list[tuple[int, float | None]]:
        """Return a list of ``(step_index, confidence_after)`` pairs.

        Useful for plotting how confidence evolved over the trace.
        """
        return [
            (idx, step.confidence_after)
            for idx, step in enumerate(self.steps)
            if step.confidence_after is not None
        ]


# ═══════════════════════════════════════════════════════════════════════
# 5. PropagationRecord
# ═══════════════════════════════════════════════════════════════════════

class PropagationRecord(_BaseCognitiveModel):
    """A record of multi-agent knowledge sharing across namespaces using a
    controlled trust reduction factor.

    When an agent learns a new belief, it may propagate that belief to
    other agents (namespaces).  Because cross-agent trust is imperfect,
    the propagated confidence is reduced by a ``trust_factor`` in
    ``[0.0, 1.0]``.

    Parameters
    ----------
    source_namespace:
        Namespace of the sending agent.
    target_namespace:
        Namespace of the receiving agent.
    belief_id:
        ID of the belief being propagated.
    belief_claim:
        A snapshot of the belief's claim at propagation time.
    original_confidence:
        The belief's confidence in the source namespace.
    trust_factor:
        Multiplier applied to the confidence during propagation.
        ``1.0`` = full trust; ``0.5`` = half confidence.
    propagated_confidence:
        ``original_confidence * trust_factor`` (computed automatically).
    status:
        Outcome of the propagation.  See :class:`PropagationStatus`.
    related_trace_id:
        ID of the reasoning trace that initiated this propagation.
    """

    id: str = Field(
        default_factory=lambda: _generate_id("propagation"),
        description="Unique propagation record identifier.",
    )

    source_namespace: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Namespace of the sending agent.",
    )
    target_namespace: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Namespace of the receiving agent.",
    )
    belief_id: str = Field(
        ...,
        min_length=1,
        description="ID of the belief being propagated.",
    )
    belief_claim: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="Snapshot of the belief's claim at propagation time.",
    )
    original_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The belief's confidence in the source namespace.",
    )
    trust_factor: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence multiplier applied during propagation.",
    )
    propagated_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="original_confidence * trust_factor.",
    )
    status: PropagationStatus = Field(
        default=PropagationStatus.PENDING,
        description="Outcome of the propagation.",
    )
    related_trace_id: str | None = Field(
        default=None,
        description="ID of the reasoning trace that initiated this propagation.",
    )
    attempted_at: datetime | None = Field(
        default=None,
        description="UTC timestamp of the first delivery attempt.",
    )
    delivered_at: datetime | None = Field(
        default=None,
        description="UTC timestamp of successful delivery.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of delivery retries attempted.",
    )
    failure_reason: str | None = Field(
        default=None,
        description="Reason for failure if status is FAILED/REJECTED.",
    )

    @model_validator(mode="after")
    def _compute_propagated_confidence(self) -> PropagationRecord:
        """Auto-compute ``propagated_confidence`` from the inputs."""
        computed = self.original_confidence * self.trust_factor
        computed = max(0.0, min(1.0, computed))
        # Only overwrite if not explicitly set to a meaningful value
        if self.propagated_confidence == 0.0 or self.propagated_confidence != computed:
            object.__setattr__(self, "propagated_confidence", computed)
        return self

    @field_validator("attempted_at", "delivered_at")
    @classmethod
    def _ensure_utc_optional(cls, v: datetime | None) -> datetime | None:
        return _ensure_utc(v)

    @model_validator(mode="after")
    def _validate_namespaces_differ(self) -> PropagationRecord:
        """Source and target namespaces must differ."""
        if self.source_namespace == self.target_namespace:
            raise ValueError(
                "source_namespace and target_namespace must differ for "
                "a propagation record."
            )
        return self

    @property
    def confidence_loss(self) -> float:
        """Return the confidence lost due to trust reduction."""
        return max(0.0, self.original_confidence - self.propagated_confidence)

    @property
    def is_complete(self) -> bool:
        """``True`` if the propagation has reached a terminal state."""
        return self.status in (
            PropagationStatus.DELIVERED,
            PropagationStatus.REJECTED,
            PropagationStatus.FAILED,
        )

    def mark_delivered(self) -> None:
        """Mark this propagation as successfully delivered."""
        self.status = PropagationStatus.DELIVERED
        self.delivered_at = _utcnow()
        self.touch()

    def mark_rejected(self, reason: str = "trust_threshold") -> None:
        """Mark this propagation as rejected."""
        self.status = PropagationStatus.REJECTED
        self.failure_reason = reason
        self.touch()

    def mark_failed(self, reason: str, max_retries: int = 3) -> bool:
        """Record a failed delivery attempt.

        Increments ``retry_count`` and returns ``True`` if the engine
        should retry (i.e. retries remain), or ``False`` if the
        propagation is now terminally failed.
        """
        self.retry_count += 1
        self.failure_reason = reason
        if self.retry_count >= max_retries:
            self.status = PropagationStatus.FAILED
            self.touch()
            return False
        self.touch()
        return True


# ═══════════════════════════════════════════════════════════════════════
# Public re-exports
# ═══════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    # Enums
    "BeliefStatus",
    "ContradictionResolution",
    "NegativeMemorySeverity",
    "PropagationStatus",
    "TraceStepType",
    # Base
    "ReasoningStep",
    # Models
    "BeliefNode",
    "ContradictionEvent",
    "NegativeMemory",
    "ReasoningTrace",
    "PropagationRecord",
]
