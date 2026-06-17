"""Pydantic v2 data models for the context compression layer.

This module defines the serialisable schemas produced and consumed by the
compression pipeline.  Every model is designed to round-trip cleanly
through JSON, making it straightforward to persist into ChromaDB metadata
or Kuzu graph properties.

Models
-------
1. :class:`MemorySnapshot` — a compressed representation of a raw memory
   fragment, retaining a summary, extracted entities, keywords, and a
   back-reference to the original text.
2. :class:`LogCompressionOutput` — the structured result of compressing a
   batch of log entries.
3. :class:`ConversationCompressionOutput` — the structured result of
   compressing a multi-turn conversation.
4. :class:`CompressionMetrics` — quantitative measurements for a single
   compression operation (token counts, ratio, retrieval frequency).

Design rationale
-----------------
These models deliberately **do not** extend
:class:`neuromem.core.models._BaseCognitiveModel`.  They are output-only
data-transfer objects (DTOs) produced by the compression pipeline rather
than first-class cognitive records managed by the engine.  This keeps the
compression layer decoupled from engine lifecycle concerns (namespaces,
decay, contradiction tracking).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _utcnow() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    """Coerce a naive datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════
# 1. MemorySnapshot
# ═══════════════════════════════════════════════════════════════════════

class MemorySnapshot(BaseModel):
    """A compressed representation of a raw memory fragment.

    The snapshot captures the essential information from a piece of
    raw memory (e.g. a long conversation turn or log burst) in a compact
    form suitable for vector search and graph storage.  The
    ``raw_reference`` field allows the original text to be retrieved from
    a backing store when full fidelity is needed.

    Parameters
    ----------
    id : str
        Globally unique identifier for this snapshot.
    summary : str
        Human-readable prose summary of the compressed content.
    keywords : list[str]
        High-signal tokens extracted for indexing and retrieval.
    entities : list[str]
        Named entities (people, organisations, locations, concepts)
        discovered in the original content.
    importance : float
        Normalised importance score in the range ``[0.0, 1.0]``,
        derived from confidence, frequency, or explicit weighting.
    compression_ratio : float
        Ratio of compressed size to original size.  A value of ``0.25``
        means the snapshot is 25 % of the original token count.
    raw_reference : str
        An opaque reference (e.g. storage key, URI, or hash) pointing to
        the original uncompressed content.
    created_at : datetime
        UTC timestamp of when this snapshot was produced.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    id: str = Field(..., min_length=1, description="Globally unique identifier for this snapshot.")
    summary: str = Field(
        ..., min_length=1, description="Human-readable prose summary of the compressed content."
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="High-signal tokens extracted for indexing and retrieval.",
    )
    entities: list[str] = Field(
        default_factory=list,
        description=(
            "Named entities (people, organisations, locations, concepts) "
            "discovered in the original content."
        ),
    )
    importance: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Normalised importance score in [0.0, 1.0], derived from "
            "confidence, frequency, or explicit weighting."
        ),
    )
    compression_ratio: float = Field(
        ..., gt=0.0, le=1.0,
        description=(
            "Ratio of compressed size to original size.  "
            "0.25 means the snapshot is 25 % of the original token count."
        ),
    )
    raw_reference: str = Field(
        ..., min_length=1,
        description=(
            "Opaque reference (storage key, URI, or hash) pointing to "
            "the original uncompressed content."
        ),
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of when this snapshot was produced.",
    )

    @model_validator(mode="after")
    def _normalise_timestamp(self) -> MemorySnapshot:
        """Ensure ``created_at`` is timezone-aware UTC."""
        object.__setattr__(self, "created_at", _ensure_utc(self.created_at))
        return self


# ═══════════════════════════════════════════════════════════════════════
# 2. LogCompressionOutput
# ═══════════════════════════════════════════════════════════════════════

_VALID_SEVERITIES = frozenset({"debug", "info", "warning", "error", "critical"})


class LogCompressionOutput(BaseModel):
    """Structured result of compressing a batch of log entries.

    The pipeline processes raw log lines and emits a concise summary,
    a deduplicated list of error messages, an overall severity
    classification, and a timeline of key events.

    Parameters
    ----------
    summary : str
        Prose summary of the log batch activity.
    errors : list[str]
        Deduplicated error messages extracted from the log entries.
    severity : str
        Highest severity level observed in the batch.  Must be one of
        ``debug``, ``info``, ``warning``, ``error``, ``critical``.
    key_events : list[str]
        Chronologically ordered descriptions of notable events.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    summary: str = Field(
        ..., min_length=1,
        description="Prose summary of the log batch activity.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Deduplicated error messages extracted from the log entries.",
    )
    severity: str = Field(
        ..., min_length=1,
        description=(
            "Highest severity level observed in the batch. "
            "One of: debug, info, warning, error, critical."
        ),
    )
    key_events: list[str] = Field(
        default_factory=list,
        description="Chronologically ordered descriptions of notable events.",
    )

    @field_validator("severity")
    @classmethod
    def _validate_severity(cls, v: str) -> str:
        """Ensure severity is a recognised log level."""
        v_lower = v.lower()
        if v_lower not in _VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(_VALID_SEVERITIES)}, "
                f"got {v_lower!r}"
            )
        return v_lower


# ═══════════════════════════════════════════════════════════════════════
# 3. ConversationCompressionOutput
# ═══════════════════════════════════════════════════════════════════════

class ConversationCompressionOutput(BaseModel):
    """Structured result of compressing a multi-turn conversation.

    The pipeline analyses the conversation transcript and extracts a
    summary, factual statements, outstanding action items, decisions
    reached, and named entities referenced.

    Parameters
    ----------
    summary : str
        Prose summary capturing the overall conversation arc.
    important_facts : list[str]
        Factual claims or pieces of information exchanged.
    open_tasks : list[str]
        Action items or tasks that remain unresolved.
    decisions : list[str]
        Explicit decisions or conclusions reached during the conversation.
    entities : list[str]
        Named entities mentioned in the conversation.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    summary: str = Field(
        ..., min_length=1,
        description="Prose summary capturing the overall conversation arc.",
    )
    important_facts: list[str] = Field(
        default_factory=list,
        description="Factual claims or pieces of information exchanged.",
    )
    open_tasks: list[str] = Field(
        default_factory=list,
        description="Action items or tasks that remain unresolved.",
    )
    decisions: list[str] = Field(
        default_factory=list,
        description="Explicit decisions or conclusions reached during the conversation.",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Named entities mentioned in the conversation.",
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. CompressionMetrics
# ═══════════════════════════════════════════════════════════════════════

class CompressionMetrics(BaseModel):
    """Quantitative measurements for a single compression operation.

    Captures the before/after token counts, the resulting compression
    ratio, and how many times the compressed artefact has been retrieved
    since creation.  These metrics are stored alongside the compressed
    content to enable adaptive compression tuning and cache eviction.

    Parameters
    ----------
    tokens_before : int
        Token count of the original (uncompressed) content.
    tokens_after : int
        Token count of the compressed representation.
    compression_ratio : float
        ``tokens_after / tokens_before``.  Must be in ``(0.0, 1.0]``.
    retrieval_count : int
        Number of times the compressed artefact has been recalled or
        queried since creation.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    tokens_before: int = Field(
        ..., ge=1,
        description="Token count of the original (uncompressed) content.",
    )
    tokens_after: int = Field(
        ..., ge=0,
        description="Token count of the compressed representation.",
    )
    compression_ratio: float = Field(
        ..., gt=0.0, le=1.0,
        description="``tokens_after / tokens_before``.  Must be in (0.0, 1.0].",
    )
    retrieval_count: int = Field(
        ..., ge=0,
        description=(
            "Number of times the compressed artefact has been recalled "
            "or queried since creation."
        ),
    )

    @model_validator(mode="after")
    def _check_consistent_ratio(self) -> CompressionMetrics:
        """Verify that ``compression_ratio`` matches the token counts."""
        if self.tokens_before > 0:
            expected = self.tokens_after / self.tokens_before
            if abs(expected - self.compression_ratio) > 1e-6:
                raise ValueError(
                    f"compression_ratio ({self.compression_ratio:.4f}) does not "
                    f"match tokens_after/tokens_before ({expected:.4f})."
                )
        return self


# ═══════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    "MemorySnapshot",
    "LogCompressionOutput",
    "ConversationCompressionOutput",
    "CompressionMetrics",
]
