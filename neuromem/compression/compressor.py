"""The top-level orchestration layer for context compression.

The :class:`CompressionEngine` is the public entry point of the
compression sub-package.  It glues together the four collaborating
components:

1. :class:`~neuromem.compression.router.ContentRouter` — classifies the
   incoming raw text into one of six content types.
2. :class:`~neuromem.compression.summarizer.ContextCompressor` — applies
   the domain-specific compression strategy (logs / conversation / RAG /
   code) selected by the router's verdict.
3. :class:`~neuromem.compression.reversible_store.ReversibleStore` —
   durably archives the **original, uncompressed** text so every
   snapshot can be expanded back to full fidelity at any time.
4. :class:`~neuromem.compression.models.MemorySnapshot` — the compact,
   searchable artefact handed back to the caller.

In addition to producing :class:`MemorySnapshot` objects, the engine
maintains a running set of cumulative metrics — ``tokens_saved``,
``compression_ratio``, and ``stored_memories_count`` — exposed in
real time through :meth:`CompressionEngine.get_stats`.

Design rationale
----------------
- **Strategy selection is data-driven.**  A class-level ``_STRATEGIES``
  mapping wires each :data:`~neuromem.compression.router.ContentType`
  to the :class:`~neuromem.compression.summarizer.ContextCompressor`
  method that should handle it.  Adding a new strategy is a one-line
  change to the table rather than another ``if/elif`` branch.
- **Heterogeneous outputs are normalised once.**  The four strategic
  methods return three different shapes (``LogCompressionOutput``,
  ``ConversationCompressionOutput``, or a plain ``str`` for code/RAG).
  A single private normaliser collapses them into the
  ``summary / keywords / entities / importance`` tuple that
  :class:`MemorySnapshot` needs, so the orchestration path stays flat.
- **Token estimation is dependency-free.**  NeuroMem does not pin
  ``tiktoken``, so :func:`estimate_tokens` provides a deterministic
  whitespace + punctuation heuristic that tracks BPE-sized token counts
  closely enough for budgeting, metrics, and tests, while remaining
  fully offline and reproducible.
- **The engine never loses data.**  Every accepted input is persisted to
  the :class:`ReversibleStore` *before* the snapshot is returned, and the
  snapshot's ``raw_reference`` carries the very id used to store it.
- **Statistics are computed, not guessed.**  ``stored_memories_count``
  is sourced from the backing :class:`ReversibleStore` when possible and
  falls back to an in-process counter for short-lived / in-memory
  stores, so :meth:`get_stats` reflects reality either way.

Example
-------
::

    from neuromem.compression import CompressionEngine, ReversibleStore

    with ReversibleStore("./raw_archive") as store:
        engine = CompressionEngine(reversible_store=store)
        snapshot = engine.compress("User: hello\\nAssistant: hi there")
        print(snapshot.summary)
        print(engine.get_stats())
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Callable

from loguru import logger

from neuromem.compression.models import (
    ConversationCompressionOutput,
    LogCompressionOutput,
    MemorySnapshot,
)
from neuromem.compression.reversible_store import ReversibleStore
from neuromem.compression.router import ContentRouter, ContentType
from neuromem.compression.summarizer import (
    BaseLLMProvider,
    ContextCompressor,
)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

_SNAPSHOT_PREFIX = "snap"
"""Semantic prefix for memory-snapshot ids, mirroring ``engine.py``."""

_DEFAULT_IMPORTANCE = 0.5
"""Importance assigned when a strategy yields no explicit signal."""

_RAG_IMPORTANCE = 0.6
"""Default importance for free-form prose / RAG / markdown content."""

_LOG_SEVERITY_IMPORTANCE: dict[str, float] = {
    "debug": 0.2,
    "info": 0.35,
    "warning": 0.55,
    "error": 0.75,
    "critical": 0.95,
}
"""Map a log batch's dominant severity to a normalised importance score."""

# Default per-metric snapshot returned before any compression has run.
_ZERO_STATS: dict[str, float | int] = {
    "tokens_saved": 0,
    "compression_ratio": 0.0,
    "stored_memories_count": 0,
}

# A regex that matches maximal runs of *word* characters.  Used by the
# token estimator to split CJK-free text into BPE-ish word fragments.
_RE_WORD = re.compile(r"\S+")


# ═══════════════════════════════════════════════════════════════════════
# Token estimation (dependency-free)
# ═══════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """Return a deterministic, offline estimate of *text*'s token count.

    NeuroMem does not depend on ``tiktoken`` or any specific model's
    tokenizer, so this function provides a stable proxy that:

    * Counts whitespace-separated word fragments **and** punctuation
      runs as separate tokens (mirroring how BPE tokenisers split on
      word/punctuation boundaries).
    * Charges roughly ``1 token per 4 characters`` for CJK and other
      non-space-delimited scripts, so multi-lingual content still
      scales sensibly.

    The result is monotonic in content length and fully reproducible,
    which is what the metrics layer (ratios, ``tokens_saved``) and the
    test suite actually need.  ``tiktoken`` can be dropped in later by
    swapping this single function without touching call sites.

    Parameters
    ----------
    text:
        The string to estimate.  ``None`` / empty → ``0``.

    Returns
    -------
    int
        A non-negative token estimate.
    """
    if not text:
        return 0
    count = 0
    for fragment in _RE_WORD.findall(text):
        # Each whitespace-delimited fragment contributes its word
        # token(s) plus the punctuation tokens attached to it.
        count += 1
        # Split off leading/trailing punctuation as separate tokens,
        # but never let the per-fragment charge fall below 1.
        inner = re.sub(r"^\W+|\W+$", "", fragment)
        if inner:
            # Heuristic: word-internal punctuation (dashes, dots in
            # identifiers) tends to break into extra tokens.
            count += len(re.findall(r"[\-_/]", inner))
        # CJK / non-ASCII surcharge: ~1 token per 4 code points, which
        # matches common BPE behaviour for languages without spaces.
        non_ascii = sum(1 for ch in fragment if ord(ch) > 127)
        if non_ascii:
            count += non_ascii // 4
    return max(count, 1) if text.strip() else 0


# ═══════════════════════════════════════════════════════════════════════
# Internal result container
# ═══════════════════════════════════════════════════════════════════════

class _NormalisedResult:
    """Uniform view over the heterogeneous strategy outputs.

    The four strategic methods on :class:`ContextCompressor` return one
    of three shapes (two Pydantic DTOs and a plain ``str``).  This tiny
    dataclass-flavoured container is the single place where those shapes
    are collapsed into the ``(summary, keywords, entities, importance)``
    tuple consumed by :class:`MemorySnapshot`.
    """

    __slots__ = ("summary", "keywords", "entities", "importance")

    def __init__(
        self,
        *,
        summary: str,
        keywords: list[str] | None = None,
        entities: list[str] | None = None,
        importance: float = _DEFAULT_IMPORTANCE,
    ) -> None:
        self.summary: str = summary
        self.keywords: list[str] = keywords or []
        self.entities: list[str] = entities or []
        self.importance: float = importance


# ═══════════════════════════════════════════════════════════════════════
# CompressionEngine
# ═══════════════════════════════════════════════════════════════════════

class CompressionEngine:
    """Top-level orchestrator for the context-compression pipeline.

    The engine ingests raw text, routes it to the right strategy,
    persists the original to the reversible store, and returns a
    :class:`MemorySnapshot` plus token-tracking metrics.

    Parameters
    ----------
    reversible_store:
        The :class:`ReversibleStore` used to durably archive the
        original, uncompressed content.  It **must** already be
        initialised (``initialize()`` called) when :meth:`compress` is
        invoked; pass an already-open store or use it as a context
        manager.
    content_router:
        Optional :class:`ContentRouter` instance.  A default one is
        created when omitted.
    context_compressor:
        Optional :class:`ContextCompressor` instance.  A default one
        (using the offline :class:`MockLLMProvider`) is created when
        omitted.
    llm:
        Convenience parameter: an :class:`BaseLLMProvider` to wire into
        a freshly-built :class:`ContextCompressor`.  Ignored when
        *context_compressor* is supplied explicitly.

    Example
    -------
    ::

        with ReversibleStore("./archive") as store:
            engine = CompressionEngine(reversible_store=store)
            snap = engine.compress(some_long_text)
            assert engine.get_stats()["stored_memories_count"] >= 1
    """

    # Maps each content type to the ContextCompressor method that
    # handles it.  ``text``/``markdown``/``json`` are funnelled through
    # ``compress_rag`` (treating the input as a single retrieval chunk),
    # since they have no dedicated structural strategy.  Keeping this as
    # a class-level table makes the dispatch a pure lookup.
    _STRATEGIES: dict[ContentType, str] = {
        "logs": "compress_logs",
        "conversation": "compress_conversation",
        "code": "compress_code",
        "json": "compress_rag",
        "markdown": "compress_rag",
        "text": "compress_rag",
    }

    def __init__(
        self,
        reversible_store: ReversibleStore,
        *,
        content_router: ContentRouter | None = None,
        context_compressor: ContextCompressor | None = None,
        llm: BaseLLMProvider | None = None,
    ) -> None:
        if reversible_store is None:
            raise ValueError("reversible_store is required")
        # Refuse to operate on a store that has not been initialised;
        # every downstream call would raise anyway, but failing fast
        # here gives a much clearer error.
        if not reversible_store.is_ready:
            raise ValueError(
                "reversible_store must be initialised before use; "
                "call initialize() or enter it as a context manager",
            )

        self._store: ReversibleStore = reversible_store
        self._router: ContentRouter = content_router or ContentRouter()
        self._compressor: ContextCompressor = (
            context_compressor
            if context_compressor is not None
            else ContextCompressor(llm=llm)
        )

        # ── Cumulative statistics ──────────────────────────────────
        # Two of the three metrics are derived from the engine's own
        # running totals; ``stored_memories_count`` is sourced from the
        # backing store when possible and falls back to a local
        # counter for ephemeral / in-memory scenarios.
        self._total_tokens_before: int = 0
        self._total_tokens_after: int = 0
        self._tokens_saved: int = 0
        # Local counter used as a fallback when the store cannot report
        # its own count (e.g. a mock or in-memory substitute).
        self._local_memory_count: int = 0

    # ── Public properties ───────────────────────────────────────────

    @property
    def reversible_store(self) -> ReversibleStore:
        """The backing :class:`ReversibleStore` for original content."""
        return self._store

    @property
    def router(self) -> ContentRouter:
        """The :class:`ContentRouter` used for content-type detection."""
        return self._router

    @property
    def compressor(self) -> ContextCompressor:
        """The :class:`ContextCompressor` applying per-type strategies."""
        return self._compressor

    # ── Core API ────────────────────────────────────────────────────

    def compress(
        self,
        content: str,
        *,
        memory_id: str | None = None,
        importance: float | None = None,
    ) -> MemorySnapshot:
        """Compress *content* into a :class:`MemorySnapshot`.

        The pipeline runs in four fixed stages:

        1. **Route** — :meth:`ContentRouter.detect_content_type` decides
           which strategic method to call.
        2. **Compress** — the matching method on
           :class:`ContextCompressor` produces the compact representation.
        3. **Store** — the *original, uncompressed* text is persisted to
           the :class:`ReversibleStore` so it can be recovered later.
        4. **Snapshot** — the compressed output is normalised into a
           :class:`MemorySnapshot` whose ``raw_reference`` points back
           to the stored original.

        Cumulative statistics (``tokens_saved``, ``compression_ratio``,
        ``stored_memories_count``) are updated as a side effect.

        Parameters
        ----------
        content:
            The raw text to compress.  Must be a non-empty string.
        memory_id:
            Optional explicit identifier for the stored original.  A
            ``snap_…`` id is generated when omitted.  Useful when the
            caller wants the snapshot id to match an existing graph /
            vector node.
        importance:
            Optional override for the snapshot's importance score in
            ``[0.0, 1.0]``.  When omitted, importance is derived from
            the strategy's output (e.g. log severity, entity density).

        Returns
        -------
        MemorySnapshot
            The generated snapshot.  Its ``raw_reference`` is the id
            under which the original was archived.

        Raises
        ------
        ValueError
            If *content* is empty / whitespace-only, or *importance*
            is out of range.
        NeuroMemError
            If storing the original content fails.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        if importance is not None and not 0.0 <= importance <= 1.0:
            raise ValueError("importance must be in [0.0, 1.0]")

        # ── Stage 1: Route ──────────────────────────────────────────
        content_type = self._router.detect_content_type(content)
        logger.debug("CompressionEngine: routed content as {!r}", content_type)

        # ── Stage 2: Compress via the selected strategy ─────────────
        result = self._run_strategy(content_type, content)

        # ── Stage 3: Persist the original (reversibility backbone) ──
        snapshot_id = memory_id or self._generate_id()
        self._store.store_original(snapshot_id, content)
        # Mirror the local counter so get_stats() is accurate even for
        # stores that do not implement count().
        self._local_memory_count += 1

        # ── Stage 4: Build the snapshot + update metrics ────────────
        effective_importance = (
            importance if importance is not None else result.importance
        )
        # Clamp defensively: a misbehaving strategy should never produce
        # a snapshot that fails Pydantic validation downstream.
        effective_importance = max(0.0, min(1.0, effective_importance))

        tokens_before = estimate_tokens(content)
        tokens_after = estimate_tokens(result.summary)
        ratio = self._safe_ratio(tokens_after, tokens_before)

        snapshot = MemorySnapshot(
            id=snapshot_id,
            summary=result.summary,
            keywords=result.keywords,
            entities=result.entities,
            importance=effective_importance,
            compression_ratio=ratio,
            raw_reference=snapshot_id,
        )

        self._update_stats(tokens_before, tokens_after)

        logger.info(
            "CompressionEngine: compressed {} → snapshot {} "
            "(type={}, {}→{} tokens, ratio={:.3f})",
            content_type, snapshot_id, content_type,
            tokens_before, tokens_after, ratio,
        )
        return snapshot

    # ── Statistics ──────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return real-time cumulative compression statistics.

        The returned dict always contains exactly three keys:

        - ``tokens_saved`` (``int``): cumulative tokens removed across
          every successful :meth:`compress` call
          (``Σ(tokens_before − tokens_after)``).
        - ``compression_ratio`` (``float`` in ``[0.0, 1.0]``): running
          ``Σ(tokens_after) / Σ(tokens_before)``.  ``0.0`` before any
          compression has run.
        - ``stored_memories_count`` (``int``): number of originals held
          by the backing store.  Sourced from
          :meth:`ReversibleStore.count` when available, otherwise from
          the engine's own counter.

        Returns
        -------
        dict
            A fresh dictionary; mutating it does not affect the engine.
        """
        ratio = (
            self._total_tokens_after / self._total_tokens_before
            if self._total_tokens_before > 0
            else 0.0
        )
        # Guard against float drift pushing the ratio a hair above 1.0
        # after many accumulations (sum-of-rounded values).
        ratio = max(0.0, min(1.0, ratio))

        return {
            "tokens_saved": self._tokens_saved,
            "compression_ratio": ratio,
            "stored_memories_count": self._stored_memory_count(),
        }

    def reset_stats(self) -> None:
        """Reset the cumulative statistics tracked by this engine.

        This affects only the in-process counters — it does **not**
        delete anything from the backing :class:`ReversibleStore`.
        Useful for tests and for scoping metrics to a session.
        """
        self._total_tokens_before = 0
        self._total_tokens_after = 0
        self._tokens_saved = 0
        # The local counter is reset too so that get_stats() reflects
        # only post-reset activity for ephemeral stores.  Real stores
        # keep reporting their true on-disk count.
        self._local_memory_count = 0
        logger.debug("CompressionEngine: statistics reset")

    # ── Strategy dispatch ───────────────────────────────────────────

    def _run_strategy(
        self,
        content_type: ContentType,
        content: str,
    ) -> _NormalisedResult:
        """Dispatch *content* to the strategy registered for *content_type*.

        Looks up the method name in :data:`_STRATEGIES`, invokes it on
        the :class:`ContextCompressor`, and normalises the heterogeneous
        return value into a :class:`_NormalisedResult`.

        Raises
        ------
        RuntimeError
            If the dispatch table is somehow missing a strategy for the
            resolved content type (indicates a programming error).
        """
        method_name = self._STRATEGIES.get(content_type)
        if method_name is None:
            # Defensive: every ContentType should be mapped.  Falling
            # back to RAG keeps the pipeline functional even so.
            logger.warning(
                "No strategy registered for content type {!r}; "
                "falling back to compress_rag",
                content_type,
            )
            method_name = "compress_rag"

        method: Callable[..., Any] = getattr(self._compressor, method_name)

        # The RAG strategy takes a list of chunks; everything else takes
        # a single string.  We wrap prose-like inputs as a one-element
        # chunk list so compress_rag's dedup/citation machinery runs.
        if method_name == "compress_rag":
            raw_output = method([content])
        else:
            raw_output = method(content)

        return self._normalise(content_type, raw_output)

    # ── Output normalisation ────────────────────────────────────────

    @staticmethod
    def _normalise(
        content_type: ContentType,
        raw_output: Any,
    ) -> _NormalisedResult:
        """Collapse a strategy output into a :class:`_NormalisedResult`.

        Each strategy returns a different shape; this helper is the
        single choke-point that adapts them:

        * ``compress_logs``         → :class:`LogCompressionOutput`
        * ``compress_conversation`` → :class:`ConversationCompressionOutput`
        * ``compress_code``         → ``str`` (markdown skeleton)
        * ``compress_rag``          → ``str`` (merged passage)
        """
        if isinstance(raw_output, LogCompressionOutput):
            # Keywords = deduplicated error snippets (truncated) plus the
            # dominant severity, giving retrievers something to index on.
            keywords: list[str] = [raw_output.severity.upper()]
            keywords.extend(
                _truncate(err, 60) for err in raw_output.errors[:8]
            )
            importance = _LOG_SEVERITY_IMPORTANCE.get(
                raw_output.severity, _DEFAULT_IMPORTANCE,
            )
            return _NormalisedResult(
                summary=raw_output.summary,
                keywords=keywords,
                entities=list(raw_output.key_events[:10]),
                importance=importance,
            )

        if isinstance(raw_output, ConversationCompressionOutput):
            # Entities come straight from the extractor; keywords are
            # the most salient facts + decisions.
            keywords = list(raw_output.important_facts[:8])
            keywords.extend(raw_output.decisions[:4])
            importance = _DEFAULT_IMPORTANCE
            if raw_output.decisions:
                # A conversation that produced explicit decisions is
                # worth slightly more than one that only exchanged facts.
                importance = 0.7
            return _NormalisedResult(
                summary=raw_output.summary,
                keywords=keywords,
                entities=list(raw_output.entities),
                importance=importance,
            )

        if isinstance(raw_output, str):
            # code / rag / text / markdown / json all return a string.
            summary = raw_output.strip() or "(empty)"
            return _NormalisedResult(
                summary=summary,
                keywords=_extract_keywords(summary),
                entities=[],
                importance=_RAG_IMPORTANCE,
            )

        # Unknown shape — degrade gracefully rather than crash.
        logger.warning(
            "Unrecognised strategy output type {!r} for content {!r}; "
            "stringifying as a fallback",
            type(raw_output).__name__, content_type,
        )
        return _NormalisedResult(
            summary=str(raw_output).strip() or "(empty)",
            importance=_DEFAULT_IMPORTANCE,
        )

    # ── Metrics helpers ─────────────────────────────────────────────

    def _update_stats(self, tokens_before: int, tokens_after: int) -> None:
        """Accumulate the per-call token deltas into the running totals."""
        self._total_tokens_before += tokens_before
        self._total_tokens_after += tokens_after
        self._tokens_saved += max(0, tokens_before - tokens_after)

    def _stored_memory_count(self) -> int:
        """Return the number of stored originals, preferring the store's own count."""
        try:
            count = self._store.count()
            # Stores that don't track state may legitimately return 0
            # even after writes; in that case trust our local counter.
            if count > 0:
                return count
        except Exception as exc:  # noqa: BLE001 — best-effort metric
            logger.debug(
                "ReversibleStore.count() unavailable ({}); "
                "using local counter",
                exc,
            )
        return self._local_memory_count

    @staticmethod
    def _safe_ratio(after: int, before: int) -> float:
        """Return ``after / before`` clamped to the snapshot's legal range.

        ``MemorySnapshot`` requires ``compression_ratio`` in ``(0.0, 1.0]``.
        When compression yields an empty summary (``after == 0``) we
        floor the ratio to a tiny positive value so the snapshot stays
        valid while still signalling near-total compression.
        """
        if before <= 0:
            return 1.0
        ratio = after / before
        if ratio <= 0.0:
            return 1e-6
        return min(1.0, ratio)

    # ── Misc helpers ────────────────────────────────────────────────

    @staticmethod
    def _generate_id() -> str:
        """Generate a ``snap_<hex>`` id, matching engine.py's convention."""
        return f"{_SNAPSHOT_PREFIX}_{uuid.uuid4().hex[:16]}"


# ═══════════════════════════════════════════════════════════════════════
# Pure helpers (module-level for testability)
# ═══════════════════════════════════════════════════════════════════════

_RE_KEYWORD_CANDIDATES = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
"""Matches candidate keyword tokens: alphanumeric identifiers of length ≥ 3."""

_KEYWORD_STOPWORDS: frozenset[str] = frozenset({
    # Generic English filler that carries no indexing signal.
    "the", "and", "for", "that", "this", "with", "from", "have", "has",
    "are", "was", "were", "been", "being", "will", "would", "could",
    "should", "shall", "may", "might", "must", "can", "into", "your",
    "you", "they", "them", "their", "what", "which", "when", "where",
    "there", "here", "about", "after", "before", "between", "during",
    "through", "over", "under", "than", "then", "once",
    # Generic code/markdown scaffold words.
    "def", "class", "import", "return", "true", "false", "none",
})
"""Low-information tokens excluded from keyword extraction."""


def _truncate(text: str, max_len: int) -> str:
    """Return *text* trimmed to *max_len* characters with an ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _extract_keywords(text: str, *, max_keywords: int = 10) -> list[str]:
    """Extract a small, deduplicated keyword list from *text*.

    Uses a frequency-ranked, stopword-filtered token scan.  Order is
    preserved by first occurrence, which keeps keywords stable and
    deterministic for a given input — important for snapshot
    reproducibility in tests.
    """
    if not text:
        return []
    seen: set[str] = set()
    keywords: list[str] = []
    for match in _RE_KEYWORD_CANDIDATES.finditer(text):
        token = match.group().lower()
        if token in _KEYWORD_STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


# ═══════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    "CompressionEngine",
    "estimate_tokens",
]
