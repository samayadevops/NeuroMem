"""Tests for compression statistics and token accounting.

The :class:`CompressionEngine` exposes cumulative metrics through
``get_stats()``:

* ``tokens_saved`` — ``Σ(tokens_before − tokens_after)`` across calls.
* ``compression_ratio`` — running ``Σ(tokens_after) / Σ(tokens_before)``,
  clamped to ``[0.0, 1.0]``.
* ``stored_memories_count`` — entries held by the backing store.

Because the engine derives these from the public :func:`estimate_tokens`
helper, the tests **reproduce the arithmetic independently** and assert
exact equality (no floating-point tolerance beyond ``1e-9``), which is the
definition of "calculations are exact".  Per-snapshot
``compression_ratio`` and the :class:`CompressionMetrics` model's internal
consistency check are covered too.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuromem.compression import CompressionEngine, ReversibleStore, estimate_tokens
from neuromem.compression.models import CompressionMetrics


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture()
def engine(tmp_path) -> CompressionEngine:
    """Return a CompressionEngine backed by a fresh tmp store."""
    with ReversibleStore(tmp_path / "archive") as store:
        yield CompressionEngine(reversible_store=store)


# ═══════════════════════════════════════════════════════════════════════
# estimate_tokens — the basis of all token math
# ═══════════════════════════════════════════════════════════════════════

class TestEstimateTokens:
    def test_empty_returns_zero(self) -> None:
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0  # type: ignore[arg-type]

    def test_whitespace_only_returns_zero(self) -> None:
        assert estimate_tokens("   ") == 0
        assert estimate_tokens("\n\n\t  ") == 0

    def test_deterministic(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        assert estimate_tokens(text) == estimate_tokens(text)

    def test_monotonic_in_length(self) -> None:
        short = "one two three"
        long_ = "one two three four five six seven eight nine ten"
        assert estimate_tokens(short) < estimate_tokens(long_)

    def test_cjk_surcharged(self) -> None:
        """Non-ASCII content incurs the ~1-token-per-4-chars surcharge."""
        ascii_tokens = estimate_tokens("aaaa")
        cjk_tokens = estimate_tokens("世界测试")  # 4 CJK chars
        assert cjk_tokens >= ascii_tokens

    def test_punctuation_counts_as_tokens(self) -> None:
        # "word." -> base 1 + trailing punctuation handled; >= 1 always.
        assert estimate_tokens("hello.") >= 1

    def test_minimum_one_for_non_whitespace(self) -> None:
        assert estimate_tokens("x") >= 1


# ═══════════════════════════════════════════════════════════════════════
# Initial stats (zero state)
# ═══════════════════════════════════════════════════════════════════════

class TestInitialStats:
    def test_zero_before_any_compression(self, engine: CompressionEngine) -> None:
        stats = engine.get_stats()
        assert stats["tokens_saved"] == 0
        assert stats["compression_ratio"] == 0.0
        assert stats["stored_memories_count"] == 0

    def test_stats_keys_contract(self, engine: CompressionEngine) -> None:
        stats = engine.get_stats()
        assert set(stats) == {"tokens_saved", "compression_ratio", "stored_memories_count"}

    def test_stats_returns_fresh_dict(self, engine: CompressionEngine) -> None:
        s1 = engine.get_stats()
        s1["tokens_saved"] = 99999  # mutate the copy
        s2 = engine.get_stats()
        assert s2["tokens_saved"] == 0  # engine unaffected


# ═══════════════════════════════════════════════════════════════════════
# Exact per-call token math
# ═══════════════════════════════════════════════════════════════════════

class TestExactTokenAccounting:
    """``tokens_before`` / ``tokens_after`` mirror ``estimate_tokens`` exactly."""

    def test_snapshot_ratio_matches_token_counts(
        self, engine: CompressionEngine
    ) -> None:
        """Snapshot ``compression_ratio`` must equal ``tokens_after / tokens_before``.

        Uses a payload large enough that the compressed summary is
        strictly shorter (``after < before``), ensuring the ratio is
        below 1.0 and the ``_safe_ratio`` clamp is not triggered.
        """
        content = "\n".join(
            f"2024-01-01 10:00:{i:02d} INFO message number {i}"
            for i in range(50)
        )
        snap = engine.compress(content, memory_id="s1")
        before = estimate_tokens(content)
        after = estimate_tokens(snap.summary)
        # Per-snapshot ratio is after/before (clamped to (0,1]).
        assert snap.compression_ratio == pytest.approx(after / before, abs=1e-9)
        assert after < before  # guard: ratio is genuinely < 1

    def test_tokens_saved_single_call(self, engine: CompressionEngine) -> None:
        content = (
            "User: decided to ship.\n"
            "Assistant: NeuroMem is a memory engine for agents.\n"
            "User: task: configure the database.\n"
        )
        snap = engine.compress(content, memory_id="c1")
        before = estimate_tokens(content)
        after = estimate_tokens(snap.summary)
        stats = engine.get_stats()
        assert stats["tokens_saved"] == max(0, before - after)

    def test_accumulation_is_exact_sum(self, engine: CompressionEngine) -> None:
        """Running totals equal the exact independent recomputation."""
        items = [
            "2024-01-01 10:00:00 INFO Server started\n2024-01-01 10:01:00 ERROR fail\n",
            "User: what?\nAssistant: NeuroMem is an engine.\n",
            "def f(a, b):\n    return a + b\n",
            "Just some plain prose about memory and graphs.",
        ]
        expected_before = 0
        expected_after = 0
        expected_saved = 0
        for i, content in enumerate(items):
            snap = engine.compress(content, memory_id=f"item_{i}")
            tb = estimate_tokens(content)
            ta = estimate_tokens(snap.summary)
            expected_before += tb
            expected_after += ta
            expected_saved += max(0, tb - ta)

        stats = engine.get_stats()
        # tokens_saved is an integer — must match exactly.
        assert stats["tokens_saved"] == expected_saved
        # compression_ratio is after/before — exact within float precision.
        assert stats["compression_ratio"] == pytest.approx(
            expected_after / expected_before, abs=1e-9
        )
        assert stats["stored_memories_count"] == len(items)


# ═══════════════════════════════════════════════════════════════════════
# Global compression ratio properties
# ═══════════════════════════════════════════════════════════════════════

class TestGlobalRatio:
    def test_ratio_in_unit_interval(self, engine: CompressionEngine) -> None:
        for i in range(5):
            engine.compress(f"sample content number {i} with detail.", memory_id=f"r{i}")
        ratio = engine.get_stats()["compression_ratio"]
        assert 0.0 <= ratio <= 1.0

    def test_ratio_is_zero_before_compression(
        self, engine: CompressionEngine
    ) -> None:
        assert engine.get_stats()["compression_ratio"] == 0.0

    def test_ratio_decreases_with_more_compression(
        self, engine: CompressionEngine
    ) -> None:
        """Highly compressible logs lower the running ratio."""
        # First a hard-to-compress snippet.
        engine.compress("alpha bravo charlie delta echo foxtrot", memory_id="a")
        ratio_before = engine.get_stats()["compression_ratio"]
        # Then a large, compressible log batch.
        big_logs = "\n".join(
            f"2024-01-01 10:00:{s:02d} INFO message number {s}" for s in range(50)
        )
        engine.compress(big_logs, memory_id="big")
        ratio_after = engine.get_stats()["compression_ratio"]
        assert ratio_after < ratio_before


# ═══════════════════════════════════════════════════════════════════════
# reset_stats
# ═══════════════════════════════════════════════════════════════════════

class TestResetStats:
    def test_reset_clears_token_counters(
        self, engine: CompressionEngine
    ) -> None:
        engine.compress(
            "User: hi\nAssistant: hello there NeuroMem.", memory_id="x"
        )
        assert engine.get_stats()["tokens_saved"] > 0
        engine.reset_stats()
        stats = engine.get_stats()
        assert stats["tokens_saved"] == 0
        assert stats["compression_ratio"] == 0.0

    def test_reset_does_not_delete_stored_originals(
        self, engine: CompressionEngine
    ) -> None:
        engine.compress("some content to compress", memory_id="keep_me")
        engine.reset_stats()
        # The original is still recoverable from the backing store.
        assert engine.reversible_store.exists("keep_me")
        # stored_memories_count reflects the real store, not the reset counter.
        assert engine.get_stats()["stored_memories_count"] == 1


# ═══════════════════════════════════════════════════════════════════════
# stored_memories_count sourcing
# ═══════════════════════════════════════════════════════════════════════

class TestStoredMemoriesCount:
    def test_count_matches_compressions(self, engine: CompressionEngine) -> None:
        for i in range(3):
            engine.compress(f"content block {i}", memory_id=f"m{i}")
        assert engine.get_stats()["stored_memories_count"] == 3

    def test_falls_back_to_local_counter_when_store_count_unavailable(
        self, tmp_path
    ) -> None:
        """If the store's ``count()`` is unavailable, the local counter is used."""

        class StubStore(ReversibleStore):
            def count(self) -> int:  # type: ignore[override]
                raise RuntimeError("count unavailable")

        with StubStore(tmp_path / "stub") as store:
            eng = CompressionEngine(reversible_store=store)
            eng.compress("some content", memory_id="s1")
            eng.compress("more content", memory_id="s2")
            # Store.count() raises -> engine uses its own counter.
            assert eng.get_stats()["stored_memories_count"] == 2


# ═══════════════════════════════════════════════════════════════════════
# CompressionMetrics model — ratio consistency invariant
# ═══════════════════════════════════════════════════════════════════════

class TestCompressionMetricsModel:
    def test_consistent_ratio_accepted(self) -> None:
        m = CompressionMetrics(
            tokens_before=100, tokens_after=25,
            compression_ratio=0.25, retrieval_count=0,
        )
        assert m.compression_ratio == 0.25

    def test_inconsistent_ratio_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CompressionMetrics(
                tokens_before=100, tokens_after=25,
                compression_ratio=0.5,   # wrong!
                retrieval_count=0,
            )

    def test_ratio_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            CompressionMetrics(
                tokens_before=10, tokens_after=0,
                compression_ratio=0.0, retrieval_count=0,
            )

    def test_tokens_before_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            CompressionMetrics(
                tokens_before=0, tokens_after=0,
                compression_ratio=1.0, retrieval_count=0,
            )

    def test_full_compression_ratio_one(self) -> None:
        """tokens_after == tokens_before => ratio 1.0 is valid."""
        m = CompressionMetrics(
            tokens_before=10, tokens_after=10,
            compression_ratio=1.0, retrieval_count=0,
        )
        assert m.compression_ratio == 1.0


# ═══════════════════════════════════════════════════════════════════════
# Branch-coverage: store-count sourcing and importance derivation
# ═══════════════════════════════════════════════════════════════════════

class TestStoredCountSourcing:
    def test_uses_store_count_when_positive(self, tmp_path) -> None:
        """When ``store.count()`` returns a positive value, that is reported."""

        class CountingStore(ReversibleStore):
            def count(self) -> int:  # type: ignore[override]
                return 7  # pretend 7 entries exist

        with CountingStore(tmp_path / "c") as store:
            eng = CompressionEngine(reversible_store=store)
            # Even with zero compressions, the store's count wins.
            assert eng.get_stats()["stored_memories_count"] == 7


class TestImportanceDerivation:
    """Importance is derived per-content-type from the strategy output."""

    def test_log_severity_drives_importance(self, engine: CompressionEngine) -> None:
        """A CRITICAL log batch yields high importance (0.95)."""
        logs = (
            "2024-01-01 10:00:00 INFO ok\n"
            "2024-01-01 10:01:00 CRITICAL system on fire\n"
        )
        snap = engine.compress(logs, memory_id="crit")
        assert snap.importance == pytest.approx(0.95)

    def test_conversation_with_decisions_has_higher_importance(
        self, engine: CompressionEngine
    ) -> None:
        conv = (
            "User: we decided to ship now.\n"
            "Assistant: NeuroMem is a memory engine.\n"
        )
        snap = engine.compress(conv, memory_id="dec")
        # Decisions present -> importance 0.7.
        assert snap.importance == pytest.approx(0.7)

    def test_rag_importance_default(self, engine: CompressionEngine) -> None:
        """Free-form prose gets the RAG default importance (0.6)."""
        snap = engine.compress("Some plain prose about memory engines.", memory_id="p")
        assert snap.importance == pytest.approx(0.6)

    def test_explicit_importance_overrides_strategy(
        self, engine: CompressionEngine
    ) -> None:
        logs = "2024-01-01 10:00:00 CRITICAL boom\n"
        snap = engine.compress(logs, memory_id="o", importance=0.1)
        # Explicit override wins over the strategy-derived 0.95.
        assert snap.importance == pytest.approx(0.1)

    def test_importance_below_zero_rejected(self, engine: CompressionEngine) -> None:
        with pytest.raises(ValueError, match="importance"):
            engine.compress("content", importance=-0.01)

    def test_log_keywords_populated_from_errors(
        self, engine: CompressionEngine
    ) -> None:
        """The snapshot keyword list includes the dominant severity + error snippets."""
        logs = (
            "2024-01-01 10:00:00 ERROR Connection failed\n"
            "2024-01-01 10:01:00 ERROR Disk full\n"
        )
        snap = engine.compress(logs, memory_id="kw")
        assert "ERROR" in snap.keywords  # severity upper-cased
        assert any("Connection failed" in k for k in snap.keywords)


class TestRawReferenceContract:
    def test_raw_reference_equals_snapshot_id(
        self, engine: CompressionEngine
    ) -> None:
        snap = engine.compress("content to store", memory_id="ref_id")
        assert snap.raw_reference == snap.id == "ref_id"

    def test_generated_id_when_none(self, engine: CompressionEngine) -> None:
        snap = engine.compress("content without explicit id")
        assert snap.id.startswith("snap_")
        assert snap.raw_reference == snap.id

    def test_original_recoverable_after_compress(
        self, engine: CompressionEngine
    ) -> None:
        original = "the exact original text\nwith newlines"
        snap = engine.compress(original, memory_id="rec")
        assert engine.reversible_store.retrieve_original(snap.raw_reference) == original
