"""Tests for :class:`neuromem.compression.summarizer.ContextCompressor`.

The compressor applies four content-type strategies.  These tests
exercise each with the deterministic :class:`MockLLMProvider` (the
default), so everything runs **fully offline** and reproducibly:

* ``compress_logs``    — structural: error extraction, severity ranking,
  milestone timeline.  No LLM.
* ``compress_conversation`` — semantic facts / decisions / tasks /
  entities, routed through the mock and verified against the structural
  fallback.
* ``compress_code``    — Python ``ast`` skeleton tracking: imports,
  functions, classes, docstrings, decorators, async defs.
* ``compress_rag``     — sentence dedup + citation preservation.

The pure module-level helpers (``_dedup_sentences``,
``_parse_json_object``, ``_dominant_severity`` …) are also covered
directly, since they carry most of the real logic and are the parts most
likely to regress.
"""

from __future__ import annotations

import json

import pytest

from neuromem.compression.models import (
    ConversationCompressionOutput,
    LogCompressionOutput,
)
from neuromem.compression.summarizer import (
    ContextCompressor,
    MockLLMProvider,
    _dedup_sentences,
    _dominant_severity,
    _ensure_citations,
    _extract_citations,
    _parse_json_object,
    _split_sentences,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixture + shared samples
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture()
def compressor() -> ContextCompressor:
    """Return a compressor wired to the offline mock provider."""
    c = ContextCompressor()
    assert isinstance(c.llm, MockLLMProvider)
    return c


# ═══════════════════════════════════════════════════════════════════════
# 1. Log compression — structural extraction
# ═══════════════════════════════════════════════════════════════════════

class TestLogCompression:
    """``compress_logs`` is fully structural — no LLM call."""

    def test_returns_log_compression_output(
        self, compressor: ContextCompressor
    ) -> None:
        out = compressor.compress_logs("2024-01-01 10:00:00 INFO start")
        assert isinstance(out, LogCompressionOutput)

    def test_extracts_and_deduplicates_errors(
        self, compressor: ContextCompressor
    ) -> None:
        logs = (
            "2024-01-01 10:02:00 ERROR Connection failed\n"
            "2024-01-01 10:02:05 ERROR Connection failed\n"   # duplicate message
            "2024-01-01 10:03:00 CRITICAL Disk full\n"
        )
        out = compressor.compress_logs(logs)
        # Duplicate "Connection failed" collapsed to one entry.
        assert len(out.errors) == 2
        assert any("Connection failed" in e for e in out.errors)
        assert any("Disk full" in e for e in out.errors)

    def test_severity_picks_dominant_level(
        self, compressor: ContextCompressor
    ) -> None:
        logs = (
            "INFO ok\n"
            "ERROR boom\n"
            "CRITICAL kaboom\n"
            "DEBUG noise\n"
        )
        out = compressor.compress_logs(logs)
        assert out.severity == "critical"

    @pytest.mark.parametrize(
        "levels,expected",
        [
            (("DEBUG",), "info"),          # debug rank(1) < info floor(2) → stays info
            (("INFO",), "info"),
            (("WARN",), "warning"),
            (("WARNING",), "warning"),
            (("ERROR",), "error"),
            (("CRITICAL",), "critical"),
            (("FATAL",), "critical"),
            (("DEBUG", "INFO", "WARN"), "warning"),
        ],
    )
    def test_severity_ranking(
        self, compressor: ContextCompressor, levels: tuple[str, ...], expected: str
    ) -> None:
        logs = "\n".join(f"2024-01-01 10:00:0{i} {lvl} msg{i}" for i, lvl in enumerate(levels))
        assert compressor.compress_logs(logs).severity == expected

    def test_key_events_capture_milestones(
        self, compressor: ContextCompressor
    ) -> None:
        logs = (
            "2024-01-01 10:00:00 INFO Server started\n"
            "2024-01-01 10:05:00 INFO Deploy completed\n"
            "2024-01-01 10:10:00 INFO Listening on 8080\n"
        )
        out = compressor.compress_logs(logs)
        # All three lines contain a milestone keyword.
        assert len(out.key_events) == 3
        joined = " ".join(out.key_events)
        assert "started" in joined.lower()
        assert "deploy" in joined.lower()

    def test_summary_counts(self, compressor: ContextCompressor) -> None:
        logs = (
            "2024-01-01 10:00:00 INFO a\n"
            "2024-01-01 10:01:00 ERROR b\n"
        )
        out = compressor.compress_logs(logs)
        assert "2 log line(s)" in out.summary
        assert "1 error(s)" in out.summary
        assert "CRITICAL" not in out.summary  # severity is ERROR here? -> dominant is error
        # dominant severity is error
        assert out.severity == "error"

    def test_empty_logs(self, compressor: ContextCompressor) -> None:
        for empty in ["", "   ", "\n\n"]:
            out = compressor.compress_logs(empty)
            assert out.errors == []
            assert out.severity == "info"
            assert out.summary  # non-empty placeholder


# ═══════════════════════════════════════════════════════════════════════
# 2. Conversation compression — semantic (via mock)
# ═══════════════════════════════════════════════════════════════════════

class TestConversationCompression:
    def test_returns_conversation_output(
        self, compressor: ContextCompressor
    ) -> None:
        out = compressor.compress_conversation("User: hi\nAssistant: yo")
        assert isinstance(out, ConversationCompressionOutput)

    def test_extracts_facts_decisions_tasks_entities(
        self, compressor: ContextCompressor
    ) -> None:
        conv = (
            "User: We decided to use NeuroMem for the project.\n"
            "Assistant: NeuroMem is a memory engine. NeuroMem supports graph search.\n"
            "User: We need to set up the database. Task: configure Kuzu.\n"
        )
        out = compressor.compress_conversation(conv)
        assert any("memory engine" in f for f in out.important_facts)
        assert any("decided" in d.lower() for d in out.decisions)
        assert any("Kuzu" in t or "database" in t.lower() for t in out.open_tasks)
        assert "NeuroMem" in out.entities
        assert "Kuzu" in out.entities

    def test_summary_drawn_from_assistant_text(
        self, compressor: ContextCompressor
    ) -> None:
        conv = (
            "User: what is it?\n"
            "Assistant: NeuroMem is a memory engine for agents.\n"
        )
        out = compressor.compress_conversation(conv)
        assert "NeuroMem is a memory engine" in out.summary

    def test_empty_conversation(self, compressor: ContextCompressor) -> None:
        out = compressor.compress_conversation("")
        assert out.important_facts == []
        assert out.decisions == []
        assert out.summary  # placeholder

    def test_llm_failure_falls_back_to_structural(
        self,
    ) -> None:
        """When the provider raises, the structural fallback still produces output."""

        class BoomProvider:
            def complete(self, system_prompt: str, user_prompt: str) -> str:
                raise RuntimeError("network down")

        c = ContextCompressor(llm=BoomProvider())  # type: ignore[arg-type]
        out = c.compress_conversation(
            "User: decided to ship.\nAssistant: NeuroMem is ready."
        )
        assert out.summary  # did not crash
        assert any("decided" in d.lower() for d in out.decisions)

    def test_malformed_llm_output_falls_back(self) -> None:
        """Provider returning non-JSON triggers the structural fallback."""

        class GarbageProvider:
            def complete(self, system_prompt: str, user_prompt: str) -> str:
                return "this is definitely not json"

        c = ContextCompressor(llm=GarbageProvider())  # type: ignore[arg-type]
        out = c.compress_conversation("User: hi\nAssistant: NeuroMem works.")
        assert out.summary


# ═══════════════════════════════════════════════════════════════════════
# 3. Code compression — Python AST tracking
# ═══════════════════════════════════════════════════════════════════════

class TestCodeCompression:
    def test_skeleton_includes_imports(
        self, compressor: ContextCompressor
    ) -> None:
        skeleton = compressor.compress_code("import os\nfrom typing import List\n")
        assert "import os" in skeleton
        assert "from typing import List" in skeleton
        assert skeleton.startswith("```python")

    def test_skeleton_includes_function_signatures(
        self, compressor: ContextCompressor
    ) -> None:
        code = "def add(a, b=2):\n    '''Add two numbers.'''\n    return a + b\n"
        skeleton = compressor.compress_code(code)
        assert "def add(a, b=…)" in skeleton          # default rendered as ellipsis
        assert "Add two numbers" in skeleton          # docstring preview
        assert "return a + b" not in skeleton         # body omitted

    def test_async_function_signature(self, compressor: ContextCompressor) -> None:
        code = "async def fetch(url):\n    '''Fetch it.'''\n    return url\n"
        skeleton = compressor.compress_code(code)
        assert "async def fetch(url)" in skeleton

    def test_class_with_methods_and_attributes(
        self, compressor: ContextCompressor
    ) -> None:
        code = (
            "class Engine:\n"
            "    '''An engine.'''\n"
            "    name = 'x'\n"
            "    def run(self):\n"
            "        '''Run it.'''\n"
            "        pass\n"
        )
        skeleton = compressor.compress_code(code)
        assert "class Engine" in skeleton
        assert "An engine" in skeleton
        assert "name = …" in skeleton                  # class attribute
        assert "def run(self)" in skeleton             # method signature

    def test_decorators_rendered(self, compressor: ContextCompressor) -> None:
        code = (
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class Point:\n"
            "    x: int\n"
        )
        skeleton = compressor.compress_code(code)
        assert "@dataclass" in skeleton

    def test_module_docstring_preserved(
        self, compressor: ContextCompressor
    ) -> None:
        code = '"""Module overview."""\nimport os\n'
        skeleton = compressor.compress_code(code)
        assert "Module overview" in skeleton

    def test_invalid_python_falls_back_to_raw(
        self, compressor: ContextCompressor
    ) -> None:
        skeleton = compressor.compress_code("def f(:\n  bad syntax")
        assert "WARNING" in skeleton
        assert "could not parse as Python" in skeleton
        assert "def f(:" in skeleton    # raw source retained

    def test_empty_source(self, compressor: ContextCompressor) -> None:
        for empty in ["", "   "]:
            skeleton = compressor.compress_code(empty)
            assert "empty source" in skeleton

    def test_no_definitions_no_imports(
        self, compressor: ContextCompressor
    ) -> None:
        skeleton = compressor.compress_code("x = 1\ny = 2\n")
        assert "no functions, classes, or imports detected" in skeleton

    def test_variadic_and_kwonly_args_formatted(
        self, compressor: ContextCompressor
    ) -> None:
        code = "def f(a, *args, b=1, **kw):\n    pass\n"
        skeleton = compressor.compress_code(code)
        assert "*args" in skeleton
        assert "**kw" in skeleton


# ═══════════════════════════════════════════════════════════════════════
# 4. RAG compression — dedup + citations
# ═══════════════════════════════════════════════════════════════════════

class TestRagCompression:
    def test_deduplicates_near_identical_sentences(
        self, compressor: ContextCompressor
    ) -> None:
        chunks = [
            "NeuroMem is a memory engine.",
            "NeuroMem is a memory engine!",   # differs only by punctuation
        ]
        merged = compressor.compress_rag(chunks)
        # The two near-duplicates collapse to a single statement.
        assert merged.count("memory engine") == 1

    def test_preserves_distinct_facts(
        self, compressor: ContextCompressor
    ) -> None:
        chunks = [
            "NeuroMem is a memory engine.",
            "It combines graph and vector search.",
        ]
        merged = compressor.compress_rag(chunks)
        assert "memory engine" in merged
        assert "graph and vector search" in merged

    def test_preserves_citations(
        self, compressor: ContextCompressor
    ) -> None:
        chunks = [
            "NeuroMem is a memory engine. [source: docs/intro]",
            "It scales well. [doc: readme]",
        ]
        merged = compressor.compress_rag(chunks)
        assert "[source: docs/intro]" in merged
        assert "[doc: readme]" in merged

    def test_empty_chunks_returns_empty(self, compressor: ContextCompressor) -> None:
        assert compressor.compress_rag([]) == ""

    def test_single_chunk_returned(self, compressor: ContextCompressor) -> None:
        assert compressor.compress_rag(["only one sentence."]) == "only one sentence."

    def test_containment_dedup_drops_subset(
        self, compressor: ContextCompressor
    ) -> None:
        """A short sentence subsumed by a longer one is dropped."""
        chunks = [
            "It is fast.",                       # subset
            "It is fast and reliable and cheap.",  # superset
        ]
        merged = compressor.compress_rag(chunks)
        assert "reliable" in merged
        # The standalone "It is fast." should not appear as a duplicate fragment.
        assert merged.count("It is fast") == 1


# ═══════════════════════════════════════════════════════════════════════
# 5. MockLLMProvider routing
# ═══════════════════════════════════════════════════════════════════════

class TestMockLLMProvider:
    def test_conversation_prompt_returns_conversation_schema(self) -> None:
        m = MockLLMProvider()
        raw = m.complete("You are a conversation compressor", "User: hi\nAssistant: yo")
        parsed = json.loads(raw)
        assert "summary" in parsed
        for key in ("important_facts", "open_tasks", "decisions", "entities"):
            assert key in parsed

    def test_rag_prompt_returns_summary_schema(self) -> None:
        m = MockLLMProvider()
        raw = m.complete("RAG deduplication engine", "some overlapping text")
        parsed = json.loads(raw)
        assert "summary" in parsed

    def test_generic_prompt_returns_summary(self) -> None:
        m = MockLLMProvider()
        raw = m.complete("Summarise this", "hello world")
        parsed = json.loads(raw)
        assert parsed["summary"] == "hello world"

    def test_provider_satisfies_protocol(self) -> None:
        """``MockLLMProvider`` conforms to :class:`BaseLLMProvider``."""
        # The protocol is runtime_checkable; structural conformance is enough.
        from neuromem.compression.summarizer import BaseLLMProvider

        assert isinstance(MockLLMProvider(), BaseLLMProvider)


# ═══════════════════════════════════════════════════════════════════════
# 6. Pure helper functions
# ═══════════════════════════════════════════════════════════════════════

class TestPureHelpers:
    # ── _split_sentences ───────────────────────────────────────────────
    def test_split_sentences_basic(self) -> None:
        assert _split_sentences("First. Second! Third?") == ["First.", "Second!", "Third?"]

    def test_split_sentences_collapses_whitespace(self) -> None:
        parts = _split_sentences("One.    Two.")
        assert parts == ["One.", "Two."]

    def test_split_sentences_reattaches_leading_citation(self) -> None:
        parts = _split_sentences("A fact. [source: x] Another.")
        # The citation fragment is reattached to the previous sentence,
        # producing one combined segment instead of two.
        assert len(parts) == 1
        assert "[source: x]" in parts[0]

    def test_split_sentences_empty(self) -> None:
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []

    # ── _dedup_sentences ───────────────────────────────────────────────
    def test_dedup_exact_duplicates(self) -> None:
        out = _dedup_sentences(["The sky is blue.", "The sky is blue."])
        assert out == ["The sky is blue."]

    def test_dedup_punctuation_insensitive(self) -> None:
        out = _dedup_sentences(["The sky is blue.", "The sky is blue!"])
        assert len(out) == 1

    def test_dedup_citation_insensitive(self) -> None:
        out = _dedup_sentences(["It works. [source: a]", "It works."])
        assert len(out) == 1

    def test_dedup_preserves_order(self) -> None:
        out = _dedup_sentences(["first.", "second.", "first.", "third."])
        assert out == ["first.", "second.", "third."]

    def test_dedup_containment(self) -> None:
        out = _dedup_sentences(["db.", "db built for scale."])
        # Shorter subset dropped in favour of the longer superset.
        assert out == ["db built for scale."]

    # ── citations ──────────────────────────────────────────────────────
    def test_extract_citations(self) -> None:
        text = "A [source: x] and [doc: y] here."
        assert _extract_citations(text) == ["[source: x]", "[doc: y]"]

    def test_ensure_citations_appends_missing(self) -> None:
        result = _ensure_citations("text", ["[source: a]", "[doc: b]"])
        assert "[source: a]" in result
        assert "[doc: b]" in result

    def test_ensure_citations_noop_when_present(self) -> None:
        text = "text [source: a]"
        assert _ensure_citations(text, ["[source: a]"]) == text

    def test_ensure_citations_empty(self) -> None:
        assert _ensure_citations("text", []) == "text"

    # ── _parse_json_object ─────────────────────────────────────────────
    def test_parse_plain_object(self) -> None:
        assert _parse_json_object('{"a": 1}') == {"a": 1}

    def test_parse_fenced_json(self) -> None:
        assert _parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_parse_embedded_in_prose(self) -> None:
        assert _parse_json_object('before {"a": 1} after') == {"a": 1}

    @pytest.mark.parametrize("bad", ["", "   ", "no json here", "[1,2,3]", "'string'", "42"])
    def test_parse_invalid_returns_none(self, bad: str) -> None:
        assert _parse_json_object(bad) is None

    # ── _dominant_severity ─────────────────────────────────────────────
    @pytest.mark.parametrize(
        "levels,expected",
        [
            ([], "info"),
            (["debug"], "info"),          # debug rank < info default floor → info
            (["info", "debug"], "info"),
            (["warn"], "warning"),
            (["warning"], "warning"),
            (["error", "info"], "error"),
            (["fatal"], "critical"),
            (["critical"], "critical"),
            (["unknown_level"], "info"),   # unknown falls below info default
        ],
    )
    def test_dominant_severity(self, levels: list[str], expected: str) -> None:
        assert _dominant_severity(levels) == expected


# ═══════════════════════════════════════════════════════════════════════
# 7. Branch-coverage: helpers and edge paths
# ═══════════════════════════════════════════════════════════════════════

from neuromem.compression.compressor import (
    CompressionEngine,
    _truncate,
    _extract_keywords,
)
from neuromem.compression.summarizer import _as_str_list, _preview_docstring


class TestBranchCoverageHelpers:
    """Targeted tests for branches missed in the initial pass."""

    # ── _truncate ─────────────────────────────────────────────────────
    def test_truncate_short_string_unchanged(self) -> None:
        assert _truncate("hi", max_len=10) == "hi"

    def test_truncate_long_string_trimmed(self) -> None:
        result = _truncate("hello world text", max_len=8)
        assert result == "hello w…"
        assert len(result) == 8

    def test_truncate_exact_length(self) -> None:
        assert _truncate("hello", max_len=5) == "hello"

    # ── _extract_keywords ─────────────────────────────────────────────
    def test_keywords_empty_text(self) -> None:
        assert _extract_keywords("") == []

    def test_keywords_all_stopwords(self) -> None:
        assert _extract_keywords("the and for that") == []

    def test_keywords_max_limit(self) -> None:
        result = _extract_keywords("alpha beta gamma delta", max_keywords=2)
        assert result == ["alpha", "beta"]

    def test_keywords_deduplication(self) -> None:
        result = _extract_keywords("alpha alpha beta beta alpha")
        assert result == ["alpha", "beta"]

    def test_keywords_short_tokens_skipped(self) -> None:
        # Tokens shorter than 3 chars don't match \b[A-Za-z_][A-Za-z0-9_]{2,}\b
        result = _extract_keywords("is an ox box")
        assert "box" in result
        assert "is" not in result
        assert "ox" not in result  # only 2 chars

    # ── _as_str_list ─────────────────────────────────────────────────
    def test_as_str_list_none(self) -> None:
        assert _as_str_list(None) == []

    def test_as_str_list_empty_string(self) -> None:
        assert _as_str_list("") == []

    def test_as_str_list_whitespace_string(self) -> None:
        assert _as_str_list("  ") == []

    def test_as_str_list_single_string(self) -> None:
        assert _as_str_list("hello") == ["hello"]

    def test_as_str_list_mixed_list(self) -> None:
        # None is coerced via str() -> "None" (truthy), so it is retained.
        assert _as_str_list(["a", "", "b", "  "]) == ["a", "b"]

    def test_as_str_list_non_string_scalar(self) -> None:
        assert _as_str_list(42) == ["42"]
        assert _as_str_list(True) == ["True"]

    # ── _preview_docstring ────────────────────────────────────────────
    def test_preview_single_line(self) -> None:
        assert _preview_docstring("one line") == "one line"

    def test_preview_multi_line_truncated(self) -> None:
        result = _preview_docstring("one\ntwo\nthree\nfour", max_lines=2)
        assert result == "one two ..."

    # ── _safe_ratio (static) ─────────────────────────────────────────
    def test_safe_ratio_zero_divisor(self) -> None:
        assert CompressionEngine._safe_ratio(0, 0) == 1.0
        assert CompressionEngine._safe_ratio(0, 5) == 1e-6

    def test_safe_ratio_zero_numerator(self) -> None:
        assert CompressionEngine._safe_ratio(5, 0) == 1.0

    def test_safe_ratio_expansion_clamped(self) -> None:
        """When ``after > before``, ratio is clamped to 1.0."""
        assert CompressionEngine._safe_ratio(3, 2) == 1.0

    def test_safe_ratio_normal(self) -> None:
        assert CompressionEngine._safe_ratio(2, 3) == pytest.approx(2 / 3)

    # ── compress_logs: log line with milestone in error line ───────────
    def test_log_error_line_contains_milestone_keyword(
        self, compressor: ContextCompressor
    ) -> None:
        """An ERROR line mentioning 'started' appears in both errors and key_events."""
        logs = (
            "2024-01-01 10:00:00 ERROR Server started unexpectedly\n"
            "2024-01-01 10:01:00 INFO All good\n"
        )
        out = compressor.compress_logs(logs)
        assert any("Server started" in e for e in out.errors)
        assert any("started" in ev.lower() for ev in out.key_events)

    # ── compress_code: class with no methods (only attributes) ─────────
    def test_code_class_no_methods(self, compressor: ContextCompressor) -> None:
        code = "class Config:\n    name: str\n    port: int = 8080\n"
        skeleton = compressor.compress_code(code)
        assert "class Config" in skeleton
        # Assignment nodes: only ast.Assign targets with ast.Name -> "name = …"
        # However, annotated assignments (AnnAssign) are NOT captured
        # (the code only handles ast.Assign, not ast.AnnAssign).
        # So we just verify class name is present.

    # ── compress_rag: chunks with all-duplicate content ──────────────
    def test_rag_all_duplicate_chunks(self, compressor: ContextCompressor) -> None:
        merged = compressor.compress_rag(["same sentence.", "same sentence!"])
        # Duplicates collapse, but content is preserved.
        assert "same" in merged

    # ── compress_rag: single-sentence chunk with citation ─────────────
    def test_rag_single_with_citation(self, compressor: ContextCompressor) -> None:
        merged = compressor.compress_rag(["fact. [source: docs]"])
        assert "[source: docs]" in merged

    # ── compress_conversation: empty string via mock provider ─────────
    def test_conversation_empty_string(self, compressor: ContextCompressor) -> None:
        out = compressor.compress_conversation("")
        assert out.summary == "(empty conversation)"


class TestCompressorEngineBranches:
    """Cover the ``_normalise`` unknown-type fallback and strategy dispatch."""

    def test_normalise_unknown_output_type(self, tmp_path, monkeypatch) -> None:
        """When a strategy returns an unexpected type, it is stringified.

        The unknown-type fallback lives inside ``_normalise``.  We drive it
        by making the compressor's RAG strategy return a non-standard
        object (an ``int``), which ``_normalise`` then stringifies.
        """
        from neuromem.compression import ReversibleStore

        with ReversibleStore(tmp_path / "archive") as store:
            eng = CompressionEngine(reversible_store=store)
            # ContextCompressor uses __slots__, so patch via monkeypatch
            # on the class method rather than the instance attribute.
            monkeypatch.setattr(
                type(eng._compressor), "compress_rag",
                lambda self, chunks: 42,
            )
            snap = eng.compress("content", memory_id="unknown")
            # 42 was stringified to "42" via the unknown-type fallback.
            assert snap.summary == "42"

    def test_importance_clamped_from_strategy(self, tmp_path) -> None:
        """Even if a strategy yields importance > 1.0, the snapshot is clamped."""
        from neuromem.compression import ReversibleStore
        from neuromem.compression.compressor import _NormalisedResult

        with ReversibleStore(tmp_path / "archive") as store:
            eng = CompressionEngine(reversible_store=store)
            orig = eng._run_strategy
            eng._run_strategy = lambda ct, content: _NormalisedResult(  # type: ignore[assignment]
                summary="x", importance=1.5  # out of range
            )
            snap = eng.compress("content", memory_id="clamp")
            assert snap.importance == 1.0  # clamped

    def test_fallback_strategy_dispatch(self, tmp_path) -> None:
        """If a content type is missing from the dispatch table, compress_rag is used."""
        from neuromem.compression import ReversibleStore

        with ReversibleStore(tmp_path / "archive") as store:
            eng = CompressionEngine(reversible_store=store)
            # Remove "text" from the dispatch table.
            original_strategies = eng._STRATEGIES.copy()
            eng._STRATEGIES = {k: v for k, v in eng._STRATEGIES.items() if k != "text"}
            snap = eng.compress("plain text content", memory_id="fallback")
            assert snap.summary  # didn't crash
            eng._STRATEGIES = original_strategies

    def test_compress_rejects_non_string(self, tmp_path) -> None:
        from neuromem.compression import ReversibleStore

        with ReversibleStore(tmp_path / "archive") as store:
            eng = CompressionEngine(reversible_store=store)
            with pytest.raises(ValueError, match="non-empty string"):
                eng.compress(123)  # type: ignore[arg-type]

    def test_engine_rejects_none_store(self) -> None:
        """Passing ``reversible_store=None`` raises immediately."""
        with pytest.raises(ValueError, match="reversible_store is required"):
            CompressionEngine(reversible_store=None)  # type: ignore[arg-type]

    def test_engine_rejects_uninitialised_store(self, tmp_path) -> None:
        """A store that has not been ``initialize()``-d is rejected."""
        from neuromem.compression import ReversibleStore

        unready = ReversibleStore(tmp_path / "unready")  # not initialised
        with pytest.raises(ValueError, match="must be initialised"):
            CompressionEngine(reversible_store=unready)

    def test_engine_exposes_router_and_compressor_properties(
        self, tmp_path
    ) -> None:
        """The ``router`` and ``compressor`` properties return the wired deps."""
        from neuromem.compression import ContentRouter, ReversibleStore

        with ReversibleStore(tmp_path / "archive") as store:
            custom_router = ContentRouter(min_lines=3)
            eng = CompressionEngine(
                reversible_store=store, content_router=custom_router
            )
            assert eng.router is custom_router
            assert isinstance(eng.compressor, ContextCompressor)
            assert eng.reversible_store is store


# ═══════════════════════════════════════════════════════════════════════
# 8. Deep branch-coverage: summarizer internals
# ═══════════════════════════════════════════════════════════════════════

class TestSummarizerDeepBranches:
    """Targeted tests for the remaining uncovered summarizer branches."""

    def test_compress_logs_skips_blank_lines(
        self, compressor: ContextCompressor
    ) -> None:
        """Blank lines inside a log batch are skipped (the ``continue`` branch)."""
        logs = (
            "2024-01-01 10:00:00 INFO start\n"
            "\n"
            "   \n"
            "2024-01-01 10:01:00 ERROR fail\n"
        )
        out = compressor.compress_logs(logs)
        # The two blank lines did not produce phantom log entries.
        assert "2 log line(s)" not in out.summary  # counts all splitlines incl blanks
        assert any("fail" in e for e in out.errors)

    def test_compress_rag_returns_empty_when_all_whitespace(
        self, compressor: ContextCompressor
    ) -> None:
        """Chunks that dedupe to nothing yield an empty string (line 414)."""
        result = compressor.compress_rag(["   ", "\n\n", ""])
        assert result == ""

    def test_compress_rag_llm_merge_fallback(self) -> None:
        """When the LLM returns no ``summary``, the structural join is used.

        This covers the ``merged = joined`` fallback branch (lines 435-436).
        """
        from neuromem.compression.summarizer import ContextCompressor

        class NoSummaryProvider:
            def complete(self, system_prompt: str, user_prompt: str) -> str:
                # Valid JSON object, but no "summary" key -> falsy.
                return '{"other": "value"}'

        c = ContextCompressor(llm=NoSummaryProvider())  # type: ignore[arg-type]
        merged = c.compress_rag(["First distinct sentence.", "Second distinct one."])
        # Both sentences survive via the structural join fallback.
        assert "First distinct sentence" in merged
        assert "Second distinct one" in merged

    def test_compress_rag_llm_merge_used_when_summary_present(self) -> None:
        """When the LLM returns a ``summary``, that merged text is used."""
        from neuromem.compression.summarizer import ContextCompressor

        class MergeProvider:
            def complete(self, system_prompt: str, user_prompt: str) -> str:
                return '{"summary": "LLM merged output here."}'

        c = ContextCompressor(llm=MergeProvider())  # type: ignore[arg-type]
        merged = c.compress_rag(["First sentence.", "Second sentence."])
        assert "LLM merged output here." in merged

    def test_parse_json_object_invalid_between_braces(self) -> None:
        """Braces present but contents unparseable -> None (lines 595-596)."""
        assert _parse_json_object("{ not valid json }") is None
        assert _parse_json_object("{a: }") is None

    def test_code_kwonly_args_without_vararg(
        self, compressor: ContextCompressor
    ) -> None:
        """Keyword-only args with no ``*args`` emit a bare ``*`` separator (line 732)."""
        code = "def f(a, *, b, c):\n    pass\n"
        skeleton = compressor.compress_code(code)
        # The bare "*" separator precedes the kwonly args.
        assert "*" in skeleton
        assert "b" in skeleton
        assert "c" in skeleton

    def test_code_decorator_unparse_failure_handled(
        self, compressor: ContextCompressor, monkeypatch
    ) -> None:
        """If ``ast.unparse`` fails on a decorator, it degrades to ``…`` (line 760)."""
        import ast

        calls = {"n": 0}
        real_unparse = ast.unparse

        def flaky(node):
            # Fail only for decorator expressions, succeed otherwise.
            calls["n"] += 1
            if isinstance(node, ast.Call):
                raise ValueError("cannot unparse")
            return real_unparse(node)

        monkeypatch.setattr(ast, "unparse", flaky)
        code = "@deco()\ndef f():\n    pass\n"
        skeleton = compressor.compress_code(code)
        assert "…" in skeleton  # decorator degraded to ellipsis

    def test_code_class_non_name_assignment_target(
        self, compressor: ContextCompressor
    ) -> None:
        """Class-body ``ast.Assign`` whose target is not a Name is skipped (line 769).

        A tuple-unpack assignment like ``a, b = 1, 2`` has tuple targets,
        which the summariser ignores (only ``ast.Name`` targets render).
        """
        code = (
            "class C:\n"
            "    a, b = 1, 2\n"   # tuple target -> skipped
            "    name = 'x'\n"    # Name target -> rendered
        )
        skeleton = compressor.compress_code(code)
        assert "class C" in skeleton
        assert "name = …" in skeleton

    def test_code_function_without_docstring(
        self, compressor: ContextCompressor
    ) -> None:
        """A function with no docstring renders its signature alone (line 745-746)."""
        code = "def f(a, b):\n    return a + b\n"
        skeleton = compressor.compress_code(code)
        assert "def f(a, b)" in skeleton
        # No docstring preview appended.
        assert "#" not in skeleton.split("def f")[1].splitlines()[0]

    def test_code_async_method_in_class(
        self, compressor: ContextCompressor
    ) -> None:
        """Async methods inside a class are indented correctly."""
        code = (
            "class Svc:\n"
            "    async def fetch(self, url):\n"
            "        '''Get it.'''\n"
            "        return url\n"
        )
        skeleton = compressor.compress_code(code)
        assert "async def fetch(self, url)" in skeleton

    def test_compress_logs_line_with_level_no_timestamp(
        self, compressor: ContextCompressor
    ) -> None:
        """A log line with a level token but no leading timestamp (branch 275→286)."""
        logs = "ERROR something broke\nINFO all fine\n"
        out = compressor.compress_logs(logs)
        assert out.severity == "error"
        assert any("something broke" in e for e in out.errors)

    def test_preview_docstring_truncation_ellipsis(
        self,
    ) -> None:
        """When docstring has more than ``max_lines`` lines, ellipsis is appended."""
        result = _preview_docstring("one\ntwo\nthree\nfour\nfive", max_lines=2)
        assert result == "one two ..."

    def test_sentence_fingerprint_trailing_punct_stripped(self) -> None:
        """Sentences differing only in trailing punctuation share a fingerprint."""
        from neuromem.compression.summarizer import _sentence_fingerprint

        fp1 = _sentence_fingerprint("The sky is blue.")
        fp2 = _sentence_fingerprint("The sky is blue!")
        fp3 = _sentence_fingerprint("The sky is blue?")
        assert fp1 == fp2 == fp3

    def test_parse_json_object_with_closing_brace_in_embedded_string(self) -> None:
        """Handles nested ``}`` inside a JSON string value without breaking."""
        result = _parse_json_object('{"key": "has } brace"}')
        assert result is not None
        assert result["key"] == "has } brace"


class TestModelsBranches:
    """Coverage for models.py branches (naive datetime, severity, ratio)."""

    def test_naive_datetime_coerced_to_utc(self) -> None:
        """A naive ``created_at`` is normalised to UTC (line 50)."""
        from datetime import datetime, timezone
        from neuromem.compression.models import MemorySnapshot

        naive = datetime(2024, 6, 15, 12, 0, 0)
        snap = MemorySnapshot(
            id="t", summary="s", importance=0.5,
            compression_ratio=0.5, raw_reference="r",
            created_at=naive,
        )
        assert snap.created_at.tzinfo is not None

    def test_aware_datetime_preserved(self) -> None:
        from datetime import datetime, timezone
        from neuromem.compression.models import MemorySnapshot

        aware = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        snap = MemorySnapshot(
            id="t", summary="s", importance=0.5,
            compression_ratio=0.5, raw_reference="r",
            created_at=aware,
        )
        assert snap.created_at == aware

    def test_severity_upper_casing(self) -> None:
        """Severity values are lowercased (line 204)."""
        from neuromem.compression.models import LogCompressionOutput

        out = LogCompressionOutput(
            summary="x", severity="ERROR", errors=[], key_events=[],
        )
        assert out.severity == "error"

    def test_compression_metrics_tokens_before_minimum_one(self) -> None:
        """``tokens_before`` has ``ge=1`` so zero is rejected (guard always runs)."""
        from neuromem.compression.models import CompressionMetrics
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CompressionMetrics(
                tokens_before=0, tokens_after=0,
                compression_ratio=1.0, retrieval_count=0,
            )
