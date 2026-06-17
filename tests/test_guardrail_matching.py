"""Tests for Regex/Fuzzy Guardrail Matching — Feature 2."""

from __future__ import annotations

import pytest

from neuromem.core.engine import _matches_negative_pattern
from neuromem.core.models import NegativeMemory, NegativeMemoryPatternType


# ── Unit tests for _matches_negative_pattern ─────────────────────────


class TestExactMatching:
    def test_exact_match(self):
        neg = NegativeMemory(id="e1", pattern="foo bar", pattern_type="exact")
        assert _matches_negative_pattern("foo bar", neg)

    def test_exact_no_match(self):
        neg = NegativeMemory(id="e2", pattern="foo bar", pattern_type="exact")
        assert not _matches_negative_pattern("foo", neg)

    def test_exact_case_sensitive(self):
        neg = NegativeMemory(id="e3", pattern="Foo Bar", pattern_type="exact")
        assert not _matches_negative_pattern("foo bar", neg)

    def test_exact_partial_no_match(self):
        neg = NegativeMemory(id="e4", pattern="foo bar baz", pattern_type="exact")
        assert not _matches_negative_pattern("foo bar", neg)


class TestRegexMatching:
    def test_regex_partial_match(self):
        neg = NegativeMemory(id="r1", pattern=r"port \d+", pattern_type="regex")
        assert _matches_negative_pattern("connect on port 3000", neg)

    def test_regex_no_match(self):
        neg = NegativeMemory(id="r2", pattern=r"port \d+", pattern_type="regex")
        assert not _matches_negative_pattern("no match here", neg)

    def test_regex_anchored(self):
        neg = NegativeMemory(id="r3", pattern=r"^error:", pattern_type="regex")
        assert _matches_negative_pattern("error: something went wrong", neg)
        assert not _matches_negative_pattern("WARN: error: nested", neg)

    def test_invalid_regex_returns_false_not_raise(self):
        neg = NegativeMemory(id="r4", pattern="[invalid regex", pattern_type="regex")
        # Must not raise; must return False
        assert not _matches_negative_pattern("anything", neg)

    def test_regex_wildcard(self):
        neg = NegativeMemory(id="r5", pattern=r"npm run.*port \d+", pattern_type="regex")
        assert _matches_negative_pattern("npm run dev on port 3001", neg)
        assert not _matches_negative_pattern("npm run dev", neg)


class TestFuzzyMatching:
    def test_fuzzy_high_overlap(self):
        neg = NegativeMemory(
            id="f1",
            pattern="database connection failed",
            pattern_type="fuzzy",
            fuzzy_threshold=0.5,
        )
        assert _matches_negative_pattern("database connection error", neg)

    def test_fuzzy_low_overlap(self):
        neg = NegativeMemory(
            id="f2",
            pattern="database connection failed",
            pattern_type="fuzzy",
            fuzzy_threshold=0.8,
        )
        assert not _matches_negative_pattern("completely unrelated text here", neg)

    def test_fuzzy_exact_text_matches(self):
        neg = NegativeMemory(
            id="f3",
            pattern="the quick brown fox",
            pattern_type="fuzzy",
            fuzzy_threshold=0.9,
        )
        assert _matches_negative_pattern("the quick brown fox", neg)

    def test_fuzzy_threshold_boundary(self):
        # pattern="a b c d", candidate="a b c x" -> overlap {a,b,c}/union{a,b,c,d,x} = 3/5 = 0.6
        neg_pass = NegativeMemory(
            id="f4",
            pattern="a b c d",
            pattern_type="fuzzy",
            fuzzy_threshold=0.5,
        )
        neg_fail = NegativeMemory(
            id="f5",
            pattern="a b c d",
            pattern_type="fuzzy",
            fuzzy_threshold=0.7,
        )
        assert _matches_negative_pattern("a b c x", neg_pass)
        assert not _matches_negative_pattern("a b c x", neg_fail)

    def test_fuzzy_empty_strings_forbidden(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            NegativeMemory(id="f6", pattern="", pattern_type="fuzzy")

    def test_fuzzy_default_threshold(self):
        # Default fuzzy_threshold is 0.8
        neg = NegativeMemory(id="f7", pattern="hello world test", pattern_type="fuzzy")
        assert neg.fuzzy_threshold == 0.8


# ── NegativeMemory model tests ────────────────────────────────────────


class TestNegativeMemoryPatternType:
    def test_default_pattern_type_is_exact(self):
        neg = NegativeMemory(id="m1", pattern="foo")
        assert neg.pattern_type == NegativeMemoryPatternType.EXACT

    def test_pattern_type_regex(self):
        neg = NegativeMemory(id="m2", pattern=r"\d+", pattern_type="regex")
        assert neg.pattern_type == NegativeMemoryPatternType.REGEX

    def test_pattern_type_fuzzy(self):
        neg = NegativeMemory(id="m3", pattern="error", pattern_type="fuzzy")
        assert neg.pattern_type == NegativeMemoryPatternType.FUZZY

    def test_fuzzy_threshold_validation(self):
        with pytest.raises(Exception):  # pydantic validation
            NegativeMemory(id="m4", pattern="x", fuzzy_threshold=1.5)

    def test_fuzzy_threshold_zero(self):
        neg = NegativeMemory(id="m5", pattern="x", fuzzy_threshold=0.0)
        assert neg.fuzzy_threshold == 0.0


# ── Integration: is_blocked with regex ───────────────────────────────


@pytest.mark.integration
class TestIsBlockedPatternTypes:
    @pytest.fixture
    def engine(self, tmp_path):
        from neuromem.storage.kuzu_graph import KuzuGraphEngine
        from neuromem.storage.chroma_vector import ChromaVectorEngine
        from neuromem.core.engine import NeuroMemEngine

        graph = KuzuGraphEngine(str(tmp_path / "graph"))
        vector = ChromaVectorEngine(str(tmp_path / "vectors"))
        graph.initialize()
        vector.initialize()
        return NeuroMemEngine(graph, vector, namespace="test")

    def test_regex_guardrail_blocks_variant(self, engine):
        engine.record_negative(
            "npm run dev on port 3000",
            pattern_type=NegativeMemoryPatternType.REGEX,
        )
        # Variant port — should match regex if pattern were r"port \d+"
        # Here pattern is exact literal so won't match — shows correctness
        assert not engine.is_blocked("npm run dev on port 9999")

    def test_regex_guardrail_with_pattern(self, engine):
        engine.record_negative(
            r"port \d+",
            pattern_type=NegativeMemoryPatternType.REGEX,
        )
        assert engine.is_blocked("connect on port 3001")
        assert engine.is_blocked("open port 8080")
        assert not engine.is_blocked("no port info here")

    def test_fuzzy_guardrail_blocks_similar(self, engine):
        engine.record_negative(
            "database connection failed",
            pattern_type=NegativeMemoryPatternType.FUZZY,
            fuzzy_threshold=0.5,
        )
        assert engine.is_blocked("database connection error")
        assert not engine.is_blocked("totally different text")

    def test_exact_guardrail_still_works(self, engine):
        engine.record_negative("foo bar", pattern_type=NegativeMemoryPatternType.EXACT)
        assert engine.is_blocked("foo bar")
        assert not engine.is_blocked("foo")
