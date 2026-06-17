"""Unit tests for :class:`neuromem.compression.router.ContentRouter`.

These tests exercise **edge-case detection** across every content type the
router knows about, with particular attention to the tricky boundaries:

* **Markdown tables** — a lone GFM pipe table carries none of the router's
  six markdown signals, so it correctly degrades to ``text``; the same
  table promoted with a second signal (heading, list, code fence) flips
  back to ``markdown``.
* **Complex JSON** — nested objects, arrays, quoted strings.
* **Python code** — keyword-led, bracket-dense, and code-fenced variants.
* **Unstructured logs** — ISO timestamps vs. bare log-level tokens, plus
  the ``min_lines`` short-circuit.

All assertions are pinned to the *empirically observed* behaviour of the
router (verified before the suite was written) so the tests are
deterministic rather than aspirational.
"""

from __future__ import annotations

import pytest

from neuromem.compression.router import ContentRouter, ContentType


# ═══════════════════════════════════════════════════════════════════════
# Fixtures / shared samples
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture()
def router() -> ContentRouter:
    """Return a router with the library defaults."""
    return ContentRouter()


# ═══════════════════════════════════════════════════════════════════════
# Markdown — including the pipe-table edge case
# ═══════════════════════════════════════════════════════════════════════

class TestMarkdownDetection:
    """Markdown edge cases, centred on GFM pipe tables."""

    def test_rich_markdown_document_detected(self, router: ContentRouter) -> None:
        """A document with two+ markdown signals classifies as ``markdown``."""
        doc = (
            "# Release Notes\n\n"
            "See the [changelog](https://example.com/changelog) for details.\n\n"
            "- New graph engine\n"
            "- Faster recall\n"
        )
        assert router.detect_content_type(doc) == "markdown"

    def test_lone_pipe_table_is_text_not_markdown(self, router: ContentRouter) -> None:
        """A GFM pipe table alone has no recognised signal -> ``text``.

        This is the documented behaviour: the router looks for headings,
        links, lists, blockquotes, rules, and code fences — **not** pipe
        tables.  Pinning this guards against silent regressions if pipe
        support is ever added (the assertion would then need updating).
        """
        table = "| name | value |\n| --- | ---: |\n| alpha | 1 |\n| beta | 2 |"
        assert router.detect_content_type(table) == "text"

    @pytest.mark.parametrize(
        "doc",
        [
            # heading + unordered list (pipe table included but isn't a signal)
            "# Report\n\n- intro\n\n| a | b |\n| --- | --- |\n| 1 | 2 |",
            # heading + link (pipe table present but irrelevant)
            "# Data\n\nSee [docs](https://x).\n\n| k | v |\n| --- | --- |",
            # heading + blockquote (table is decorative)
            "# Summary\n\n> note\n\n| x | y |\n| --- | --- |",
            # list + link
            "- item one\n- item two\n\n[docs](https://x)\n\n| t | t |",
        ],
    )
    def test_markdown_with_two_or_more_signals(
        self, router: ContentRouter, doc: str
    ) -> None:
        """Two or more recognised markdown signals classify as ``markdown``.

        The pipe table is *not* a recognised markdown signal on its own,
        but when paired with heading/list/link/blockquote the document
        reaches the threshold of 2.
        """
        assert router.detect_content_type(doc) == "markdown"

    def test_code_fence_plus_table_routes_as_code_not_markdown(
        self, router: ContentRouter
    ) -> None:
        """Code fences have higher priority than markdown signals."""
        doc = "```python\nx = 1\n```\n\n| a | b |\n| --- | --- |\n| 1 | 2 |"
        assert router.detect_content_type(doc) == "code"

    def test_single_markdown_signal_is_not_enough(self, router: ContentRouter) -> None:
        """A single signal (e.g. one heading) does NOT classify as markdown."""
        assert router.detect_content_type("# Just a heading line") == "text"


# ═══════════════════════════════════════════════════════════════════════
# JSON — complex / nested
# ═══════════════════════════════════════════════════════════════════════

class TestJsonDetection:
    """Detection of complex JSON strings (highest priority)."""

    def test_nested_object(self, router: ContentRouter) -> None:
        payload = (
            '{"name": "NeuroMem", "version": 3, '
            '"nested": {"k": [1, 2, 3]}, "flags": {"on": true}}'
        )
        assert router.detect_content_type(payload) == "json"

    def test_top_level_array_of_objects(self, router: ContentRouter) -> None:
        payload = '[{"a": 1}, {"b": 2}, {"c": [3, 4]}]'
        assert router.detect_content_type(payload) == "json"

    def test_json_with_quoted_special_chars(self, router: ContentRouter) -> None:
        """Quoted braces/brackets must not trip the code detector."""
        payload = '{"code": "def f():\\n    return {1: [2, 3]}", "ok": true}'
        assert router.detect_content_type(payload) == "json"

    def test_json_role_keys_still_json_not_conversation(
        self, router: ContentRouter
    ) -> None:
        """JSON takes priority over the ``"role"`` conversation heuristic."""
        payload = (
            '{"messages": [{"role": "user", "content": "hi"}, '
            '{"role": "assistant", "content": "hello"}]}'
        )
        assert router.detect_content_type(payload) == "json"

    @pytest.mark.parametrize(
        "invalid",
        [
            "{not valid json}",            # unquoted keys
            '{"a": 1, }',                  # trailing comma
            "[1, 2, ",                      # truncated
            "{'single': 'quotes'}",         # single quotes
        ],
    )
    def test_invalid_json_falls_through(self, router: ContentRouter, invalid: str) -> None:
        """Malformed JSON is rejected by ``json.loads`` and falls through the pipeline."""
        # All four samples happen to have no other matching signal -> ``text``.
        assert router.detect_content_type(invalid) == "text"

    def test_json_scalar_rejected(self, router: ContentRouter) -> None:
        """A bare JSON scalar (string/number) is NOT an object/array -> not ``json``."""
        assert router.detect_content_type('"just a string"') != "json"
        assert router.detect_content_type("42") != "json"


# ═══════════════════════════════════════════════════════════════════════
# Python code snippets
# ═══════════════════════════════════════════════════════════════════════

class TestCodeDetection:
    """Detection of Python (and code-shaped) snippets."""

    def test_keyword_led_python(self, router: ContentRouter) -> None:
        code = (
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        )
        assert router.detect_content_type(code) == "code"

    def test_class_definition(self, router: ContentRouter) -> None:
        code = (
            "class Engine:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n"
        )
        assert router.detect_content_type(code) == "code"

    def test_code_fence_without_keywords(self, router: ContentRouter) -> None:
        """A fenced block triggers the code detector even without keywords."""
        code = "```python\nresult = compute(data)\nprint(result)\n```"
        assert router.detect_content_type(code) == "code"

    def test_bracket_dense_line(self, router: ContentRouter) -> None:
        """Three bracket-type chars separated by content flag code.

        The regex requires ``bracket … bracket … bracket`` spread across
        the line, so nested braces like ``{'a': [1, (2, 3)]}`` trigger it
        while ``func(a, b)`` (only 2 brackets) does not.
        """
        code = "{a: [1, (2, 3)]}"
        assert router.detect_content_type(code) == "code"

    def test_markdown_links_do_not_falsely_trigger_code(
        self, router: ContentRouter
    ) -> None:
        """A line of markdown links must not look like bracket-dense code."""
        line = "Read the [docs](https://x) and the [faq](https://y) today."
        assert router.detect_content_type(line) == "text"

    def test_import_statement(self, router: ContentRouter) -> None:
        assert router.detect_content_type("from neuromem import NeuroMemClient") == "code"


# ═══════════════════════════════════════════════════════════════════════
# Unstructured / structured logs
# ═══════════════════════════════════════════════════════════════════════

class TestLogDetection:
    """Detection of application log output."""

    def test_iso_timestamps_with_levels(self, router: ContentRouter) -> None:
        logs = (
            "2024-01-15 14:23:00 INFO Server started\n"
            "2024-01-15 14:24:00 ERROR Connection failed\n"
            "2024-01-15 14:25:00 WARN Retry attempt\n"
        )
        assert router.detect_content_type(logs) == "logs"

    def test_bare_log_level_token(self, router: ContentRouter) -> None:
        """A single ERROR-level line is enough (the level token fires)."""
        logs = "Something went wrong here.\nERROR unable to reach database"
        assert router.detect_content_type(logs) == "logs"

    @pytest.mark.parametrize("level", ["INFO", "DEBUG", "WARN", "WARNING", "CRITICAL", "FATAL", "TRACE"])
    def test_each_level_token_detected(self, router: ContentRouter, level: str) -> None:
        logs = f"first line of context\n{level} a notable event occurred"
        assert router.detect_content_type(logs) == "logs"

    def test_two_timestamps_without_level(self, router: ContentRouter) -> None:
        """Two timestamps alone (no level token) still classify as logs."""
        logs = "12:00:00 boot sequence\n12:00:05 ready to serve"
        assert router.detect_content_type(logs) == "logs"

    def test_single_timestamp_not_enough(self, router: ContentRouter) -> None:
        """One timestamp, no level, default threshold=2 -> not logs."""
        logs = "12:00:00 something happened"
        assert router.detect_content_type(logs) != "logs"

    def test_short_log_batch_below_min_lines(self, router: ContentRouter) -> None:
        """A single-line log is rejected by the ``min_lines`` gate."""
        # Only one non-empty line; even with a level token the gate fails.
        assert router.detect_content_type("ERROR boom") == "text"

    def test_timestamp_hits_threshold_tunable(self) -> None:
        """``timestamp_hits=1`` makes a single timestamp sufficient."""
        r = ContentRouter(timestamp_hits=1)
        logs = "12:00:00 boot\n12:00:05 ready"
        assert r.detect_content_type(logs) == "logs"


# ═══════════════════════════════════════════════════════════════════════
# Conversation + text fallback + empty handling
# ═══════════════════════════════════════════════════════════════════════

class TestConversationAndFallback:
    """Conversation role markers and the plain-text fallback."""

    def test_two_role_markers(self, router: ContentRouter) -> None:
        conv = "User: What is NeuroMem?\nAssistant: A memory engine."
        assert router.detect_content_type(conv) == "conversation"

    def test_single_role_marker_is_text(self, router: ContentRouter) -> None:
        """Conversation detection needs >= 2 role-marker lines."""
        assert router.detect_content_type("User: hello there") == "text"

    def test_plain_text_fallback(self, router: ContentRouter) -> None:
        assert router.detect_content_type("The quick brown fox jumps over the lazy dog.") == "text"

    @pytest.mark.parametrize("empty", ["", "   ", "\n\n\t  \n"])
    def test_empty_and_whitespace_is_text(self, router: ContentRouter, empty: str) -> None:
        assert router.detect_content_type(empty) == "text"


# ═══════════════════════════════════════════════════════════════════════
# Construction / configuration validation
# ═══════════════════════════════════════════════════════════════════════

class TestRouterConfiguration:
    """Constructor guardrails and the public type alias."""

    def test_default_min_lines_and_timestamp_hits(self) -> None:
        r = ContentRouter()
        # Defaults are exposed via behaviour; assert the documented contract.
        assert r.detect_content_type("ERROR one") == "text"           # 1 line < min_lines
        assert r.detect_content_type("x\nERROR two") == "logs"        # level fires

    def test_invalid_min_lines_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_lines"):
            ContentRouter(min_lines=0)

    def test_invalid_timestamp_hits_rejected(self) -> None:
        with pytest.raises(ValueError, match="timestamp_hits"):
            ContentRouter(timestamp_hits=0)

    def test_content_type_alias_covers_all_six(self) -> None:
        """Sanity: every classification result is a member of the literal set."""
        allowed = {"json", "code", "logs", "conversation", "markdown", "text"}
        # ContentType is a typing.Literal; introspect its args.
        literal_args = set(ContentType.__args__)  # type: ignore[attr-defined]
        assert literal_args == allowed


# ═══════════════════════════════════════════════════════════════════════
# Detection priority ordering
# ═══════════════════════════════════════════════════════════════════════

class TestDetectionPriority:
    """The fixed priority order: json > code > logs > conversation > markdown > text."""

    def test_json_beats_code(self, router: ContentRouter) -> None:
        """Valid JSON containing code-like brackets classifies as ``json``."""
        assert router.detect_content_type('{"fn": "def f(): return [1, 2, 3]"}') == "json"

    def test_code_beats_logs(self, router: ContentRouter) -> None:
        """Code keywords win over an embedded log-level token."""
        snippet = (
            "def handle():\n"
            "    if response.status >= 400:\n"
            "        log.warning('bad status')"
        )
        assert router.detect_content_type(snippet) == "code"

    def test_logs_beats_conversation(self, router: ContentRouter) -> None:
        """Timestamped log lines win over role markers."""
        mixed = (
            "2024-01-01 10:00:00 INFO start\n"
            "User: hello\n"
            "Assistant: hi\n"
        )
        assert router.detect_content_type(mixed) == "logs"

    def test_statelessness(self, router: ContentRouter) -> None:
        """Repeated calls on the same input are identical (no mutable state)."""
        sample = "2024-01-01 10:00:00 INFO start\n2024-01-01 10:01:00 ERROR fail"
        first = router.detect_content_type(sample)
        for _ in range(5):
            assert router.detect_content_type(sample) == first


# ═══════════════════════════════════════════════════════════════════════
# Branch-coverage: individual markdown signals (HR, blockquote, tilde)
# ═══════════════════════════════════════════════════════════════════════

class TestMarkdownSignalBranches:
    """Cover every individual signal and pair used in ``_looks_like_markdown``."""

    def test_heading_plus_hr_is_markdown(self, router: ContentRouter) -> None:
        assert router.detect_content_type("# Title\n\n---\n\nBody.") == "markdown"

    def test_blockquote_plus_link_is_markdown(self, router: ContentRouter) -> None:
        assert router.detect_content_type("> quote\n\n[docs](https://x)") == "markdown"

    def test_tilde_fence_detects_code(self, router: ContentRouter) -> None:
        assert router.detect_content_type("~~~python\nx = 1\n~~~") == "code"

    def test_hr_plus_list_is_markdown(self, router: ContentRouter) -> None:
        assert router.detect_content_type("---\n\n- item\n- two") == "markdown"

    def test_list_plus_code_fence_plus_heading(self, router: ContentRouter) -> None:
        doc = "# H\n\n- item\n\n```bash\necho hi\n```"
        assert router.detect_content_type(doc) == "code"  # code > markdown

    def test_asterisk_list_item(self, router: ContentRouter) -> None:
        assert router.detect_content_type("# H\n\n* item\n* two") == "markdown"

    def test_link_in_code_fence_is_code(self, router: ContentRouter) -> None:
        """Link inside a code fence → code (code detector fires first)."""
        code = "```\n[text](url)\n```"
        assert router.detect_content_type(code) == "code"

    def test_solo_heading_below_min_lines(self, router: ContentRouter) -> None:
        """A single heading line: 1 non-empty line < min_lines(2) → not markdown."""
        assert router.detect_content_type("# Just a heading") == "text"

    def test_solo_link_below_min_lines(self, router: ContentRouter) -> None:
        assert router.detect_content_type("[docs](https://x)") == "text"

    def test_solo_code_fence_always_code(self, router: ContentRouter) -> None:
        """Code fences bypass the min_lines gate (checked before markdown)."""
        assert router.detect_content_type("```\nx\n```") == "code"

    def test_conversation_beats_markdown(self, router: ContentRouter) -> None:
        """Two role markers beat markdown signals."""
        doc = "# Notes\n\n- item\n\nUser: hello\nAssistant: hi"
        assert router.detect_content_type(doc) == "conversation"

    def test_logs_beats_markdown(self, router: ContentRouter) -> None:
        """Log-level token beats markdown signals."""
        doc = "# Title\n\n- item\n\nERROR something bad happened"
        assert router.detect_content_type(doc) == "logs"

    def test_solo_hr_below_min_lines(self, router: ContentRouter) -> None:
        """A horizontal rule alone: 1 non-empty line < min_lines → not markdown."""
        assert router.detect_content_type("---") == "text"

    def test_code_beats_markdown_even_with_two_signals(
        self, router: ContentRouter
    ) -> None:
        """A keyword line makes it code even with heading+list."""
        doc = "# Title\n\n- item\n\ndef foo():\n    pass"
        assert router.detect_content_type(doc) == "code"
