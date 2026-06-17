"""Heuristic content-type detection and routing for the compression layer.

The :class:`ContentRouter` classifies raw text strings into one of six
content types so that the compression pipeline can select the optimal
compression strategy for each format.

Detection order (highest priority first)
------------------------------------------
1. ``json``     — structural parse via :mod:`json`; must be a valid object
                 or array at the top level.
2. ``code``     — strong indicators of programming-language syntax
                 (brackets, language keywords, code-fence markers).
3. ``logs``     — repeated timestamp patterns or standard log-level
                 tokens (``INFO``, ``WARN``, ``ERROR``, ``DEBUG``, …).
4. ``conversation`` — role markers such as ``User:``, ``Assistant:``,
                 ``Human:``, ``AI:``, or ``role`` keys inside JSON.
5. ``markdown`` — headings (``#``), links (``[text](url)``), unordered
                 lists (``- `` or ``* ``), or fenced code blocks.
6. ``text``     — fallback for anything that matches no specialised
                 detector.

Design rationale
-----------------
Detection is deliberately heuristic and ordered by priority.  Structural
formats (JSON, code) are tested first because they have unambiguous
syntactic fingerprints.  Semi-structured formats (logs, conversations)
come next, followed by lightweight markup (markdown).  Plain text is the
default fallback.

Each detector is a private method prefixed with ``_looks_like_`` so that
individual heuristics can be unit-tested, extended, or replaced without
touching the public API.
"""

from __future__ import annotations

import json
import re
from typing import Literal


# ═══════════════════════════════════════════════════════════════════════
# Public type alias
# ═══════════════════════════════════════════════════════════════════════

ContentType = Literal["json", "code", "logs", "conversation", "markdown", "text"]
"""The set of content types recognised by the :class:`ContentRouter`."""

# ═══════════════════════════════════════════════════════════════════════
# Pre-compiled patterns
# ═══════════════════════════════════════════════════════════════════════

# JSON: fast-reject whitespace check before handing off to json.loads.
# A valid JSON document starts with {, [, or a quote after stripping.
_RE_JSON_START = re.compile(r"\s*[\[\{\"]")

# ── Code indicators ─────────────────────────────────────────────────

# Code-fence markers: ``` or ~~~ optionally followed by a language tag.
_RE_CODE_FENCE = re.compile(r"(`{3,}|~{3,})\s*\w*")

# Common language keywords (line-anchored, word-boundary).  A bare
# ``def``, ``class``, ``import``, ``function``, ``const``, ``let``,
# ``var``, ``fn``, ``pub``, ``impl``, ``use``, ``package`` at the start
# of a line is a strong code signal.
_RE_CODE_KEYWORD = re.compile(
    r"^\s*(?:def|class|import|from|return|if|else|elif|for|while|"
    r"try|except|finally|with|async|await|yield|raise|pass|break|"
    r"continue|lambda|global|nonlocal|"
    r"function|const|let|var|if|else|for|while|switch|case|"
    r"fn|pub|impl|use|mod|crate|struct|enum|trait|type|"
    r"package|import|func|interface|public|private|protected|"
    r"static|void|int|float|double|char|string|bool|"
    r"SELECT|FROM|WHERE|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b",
    re.MULTILINE,
)

# Bracket-heavy lines — at least 3 opening or closing brackets on a
# single line (covers most code even without keywords).
_RE_BRACKET_HEAVY = re.compile(r"[{}()\[\]<>].*[{}()\[\]<>].*[{}()\[\]<>]")

# ── Log indicators ──────────────────────────────────────────────────

# ISO-8601-ish timestamps: 2024-01-15T14:23:00, 2024-01-15 14:23:00,
# 14:23:00.123, etc.  Requires at least two timestamp-like occurrences
# to distinguish from a single embedded time.
_RE_TIMESTAMP = re.compile(
    r"(?:\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}(?::\d{2})?"
    r"|\d{2}:\d{2}:\d{2}(?:\.\d+)?)",
)

# Standard log-level tokens, case-insensitive.
_RE_LOG_LEVEL = re.compile(
    r"\b(?:TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL|"
    r"EMERGENCY|SUCCESS)\b",
    re.IGNORECASE,
)

# ── Conversation indicators ─────────────────────────────────────────

# Common role-label patterns at the start of a line.  Covers both the
# simple ``User:`` style and the indented/block variants seen in chat
# transcripts and LLM output logs.
_RE_ROLE_MARKER = re.compile(
    r"^\s*(?:User|Assistant|Human|AI|System|Bot|Model|"
    r"Customer|Agent|Operator|You|Me)\s*:",
    re.MULTILINE,
)

# ``"role": "..."`` style inside JSON (multi-turn datasets).
_RE_JSON_ROLE = re.compile(
    r"""["']role["']\s*:\s*["'](?:user|assistant|system|human|ai)["']""",
    re.IGNORECASE,
)

# ── Markdown indicators ──────────────────────────────────────────────

# ATX headings: ``#``, ``##``, ``###``, etc. at line start.
_RE_HEADING = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)

# Inline links: ``[text](url)``.
_RE_LINK = re.compile(r"\[[^\]]+\]\([^)]+\)")

# Unordered list items: ``- `` or ``* `` at line start.
_RE_UNORDERED_LIST = re.compile(r"^\s*[-*]\s+\S", re.MULTILINE)

# Blockquotes: ``> `` at line start.
_RE_BLOCKQUOTE = re.compile(r"^\s*>\s+\S", re.MULTILINE)

# Horizontal rules: three or more ``-``, ``*``, or ``_`` alone on a line.
_RE_HR = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)


# ═══════════════════════════════════════════════════════════════════════
# ContentRouter
# ═══════════════════════════════════════════════════════════════════════

class ContentRouter:
    """Heuristic content-type detector for the compression pipeline.

    Analyses a string and returns one of :data:`ContentType` values
    indicating the most likely format of the content.  Detection is
    performed by applying a prioritised sequence of structural and
    heuristic checks.

    The router is stateless — all methods are pure functions of their
    input — so a single instance can be safely shared across threads or
    reused across many calls.

    Parameters
    ----------
    min_lines : int
        Minimum number of non-empty lines required before line-based
        heuristics (logs, conversation, markdown) are considered.  A
        low value makes detection more sensitive but increases false-
        positive risk.  Default is ``2``.
    timestamp_hits : int
        Number of timestamp matches required to classify content as
        ``logs``.  Default is ``2``.

    Example
    -------
    ::

        router = ContentRouter()
        print(router.detect_content_type('{"key": "value"}'))
        # -> json

        print(router.detect_content_type('def hello():\\n    return 42'))
        # -> code
    """

    __slots__ = ("_min_lines", "_timestamp_hits")

    def __init__(
        self,
        *,
        min_lines: int = 2,
        timestamp_hits: int = 2,
    ) -> None:
        if min_lines < 1:
            raise ValueError(f"min_lines must be >= 1, got {min_lines}")
        if timestamp_hits < 1:
            raise ValueError(f"timestamp_hits must be >= 1, got {timestamp_hits}")
        self._min_lines = min_lines
        self._timestamp_hits = timestamp_hits

    # ── Public API ────────────────────────────────────────────────────

    def detect_content_type(self, content: str) -> ContentType:
        """Classify *content* into one of the recognised content types.

        The detection pipeline runs checks in a fixed priority order:

        1. **JSON** — structural parse via :func:`json.loads`.
        2. **Code** — keywords, brackets, code fences.
        3. **Logs** — repeated timestamps and log-level tokens.
        4. **Conversation** — role markers or ``"role"`` keys.
        5. **Markdown** — headings, links, lists.
        6. **Text** — default fallback.

        Parameters
        ----------
        content : str
            The raw text to classify.  Empty or whitespace-only strings
            are classified as ``text``.

        Returns
        -------
        ContentType
            One of ``json``, ``code``, ``logs``, ``conversation``,
            ``markdown``, or ``text``.
        """
        if not content or not content.strip():
            return "text"

        if self._looks_like_json(content):
            return "json"
        if self._looks_like_code(content):
            return "code"
        if self._looks_like_logs(content):
            return "logs"
        if self._looks_like_conversation(content):
            return "conversation"
        if self._looks_like_markdown(content):
            return "markdown"
        return "text"

    # ── Detector implementations ───────────────────────────────────────

    def _looks_like_json(self, content: str) -> bool:
        """Return ``True`` if *content* is a valid JSON object or array.

        A fast-reject regex is applied first to avoid the cost of
        :func:`json.loads` on clearly non-JSON input.
        """
        if not _RE_JSON_START.match(content):
            return False
        try:
            parsed = json.loads(content)
            return isinstance(parsed, dict | list)
        except (json.JSONDecodeError, ValueError):
            return False

    def _looks_like_code(self, content: str) -> bool:
        """Return ``True`` if *content* strongly resembles source code.

        Detection relies on three independent signals:

        * **Code fences** — `````lang`` or ``~~~lang`` markers.
        * **Language keywords** — common keywords at the start of a line.
        * **Bracket density** — three or more bracket characters on a
          single line.

        At least one signal must fire.
        """
        if _RE_CODE_FENCE.search(content):
            return True
        if _RE_CODE_KEYWORD.search(content):
            return True
        # Check bracket density on a line-by-line basis; at least
        # one line must be bracket-heavy.  Markdown link syntax
        # ``[text](url)`` is stripped first to avoid false positives
        # on documents that are primarily markdown with embedded links.
        for line in content.splitlines():
            cleaned = _RE_LINK.sub("", line)
            if _RE_BRACKET_HEAVY.match(cleaned):
                return True
        return False

    def _looks_like_logs(self, content: str) -> bool:
        """Return ``True`` if *content* resembles application log output.

        Content is classified as logs when **both** of the following are
        true:

        * The number of non-empty lines meets the ``min_lines`` threshold.
        * At least ``timestamp_hits`` timestamp patterns are found **or**
          at least one log-level token is found.
        """
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if len(lines) < self._min_lines:
            return False

        ts_matches = len(_RE_TIMESTAMP.findall(content))
        has_level = bool(_RE_LOG_LEVEL.search(content))

        return (ts_matches >= self._timestamp_hits) or has_level

    def _looks_like_conversation(self, content: str) -> bool:
        """Return ``True`` if *content* resembles a chat transcript.

        Detection fires when at least two role-marker lines are found,
        **or** when ``"role": "user"`` / ``"role": "assistant"`` style
        keys are present (common in LLM training datasets).

        A ``min_lines`` check is applied first to avoid false positives
        on short strings.
        """
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if len(lines) < self._min_lines:
            return False

        role_hits = len(_RE_ROLE_MARKER.findall(content))
        has_json_role = bool(_RE_JSON_ROLE.search(content))

        return (role_hits >= 2) or has_json_role

    def _looks_like_markdown(self, content: str) -> bool:
        """Return ``True`` if *content* contains Markdown formatting.

        A ``min_lines`` check is applied first.  Then the content must
        exhibit at least **two** of the following signals:

        * ATX headings (``#``)
        * Inline links (``[text](url)``)
        * Unordered list items (``- `` / ``* ``)
        * Blockquotes (``> ``)
        * Horizontal rules (``---``)
        * Code fences (`````lang`` / ``~~~lang``)
        """
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if len(lines) < self._min_lines:
            return False

        signals = 0
        if _RE_HEADING.search(content):
            signals += 1
        if _RE_LINK.search(content):
            signals += 1
        if _RE_UNORDERED_LIST.search(content):
            signals += 1
        if _RE_BLOCKQUOTE.search(content):
            signals += 1
        if _RE_HR.search(content):
            signals += 1
        if _RE_CODE_FENCE.search(content):
            signals += 1

        return signals >= 2


# ═══════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    "ContentRouter",
    "ContentType",
]
