"""Domain-specific summarization and extraction for the compression layer.

The :class:`ContextCompressor` applies the right summarisation strategy
for each content type recognised by the
:class:`~neuromem.compression.router.ContentRouter`:

- :meth:`compress_logs`         — extracts errors, severity, and a
  structural timeline of key milestones from log output.
- :meth:`compress_conversation` — captures facts, decisions, open
  tasks, and entities from a multi-turn chat transcript.
- :meth:`compress_rag`          — deduplicates overlapping retrieval
  chunks, removes semantic redundancy, and preserves source citations.
- :meth:`compress_code`         — uses Python's standard :mod:`ast`
  module to extract functions, classes, imports, and docstrings,
  omitting raw implementation detail.

LLM integration
---------------
Semantic extraction (conversation facts/decisions, RAG deduplication)
is delegated to an LLM provider so callers can inject any backend
(OpenAI, Anthropic, a local model, etc.).  The provider contract is
defined by :class:`BaseLLMProvider`.  When no provider is supplied the
compressor falls back to a deterministic :class:`MockLLMProvider` whose
heuristics work entirely offline — enough to populate the output
schemas and to keep the pipeline functional in tests and CI.

Deterministic fallbacks
-----------------------
:meth:`compress_logs` and :meth:`compress_code` are fully structural
and never invoke the LLM — they rely on regex and :mod:`ast`
respectively.  This makes them fast, cheap, and reproducible.  Only
:meth:`compress_conversation` and :meth:`compress_rag` consult the LLM,
and both degrade gracefully to structural heuristics when the provider
is a mock or when an LLM call fails.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from neuromem.compression.models import (
    ConversationCompressionOutput,
    LogCompressionOutput,
)


# ═══════════════════════════════════════════════════════════════════════
# LLM provider abstraction
# ═══════════════════════════════════════════════════════════════════════

@runtime_checkable
class BaseLLMProvider(Protocol):
    """Minimal LLM invocation contract for semantic extraction.

    Implementations need only expose a single :meth:`complete` method.
    The compressor treats the model as a stateless text-in / text-out
    function and performs all structured parsing itself, so providers
    are free to return plain text (JSON is parsed defensively).

    To wire in a real backend, implement this protocol — for example::

        class OpenAIProvider:
            def __init__(self, client) -> None:
                self._client = client

            def complete(self, system_prompt: str, user_prompt: str) -> str:
                resp = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return resp.choices[0].message.content
    """

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's completion for the given prompts.

        Parameters
        ----------
        system_prompt:
            Task instructions and output-format specification.
        user_prompt:
            The concrete content to process.

        Returns
        -------
        str
            The raw model output.  When JSON is requested, the
            compressor will attempt to parse it defensively and fall
            back to heuristics on failure.
        """
        ...


class MockLLMProvider:
    """Deterministic, offline stand-in for :class:`BaseLLMProvider`.

    Produces structured JSON whose fields are derived from simple
    heuristics over the input.  This keeps the compression pipeline
    functional without network access or API keys, and gives tests a
    predictable provider to assert against.

    The mock inspects the *system prompt* to decide which extraction
    schema to emit, so the compressor's prompt strings double as a
    lightweight routing signal.
    """

    __slots__ = ()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return a heuristic JSON completion derived from *user_prompt*."""
        prompt_lower = system_prompt.lower()
        if "conversation" in prompt_lower:
            return json.dumps(_structural_conversation(user_prompt))
        if "rag" in prompt_lower or "deduplicat" in prompt_lower:
            return json.dumps({"summary": _mock_rag_summary(user_prompt)})
        # Generic fallback: echo a trimmed version of the input.
        return json.dumps({"summary": user_prompt.strip()[:500]})


# ═══════════════════════════════════════════════════════════════════════
# Pre-compiled patterns for log parsing
# ═══════════════════════════════════════════════════════════════════════

# Severity ranking (lowest → highest).  Used to pick the dominant
# severity for a log batch.
_SEVERITY_RANK: dict[str, int] = {
    "trace": 0,
    "debug": 1,
    "info": 2,
    "warn": 3,
    "warning": 3,
    "error": 4,
    "critical": 5,
    "fatal": 5,
}

# A log line typically looks like:
#   2024-01-15 14:23:00 ERROR  Something broke
# We capture the optional leading timestamp and the level token.
_RE_LOG_LINE = re.compile(
    r"^(?P<ts>(?:\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?|\d{2}:\d{2}:\d{2}(?:\.\d+)?)?\s*)?"
    r"(?P<level>TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)"
    r"\b[:\s]*",
    re.IGNORECASE | re.MULTILINE,
)

# Lines that mark structural milestones — startup, shutdown, phase
# transitions, scaling events, connections, etc.
_RE_MILESTONE = re.compile(
    r"\b(?:start(?:ed|ing)?|shutdown|stopping|ready|listening|"
    r"connected|disconnected|deploy(?:ed|ing)?|"
    r"scal(?:e|ing)(?:d|up|down)?|reload(?:ed|ing)?|restart(?:ed|ing)?|"
    r"init(?:ializ(?:ed|ing))?|boot(?:ed|ing)?|"
    r"complet(?:ed|ing)|finish(?:ed|ing)?|fail(?:ed|ing)?|"
    r"crash(?:ed)?|recover(?:ed|ing)?)\b",
    re.IGNORECASE,
)

# ── Conversation patterns ────────────────────────────────────────────

# Role markers at the start of a line: "User:", "Assistant:", etc.
_RE_ROLE_TURN = re.compile(
    r"^\s*(?P<role>User|Assistant|Human|AI|System|Bot|Model|Customer|Agent)\s*:\s*(?P<text>.*)$",
    re.MULTILINE,
)

# Citation markers embedded in retrieved chunks: [source: …], [doc: …].
_RE_CITATION = re.compile(r"\[(?:source|doc):\s*[^\]]*\]", re.IGNORECASE)

# ── Code summarisation helpers ──────────────────────────────────────

# Maximum number of leading docstring/comment lines retained per node.
_CODE_DOC_PREVIEW_LINES = 3


# ═══════════════════════════════════════════════════════════════════════
# ContextCompressor
# ═══════════════════════════════════════════════════════════════════════

class ContextCompressor:
    """Domain-specific compressor that picks a strategy per content type.

    The compressor combines **deterministic structural parsing** (logs,
    code) with **LLM-backed semantic extraction** (conversation, RAG).
    An LLM provider is injected via the constructor; when omitted, a
    :class:`MockLLMProvider` is used so the pipeline runs fully offline.

    Parameters
    ----------
    llm:
        Optional :class:`BaseLLMProvider` implementation.  When
        ``None`` (the default), a :class:`MockLLMProvider` is used.

    Example
    -------
    ::

        compressor = ContextCompressor()  # offline mock
        out = compressor.compress_code("def f(x):\\n    return x")
        print(out)

        # With a real provider:
        compressor = ContextCompressor(llm=MyOpenAIProvider(client))
    """

    __slots__ = ("_llm",)

    def __init__(self, *, llm: BaseLLMProvider | None = None) -> None:
        # Inject a provider or fall back to the deterministic mock.
        self._llm: BaseLLMProvider = llm if llm is not None else MockLLMProvider()

    @property
    def llm(self) -> BaseLLMProvider:
        """The LLM provider currently wired into the compressor."""
        return self._llm

    # ── 1. Logs ────────────────────────────────────────────────────────

    def compress_logs(self, logs: str) -> LogCompressionOutput:
        """Compress raw log output into errors, severity, and milestones.

        This method is **fully structural** — it never invokes the LLM.
        It scans each line for:

        * **Errors** — any line whose level is ``ERROR``, ``CRITICAL``,
          or ``FATAL`` (deduplicated, order-preserving).
        * **Severity** — the highest level observed across the batch,
          normalised to one of ``debug``/``info``/``warning``/``error``/
          ``critical``.
        * **Key events** — lines mentioning structural milestones
          (startup, shutdown, scaling, deploy, recovery, …), preserving
          their timestamp for a compact timeline.

        Parameters
        ----------
        logs:
            Raw log text, one entry per line.

        Returns
        -------
        LogCompressionOutput
        """
        if not logs or not logs.strip():
            return LogCompressionOutput(
                summary="(empty log batch)",
                errors=[],
                severity="info",
                key_events=[],
            )

        lines = logs.splitlines()
        errors: list[str] = []
        seen_errors: set[str] = set()
        observed_levels: list[str] = []
        key_events: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            match = _RE_LOG_LINE.match(stripped)
            level = match.group("level").lower() if match else None
            ts = (match.group("ts").strip() if match and match.group("ts") else "")
            message = stripped[match.end():].strip() if match else stripped

            if level is not None:
                observed_levels.append(level)
                if level in {"error", "critical", "fatal"}:
                    error_line = f"{ts} {level.upper()} {message}".strip()
                    # Dedup on normalised message text (ignoring timestamp)
                    # so the same error repeated over time is reported once.
                    dedup_key = re.sub(r"\s+", " ", message.lower()).strip()
                    if dedup_key not in seen_errors:
                        seen_errors.add(dedup_key)
                        errors.append(error_line)

            if _RE_MILESTONE.search(stripped):
                key_events.append(stripped)

        severity = _dominant_severity(observed_levels)
        summary = _log_summary(lines, errors, severity, key_events)

        logger.debug(
            "compress_logs: {} lines → {} errors, severity={}, {} milestones",
            len(lines), len(errors), severity, len(key_events),
        )

        return LogCompressionOutput(
            summary=summary,
            errors=errors,
            severity=severity,
            key_events=key_events,
        )

    # ── 2. Conversation ───────────────────────────────────────────────

    def compress_conversation(self, history: str) -> ConversationCompressionOutput:
        """Compress a multi-turn conversation transcript.

        Extracts a prose summary plus structured lists of important
        facts, decisions, open tasks, and entities.  Extraction is
        delegated to the LLM provider with a strict JSON schema; if the
        provider is a mock or the call fails, a structural fallback
        parses the transcript by role turns and keyword heuristics.

        Parameters
        ----------
        history:
            Conversation text with role markers (``User:``,
            ``Assistant:``, etc.) or a JSON array of
            ``{"role", "content"}`` messages.

        Returns
        -------
        ConversationCompressionOutput
        """
        if not history or not history.strip():
            return ConversationCompressionOutput(
                summary="(empty conversation)",
                important_facts=[],
                open_tasks=[],
                decisions=[],
                entities=[],
            )

        system_prompt = (
            "You are a conversation compression engine. Extract a concise "
            "summary and structured fields from the conversation. Respond "
            "ONLY with valid JSON matching this schema:\n"
            "{\n"
            '  "summary": str,\n'
            '  "important_facts": [str],\n'
            '  "open_tasks": [str],\n'
            '  "decisions": [str],\n'
            '  "entities": [str]\n'
            "}\n"
            "Omit any field you cannot populate rather than guessing."
        )

        extracted = self._safe_llm_json(system_prompt, history)
        if extracted is None:
            logger.debug("compress_conversation: LLM unavailable, using structural fallback")
            extracted = _structural_conversation(history)

        return ConversationCompressionOutput(
            summary=str(extracted.get("summary", "(no summary)")).strip() or "(no summary)",
            important_facts=_as_str_list(extracted.get("important_facts")),
            open_tasks=_as_str_list(extracted.get("open_tasks")),
            decisions=_as_str_list(extracted.get("decisions")),
            entities=_as_str_list(extracted.get("entities")),
        )

    # ── 3. RAG ─────────────────────────────────────────────────────────

    def compress_rag(self, chunks: list[str]) -> str:
        """Deduplicate and merge overlapping retrieval chunks.

        Given a list of retrieved text chunks (typically from a vector
        search), this method removes semantic redundancy and produces a
        single coherent passage.  Source citations are preserved: any
        ``[source: …]`` or ``[doc: …]`` markers present in the input are
        retained in the output.

        The deduplication strategy is hybrid:

        1. **Structural dedup** — exact and near-exact duplicate
           sentences are removed using normalised fingerprinting.  This
           always runs and needs no LLM.
        2. **LLM merge** — the LLM is asked to fuse the remaining
           sentences into fluent prose.  When the provider is the mock
           (or the call fails), the structurally-deduplicated sentences
           are joined directly.

        Parameters
        ----------
        chunks:
            List of retrieved text passages, possibly overlapping.

        Returns
        -------
        str
            A single merged, deduplicated passage with citations.
        """
        if not chunks:
            return ""

        # Normalise and split into sentences across all chunks.
        sentences: list[str] = []
        for chunk in chunks:
            for sentence in _split_sentences(chunk):
                norm = sentence.strip()
                if norm:
                    sentences.append(norm)

        # Collect citations from the ORIGINAL sentences up front, before
        # deduplication.  A near-duplicate sentence that gets dropped may
        # be the one carrying a unique ``[source: …]`` / ``[doc: …]``
        # marker, so capturing them here lets us reattach any citation
        # that the dedup pass (or the LLM merge below) drops.
        citations = _extract_citations(" ".join(sentences))

        # Structural dedup by normalised fingerprint.
        deduped = _dedup_sentences(sentences)
        if not deduped:
            return ""

        if len(deduped) == 1:
            # Honour citation preservation even on the single-sentence path.
            return _ensure_citations(deduped[0], citations)

        # Preserve citations before/after LLM merge.
        joined = " ".join(deduped)

        system_prompt = (
            "You are a retrieval-augmented generation (RAG) deduplication "
            "engine. The user message contains overlapping retrieved "
            "sentences. Merge them into a single coherent passage that "
            "removes redundancy while preserving all distinct facts and "
            "any [source: ...] / [doc: ...] citations. Respond ONLY with "
            "valid JSON: {\"summary\": str}"
        )
        extracted = self._safe_llm_json(system_prompt, joined)
        if extracted is not None and extracted.get("summary"):
            merged = str(extracted["summary"]).strip()
        else:
            logger.debug("compress_rag: LLM unavailable, using structural join")
            merged = joined

        # Re-attach any citations that the merge may have dropped.
        merged = _ensure_citations(merged, citations)
        return merged

    # ── 4. Code ────────────────────────────────────────────────────────

    def compress_code(self, code: str) -> str:
        """Summarise Python source via the standard :mod:`ast` module.

        Walks the parse tree and emits a compact skeleton containing:

        * **Module docstring** (if present).
        * **Imports** — one line per ``import`` / ``from … import``.
        * **Functions** — signature (name + arguments) and a short
          docstring preview.  Implementation bodies are omitted.
        * **Classes** — name, bases, docstring preview, and nested
          methods (signatures only).

        Non-Python code or syntactically invalid input falls back to a
        header line followed by the raw source, so the method never
        raises on unparseable input.

        Parameters
        ----------
        code:
            Python source code text.

        Returns
        -------
        str
            A markdown-formatted structural summary.
        """
        if not code or not code.strip():
            return "```\n(empty source)\n```"

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            logger.debug("compress_code: parse failed ({}), returning raw", exc.msg)
            return f"```\n# WARNING: could not parse as Python ({exc.msg})\n{code.strip()}\n```"

        parts: list[str] = ["```python"]

        module_doc = ast.get_docstring(tree)
        if module_doc:
            preview = _preview_docstring(module_doc)
            parts.append(f'"""{preview}"""')
            parts.append("")

        imports = _extract_imports(tree)
        if imports:
            parts.append("# Imports")
            parts.extend(imports)
            parts.append("")

        has_top_level = False
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                has_top_level = True
                parts.append(_summarize_function(node))
            elif isinstance(node, ast.ClassDef):
                has_top_level = True
                parts.append(_summarize_class(node))

        if not has_top_level and not imports and not module_doc:
            parts.append("# (no functions, classes, or imports detected)")

        parts.append("```")
        return "\n".join(parts)

    # ── Internal helpers ────────────────────────────────────────────────

    def _safe_llm_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        """Call the LLM and parse JSON defensively.

        Returns ``None`` if the provider raises or the output is not
        valid JSON / not a JSON object.  Callers must handle ``None`` by
        falling back to structural heuristics.
        """
        try:
            raw = self._llm.complete(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 — provider-agnostic
            logger.warning("LLM call failed ({}); falling back to heuristics", exc)
            return None
        return _parse_json_object(raw)


# ═══════════════════════════════════════════════════════════════════════
# Pure helper functions (module-level for testability)
# ═══════════════════════════════════════════════════════════════════════

def _dominant_severity(levels: list[str]) -> str:
    """Return the highest severity among *levels*, defaulting to ``info``."""
    if not levels:
        return "info"
    best = "info"
    best_rank = _SEVERITY_RANK["info"]
    for lvl in levels:
        rank = _SEVERITY_RANK.get(lvl.lower(), -1)
        if rank > best_rank:
            best_rank = rank
            best = "warning" if lvl.lower() in {"warn", "warning"} else lvl.lower()
    # Normalise aliases.
    if best in {"warn", "warning"}:
        return "warning"
    if best in {"fatal"}:
        return "critical"
    return best


def _log_summary(
    lines: list[str],
    errors: list[str],
    severity: str,
    key_events: list[str],
) -> str:
    """Build a one-paragraph prose summary of a log batch."""
    return (
        f"Analysed {len(lines)} log line(s). "
        f"Dominant severity: {severity.upper()}. "
        f"Found {len(errors)} error(s) and {len(key_events)} key milestone(s)."
    )


def _as_str_list(value: Any) -> list[str]:
    """Coerce an arbitrary JSON value into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    """Parse *raw* as a JSON object, tolerating surrounding prose.

    Looks for the first ``{`` and the last ``}`` to extract a JSON
    substring, which handles models that wrap JSON in markdown fences or
    conversational text.  Returns ``None`` on any parse failure.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    # Extract the outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences on ``.``/``!``/``?`` boundaries.

    A trailing ``[source: …]`` / ``[doc: …]`` citation marker that
    follows a sentence ender is **reattached** to the preceding sentence
    rather than emitted as a standalone fragment.  This keeps each
    sentence and its citation together through deduplication.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    raw = re.split(r"(?<=[.!?])\s+", cleaned)
    out: list[str] = []
    for frag in raw:
        frag = frag.strip()
        if not frag:
            continue
        # Reattach a leading citation fragment to the previous sentence.
        if _RE_CITATION.match(frag) and out:
            out[-1] = f"{out[-1]} {frag}"
        else:
            out.append(frag)
    return out


def _sentence_fingerprint(sentence: str) -> str:
    """Return a normalised fingerprint for near-duplicate detection.

    Lowercases, collapses whitespace, and strips citation markers **and
    trailing sentence-ending punctuation** so that sentences differing
    only by a final period, exclamation, or question mark — or by a
    citation suffix — share a fingerprint.  Without the trailing-punct
    strip, ``"The sky is blue."`` and ``"The sky is blue!"`` would hash
    to different fingerprints and slip past dedup, and the containment
    check would fail because the embedded period breaks the substring
    match (``"...database." in "...database built for scale."`` is False).
    """
    no_cite = _RE_CITATION.sub("", sentence)
    norm = re.sub(r"\s+", " ", no_cite.lower()).strip()
    return norm.rstrip(".!?;:,\u2026")


def _dedup_sentences(sentences: list[str]) -> list[str]:
    """Remove duplicate and near-duplicate sentences.

    Two strategies are combined:

    * **Exact fingerprint dedup** — sentences with the same normalised
      fingerprint (citation-stripped, lowercased) collapse to the first
      occurrence, which retains its citation.
    * **Containment dedup** — a sentence whose fingerprint is a proper
      substring of a later sentence's fingerprint is dropped, since the
      longer sentence is the more informative superset.
    """
    fingerprinted = [_sentence_fingerprint(s) for s in sentences]
    n = len(sentences)

    # Mark sentences to keep (default: keep the first of exact dups).
    keep = [True] * n
    seen: set[str] = set()
    for i in range(n):
        fp = fingerprinted[i]
        if not fp:
            keep[i] = False
            continue
        if fp in seen:
            keep[i] = False
        else:
            seen.add(fp)

    # Containment pass: drop a shorter sentence subsumed by a longer one.
    for i in range(n):
        if not keep[i]:
            continue
        fp_i = fingerprinted[i]
        for j in range(n):
            if i == j or not keep[j]:
                continue
            fp_j = fingerprinted[j]
            # Drop i if it is a proper substring of j (j is more specific).
            if len(fp_i) < len(fp_j) and fp_i in fp_j:
                keep[i] = False
                break

    return [sentences[i] for i in range(n) if keep[i]]


def _extract_citations(text: str) -> list[str]:
    """Return all ``[source: …]`` / ``[doc: …]`` markers in *text*."""
    return _RE_CITATION.findall(text)


def _ensure_citations(text: str, citations: list[str]) -> str:
    """Re-append any *citations* missing from *text*."""
    if not citations:
        return text
    present = set(_extract_citations(text))
    missing = [c for c in citations if c not in present]
    if not missing:
        return text
    return text.rstrip() + " " + " ".join(missing)


def _preview_docstring(doc: str, max_lines: int = _CODE_DOC_PREVIEW_LINES) -> str:
    """Return the first *max_lines* of a docstring, single-line collapsed."""
    lines = [ln.strip() for ln in doc.strip().splitlines() if ln.strip()]
    preview = " ".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += " ..."
    return preview


def _format_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a function's argument list as a signature string."""
    args: list[str] = []
    # Positional args (including those with defaults handled below).
    all_args = node.args
    pos = all_args.posonlyargs + all_args.args
    defaults = all_args.defaults
    # defaults align to the tail of pos.
    n_without_default = len(pos) - len(defaults)
    for i, a in enumerate(pos):
        if i >= n_without_default:
            args.append(f"{a.arg}=…")
        else:
            args.append(a.arg)
    if all_args.vararg is not None:
        args.append(f"*{all_args.vararg.arg}")
    if all_args.kwarg is not None:
        args.append(f"**{all_args.kwarg.arg}")
    if all_args.kwonlyargs:
        if all_args.vararg is None:
            args.append("*")
        for a in all_args.kwonlyargs:
            args.append(a.arg)
    return ", ".join(args)


def _summarize_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a single function as a signature + docstring preview."""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args_str = _format_args(node)
    line = f"{prefix} {node.name}({args_str})"
    # Decorators hint at intent (e.g. @property, @staticmethod).
    if node.decorator_list:
        decos = ", ".join(_unparse_decorators(node.decorator_list))
        line = f"@{decos}\n{line}"
    doc = ast.get_docstring(node)
    if doc:
        line += f"  # {_preview_docstring(doc)}"
    return line


def _unparse_decorators(decos: list[ast.expr]) -> list[str]:
    """Best-effort string rendering of decorator expressions."""
    out: list[str] = []
    for d in decos:
        try:
            out.append(ast.unparse(d))
        except Exception:  # noqa: BLE001
            out.append("…")
    return out


def _summarize_class(node: ast.ClassDef) -> str:
    """Render a class with its bases, docstring, and method signatures."""
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    header = f"class {node.name}"
    if bases:
        header += f"({bases})"
    if node.decorator_list:
        decos = ", ".join(_unparse_decorators(node.decorator_list))
        header = f"@{decos}\n{header}"

    lines = [header]
    doc = ast.get_docstring(node)
    if doc:
        lines.append(f'    """{_preview_docstring(doc)}"""')

    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Reuse _summarize_function for full decorator support.
            method_line = _summarize_function(child)
            # Indent the method (and any decorator line above it) into the class body.
            for mline in method_line.splitlines():
                lines.append(f"    {mline}")
        elif isinstance(child, ast.Assign):
            # Class-level attribute annotations.
            for target in child.targets:
                if isinstance(target, ast.Name):
                    lines.append(f"    {target.id} = …")

    return "\n".join(lines)


def _extract_imports(tree: ast.Module) -> list[str]:
    """Render all import statements in *tree* as their source lines."""
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(alias.name for alias in node.names)
            imports.append(f"from {module} import {names}")
    return imports


# ── Structural conversation extraction ───────────────────────────────
#
# This heuristic extractor serves double duty: it is the offline
# implementation behind :class:`MockLLMProvider` and the deterministic
# fallback used by :meth:`ContextCompressor.compress_conversation` when
# the LLM is unavailable or returns unparseable output.

def _structural_conversation(history: str) -> dict[str, Any]:
    """Heuristic offline extraction of a conversation transcript.

    Parses role turns, then classifies each sentence into facts,
    decisions, or open tasks via keyword heuristics, and extracts
    capitalised tokens as candidate named entities.  Used both by the
    mock LLM provider and as the structural fallback.
    """
    turns = list(_RE_ROLE_TURN.finditer(history))
    assistant_text = " ".join(
        t.group("text").strip() for t in turns if t.group("role").lower() in {"assistant", "ai", "model", "bot"}
    )
    all_text = " ".join(t.group("text").strip() for t in turns) or history.strip()

    facts: list[str] = []
    decisions: list[str] = []
    tasks: list[str] = []

    for sentence in _split_sentences(all_text):
        low = sentence.lower()
        if any(k in low for k in ("decided", "decision", "we will use", "let's go with", "agreed to", "chose")):
            decisions.append(sentence)
        elif any(k in low for k in ("todo", "to-do", "task", "need to", "should", "must", "open item", "follow up", "pending")):
            tasks.append(sentence)
        elif any(k in low for k in ("is ", "are ", "means", "defined as", "supports", "located", "was born", "founded")):
            facts.append(sentence)

    entities = sorted(set(_extract_capitalised_entities(all_text + " " + assistant_text)))
    summary = assistant_text.strip()[:200] or all_text.strip()[:200] or "(no summary)"

    return {
        "summary": summary,
        "important_facts": facts,
        "open_tasks": tasks,
        "decisions": decisions,
        "entities": entities,
    }


_RE_CAPITALISED = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]+(?:[-:][A-Za-z0-9]+)*|[A-Z]{2,})\b")
_STOPWORDS_ENTITIES = {
    "User", "Assistant", "Human", "AI", "System", "Bot", "Model",
    "Customer", "Agent", "The", "A", "An", "I", "We", "They", "It",
    "This", "That", "These", "Those", "Yes", "No", "True", "False",
    "TODO", "INFO", "WARN", "ERROR",
}


def _extract_capitalised_entities(text: str) -> list[str]:
    """Extract candidate named entities (capitalised tokens)."""
    return [
        m.group()
        for m in _RE_CAPITALISED.finditer(text)
        if m.group() not in _STOPWORDS_ENTITIES
    ]


def _mock_rag_summary(text: str) -> str:
    """Structural RAG summary used by :class:`MockLLMProvider`.

    Deduplicates sentences and re-merges them into one passage,
    preserving citations.  This mirrors what a good LLM would do.
    """
    sentences = _dedup_sentences(_split_sentences(text))
    merged = " ".join(sentences)
    citations = _extract_citations(text)
    return _ensure_citations(merged, citations)


# ═══════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    "ContextCompressor",
    "BaseLLMProvider",
    "MockLLMProvider",
]
