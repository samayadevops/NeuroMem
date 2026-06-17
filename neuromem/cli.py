"""NeuroMem command-line interface.

Exposes every major :class:`NeuroMemClient` operation as a sub-command
so the library can be driven from **cmd, PowerShell, Claude Code,
Gemini CLI, bash**, and any other shell or agent that can run a
subprocess.

Usage
-----
::

    neuromem [GLOBAL FLAGS] <sub-command> [OPTIONS]

Output contract
---------------
* Every command prints **valid JSON** to *stdout* and exits ``0`` on
  success.
* On error: exits ``1`` and prints ``{"error": "<message>"}``.
* ``--pretty`` adds indentation for human reading.
* ``--quiet`` silences loguru output on *stderr*.
* ``--format text`` switches to plain human-readable output instead of
  JSON (useful in interactive shells).

Sub-commands
------------
learn, recall, forget, guard, is-blocked, propagate,
list, get, stats, decay, compress, retrieve
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


# ─────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────

def _dump(data: Any, *, pretty: bool, fmt: str) -> None:
    """Print *data* to stdout according to the chosen output format."""
    if fmt == "text":
        _dump_text(data)
    else:
        indent = 2 if pretty else None
        print(json.dumps(data, default=str, indent=indent))


def _dump_text(data: Any) -> None:  # noqa: C901
    """Simple human-readable renderer (best-effort)."""
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                print(f"{k}:")
                _dump_text(v)
            else:
                print(f"  {k}: {v}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print(f"[{i}]")
            _dump_text(item)
    else:
        print(data)


def _ok(data: Any, *, pretty: bool, fmt: str) -> int:
    """Print success payload and return exit code 0."""
    _dump(data, pretty=pretty, fmt=fmt)
    return 0


def _err(message: str, *, pretty: bool, fmt: str) -> int:
    """Print error payload to stdout and return exit code 1."""
    _dump({"error": message}, pretty=pretty, fmt=fmt)
    return 1


# ─────────────────────────────────────────────────────────────────────
# Client factory
# ─────────────────────────────────────────────────────────────────────

def _make_client(data_dir: str, namespace: str) -> Any:
    """Import and create a :class:`NeuroMemClient` (deferred import)."""
    from neuromem.client import NeuroMemClient  # noqa: PLC0415
    return NeuroMemClient.create(data_dir, namespace=namespace)


# ─────────────────────────────────────────────────────────────────────
# Sub-command handlers
# ─────────────────────────────────────────────────────────────────────

def _cmd_learn(args: argparse.Namespace) -> dict[str, Any]:
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
    with _make_client(args.data_dir, args.namespace) as client:
        belief = client.learn(
            args.claim,
            confidence=args.confidence,
            source=args.source,
            namespace=args.namespace,
            tags=tags,
        )
        return {
            "id": belief.id,
            "claim": belief.claim,
            "confidence": belief.confidence,
            "status": belief.status.value,
            "source": belief.source,
            "namespace": belief.namespace,
            "tags": list(belief.tags),
            "created_at": belief.created_at.isoformat(),
        }


def _cmd_recall(args: argparse.Namespace) -> dict[str, Any]:
    with _make_client(args.data_dir, args.namespace) as client:
        results = client.recall(
            query=args.query,
            n_results=args.n,
            namespace=args.namespace,
            min_confidence=args.min_confidence,
            apply_decay=not args.no_decay,
        )
        return {
            "query": args.query,
            "count": len(results),
            "results": [r.to_dict() for r in results],
        }


def _cmd_forget(args: argparse.Namespace) -> dict[str, Any]:
    with _make_client(args.data_dir, args.namespace) as client:
        found = client.forget(args.belief_id, namespace=args.namespace)
        return {"belief_id": args.belief_id, "forgotten": found}


def _cmd_guard(args: argparse.Namespace) -> dict[str, Any]:
    context: dict[str, Any] | None = None
    if args.context:
        try:
            context = json.loads(args.context)
        except json.JSONDecodeError:
            context = {"raw": args.context}

    with _make_client(args.data_dir, args.namespace) as client:
        neg = client.guard(
            args.pattern,
            severity=args.severity,
            block_threshold=args.block_threshold,
            context=context,
            related_belief_id=args.related_belief_id,
            namespace=args.namespace,
        )
        return {
            "id": neg.id,
            "pattern": neg.pattern,
            "severity": neg.severity.value if hasattr(neg.severity, "value") else str(neg.severity),
            "occurrence_count": neg.occurrence_count,
            "block_threshold": neg.block_threshold,
            "should_block": neg.should_block,
            "namespace": neg.namespace,
            "created_at": neg.created_at.isoformat(),
        }


def _cmd_is_blocked(args: argparse.Namespace) -> dict[str, Any]:
    with _make_client(args.data_dir, args.namespace) as client:
        blocked = client.is_blocked(args.pattern, namespace=args.namespace)
        return {"pattern": args.pattern, "blocked": blocked}


def _cmd_propagate(args: argparse.Namespace) -> dict[str, Any]:
    with _make_client(args.data_dir, args.namespace) as client:
        record = client.propagate(
            args.belief_id,
            args.target_namespace,
            trust_factor=args.trust_factor,
            namespace=args.namespace,
        )
        return {
            "id": record.id,
            "belief_id": record.belief_id,
            "source_namespace": record.source_namespace,
            "target_namespace": record.target_namespace,
            "propagated_confidence": record.propagated_confidence,
            "status": record.status.value if hasattr(record.status, "value") else str(record.status),
            "created_at": record.created_at.isoformat(),
        }


def _cmd_list(args: argparse.Namespace) -> dict[str, Any]:
    from neuromem.core.models import BeliefStatus  # noqa: PLC0415

    status_filter = None
    if args.status:
        try:
            status_filter = BeliefStatus(args.status)
        except ValueError:
            raise ValueError(
                f"Unknown status {args.status!r}. "
                f"Valid values: {[s.value for s in BeliefStatus]}"
            )

    with _make_client(args.data_dir, args.namespace) as client:
        beliefs = client.list_beliefs(namespace=args.namespace, status=status_filter)
        return {
            "namespace": args.namespace,
            "count": len(beliefs),
            "beliefs": [
                {
                    "id": b.id,
                    "claim": b.claim,
                    "confidence": b.confidence,
                    "status": b.status.value,
                    "source": b.source,
                    "tags": list(b.tags),
                    "created_at": b.created_at.isoformat(),
                }
                for b in beliefs
            ],
        }


def _cmd_get(args: argparse.Namespace) -> dict[str, Any]:
    with _make_client(args.data_dir, args.namespace) as client:
        belief = client.get_belief(args.belief_id, namespace=args.namespace)
        if belief is None:
            raise LookupError(f"Belief {args.belief_id!r} not found in namespace {args.namespace!r}")
        return {
            "id": belief.id,
            "claim": belief.claim,
            "confidence": belief.confidence,
            "status": belief.status.value,
            "source": belief.source,
            "namespace": belief.namespace,
            "tags": list(belief.tags),
            "evidence_count": belief.evidence_count,
            "created_at": belief.created_at.isoformat(),
        }


def _cmd_stats(args: argparse.Namespace) -> dict[str, Any]:
    with _make_client(args.data_dir, args.namespace) as client:
        return client.stats()


def _cmd_decay(args: argparse.Namespace) -> dict[str, Any]:
    with _make_client(args.data_dir, args.namespace) as client:
        deprecated = client.decay(namespace=args.namespace, advance_ticks=args.ticks)
        return {
            "namespace": args.namespace,
            "ticks_advanced": args.ticks,
            "beliefs_deprecated": deprecated,
        }


def _cmd_compress(args: argparse.Namespace) -> dict[str, Any]:
    # Accept text from argument, --file, or stdin
    text: str
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    with _make_client(args.data_dir, args.namespace) as client:
        snapshot = client.compress(text, importance=args.importance)
        return {
            "id": snapshot.id,
            "summary": snapshot.summary,
            "keywords": snapshot.keywords,
            "importance": snapshot.importance,
            "compression_ratio": snapshot.compression_ratio,
            "raw_reference": snapshot.raw_reference,
            "created_at": snapshot.created_at.isoformat() if snapshot.created_at else None,
        }


def _cmd_retrieve(args: argparse.Namespace) -> dict[str, Any]:
    with _make_client(args.data_dir, args.namespace) as client:
        original = client.retrieve_original(args.memory_id)
        return {"memory_id": args.memory_id, "original": original}


# ─────────────────────────────────────────────────────────────────────
# Argument parser construction
# ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuromem",
        description=(
            "NeuroMem — hybrid memory engine for AI agents.\n\n"
            "All commands output JSON to stdout (exit 0 on success, 1 on error).\n"
            "Use --pretty for readable indented output, --quiet to silence logs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Global flags ──────────────────────────────────────────────────
    parser.add_argument(
        "--data-dir",
        default="./neuromem_data",
        metavar="PATH",
        help="Root storage directory for graph + vector data (default: ./neuromem_data)",
    )
    parser.add_argument(
        "--namespace",
        default="default",
        metavar="NS",
        help="Agent namespace (default: default)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress loguru diagnostic output on stderr",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        dest="output_format",
        help="Output format: json (default) or text",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── learn ─────────────────────────────────────────────────────────
    p_learn = sub.add_parser(
        "learn",
        help="Store a new belief",
        description="Teach the agent a new belief (stored in graph + vector).",
    )
    p_learn.add_argument("claim", help="The semantic statement to learn")
    p_learn.add_argument(
        "--confidence", "-c", type=float, default=0.5,
        metavar="[0-1]",
        help="Initial confidence in [0, 1] (default: 0.5)",
    )
    p_learn.add_argument(
        "--source", "-s", default="cli",
        help="Origin label, e.g. user / observation (default: cli)",
    )
    p_learn.add_argument(
        "--tags", "-t", default=None,
        metavar="TAG1,TAG2",
        help="Comma-separated tags",
    )

    # ── recall ────────────────────────────────────────────────────────
    p_recall = sub.add_parser(
        "recall",
        help="Recall beliefs matching a query",
        description=(
            "Semantic search over stored beliefs using graph + vector fusion.\n"
            "Returns results sorted by fused score (highest first)."
        ),
    )
    p_recall.add_argument("query", help="Text query for semantic search")
    p_recall.add_argument(
        "--n", "-n", type=int, default=10,
        metavar="N",
        help="Maximum number of results (default: 10)",
    )
    p_recall.add_argument(
        "--min-confidence", type=float, default=0.0,
        metavar="[0-1]",
        help="Minimum fused score threshold (default: 0.0)",
    )
    p_recall.add_argument(
        "--no-decay", action="store_true",
        help="Skip temporal decay adjustment before scoring",
    )

    # ── forget ────────────────────────────────────────────────────────
    p_forget = sub.add_parser(
        "forget",
        help="Deprecate a belief by ID",
        description=(
            "Mark a belief as DEPRECATED (does not physically delete — "
            "preserves audit trail)."
        ),
    )
    p_forget.add_argument("belief_id", help="ID of the belief to forget")

    # ── guard ─────────────────────────────────────────────────────────
    p_guard = sub.add_parser(
        "guard",
        help="Record a negative-memory guardrail",
        description="Prevent the agent from repeating a failed action or decision path.",
    )
    p_guard.add_argument("pattern", help="Description of the failed path to block")
    p_guard.add_argument(
        "--severity",
        choices=["info", "warning", "error", "critical"],
        default="warning",
        help="Severity level (default: warning)",
    )
    p_guard.add_argument(
        "--block-threshold", type=int, default=1,
        metavar="N",
        help="Occurrences before this becomes a hard block (default: 1)",
    )
    p_guard.add_argument(
        "--context", default=None,
        metavar="JSON",
        help="JSON string with structured context about the failure",
    )
    p_guard.add_argument(
        "--related-belief-id", default=None,
        metavar="ID",
        help="ID of a related belief (optional)",
    )

    # ── is-blocked ────────────────────────────────────────────────────
    p_blocked = sub.add_parser(
        "is-blocked",
        help="Check if a pattern is blocked by a guardrail",
        description="Exit code 0 in both cases; check the 'blocked' field in the JSON output.",
    )
    p_blocked.add_argument("pattern", help="Pattern string to check")

    # ── propagate ─────────────────────────────────────────────────────
    p_propagate = sub.add_parser(
        "propagate",
        help="Share a belief with another namespace",
        description="Copy a belief into a target namespace with an optional trust multiplier.",
    )
    p_propagate.add_argument("belief_id", help="ID of the belief to propagate")
    p_propagate.add_argument("target_namespace", help="Receiving namespace")
    p_propagate.add_argument(
        "--trust-factor", type=float, default=None,
        metavar="[0-1]",
        help="Confidence multiplier in [0, 1] (default: engine config)",
    )

    # ── list ──────────────────────────────────────────────────────────
    p_list = sub.add_parser(
        "list",
        help="List all beliefs in a namespace",
        description="List every belief stored in the namespace, optionally filtered by status.",
    )
    p_list.add_argument(
        "--status",
        choices=["active", "deprecated", "conflicted"],
        default=None,
        help="Filter by belief status",
    )

    # ── get ───────────────────────────────────────────────────────────
    p_get = sub.add_parser(
        "get",
        help="Fetch a single belief by ID",
        description="Retrieve full detail for one belief, including evidence count and timestamps.",
    )
    p_get.add_argument("belief_id", help="ID of the belief to fetch")

    # ── stats ─────────────────────────────────────────────────────────
    sub.add_parser(
        "stats",
        help="Show unified statistics (engine + compression + storage)",
        description="Return belief counts, compression ratios, graph node/edge counts, and more.",
    )

    # ── decay ─────────────────────────────────────────────────────────
    p_decay = sub.add_parser(
        "decay",
        help="Advance the logical clock and apply temporal decay",
        description=(
            "Beliefs lose confidence each tick. Run this periodically to "
            "reflect that old information becomes less certain."
        ),
    )
    p_decay.add_argument(
        "--ticks", type=int, default=1,
        metavar="N",
        help="Number of ticks to advance (default: 1)",
    )

    # ── compress ──────────────────────────────────────────────────────
    p_compress = sub.add_parser(
        "compress",
        help="Compress text/logs into a MemorySnapshot",
        description=(
            "Auto-detects content type (logs, conversation, code, RAG) and "
            "applies the optimal compression strategy.  Text can be provided "
            "as an argument, via --file, or piped to stdin."
        ),
    )
    p_compress.add_argument(
        "text", nargs="?", default=None,
        help="Text to compress (or omit to read from --file / stdin)",
    )
    p_compress.add_argument(
        "--file", "-f", default=None,
        metavar="PATH",
        help="Read text from this file instead",
    )
    p_compress.add_argument(
        "--importance", type=float, default=None,
        metavar="[0-1]",
        help="Override auto-detected importance score",
    )

    # ── retrieve ──────────────────────────────────────────────────────
    p_retrieve = sub.add_parser(
        "retrieve",
        help="Retrieve the original uncompressed text for a snapshot ID",
        description="Losslessly restore the exact original text that was compressed.",
    )
    p_retrieve.add_argument("memory_id", help="Snapshot ID (raw_reference) to look up")

    return parser


# ─────────────────────────────────────────────────────────────────────
# Dispatch table
# ─────────────────────────────────────────────────────────────────────

_DISPATCH: dict[str, Any] = {
    "learn":      _cmd_learn,
    "recall":     _cmd_recall,
    "forget":     _cmd_forget,
    "guard":      _cmd_guard,
    "is-blocked": _cmd_is_blocked,
    "propagate":  _cmd_propagate,
    "list":       _cmd_list,
    "get":        _cmd_get,
    "stats":      _cmd_stats,
    "decay":      _cmd_decay,
    "compress":   _cmd_compress,
    "retrieve":   _cmd_retrieve,
}


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the requested sub-command, and return exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Silence loguru when --quiet is requested
    if args.quiet:
        try:
            from loguru import logger  # noqa: PLC0415
            logger.remove()
        except Exception:  # noqa: BLE001
            pass

    handler = _DISPATCH.get(args.command)
    if handler is None:
        return _err(f"Unknown command: {args.command!r}", pretty=args.pretty, fmt=args.output_format)

    try:
        result = handler(args)
        return _ok(result, pretty=args.pretty, fmt=args.output_format)
    except KeyboardInterrupt:
        return _err("Interrupted", pretty=args.pretty, fmt=args.output_format)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc), pretty=args.pretty, fmt=args.output_format)


def _entrypoint() -> None:
    """Console script entry point (calls ``sys.exit``)."""
    sys.exit(main())


if __name__ == "__main__":
    _entrypoint()
