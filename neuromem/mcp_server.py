"""MCP server exposing NeuroMem cognitive tools to any MCP-compatible agent.

Start with::

    neuromem serve --data-dir ./agent_memory --namespace default

Then add to ``~/.claude/claude_desktop_config.json`` (or equivalent)::

    {
      "mcpServers": {
        "neuromem": {
          "command": "neuromem",
          "args": ["--data-dir", "./agent_memory", "serve"]
        }
      }
    }

The server communicates over **stdio** (the MCP standard transport) and
exposes the following tools:

+--------------------+--------------------------------------------------+
| Tool               | Description                                      |
+====================+==================================================+
| ``learn``          | Store a belief in NeuroMem memory                |
+--------------------+--------------------------------------------------+
| ``recall``         | Recall beliefs matching a query                  |
+--------------------+--------------------------------------------------+
| ``guard``          | Record a negative-memory guardrail               |
+--------------------+--------------------------------------------------+
| ``is_blocked``     | Check if a pattern is blocked by a guardrail     |
+--------------------+--------------------------------------------------+
| ``compress``       | Compress text into a compact MemorySnapshot      |
+--------------------+--------------------------------------------------+
| ``retrieve``       | Retrieve original text for a snapshot ID         |
+--------------------+--------------------------------------------------+
| ``forget``         | Deprecate a belief by ID                         |
+--------------------+--------------------------------------------------+
| ``propagate``      | Share a belief with another agent namespace      |
+--------------------+--------------------------------------------------+
| ``stats``          | Return unified statistics for all subsystems     |
+--------------------+--------------------------------------------------+
| ``decay``          | Advance the clock and apply confidence decay     |
+--------------------+--------------------------------------------------+
"""

from __future__ import annotations

import json
import traceback
from typing import Any

from loguru import logger


def _require_mcp() -> Any:
    """Lazy-import the mcp package and return the module, or raise a friendly error."""
    try:
        import mcp  # type: ignore[import-untyped, import-not-found]  # noqa: PLC0415
        return mcp
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required to run the NeuroMem MCP server. "
            "Install it with:  pip install mcp\n"
            "Or add the optional group:  pip install neuromem-ai[mcp]"
        ) from exc


def _json(obj: Any) -> str:
    """Serialise *obj* to a JSON string with ISO datetimes."""
    from datetime import datetime  # noqa: PLC0415

    def _default(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")

    return json.dumps(obj, default=_default)


def build_server(data_dir: str, namespace: str) -> Any:
    """Build and return a configured MCP :class:`~mcp.server.Server`.

    Parameters
    ----------
    data_dir:
        Root storage directory passed to :meth:`NeuroMemClient.create`.
    namespace:
        Default namespace for this server instance.

    Returns
    -------
    mcp.server.Server
        A ready-to-run MCP server with all NeuroMem tools registered.
    """
    mcp_mod = _require_mcp()
    from mcp.server import Server  # type: ignore[import-untyped, import-not-found]  # noqa: PLC0415
    from mcp import types as mcp_types  # type: ignore[import-untyped, import-not-found]  # noqa: PLC0415

    from neuromem.client import NeuroMemClient  # noqa: PLC0415

    server = Server("neuromem")
    client = NeuroMemClient.create(data_dir, namespace=namespace)

    logger.info("NeuroMem MCP server ready (namespace={}, data_dir={})", namespace, data_dir)

    # ── learn ─────────────────────────────────────────────────────────────

    @server.tool()
    async def learn(
        claim: str,
        confidence: float = 0.5,
        source: str = "agent",
        tags: str = "",
    ) -> str:
        """Store a new belief in NeuroMem memory.

        Args:
            claim: The semantic content of the belief.
            confidence: Initial confidence score in [0.0, 1.0].
            source: Origin of the belief (e.g. 'agent', 'user').
            tags: Comma-separated list of labels for filtering.
        """
        try:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] or None
            belief = client.learn(claim, confidence=confidence, source=source, tags=tag_list)
            return _json({
                "id": belief.id,
                "claim": belief.claim,
                "confidence": belief.confidence,
                "evidence_count": belief.evidence_count,
                "namespace": belief.namespace,
            })
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP learn error: {}", exc)
            return _json({"error": str(exc), "traceback": traceback.format_exc()})

    # ── recall ────────────────────────────────────────────────────────────

    @server.tool()
    async def recall(
        query: str,
        n: int = 5,
        min_confidence: float = 0.0,
    ) -> str:
        """Recall beliefs matching a query from NeuroMem memory.

        Args:
            query: Semantic search query.
            n: Maximum number of results to return.
            min_confidence: Filter results below this confidence threshold.
        """
        try:
            results = client.recall(query, n_results=n, min_confidence=min_confidence)
            return _json([r.to_dict() for r in results])
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP recall error: {}", exc)
            return _json({"error": str(exc)})

    # ── guard ─────────────────────────────────────────────────────────────

    @server.tool()
    async def guard(
        pattern: str,
        severity: str = "warning",
        pattern_type: str = "exact",
        block_threshold: int = 1,
        fuzzy_threshold: float = 0.8,
    ) -> str:
        """Record a negative-memory guardrail to prevent repeating failures.

        Args:
            pattern: Description of the failed path or rejected logic.
            severity: Severity level ('info', 'warning', 'error', 'fatal').
            pattern_type: Matching strategy ('exact', 'regex', 'fuzzy').
            block_threshold: Occurrences before this becomes a hard block.
            fuzzy_threshold: Jaccard overlap threshold for fuzzy matching.
        """
        try:
            neg = client.guard(
                pattern,
                severity=severity,
                pattern_type=pattern_type,
                block_threshold=block_threshold,
                fuzzy_threshold=fuzzy_threshold,
            )
            return _json({
                "id": neg.id,
                "pattern": neg.pattern,
                "pattern_type": neg.pattern_type.value if hasattr(neg.pattern_type, "value") else str(neg.pattern_type),
                "severity": neg.severity.value if hasattr(neg.severity, "value") else str(neg.severity),
                "should_block": neg.should_block,
                "occurrence_count": neg.occurrence_count,
            })
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP guard error: {}", exc)
            return _json({"error": str(exc)})

    # ── is_blocked ────────────────────────────────────────────────────────

    @server.tool()
    async def is_blocked(pattern: str) -> str:
        """Check if a pattern is blocked by an existing guardrail.

        Args:
            pattern: The string to check against all registered guardrails.
        """
        try:
            blocked = client.is_blocked(pattern)
            return _json({"pattern": pattern, "blocked": blocked})
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP is_blocked error: {}", exc)
            return _json({"error": str(exc)})

    # ── compress ──────────────────────────────────────────────────────────

    @server.tool()
    async def compress(text: str, importance: float = 0.5) -> str:
        """Compress text into a compact MemorySnapshot.

        Args:
            text: The text to compress (e.g. a long conversation).
            importance: Importance score in [0.0, 1.0] used to guide compression.
        """
        try:
            snap = client.compress(text, importance=importance)
            return _json({
                "id": snap.id,
                "summary": snap.summary,
                "compression_ratio": snap.compression_ratio,
                "raw_reference": snap.raw_reference,
            })
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP compress error: {}", exc)
            return _json({"error": str(exc)})

    # ── retrieve ──────────────────────────────────────────────────────────

    @server.tool()
    async def retrieve(memory_id: str) -> str:
        """Retrieve the original uncompressed text for a snapshot ID.

        Args:
            memory_id: The ``raw_reference`` (snapshot ID) to look up.
        """
        try:
            original = client.retrieve_original(memory_id)
            return _json({"memory_id": memory_id, "original": original})
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP retrieve error: {}", exc)
            return _json({"error": str(exc)})

    # ── forget ────────────────────────────────────────────────────────────

    @server.tool()
    async def forget(belief_id: str) -> str:
        """Deprecate a belief by ID (soft delete — audit trail preserved).

        Args:
            belief_id: ID of the belief to deprecate.
        """
        try:
            found = client.forget(belief_id)
            return _json({"belief_id": belief_id, "forgotten": found})
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP forget error: {}", exc)
            return _json({"error": str(exc)})

    # ── propagate ─────────────────────────────────────────────────────────

    @server.tool()
    async def propagate(
        belief_id: str,
        target_namespace: str,
        trust_factor: float = 0.8,
    ) -> str:
        """Share a belief with another agent namespace.

        Args:
            belief_id: ID of the belief to propagate.
            target_namespace: Receiving namespace.
            trust_factor: Confidence multiplier in [0.0, 1.0].
        """
        try:
            record = client.propagate(belief_id, target_namespace, trust_factor=trust_factor)
            return _json({
                "id": record.id,
                "status": record.status.value if hasattr(record.status, "value") else str(record.status),
                "propagated_confidence": record.propagated_confidence,
            })
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP propagate error: {}", exc)
            return _json({"error": str(exc)})

    # ── stats ─────────────────────────────────────────────────────────────

    @server.tool()
    async def stats() -> str:
        """Return unified statistics across all NeuroMem subsystems."""
        try:
            return _json(client.stats())
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP stats error: {}", exc)
            return _json({"error": str(exc)})

    # ── decay ─────────────────────────────────────────────────────────────

    @server.tool()
    async def decay(ticks: int = 1) -> str:
        """Advance the logical clock and apply temporal confidence decay.

        Args:
            ticks: Number of logical ticks to advance (default 1).
        """
        try:
            deprecated = client.decay(advance_ticks=ticks)
            return _json({
                "ticks_advanced": ticks,
                "beliefs_deprecated": deprecated,
            })
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP decay error: {}", exc)
            return _json({"error": str(exc)})

    return server


async def run_server(data_dir: str, namespace: str) -> None:
    """Build and run the MCP server over stdio transport.

    Parameters
    ----------
    data_dir:
        Root storage directory for NeuroMem data.
    namespace:
        Default namespace for this server session.
    """
    _require_mcp()
    from mcp.server.stdio import stdio_server  # type: ignore[import-untyped, import-not-found]  # noqa: PLC0415

    server = build_server(data_dir, namespace)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
