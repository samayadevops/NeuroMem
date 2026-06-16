"""Kuzu-based implementation of :class:`BaseGraphEngine`.

Design rationale
----------------
This module wraps Kuzu's embedded Cypher engine behind NeuroMem's
abstract storage contract.  Every method translates to one or more Kuzu
Cypher statements, translates the results into the framework's plain-data
types (:class:`NodeRecord`, :class:`EdgeRecord`, :class:`QueryResult`),
and maps engine-level errors into NeuroMem's typed exception hierarchy.

Kuzu-specific notes (v0.11.x)
-------------------------------
* **Node tables**: ``CREATE NODE TABLE Label(props…) PRIMARY KEY (…)``.
  Use ``CREATE (n:Label {…})`` for inserts.
* **Rel tables**: ``CREATE REL TABLE Type(FROM SrcLabel TO DstLabel, props…)``.
  Note: no comma between ``SrcLabel`` and ``TO``.
* **No ``INSERT INTO … VALUES``** — only ``CREATE (n:Label {…})``.
* **Edges with refs**: ``DELETE n`` fails if edges reference the node;
  use ``DETACH DELETE n`` instead.
* **``IF NOT EXISTS``**: supported for ``CREATE TABLE``.
* **Parameter binding**: dict ``{"$name": value}``.
* **Node/Rel objects**: returned as dicts with internal keys ``_id``,
  ``_label``, ``_src``, ``_dst``.  We strip these before returning
  :class:`NodeRecord` / :class:`EdgeRecord`.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from neuromem.core.exceptions import (
    EdgeNotFoundError,
    GraphEngineError,
    GraphQueryError,
    NodeNotFoundError,
    SchemaViolationError,
)
from neuromem.storage.base import (
    BaseGraphEngine,
    EdgeRecord,
    EngineState,
    NodeRecord,
    QueryResult,
)

# ═══════════════════════════════════════════════════════════════════════
# Type mapping: Python → Kuzu column type strings
# ═══════════════════════════════════════════════════════════════════════

_PY_TO_KUZU_TYPE: dict[type[Any], str] = {
    str: "STRING",
    int: "INT64",
    float: "DOUBLE",
    bool: "BOOLEAN",
    datetime: "TIMESTAMP",
    list: "STRING[]",
}

# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

# Keys injected by Kuzu into node/rel dicts that are NOT user properties.
_INTERNAL_NODE_KEYS = frozenset({"_id", "_label"})
_INTERNAL_EDGE_KEYS = frozenset({"_src", "_dst", "_id", "_label"})


def _strip_internal_node(props: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *props* without Kuzu internal keys."""
    return {k: v for k, v in props.items() if k not in _INTERNAL_NODE_KEYS}


def _strip_internal_edge(props: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *props* without Kuzu internal keys."""
    return {k: v for k, v in props.items() if k not in _INTERNAL_EDGE_KEYS}


def _cypher_literal(value: Any) -> str:
    """Convert a Python value to a Cypher literal string for inline queries.

    Used for property maps in CREATE statements where Kuzu's parameter
    binding does not support map-style expansion.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, datetime):
        return f"timestamp('{value.isoformat()}')"
    if isinstance(value, list):
        inner = ", ".join(_cypher_literal(v) for v in value)
        return f"[{inner}]"
    # Fallback: string representation
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _build_property_map(props: dict[str, Any]) -> str:
    """Build a Cypher ``{key: value, …}`` property map string."""
    if not props:
        return ""
    parts = [f"{k}: {_cypher_literal(v)}" for k, v in props.items()]
    return "{" + ", ".join(parts) + "}"


def _infer_kuzu_type(py_type: type[Any]) -> str:
    """Map a Python type annotation to a Kuzu column type.

    Falls back to ``STRING`` for unrecognised types.
    """
    # Handle Optional / Union — if it's NoneType, skip
    if py_type is type(None):
        return "STRING"
    # Handle generics like list[str] → STRING[]
    origin = getattr(py_type, "__origin__", None)
    if origin is list:
        return "STRING[]"
    return _PY_TO_KUZU_TYPE.get(py_type, "STRING")


def _validate_db_path(db_path: str) -> Path:
    """Resolve and validate the database directory path."""
    path = Path(db_path).resolve()
    # If path exists and is a file (partial DB state), remove it
    if path.exists() and not path.is_dir():
        path.unlink()
    return path


# ═══════════════════════════════════════════════════════════════════════
# KuzuGraphEngine
# ═══════════════════════════════════════════════════════════════════════

class KuzuGraphEngine(BaseGraphEngine):
    """Production Kuzu-backed graph engine.

    Parameters
    ----------
    db_path:
        Filesystem path to the Kuzu database directory.  Created
        automatically if it does not exist.
    read_only:
        If ``True``, open the database in read-only mode.  Mutating
        operations will raise :class:`GraphEngineError`.
    max_query_retries:
        Number of automatic retries for transient query failures
        (e.g. busy locks).  Defaults to 1 (no retry).
    """

    def __init__(
        self,
        db_path: str = "./neuromem_graph",
        *,
        read_only: bool = False,
        max_query_retries: int = 1,
    ) -> None:
        self._db_path: str = db_path
        self._read_only: bool = read_only
        self._max_retries: int = max_query_retries
        self._db: Any | None = None  # kuzu.Database
        self._conn: Any | None = None  # kuzu.Connection
        self._state: EngineState = "uninitialized"
        self._registered_labels: set[str] = set()
        self._registered_edge_types: set[str] = set()
        self._primary_keys: dict[str, str] = {}
        self._label_schemas: dict[str, dict[str, str]] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Open the Kuzu database and prepare for operations."""
        if self._state == "ready":
            logger.debug("KuzuGraphEngine already initialized — skipping")
            return

        self._state = "initializing"
        logger.info(
            "Initializing KuzuGraphEngine",
            db_path=self._db_path,
            read_only=self._read_only,
        )

        try:
            import kuzu  # noqa: F811 (re-import is fine)

            resolved = _validate_db_path(self._db_path)

            # Detect if we should create a new DB or open existing
            db_exists = resolved.is_dir()

            if self._read_only and not db_exists:
                raise GraphEngineError(
                    f"Database path {self._db_path} does not exist and "
                    "read_only=True was specified.",
                    backend="kuzu",
                )

            db_size = 0
            if db_exists:
                db_size = sum(
                    f.stat().st_size for f in resolved.rglob("*") if f.is_file()
                )

            # Default buffer pool: 256 MB
            buffer_size = 256 * 1024 * 1024

            self._db = kuzu.Database(
                str(resolved),
                read_only=self._read_only,
                buffer_pool_size=buffer_size,
            )
            self._conn = kuzu.Connection(self._db)

            # Re-scan existing tables to populate internal registries
            self._scan_existing_tables()

            self._state = "ready"
            logger.info(
                "KuzuGraphEngine initialized",
                db_path=self._db_path,
                existing=db_exists,
                db_size_bytes=db_size,
                node_labels=len(self._registered_labels),
                edge_types=len(self._registered_edge_types),
            )

        except GraphEngineError:
            raise
        except Exception as exc:
            self._state = "closed"
            raise GraphEngineError(
                f"Failed to initialize Kuzu database: {exc}",
                backend="kuzu",
                context={"db_path": self._db_path, "error_type": type(exc).__name__},
            ) from exc

    def close(self) -> None:
        """Release the Kuzu connection and database handles."""
        if self._state == "closed":
            return
        logger.info("Closing KuzuGraphEngine", db_path=self._db_path)
        self._conn = None
        self._db = None
        self._state = "closed"

    @property
    def state(self) -> EngineState:
        """Return the current lifecycle state."""
        return self._state

    @property
    def is_ready(self) -> bool:
        """``True`` when the engine is initialized and ready for queries."""
        return self._state == "ready"

    # ── Schema management ────────────────────────────────────────────

    def create_node_label(
        self,
        label: str,
        properties: dict[str, str] | None = None,
        primary_key: str | None = None,
    ) -> None:
        """Register a Kuzu node table.

        If *primary_key* is omitted, defaults to ``"id"``.  The column
        type is always ``STRING`` for the primary key.
        """
        self._require_ready()

        if label in self._registered_labels:
            logger.debug("Node label {} already registered — skipping", label)
            return

        pk = primary_key or "id"
        props = dict(properties or {})

        # Ensure the PK column exists in the schema
        if pk not in props:
            props[pk] = "STRING"

        # Build column definitions: name TYPE[, …] PRIMARY KEY (pk)
        col_defs = ", ".join(f"{col} {typ}" for col, typ in props.items())
        ddl = f"CREATE NODE TABLE IF NOT EXISTS {label}({col_defs}, PRIMARY KEY({pk}))"

        try:
            self._execute_ddl(ddl)
        except Exception as exc:
            raise SchemaViolationError(ddl, str(exc)) from exc

        self._registered_labels.add(label)
        self._primary_keys[label] = pk
        self._label_schemas[label] = props
        logger.debug("Created node label {}", label)

    def create_edge_type(
        self,
        edge_type: str,
        src_label: str,
        dst_label: str,
        properties: dict[str, str] | None = None,
    ) -> None:
        """Register a Kuzu relationship table.

        Kuzu 0.11 syntax: ``CREATE REL TABLE Type(FROM Src TO Dst, props…)``
        """
        self._require_ready()

        if edge_type in self._registered_edge_types:
            logger.debug("Edge type {} already registered — skipping", edge_type)
            return

        props = dict(properties or {})

        # Build the column list
        col_defs = ", ".join(f"{col} {typ}" for col, typ in props.items())
        if col_defs:
            col_defs = ", " + col_defs

        ddl = (
            f"CREATE REL TABLE IF NOT EXISTS {edge_type}"
            f"(FROM {src_label} TO {dst_label}{col_defs})"
        )

        try:
            self._execute_ddl(ddl)
        except Exception as exc:
            raise SchemaViolationError(ddl, str(exc)) from exc

        self._registered_edge_types.add(edge_type)
        logger.debug("Created edge type {} ({} -> {})", edge_type, src_label, dst_label)

    def label_exists(self, label: str) -> bool:
        """Return ``True`` if the node table exists in Kuzu."""
        self._require_ready()
        if label in self._registered_labels:
            return True
        # Fallback: scan the catalog
        return self._table_exists_in_catalog(label, expected_type="NODE")

    def edge_type_exists(self, edge_type: str) -> bool:
        """Return ``True`` if the rel table exists in Kuzu."""
        self._require_ready()
        if edge_type in self._registered_edge_types:
            return True
        return self._table_exists_in_catalog(edge_type, expected_type="REL")

    # ── Node CRUD ────────────────────────────────────────────────────

    def add_node(
        self,
        label: str,
        node_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Insert or replace a node using ``CREATE`` Cypher.

        If the node with the given primary key already exists, it is
        deleted and re-created (upsert semantics).  This is a deliberate
        trade-off: Kuzu 0.11 does not support ``ON CREATE / ON MATCH``
        merge patterns with property maps, so we use delete+create.
        """
        self._require_ready()
        self._require_not_readonly()

        pk = self._primary_keys.get(label, "id")
        props = dict(properties or {})
        props[pk] = node_id

        prop_map = _build_property_map(props)

        try:
            # Check if node exists — if so, delete first (upsert)
            if self.node_exists(label, node_id):
                self.delete_node(label, node_id, cascade_edges=True)

            cypher = f"CREATE (n:{label} {prop_map})"
            self._execute_write(cypher, context={"label": label, "node_id": node_id})
            logger.debug("Upserted node {}.{} = {}", label, pk, node_id)

        except GraphEngineError:
            raise
        except Exception as exc:
            raise GraphEngineError(
                f"Failed to add node {node_id!r} to {label}: {exc}",
                backend="kuzu",
                context={"label": label, "node_id": node_id, "cypher": cypher},
            ) from exc

    def get_node(self, label: str, node_id: str) -> NodeRecord | None:
        """Fetch a node by label and primary key.  Returns ``None`` if not found."""
        self._require_ready()

        pk = self._primary_keys.get(label, "id")
        cypher = f"MATCH (n:{label} {{{pk}: $nid}}) RETURN n"
        result = self.execute_query(cypher, {"nid": node_id})

        if not result:
            return None

        node_dict = result.records[0]["n"]
        clean = _strip_internal_node(node_dict)

        created_at = clean.pop("created_at", None)
        updated_at = clean.pop("updated_at", None)

        return NodeRecord(
            node_id=node_id,
            label=label,
            properties=clean,
            created_at=created_at,
            updated_at=updated_at,
        )

    def update_node_properties(
        self,
        label: str,
        node_id: str,
        properties: dict[str, Any],
        merge: bool = True,
    ) -> None:
        """Update properties on an existing node via Cypher ``SET``."""
        self._require_ready()
        self._require_not_readonly()

        if not self.node_exists(label, node_id):
            raise NodeNotFoundError(node_id, label=label)

        pk = self._primary_keys.get(label, "id")

        if merge:
            # SET individual properties
            set_clauses = []
            params: dict[str, Any] = {}
            for idx, (key, value) in enumerate(properties.items()):
                param_name = f"_val_{idx}"
                params[param_name] = value
                set_clauses.append(f"n.{key} = ${param_name}")

            params["nid"] = node_id
            cypher = (
                f"MATCH (n:{label} {{{pk}: $nid}}) "
                f"SET {', '.join(set_clauses)}"
            )
        else:
            # Replace entire property map: delete + recreate with new props
            self.delete_node(label, node_id, cascade_edges=True)
            # Get existing properties from schema to re-create
            all_props = dict(properties)
            all_props[pk] = node_id
            prop_map = _build_property_map(all_props)
            cypher = f"CREATE (n:{label} {prop_map})"
            params = {}

        try:
            self._execute_write(cypher, params, context={"label": label, "node_id": node_id})
            logger.debug("Updated properties on {}.{} = {}", label, pk, node_id)
        except GraphEngineError:
            raise
        except Exception as exc:
            raise GraphEngineError(
                f"Failed to update properties on {label}/{node_id}: {exc}",
                backend="kuzu",
                context={"label": label, "node_id": node_id},
            ) from exc

    def delete_node(
        self,
        label: str,
        node_id: str,
        cascade_edges: bool = False,
    ) -> bool:
        """Delete a node.  Uses ``DETACH DELETE`` when *cascade_edges* is True."""
        self._require_ready()
        self._require_not_readonly()

        pk = self._primary_keys.get(label, "id")

        # Check existence first
        exists_result = self.execute_query(
            f"MATCH (n:{label} {{{pk}: $nid}}) RETURN n.{pk}",
            {"nid": node_id},
        )
        if not exists_result:
            return False

        delete_keyword = "DETACH DELETE" if cascade_edges else "DELETE"
        cypher = (
            f"MATCH (n:{label} {{{pk}: $nid}}) {delete_keyword} n"
        )

        try:
            self._execute_write(cypher, {"nid": node_id}, context={"label": label, "node_id": node_id})
            logger.debug(
                "Deleted node {}.{} (cascade={})",
                label, node_id, cascade_edges,
            )
            return True
        except GraphEngineError:
            raise
        except Exception as exc:
            raise GraphEngineError(
                f"Failed to delete node {node_id!r} from {label}: {exc}",
                backend="kuzu",
                context={"label": label, "node_id": node_id, "cascade_edges": cascade_edges},
            ) from exc

    def node_exists(self, label: str, node_id: str) -> bool:
        """Check whether a node exists without fetching full data."""
        self._require_ready()

        pk = self._primary_keys.get(label, "id")
        result = self.execute_query(
            f"MATCH (n:{label} {{{pk}: $nid}}) RETURN count(n) > 0 AS _found",
            {"nid": node_id},
        )
        # result.records[0][0] is the boolean value
        return bool(result.records[0]["_found"]) if result else False

    # ── Edge CRUD ────────────────────────────────────────────────────

    def add_edge(
        self,
        src_label: str,
        src_id: str,
        dst_label: str,
        dst_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create a directed edge between two existing nodes."""
        self._require_ready()
        self._require_not_readonly()

        src_pk = self._primary_keys.get(src_label, "id")
        dst_pk = self._primary_keys.get(dst_label, "id")

        # Verify both endpoints exist
        if not self.node_exists(src_label, src_id):
            raise NodeNotFoundError(src_id, label=src_label)
        if not self.node_exists(dst_label, dst_id):
            raise NodeNotFoundError(dst_id, label=dst_label)

        props = dict(properties or {})
        prop_map = _build_property_map(props) if props else ""

        try:
            if props:
                cypher = (
                    f"MATCH (a:{src_label} {{{src_pk}: $src_id}}), "
                    f"(b:{dst_label} {{{dst_pk}: $dst_id}}) "
                    f"CREATE (a)-[r:{edge_type} {prop_map}]->(b)"
                )
            else:
                cypher = (
                    f"MATCH (a:{src_label} {{{src_pk}: $src_id}}), "
                    f"(b:{dst_label} {{{dst_pk}: $dst_id}}) "
                    f"CREATE (a)-[r:{edge_type}]->(b)"
                )

            self._execute_write(
                cypher,
                {"src_id": src_id, "dst_id": dst_id},
                context={
                    "src_label": src_label,
                    "src_id": src_id,
                    "dst_label": dst_label,
                    "dst_id": dst_id,
                    "edge_type": edge_type,
                },
            )
            logger.debug(
                "Created edge ({}:{})-[{}]->({}:{})",
                src_label, src_id, edge_type, dst_label, dst_id,
            )

        except NodeNotFoundError:
            raise
        except GraphEngineError:
            raise
        except Exception as exc:
            raise GraphEngineError(
                f"Failed to create edge {edge_type} from "
                f"{src_label}/{src_id} to {dst_label}/{dst_id}: {exc}",
                backend="kuzu",
                context={
                    "src_label": src_label,
                    "src_id": src_id,
                    "dst_label": dst_label,
                    "dst_id": dst_id,
                    "edge_type": edge_type,
                },
            ) from exc

    def get_edge(
        self,
        src_label: str,
        src_id: str,
        dst_label: str,
        dst_id: str,
        edge_type: str,
    ) -> EdgeRecord | None:
        """Fetch a single edge.  Returns ``None`` if not found."""
        self._require_ready()

        src_pk = self._primary_keys.get(src_label, "id")
        dst_pk = self._primary_keys.get(dst_label, "id")

        cypher = (
            f"MATCH (a:{src_label} {{{src_pk}: $src_id}})"
            f"-[r:{edge_type}]->"
            f"(b:{dst_label} {{{dst_pk}: $dst_id}}) "
            f"RETURN r"
        )
        result = self.execute_query(cypher, {"src_id": src_id, "dst_id": dst_id})

        if not result:
            return None

        rel_dict = result.records[0]["r"]
        clean = _strip_internal_edge(rel_dict)
        created_at = clean.pop("created_at", None)

        return EdgeRecord(
            src_id=src_id,
            dst_id=dst_id,
            edge_type=edge_type,
            properties=clean,
            created_at=created_at,
        )

    def delete_edge(
        self,
        src_label: str,
        src_id: str,
        dst_label: str,
        dst_id: str,
        edge_type: str,
    ) -> bool:
        """Delete a specific edge."""
        self._require_ready()
        self._require_not_readonly()

        src_pk = self._primary_keys.get(src_label, "id")
        dst_pk = self._primary_keys.get(dst_label, "id")

        # Check if edge exists
        check_result = self.execute_query(
            (
                f"MATCH (a:{src_label} {{{src_pk}: $src_id}})"
                f"-[r:{edge_type}]->"
                f"(b:{dst_label} {{{dst_pk}: $dst_id}}) "
                f"RETURN count(r) > 0 AS _found"
            ),
            {"src_id": src_id, "dst_id": dst_id},
        )

        if not check_result or not bool(check_result.records[0]["_found"]):
            return False

        cypher = (
            f"MATCH (a:{src_label} {{{src_pk}: $src_id}})"
            f"-[r:{edge_type}]->"
            f"(b:{dst_label} {{{dst_pk}: $dst_id}}) "
            f"DELETE r"
        )

        try:
            self._execute_write(
                cypher,
                {"src_id": src_id, "dst_id": dst_id},
                context={"edge_type": edge_type},
            )
            logger.debug(
                "Deleted edge ({}:{})-[{}]->({}:{})",
                src_label, src_id, edge_type, dst_label, dst_id,
            )
            return True
        except GraphEngineError:
            raise
        except Exception as exc:
            raise GraphEngineError(
                f"Failed to delete edge {edge_type}: {exc}",
                backend="kuzu",
                context={"edge_type": edge_type, "src_id": src_id, "dst_id": dst_id},
            ) from exc

    def get_edges(
        self,
        label: str,
        node_id: str,
        direction: Literal["outgoing", "incoming", "both"] = "both",
        edge_type: str | None = None,
    ) -> list[EdgeRecord]:
        """Retrieve all edges connected to a node."""
        self._require_ready()

        pk = self._primary_keys.get(label, "id")

        if direction == "outgoing":
            pattern = f"(n:{label} {{{pk}: $nid}})-[r{edge_type_filter(edge_type)}]->(m)"
        elif direction == "incoming":
            pattern = f"(n:{label} {{{pk}: $nid}})<-[r{edge_type_filter(edge_type)}]-(m)"
        else:
            pattern = f"(n:{label} {{{pk}: $nid}})-[r{edge_type_filter(edge_type)}]-(m)"

        cypher = f"MATCH {pattern} RETURN n.{pk} AS src_id, m.{pk} AS dst_id, r"
        result = self.execute_query(cypher, {"nid": node_id})

        edges: list[EdgeRecord] = []
        if not result:
            return edges

        for row in result.records:
            src_id = row["src_id"] if isinstance(row, dict) else row[0]
            dst_id = row["dst_id"] if isinstance(row, dict) else row[1]
            rel_dict = row["r"] if isinstance(row, dict) else row[2]
            clean = _strip_internal_edge(rel_dict)
            created_at = clean.pop("created_at", None)

            edges.append(
                EdgeRecord(
                    src_id=str(src_id),
                    dst_id=str(dst_id),
                    edge_type=str(clean.pop("_label", edge_type or "UNKNOWN")),
                    properties=clean,
                    created_at=created_at,
                )
            )

        return edges

    # ── Query ────────────────────────────────────────────────────────

    def execute_query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Execute a Cypher query and return structured results.

        Automatically handles retries for transient failures.
        """
        self._require_ready()

        import time

        params = parameters or {}
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                start = time.perf_counter()
                result = self._conn.execute(cypher, params)
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                columns = result.get_column_names()
                records: list[dict[str, Any]] = []

                while result.has_next():
                    raw_row = result.get_next()
                    if isinstance(raw_row, dict):
                        records.append(raw_row)
                    else:
                        row_dict: dict[str, Any] = {}
                        for idx, col_name in enumerate(columns):
                            row_dict[col_name] = raw_row[idx]
                        records.append(row_dict)

                return QueryResult(
                    records=records,
                    columns=columns,
                    execution_time_ms=round(elapsed_ms, 2),
                )

            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    logger.warning(
                        "Query attempt {}/{} failed, retrying: {}",
                        attempt + 1,
                        self._max_retries,
                        exc,
                    )
                    continue
                break

        raise GraphQueryError(cypher, str(last_exc)) from last_exc

    # ── Bulk / maintenance ────────────────────────────────────────────

    def count_nodes(self, label: str | None = None) -> int:
        """Return total node count, optionally filtered by label."""
        self._require_ready()

        if label:
            result = self.execute_query(
                f"MATCH (n:{label}) RETURN count(n) AS cnt"
            )
        else:
            result = self.execute_query("MATCH (n) RETURN count(n) AS cnt")

        if result and result.records:
            val = result.records[0]["cnt"] if isinstance(result.records[0], dict) else result.records[0][0]
            return int(val)
        return 0

    def count_edges(self, edge_type: str | None = None) -> int:
        """Return total edge count, optionally filtered by type."""
        self._require_ready()

        if edge_type:
            result = self.execute_query(
                f"MATCH ()-[r:{edge_type}]->() RETURN count(r) AS cnt"
            )
        else:
            result = self.execute_query("MATCH ()-[r]->() RETURN count(r) AS cnt")

        if result and result.records:
            val = result.records[0]["cnt"] if isinstance(result.records[0], dict) else result.records[0][0]
            return int(val)
        return 0

    def clear_all(self) -> None:
        """Remove all nodes and edges.  For test teardown only."""
        self._require_ready()
        self._require_not_readonly()

        logger.warning("Clearing all data from Kuzu graph", db_path=self._db_path)

        try:
            # Detach-delete all nodes (also removes edges)
            self._conn.execute("MATCH (n) DETACH DELETE n")
        except Exception as exc:
            raise GraphEngineError(
                f"Failed to clear graph: {exc}",
                backend="kuzu",
            ) from exc

        # Re-scan to update registries (they should be empty now but
        # the tables still exist)
        self._registered_labels.clear()
        self._registered_edge_types.clear()
        self._primary_keys.clear()
        self._label_schemas.clear()

    # ── Internal helpers ─────────────────────────────────────────────

    def _require_ready(self) -> None:
        """Raise if the engine is not ready for operations."""
        if self._state != "ready":
            raise GraphEngineError(
                f"Kuzu engine is not ready (current state: {self._state!r}). "
                "Call initialize() first.",
                backend="kuzu",
            )

    def _require_not_readonly(self) -> None:
        """Raise if a mutating operation is attempted on a read-only engine."""
        if self._read_only:
            raise GraphEngineError(
                "Cannot perform write operation: engine is in read-only mode.",
                backend="kuzu",
            )

    def _execute_ddl(self, ddl: str) -> None:
        """Execute a DDL statement (CREATE TABLE, DROP TABLE, etc.)."""
        try:
            self._conn.execute(ddl)
        except Exception as exc:
            raise GraphEngineError(
                f"DDL execution failed: {exc}",
                backend="kuzu",
                context={"ddl": ddl},
            ) from exc

    def _execute_write(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Execute a mutating Cypher statement with error handling."""
        try:
            self._conn.execute(cypher, parameters or {})
        except Exception as exc:
            raise GraphEngineError(
                f"Write query failed: {exc}",
                backend="kuzu",
                context={"cypher": cypher, **(context or {})},
            ) from exc

    def _scan_existing_tables(self) -> None:
        """Scan the Kuzu catalog to populate label/edge-type registries."""
        try:
            result = self._conn.execute("CALL show_tables() RETURN *")
            while result.has_next():
                row = result.get_next()
                # row format from show_tables: [id, name, type, db_name, comment]
                table_name = row[1]
                table_type = row[2]

                if table_type == "NODE":
                    self._registered_labels.add(table_name)
                    # Infer primary key from table schema
                    # Default to "id" — Kuzu 0.11 doesn't expose PK via API
                    self._primary_keys[table_name] = "id"
                    self._label_schemas[table_name] = {}
                elif table_type == "REL":
                    self._registered_edge_types.add(table_name)
        except Exception as exc:
            logger.warning(
                "Failed to scan existing tables (non-fatal): {}",
                exc,
            )

    def _table_exists_in_catalog(
        self,
        name: str,
        expected_type: str = "NODE",
    ) -> bool:
        """Check the Kuzu catalog for a table by name and type."""
        try:
            result = self._conn.execute(
                "CALL show_tables() WHERE name = $name AND type = $type RETURN count(*) AS cnt",
                {"name": name, "type": expected_type},
            )
            while result.has_next():
                row = result.get_next()
                return int(row[0]) > 0
        except Exception as exc:
            logger.debug("Catalog lookup failed for {}: {}", name, exc)
        return False

    def drop_node_table(self, label: str, cascade: bool = False) -> None:
        """Drop a node table.  If *cascade* is True, also drops referencing rel tables."""
        self._require_ready()
        self._require_not_readonly()

        if cascade:
            # Drop all edge types that reference this label
            # (We need to check which rel tables reference it)
            try:
                result = self._conn.execute(
                    "CALL show_tables() WHERE type = 'REL' RETURN name"
                )
                while result.has_next():
                    row = result.get_next()
                    rel_name = row[0]
                    try:
                        self._conn.execute(f"DROP TABLE {rel_name}")
                        self._registered_edge_types.discard(rel_name)
                        logger.debug("Dropped rel table {} (cascade from {})", rel_name, label)
                    except Exception:
                        # If drop fails, the rel table may not reference this label
                        pass
            except Exception as exc:
                logger.debug("Cascade rel scan failed: {}", exc)

        try:
            self._conn.execute(f"DROP TABLE {label}")
            self._registered_labels.discard(label)
            self._primary_keys.pop(label, None)
            self._label_schemas.pop(label, None)
            logger.debug("Dropped node table {}", label)
        except Exception as exc:
            raise SchemaViolationError(
                f"DROP TABLE {label}",
                str(exc),
            ) from exc

    def drop_edge_table(self, edge_type: str) -> None:
        """Drop a relationship table."""
        self._require_ready()
        self._require_not_readonly()

        try:
            self._conn.execute(f"DROP TABLE {edge_type}")
            self._registered_edge_types.discard(edge_type)
            logger.debug("Dropped rel table {}", edge_type)
        except Exception as exc:
            raise SchemaViolationError(
                f"DROP TABLE {edge_type}",
                str(exc),
            ) from exc


# ═══════════════════════════════════════════════════════════════════════
# Module-level helper
# ═══════════════════════════════════════════════════════════════════════

def edge_type_filter(edge_type: str | None) -> str:
    """Return a Cypher edge-type filter, e.g. ``:TYPE`` or ```` (empty)."""
    return f":{edge_type}" if edge_type else ""


# ═══════════════════════════════════════════════════════════════════════
# Public re-exports
# ═══════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    "KuzuGraphEngine",
]
