"""Abstract base classes for NeuroMem storage backends.

Design rationale
----------------
NeuroMem decouples cognitive logic from persistence through two contracts:

1. :class:`BaseGraphEngine`  — handles structured, relational data (nodes +
   edges) powered by Cypher.  Concrete implementations wrap Kuzu, Neo4j,
   or any Cypher-compatible engine.

2. :class:`BaseVectorEngine` — handles high-dimensional embeddings and
   approximate nearest-neighbour search.  Concrete implementations wrap
   ChromaDB, Qdrant, Milvus, or similar.

Both contracts are deliberately **synchronous**.  The cognitive engine calls
storage synchronously within a single-agent step.  If async execution is
desired in a production deployment, the *caller* (e.g. an async agent loop)
should dispatch storage calls through ``asyncio.to_thread`` rather than
forcing every implementation to be async.

Type signatures use only built-in generic containers (``list``, ``dict``,
``tuple``) so the module stays free of runtime imports while remaining fully
typed for static checkers (mypy, basedpyright, pyright).

All public methods accept and return plain Python data structures — never
engine-specific objects.  This ensures that swapping backends never leaks
implementation details into cognitive or client code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


# ═══════════════════════════════════════════════════════════════════════
# Shared data containers used across both storage contracts
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class NodeRecord:
    """A row returned from a graph node read operation.

    Immutable and hashable so it can be cached or used as a dict key.
    """

    node_id: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    """A row returned from a graph edge read operation."""

    src_id: str
    dst_id: str
    edge_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """A row returned from a vector similarity search."""

    id: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    document: str = ""
    distance: float | None = None


# ═══════════════════════════════════════════════════════════════════════
# Result wrapper — a type-safe, self-documenting alternative to raw tuples
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class QueryResult:
    """Wraps a Cypher or vector query result with metadata."""

    records: list[dict[str, Any]]
    columns: list[str] = field(default_factory=list)
    count: int = field(init=False)
    execution_time_ms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "count", len(self.records))

    def __len__(self) -> int:
        return self.count

    def __bool__(self) -> bool:
        return self.count > 0

    def first(self) -> dict[str, Any] | None:
        """Return the first record, or ``None`` if the result set is empty."""
        return self.records[0] if self.records else None

    def column_values(self, column: str) -> list[Any]:
        """Extract all values for a given column name.

        Raises ``KeyError`` if the column does not exist.
        """
        if column not in self.columns:
            raise KeyError(
                f"Column {column!r} not in result set. "
                f"Available: {self.columns}"
            )
        return [row[column] for row in self.records]


# ═══════════════════════════════════════════════════════════════════════
# Lifecycle state enum (shared by both engines)
# ═══════════════════════════════════════════════════════════════════════

EngineState = Literal[
    "uninitialized",
    "initializing",
    "ready",
    "degraded",
    "closed",
]


# ═══════════════════════════════════════════════════════════════════════
# BaseGraphEngine
# ═══════════════════════════════════════════════════════════════════════

class BaseGraphEngine(ABC):
    """Abstract contract for graph-database operations.

    Implementations must be safe for **single-writer, multi-reader** usage
    within a single process.  Thread safety beyond that is the caller's
    responsibility.

    Lifecycle
    ---------
    1. Construct the instance (``__init__``).
    2. Call :meth:`initialize` to create schemas / open connections.
    3. Perform CRUD and query operations.
    4. Call :meth:`close` to release resources.

    The ``state`` property reflects the current lifecycle phase.
    """

    # ── Lifecycle ──────────────────────────────────────────────────────

    @abstractmethod
    def initialize(self) -> None:
        """Open the database connection and bootstrap required schemas.

        Idempotent — calling on an already-initialised engine is a no-op.

        Raises
        ------
        GraphEngineError
            If the connection cannot be established or schema bootstrapping
            fails.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the connection and release all resources.

        Idempotent — calling on an already-closed engine is a no-op.
        """

    @property
    @abstractmethod
    def state(self) -> EngineState:
        """Return the current lifecycle state of the engine."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Shortcut: ``True`` when ``state == "ready"``."""

    # ── Schema management ────────────────────────────────────────────

    @abstractmethod
    def create_node_label(
        self,
        label: str,
        properties: dict[str, str] | None = None,
        primary_key: str | None = None,
    ) -> None:
        """Register a node label with an optional property schema.

        Parameters
        ----------
        label:
            The node label / type name (e.g. ``"BeliefNode"``).
        properties:
            Mapping of property names to their Kuzu-compatible type strings
            (e.g. ``{"confidence": "DOUBLE", "created_at": "TIMESTAMP"}``).
            If ``None``, no property constraints are enforced.
        primary_key:
            Name of the property that serves as the unique primary key.
            If ``None``, the engine may use a built-in ``_id`` column.

        Raises
        ------
        SchemaViolationError
            If the label already exists with an incompatible schema.
        GraphEngineError
            For other engine-level failures.
        """

    @abstractmethod
    def create_edge_type(
        self,
        edge_type: str,
        src_label: str,
        dst_label: str,
        properties: dict[str, str] | None = None,
    ) -> None:
        """Register a directed edge type between two node labels.

        Parameters
        ----------
        edge_type:
            Relationship type name (e.g. ``"CONTRADICTS"``).
        src_label:
            Label of the source node.
        dst_label:
            Label of the destination node.
        properties:
            Optional property schema for the edge.

        Raises
        ------
        SchemaViolationError
            If the edge type conflicts with an existing definition.
        GraphEngineError
            For other engine-level failures.
        """

    @abstractmethod
    def label_exists(self, label: str) -> bool:
        """Return ``True`` if a node label has been registered."""

    @abstractmethod
    def edge_type_exists(self, edge_type: str) -> bool:
        """Return ``True`` if an edge type has been registered."""

    # ── Node CRUD ────────────────────────────────────────────────────

    @abstractmethod
    def add_node(
        self,
        label: str,
        node_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Insert a new node.  Silently overwrites if the node already
        exists (upsert semantics).

        Parameters
        ----------
        label:
            Node label.
        node_id:
            Unique identifier (stored as the primary key property).
        properties:
            Additional key-value pairs to store on the node.

        Raises
        ------
        SchemaViolationError
            If ``label`` has not been registered.
        GraphEngineError
            For engine-level failures.
        """

    @abstractmethod
    def get_node(self, label: str, node_id: str) -> NodeRecord | None:
        """Fetch a single node by label and ID.

        Returns ``None`` if the node does not exist (never raises
        ``NodeNotFoundError`` — the caller decides whether that is an
        error).

        Parameters
        ----------
        label:
            Node label.
        node_id:
            Primary key.
        """

    @abstractmethod
    def update_node_properties(
        self,
        label: str,
        node_id: str,
        properties: dict[str, Any],
        merge: bool = True,
    ) -> None:
        """Update properties on an existing node.

        Parameters
        ----------
        label:
            Node label.
        node_id:
            Primary key.
        properties:
            New property values.
        merge:
            If ``True`` (default), merge with existing properties.  If
            ``False``, replace the entire property map.

        Raises
        ------
        NodeNotFoundError
            If the node does not exist.
        """

    @abstractmethod
    def delete_node(self, label: str, node_id: str, cascade_edges: bool = False) -> bool:
        """Delete a node.

        Parameters
        ----------
        label:
            Node label.
        node_id:
            Primary key.
        cascade_edges:
            If ``True``, also remove all edges connected to this node.

        Returns
        -------
        bool
            ``True`` if a node was actually deleted, ``False`` if the node
            did not exist.
        """

    @abstractmethod
    def node_exists(self, label: str, node_id: str) -> bool:
        """Check whether a node exists without fetching its data."""

    # ── Edge CRUD ────────────────────────────────────────────────────

    @abstractmethod
    def add_edge(
        self,
        src_label: str,
        src_id: str,
        dst_label: str,
        dst_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create a directed edge between two existing nodes.

        Parameters
        ----------
        src_label, src_id:
            Source node identification.
        dst_label, dst_id:
            Destination node identification.
        edge_type:
            Relationship type.
        properties:
            Optional edge properties.

        Raises
        ------
        NodeNotFoundError
            If either endpoint node does not exist.
        SchemaViolationError
            If ``edge_type`` has not been registered or is incompatible with
            the source/destination labels.
        """

    @abstractmethod
    def get_edge(
        self,
        src_label: str,
        src_id: str,
        dst_label: str,
        dst_id: str,
        edge_type: str,
    ) -> EdgeRecord | None:
        """Fetch a single edge.  Returns ``None`` if not found."""

    @abstractmethod
    def delete_edge(
        self,
        src_label: str,
        src_id: str,
        dst_label: str,
        dst_id: str,
        edge_type: str,
    ) -> bool:
        """Delete an edge.

        Returns ``True`` if an edge was actually deleted, ``False`` if it
        did not exist.
        """

    @abstractmethod
    def get_edges(
        self,
        label: str,
        node_id: str,
        direction: Literal["outgoing", "incoming", "both"] = "both",
        edge_type: str | None = None,
    ) -> list[EdgeRecord]:
        """Retrieve all edges connected to a node.

        Parameters
        ----------
        label:
            Label of the anchor node.
        node_id:
            ID of the anchor node.
        direction:
            Filter by edge direction.
        edge_type:
            Optional edge type filter.  If ``None``, return all types.
        """

    # ── Query ────────────────────────────────────────────────────────

    @abstractmethod
    def execute_query(self, cypher: str, parameters: dict[str, Any] | None = None) -> QueryResult:
        """Execute a raw Cypher query and return structured results.

        Parameters
        ----------
        cypher:
            The Cypher query string.
        parameters:
            Optional bound parameters (prevents Cypher injection).

        Returns
        -------
        QueryResult
            Container with ``records``, ``columns``, and metadata.

        Raises
        ------
        GraphQueryError
            If the query fails at the engine level.
        """

    # ── Bulk / maintenance ────────────────────────────────────────────

    @abstractmethod
    def count_nodes(self, label: str | None = None) -> int:
        """Return the total number of nodes, optionally filtered by label."""

    @abstractmethod
    def count_edges(self, edge_type: str | None = None) -> int:
        """Return the total number of edges, optionally filtered by type."""

    @abstractmethod
    def clear_all(self) -> None:
        """Remove **all** nodes and edges.  Use only in test teardown."""

    def transaction(self) -> _GraphTransaction:
        """Return a context manager for atomic write operations.

        Default implementation performs no-op commits; concrete engines
        should override this when the underlying database supports true
        transactions.

        Usage::

            with engine.transaction():
                engine.add_node(...)
                engine.add_edge(...)

        Raises
        ------
        GraphEngineError
            If the engine does not support transactions or the transaction
            commit fails.
        """
        return _GraphTransaction(self)


class _GraphTransaction:
    """Minimal no-op transaction context manager.

    Concrete engines override :meth:`BaseGraphEngine.transaction` to return
    their own subclass that calls ``commit`` / ``rollback``.
    """

    def __init__(self, engine: BaseGraphEngine) -> None:
        self._engine = engine

    def __enter__(self) -> _GraphTransaction:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if exc_val is not None:
            # Propagate the exception; concrete implementations may
            # add explicit rollback logic here.
            pass


# ═══════════════════════════════════════════════════════════════════════
# BaseVectorEngine
# ═══════════════════════════════════════════════════════════════════════

class BaseVectorEngine(ABC):
    """Abstract contract for vector-database operations.

    Collections map 1:1 to NeuroMem **namespaces**, providing natural
    isolation between agents or memory domains.

    Lifecycle mirrors :class:`BaseGraphEngine`:
    ``initialize`` → CRUD/search → ``close``.
    """

    # ── Lifecycle ──────────────────────────────────────────────────────

    @abstractmethod
    def initialize(self) -> None:
        """Open the database connection and prepare internal state.

        Raises
        ------
        VectorEngineError
            If the connection or initialisation fails.
        """

    @abstractmethod
    def close(self) -> None:
        """Release all resources.  Idempotent."""

    @property
    @abstractmethod
    def state(self) -> EngineState:
        """Return the current lifecycle state."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Shortcut: ``True`` when ``state == "ready"``."""

    # ── Collection management ───────────────────────────────────────

    @abstractmethod
    def create_collection(
        self,
        name: str,
        dimension: int | None = None,
        metadata: dict[str, Any] | None = None,
        distance_metric: Literal["cosine", "l2", "ip"] = "cosine",
    ) -> None:
        """Create a new collection (namespace).

        Parameters
        ----------
        name:
            Unique collection identifier.
        dimension:
            Embedding dimensionality.  If ``None``, the engine infers it
            from the first inserted vector.
        metadata:
            Arbitrary metadata attached to the collection.
        distance_metric:
            Distance function for similarity computation.

        Raises
        ------
        VectorEngineError
            If the collection already exists or creation fails.
        """

    @abstractmethod
    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all its records.

        Returns ``True`` if the collection existed and was removed,
        ``False`` otherwise.
        """

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """Check whether a collection exists."""

    @abstractmethod
    def list_collections(self) -> list[str]:
        """Return names of all collections."""

    @abstractmethod
    def get_collection_info(self, name: str) -> dict[str, Any]:
        """Return metadata and stats for a collection.

        Returned dict **must** contain at minimum:
        - ``"name"``: ``str``
        - ``"count"``: ``int``
        - ``"dimension"``: ``int | None``
        - ``"distance_metric"``: ``str``

        Raises
        ------
        CollectionNotFoundError
            If the collection does not exist.
        """

    # ── CRUD operations ───────────────────────────────────────────────

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or update records in a collection.

        Parameters
        ----------
        collection_name:
            Target collection.
        ids:
            Unique identifiers — one per vector.  Length must equal
            ``len(embeddings)``.
        embeddings:
            Dense vectors.
        documents:
            Optional raw-text documents associated with each vector.
        metadatas:
            Optional per-record metadata dicts.

        Raises
        ------
        CollectionNotFoundError
            If the collection does not exist.
        EmbeddingDimensionError
            If any vector has an unexpected dimension.
        VectorEngineError
            For other engine-level failures.
        """

    @abstractmethod
    def get(
        self,
        collection_name: str,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[VectorRecord]:
        """Fetch records by ID or metadata filter.

        Parameters
        ----------
        collection_name:
            Target collection.
        ids:
            Specific IDs to fetch.  If ``None`` and ``where`` is also
            ``None``, returns all records up to ``limit``.
        where:
            Metadata filter dict (e.g. ``{"namespace": "agent-1"}``).
        limit:
            Maximum number of records to return.  ``None`` means no cap.

        Returns
        -------
        list[VectorRecord]
            Matching records with their embeddings.

        Raises
        ------
        CollectionNotFoundError
            If the collection does not exist.
        """

    @abstractmethod
    def delete(
        self,
        collection_name: str,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> int:
        """Delete records by ID or metadata filter.

        Parameters
        ----------
        collection_name:
            Target collection.
        ids:
            Specific IDs to delete.
        where:
            Metadata filter.

        Returns
        -------
        int
            Number of records actually deleted.

        Raises
        ------
        CollectionNotFoundError
            If the collection does not exist.
        """

    # ── Search ──────────────────────────────────────────────────────

    @abstractmethod
    def similarity_search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[VectorRecord]:
        """Find the nearest neighbours to a query vector.

        Parameters
        ----------
        collection_name:
            Target collection.
        query_embedding:
            The query vector.
        n_results:
            Maximum number of results.
        where:
            Metadata pre-filter.
        where_document:
            Document-content pre-filter (e.g. ``{"$contains": "keyword"}``).

        Returns
        -------
        list[VectorRecord]
            Results sorted by ascending distance (most similar first).
            ``distance`` is populated on each record.

        Raises
        ------
        CollectionNotFoundError
            If the collection does not exist.
        EmbeddingDimensionError
            If ``query_embedding`` has the wrong dimension.
        VectorQueryError
            If the search operation fails.
        """

    # ── Bulk / maintenance ───────────────────────────────────────────

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Return the number of records in a collection.

        Raises
        ------
        CollectionNotFoundError
            If the collection does not exist.
        """

    @abstractmethod
    def clear_all(self) -> None:
        """Delete **all** collections and their records.  Test teardown only."""


# ═══════════════════════════════════════════════════════════════════════
# Public re-exports
# ═══════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    # Data containers
    "NodeRecord",
    "EdgeRecord",
    "VectorRecord",
    "QueryResult",
    # Shared type alias
    "EngineState",
    # Abstract engines
    "BaseGraphEngine",
    "BaseVectorEngine",
]
