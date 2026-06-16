"""ChromaDB-based implementation of :class:`BaseVectorEngine`.

Design rationale
----------------
This module wraps ChromaDB's embedded vector engine behind NeuroMem's
abstract storage contract.  ChromaDB **collections** map 1:1 to NeuroMem
**namespaces**, giving each agent or memory domain a fully isolated vector
store.

Every method translates to one or more ChromaDB client calls, translates
the results into the framework's plain-data types (:class:`VectorRecord`),
and maps engine-level errors into NeuroMem's typed exception hierarchy.

ChromaDB-specific notes (v1.5.x)
--------------------------------
* **Client**: ``chromadb.PersistentClient(path=...)`` for on-disk storage.
* **Collections**: ``client.create_collection(name, metadata=...)``.
  The ``metadata`` dict can carry HNSW tuning (e.g. ``{"hnsw:space":
  "cosine"}``).
* **Add/Update**: ``collection.add(...)`` inserts (fails on duplicate IDs);
  ``collection.upsert(...)`` inserts-or-replaces.  We use ``upsert`` for
  all writes to provide idempotent semantics.
* **Get**: returns a dict with keys ``ids``, ``embeddings``, ``documents``,
  ``metadatas``.  ``embeddings`` is ``None`` unless explicitly requested
  via ``include=["embeddings"]``.
* **Query**: similarity search returns nested lists — one inner list per
  query embedding.
* **Errors**: ``chromadb.errors.NotFoundError`` for missing collections,
  ``chromadb.errors.InternalError`` for duplicates.  Dimension mismatches
  surface as ``chromadb.errors.InvalidDimensionException`` or generic
  ``ValueError``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from neuromem.core.exceptions import (
    CollectionNotFoundError,
    EmbeddingDimensionError,
    VectorEngineError,
    VectorQueryError,
)
from neuromem.storage.base import (
    BaseVectorEngine,
    EngineState,
    VectorRecord,
)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

DistanceMetric = Literal["cosine", "l2", "ip"]

# Mapping from our public distance-metric names to ChromaDB HNSW space keys.
_METRIC_TO_HNSW_SPACE: dict[str, str] = {
    "cosine": "cosine",
    "l2": "l2",
    "ip": "ip",
}

# Sentinel used to detect unset dimension values.
_UNSET_DIMENSION = -1


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _validate_storage_path(path_str: str) -> Path:
    """Resolve and validate the persistence directory path."""
    path = Path(path_str).resolve()
    if path.exists() and not path.is_dir():
        path.unlink()
    return path


def _utcnow() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _normalize_embedding(embedding: list[float] | None) -> list[float]:
    """Ensure an embedding is a list of floats.

    Raises ``EmbeddingDimensionError`` if the embedding is ``None``.
    """
    if embedding is None:
        raise EmbeddingDimensionError(0, 0)
    return [float(x) for x in embedding]


def _extract_distance_metric(metadata: dict[str, Any] | None) -> str:
    """Pull the distance metric from a collection's metadata dict."""
    if not metadata:
        return "cosine"
    return metadata.get("hnsw:space", "cosine")


def _records_from_get_result(
    result: dict[str, Any],
    include_embeddings: bool,
) -> list[VectorRecord]:
    """Translate a ChromaDB ``collection.get()`` dict into VectorRecords."""
    records: list[VectorRecord] = []
    ids = result.get("ids") or []
    documents = result.get("documents") or ["" for _ in ids]
    metadatas = result.get("metadatas") or [{} for _ in ids]
    embeddings = result.get("embeddings") if include_embeddings else None

    for idx, rid in enumerate(ids):
        emb: list[float] | None = None
        if include_embeddings and embeddings is not None:
            raw_emb = embeddings[idx]
            emb = [float(x) for x in raw_emb] if raw_emb is not None else []

        doc = documents[idx] if idx < len(documents) else ""
        meta = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}

        records.append(
            VectorRecord(
                id=str(rid),
                embedding=emb if emb is not None else [],
                metadata=dict(meta),
                document=doc if doc is not None else "",
                distance=None,
            )
        )

    return records


def _records_from_query_result(
    result: dict[str, Any],
    query_index: int = 0,
) -> list[VectorRecord]:
    """Translate a ChromaDB ``collection.query()`` dict into VectorRecords.

    ChromaDB returns nested lists — one inner list per query embedding.
    We extract the results for a single ``query_index``.
    """
    records: list[VectorRecord] = []

    ids_outer = result.get("ids") or []
    if not ids_outer or query_index >= len(ids_outer):
        return records

    ids = ids_outer[query_index] or []
    documents_outer = result.get("documents") or [[]]
    metadatas_outer = result.get("metadatas") or [[]]
    distances_outer = result.get("distances") or [[]]
    embeddings_outer = result.get("embeddings")
    if embeddings_outer is None:
        embeddings_outer = []

    documents = documents_outer[query_index] if query_index < len(documents_outer) else []
    metadatas = metadatas_outer[query_index] if query_index < len(metadatas_outer) else []
    distances = distances_outer[query_index] if query_index < len(distances_outer) else []
    embeddings = (
        embeddings_outer[query_index]
        if embeddings_outer is not None and query_index < len(embeddings_outer)
        else []
    )

    for idx, rid in enumerate(ids):
        emb: list[float] = []
        if embeddings is not None and idx < len(embeddings) and embeddings[idx] is not None:
            emb = [float(x) for x in embeddings[idx]]

        doc = documents[idx] if idx < len(documents) and documents[idx] else ""
        meta = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
        dist = distances[idx] if idx < len(distances) else None

        records.append(
            VectorRecord(
                id=str(rid),
                embedding=emb,
                metadata=dict(meta),
                document=doc if doc else "",
                distance=float(dist) if dist is not None else None,
            )
        )

    return records


# ═══════════════════════════════════════════════════════════════════════
# ChromaVectorEngine
# ═══════════════════════════════════════════════════════════════════════

class ChromaVectorEngine(BaseVectorEngine):
    """Production ChromaDB-backed vector engine.

    Collections map 1:1 to NeuroMem namespaces, providing natural
    isolation between agents or memory domains.

    Parameters
    ----------
    storage_path:
        Filesystem directory for ChromaDB's persistent storage.
        Created automatically if it does not exist.
    default_distance_metric:
        Distance function used when a collection is created without an
        explicit ``distance_metric`` argument.  Defaults to ``"cosine"``.
    anonymized_telemetry:
        If ``False`` (default), disables ChromaDB's anonymous usage
        telemetry.
    """

    def __init__(
        self,
        storage_path: str = "./neuromem_vectors",
        *,
        default_distance_metric: DistanceMetric = "cosine",
        anonymized_telemetry: bool = False,
    ) -> None:
        self._storage_path: str = storage_path
        self._default_metric: DistanceMetric = default_distance_metric
        self._anonymized_telemetry: bool = anonymized_telemetry
        self._client: Any | None = None  # chromadb.api.Client
        self._state: EngineState = "uninitialized"
        # Cache of collection metadata to avoid repeated client round-trips.
        self._collection_cache: dict[str, dict[str, Any]] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Open the ChromaDB persistent client."""
        if self._state == "ready":
            logger.debug("ChromaVectorEngine already initialized — skipping")
            return

        self._state = "initializing"
        logger.info(
            "Initializing ChromaVectorEngine",
            storage_path=self._storage_path,
            default_metric=self._default_metric,
        )

        try:
            import chromadb
            from chromadb.config import Settings

            resolved = _validate_storage_path(self._storage_path)
            resolved.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(resolved),
                settings=Settings(anonymized_telemetry=self._anonymized_telemetry),
            )

            # Prime the collection cache by scanning existing collections
            self._refresh_collection_cache()

            self._state = "ready"
            logger.info(
                "ChromaVectorEngine initialized",
                storage_path=self._storage_path,
                collection_count=len(self._collection_cache),
            )

        except VectorEngineError:
            raise
        except Exception as exc:
            self._state = "closed"
            raise VectorEngineError(
                f"Failed to initialize ChromaDB: {exc}",
                backend="chromadb",
                context={
                    "storage_path": self._storage_path,
                    "error_type": type(exc).__name__,
                },
            ) from exc

    def close(self) -> None:
        """Release the ChromaDB client handle."""
        if self._state == "closed":
            return
        logger.info("Closing ChromaVectorEngine", storage_path=self._storage_path)
        self._client = None
        self._collection_cache.clear()
        self._state = "closed"

    @property
    def state(self) -> EngineState:
        """Return the current lifecycle state."""
        return self._state

    @property
    def is_ready(self) -> bool:
        """``True`` when the engine is initialized and ready for queries."""
        return self._state == "ready"

    # ── Collection management ───────────────────────────────────────

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
            Embedding dimensionality.  ChromaDB infers this from the first
            inserted vector if ``None``, but we store it in metadata for
            later validation.
        metadata:
            Arbitrary metadata attached to the collection.  Merged with
            the HNSW distance-metric setting.
        distance_metric:
            Distance function for similarity computation.
        """
        self._require_ready()

        if self.collection_exists(name):
            logger.debug("Collection {} already exists — skipping", name)
            return

        # Build the merged metadata dict
        hnsw_space = _METRIC_TO_HNSW_SPACE.get(distance_metric, "cosine")
        merged_meta: dict[str, Any] = {"hnsw:space": hnsw_space}
        if dimension is not None and dimension > 0:
            merged_meta["neuromem:dimension"] = dimension
        if metadata:
            merged_meta.update(metadata)

        try:
            self._client.create_collection(
                name=name,
                metadata=merged_meta,
            )
        except Exception as exc:
            # Distinguish duplicate-collection from other failures
            err_msg = str(exc).lower()
            if "already exists" in err_msg:
                logger.debug("Collection {} already exists (race) — skipping", name)
            else:
                raise VectorEngineError(
                    f"Failed to create collection {name!r}: {exc}",
                    backend="chromadb",
                    context={"collection_name": name, "error_type": type(exc).__name__},
                ) from exc

        self._collection_cache[name] = {
            "name": name,
            "dimension": dimension,
            "distance_metric": distance_metric,
            "metadata": dict(merged_meta),
        }
        logger.debug(
            "Created collection {} (metric={}, dimension={})",
            name, distance_metric, dimension,
        )

    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all its records."""
        self._require_ready()

        if not self.collection_exists(name):
            return False

        try:
            self._client.delete_collection(name)
        except Exception as exc:
            err_msg = str(exc).lower()
            if "does not exist" in err_msg or "not found" in err_msg:
                return False
            raise VectorEngineError(
                f"Failed to delete collection {name!r}: {exc}",
                backend="chromadb",
                context={"collection_name": name},
            ) from exc

        self._collection_cache.pop(name, None)
        logger.debug("Deleted collection {}", name)
        return True

    def collection_exists(self, name: str) -> bool:
        """Check whether a collection exists."""
        self._require_ready()

        if name in self._collection_cache:
            return True

        # Fallback: scan the client's collection list
        try:
            collections = self._client.list_collections()
            for col in collections:
                col_name = col.name if hasattr(col, "name") else str(col)
                if col_name == name:
                    return True
            return False
        except Exception as exc:
            logger.debug("collection_exists scan failed for {}: {}", name, exc)
            return False

    def list_collections(self) -> list[str]:
        """Return names of all collections."""
        self._require_ready()

        try:
            collections = self._client.list_collections()
            names: list[str] = []
            for col in collections:
                col_name = col.name if hasattr(col, "name") else str(col)
                names.append(col_name)
            return names
        except Exception as exc:
            raise VectorEngineError(
                f"Failed to list collections: {exc}",
                backend="chromadb",
            ) from exc

    def get_collection_info(self, name: str) -> dict[str, Any]:
        """Return metadata and stats for a collection."""
        self._require_ready()

        collection = self._get_collection_or_raise(name)

        try:
            metadata = collection.metadata or {}
            count = collection.count()
            dimension = metadata.get("neuromem:dimension")
            distance_metric = _extract_distance_metric(metadata)

            return {
                "name": name,
                "count": int(count),
                "dimension": int(dimension) if dimension is not None else None,
                "distance_metric": distance_metric,
                "metadata": dict(metadata),
            }
        except Exception as exc:
            raise VectorEngineError(
                f"Failed to get info for collection {name!r}: {exc}",
                backend="chromadb",
                context={"collection_name": name},
            ) from exc

    # ── CRUD operations ───────────────────────────────────────────────

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or update records in a collection."""
        self._require_ready()

        if not ids:
            raise VectorEngineError(
                "upsert() requires at least one id",
                backend="chromadb",
                context={"collection_name": collection_name},
            )

        if len(ids) != len(embeddings):
            raise VectorEngineError(
                f"Length mismatch: {len(ids)} ids vs {len(embeddings)} embeddings",
                backend="chromadb",
                context={"collection_name": collection_name},
            )

        if documents is not None and len(documents) != len(ids):
            raise VectorEngineError(
                f"Length mismatch: {len(ids)} ids vs {len(documents)} documents",
                backend="chromadb",
                context={"collection_name": collection_name},
            )

        if metadatas is not None and len(metadatas) != len(ids):
            raise VectorEngineError(
                f"Length mismatch: {len(ids)} ids vs {len(metadatas)} metadatas",
                backend="chromadb",
                context={"collection_name": collection_name},
            )

        collection = self._get_collection_or_raise(collection_name)

        # Validate embedding dimensions against the collection's recorded dimension
        info = self.get_collection_info(collection_name)
        expected_dim = info.get("dimension")
        if expected_dim is not None:
            for idx, emb in enumerate(embeddings):
                actual_dim = len(emb)
                if actual_dim != expected_dim:
                    raise EmbeddingDimensionError(expected_dim, actual_dim)

        # Normalize embeddings to float lists
        norm_embeddings = [_normalize_embedding(emb) for emb in embeddings]

        # Default empty documents/metadatas
        norm_documents = documents if documents is not None else [""] * len(ids)
        norm_metadatas = metadatas if metadatas is not None else [{} for _ in ids]

        # If the collection had no recorded dimension, record it now
        if expected_dim is None and norm_embeddings:
            actual_dim = len(norm_embeddings[0])
            self._record_dimension(collection_name, actual_dim)

        try:
            collection.upsert(
                ids=ids,
                embeddings=norm_embeddings,
                documents=norm_documents,
                metadatas=norm_metadatas,
            )
            logger.debug(
                "Upserted {} record(s) into {}",
                len(ids), collection_name,
            )
        except Exception as exc:
            err_msg = str(exc).lower()
            if "dimension" in err_msg or "dim" in err_msg:
                # Try to extract expected/actual from the message
                raise EmbeddingDimensionError(-1, -1) from exc
            raise VectorEngineError(
                f"Failed to upsert into {collection_name!r}: {exc}",
                backend="chromadb",
                context={
                    "collection_name": collection_name,
                    "count": len(ids),
                    "error_type": type(exc).__name__,
                },
            ) from exc

    def get(
        self,
        collection_name: str,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[VectorRecord]:
        """Fetch records by ID or metadata filter."""
        self._require_ready()

        collection = self._get_collection_or_raise(collection_name)

        # Build kwargs — ChromaDB's get() accepts None for optional args
        kwargs: dict[str, Any] = {
            "include": ["embeddings", "documents", "metadatas"],
        }
        if ids is not None:
            kwargs["ids"] = ids
        if where is not None:
            kwargs["where"] = where
        if limit is not None:
            kwargs["limit"] = limit

        try:
            result = collection.get(**kwargs)
        except Exception as exc:
            raise VectorQueryError(
                "get",
                str(exc),
            ) from exc

        return _records_from_get_result(result, include_embeddings=True)

    def delete(
        self,
        collection_name: str,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> int:
        """Delete records by ID or metadata filter.

        Returns the number of records actually deleted.
        """
        self._require_ready()

        collection = self._get_collection_or_raise(collection_name)

        # Count before delete to compute the delta
        count_before = collection.count()

        if ids is None and where is None:
            raise VectorEngineError(
                f"delete() on {collection_name!r} requires ids or where",
                backend="chromadb",
                context={"collection_name": collection_name},
            )

        delete_kwargs: dict[str, Any] = {}
        if ids is not None:
            delete_kwargs["ids"] = ids
        if where is not None:
            delete_kwargs["where"] = where

        try:
            collection.delete(**delete_kwargs)
        except Exception as exc:
            raise VectorQueryError(
                "delete",
                str(exc),
            ) from exc

        count_after = collection.count()
        deleted = count_before - count_after
        logger.debug(
            "Deleted {} record(s) from {} (before={}, after={})",
            deleted, collection_name, count_before, count_after,
        )
        return deleted

    # ── Search ──────────────────────────────────────────────────────

    def similarity_search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[VectorRecord]:
        """Find the nearest neighbours to a query vector."""
        self._require_ready()

        collection = self._get_collection_or_raise(collection_name)

        # Validate query embedding dimension
        info = self.get_collection_info(collection_name)
        expected_dim = info.get("dimension")
        if expected_dim is not None and len(query_embedding) != expected_dim:
            raise EmbeddingDimensionError(expected_dim, len(query_embedding))

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [_normalize_embedding(query_embedding)],
            "n_results": n_results,
            "include": ["embeddings", "documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where
        if where_document is not None:
            query_kwargs["where_document"] = where_document

        try:
            result = collection.query(**query_kwargs)
        except Exception as exc:
            err_msg = str(exc).lower()
            if "dimension" in err_msg or "dim" in err_msg:
                raise EmbeddingDimensionError(expected_dim or -1, len(query_embedding)) from exc
            raise VectorQueryError(
                "similarity_search",
                str(exc),
            ) from exc

        return _records_from_query_result(result, query_index=0)

    # ── Bulk / maintenance ───────────────────────────────────────────

    def count(self, collection_name: str) -> int:
        """Return the number of records in a collection."""
        self._require_ready()

        collection = self._get_collection_or_raise(collection_name)
        try:
            return int(collection.count())
        except Exception as exc:
            raise VectorEngineError(
                f"Failed to count {collection_name!r}: {exc}",
                backend="chromadb",
                context={"collection_name": collection_name},
            ) from exc

    def clear_all(self) -> None:
        """Delete all collections.  For test teardown only."""
        self._require_ready()

        logger.warning("Clearing all collections from ChromaDB", storage_path=self._storage_path)

        try:
            names = self.list_collections()
            for name in names:
                try:
                    self._client.delete_collection(name)
                except Exception as exc:
                    logger.warning("Failed to delete collection {}: {}", name, exc)
            self._collection_cache.clear()
        except Exception as exc:
            raise VectorEngineError(
                f"Failed to clear all collections: {exc}",
                backend="chromadb",
            ) from exc

    # ── Internal helpers ─────────────────────────────────────────────

    def _require_ready(self) -> None:
        """Raise if the engine is not ready for operations."""
        if self._state != "ready":
            raise VectorEngineError(
                f"ChromaDB engine is not ready (current state: {self._state!r}). "
                "Call initialize() first.",
                backend="chromadb",
            )

    def _get_collection_or_raise(self, name: str) -> Any:
        """Fetch a ChromaDB collection handle, raising on missing."""
        try:
            return self._client.get_collection(name)
        except Exception as exc:
            err_msg = str(exc).lower()
            if "does not exist" in err_msg or "not found" in err_msg:
                raise CollectionNotFoundError(name) from exc
            raise VectorEngineError(
                f"Failed to access collection {name!r}: {exc}",
                backend="chromadb",
                context={"collection_name": name},
            ) from exc

    def _refresh_collection_cache(self) -> None:
        """Rebuild the in-memory collection metadata cache."""
        self._collection_cache.clear()
        try:
            collections = self._client.list_collections()
            for col in collections:
                name = col.name if hasattr(col, "name") else str(col)
                metadata = col.metadata or {} if hasattr(col, "metadata") else {}
                dimension = metadata.get("neuromem:dimension")
                metric = _extract_distance_metric(metadata)
                self._collection_cache[name] = {
                    "name": name,
                    "dimension": int(dimension) if dimension is not None else None,
                    "distance_metric": metric,
                    "metadata": dict(metadata) if metadata else {},
                }
        except Exception as exc:
            logger.warning("Failed to refresh collection cache (non-fatal): {}", exc)

    def _record_dimension(self, collection_name: str, dimension: int) -> None:
        """Record the embedding dimension in a collection's metadata.

        ChromaDB does not natively store dimension as a first-class
        concept, so we persist it in the collection metadata under
        ``neuromem:dimension``.  This requires updating the collection's
        metadata, which ChromaDB does via ``update_collection`` if
        available, otherwise we keep it in our in-memory cache only.
        """
        # Update in-memory cache
        if collection_name in self._collection_cache:
            self._collection_cache[collection_name]["dimension"] = dimension
        else:
            self._collection_cache[collection_name] = {
                "name": collection_name,
                "dimension": dimension,
                "distance_metric": self._default_metric,
                "metadata": {},
            }

        # Attempt to persist via update_collection (ChromaDB >= 0.5)
        try:
            collection = self._client.get_collection(collection_name)
            if hasattr(collection, "metadata") and collection.metadata is not None:
                updated_meta = dict(collection.metadata)
                updated_meta["neuromem:dimension"] = dimension
                # ChromaDB 1.x: update_collection may not exist; fall back gracefully
                if hasattr(self._client, "update_collection"):
                    self._client.update_collection(
                        name=collection_name,
                        metadata=updated_meta,
                    )
        except Exception as exc:
            # Non-fatal — dimension is cached in-memory for this session
            logger.debug(
                "Could not persist dimension for {} (non-fatal): {}",
                collection_name, exc,
            )


# ═══════════════════════════════════════════════════════════════════════
# Public re-exports
# ═══════════════════════════════════════════════════════════════════════

__all__: list[str] = [
    "ChromaVectorEngine",
    "DistanceMetric",
]
