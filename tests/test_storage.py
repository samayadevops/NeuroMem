"""Integration tests for the storage backends (Kuzu + ChromaDB).

These validate the concrete storage engine implementations against the
abstract contracts defined in :mod:`neuromem.storage.base`.
"""

from __future__ import annotations

import pytest

from neuromem.core.exceptions import (
    CollectionNotFoundError,
    EmbeddingDimensionError,
    GraphEngineError,
    GraphQueryError,
    NodeNotFoundError,
    SchemaViolationError,
    VectorEngineError,
    VectorQueryError,
)
from neuromem.storage.base import NodeRecord, QueryResult, VectorRecord
from neuromem.storage.kuzu_graph import KuzuGraphEngine
from neuromem.storage.chroma_vector import ChromaVectorEngine


# ═══════════════════════════════════════════════════════════════════════
# KuzuGraphEngine
# ═══════════════════════════════════════════════════════════════════════

class TestKuzuGraphEngine:
    """Tests for the Kuzu graph storage engine."""

    def test_initialize_and_close(self, graph_engine) -> None:
        assert graph_engine.is_ready
        graph_engine.close()
        assert graph_engine.state == "closed"

    def test_create_node_label(self, graph_engine) -> None:
        graph_engine.create_node_label(
            "TestNode", {"name": "STRING", "value": "INT64"},
        )
        assert graph_engine.label_exists("TestNode")

    def test_create_edge_type(self, graph_engine) -> None:
        graph_engine.create_node_label("Src", {"name": "STRING"})
        graph_engine.create_node_label("Dst", {"name": "STRING"})
        graph_engine.create_edge_type("LINKS", "Src", "Dst", {"weight": "DOUBLE"})
        assert graph_engine.edge_type_exists("LINKS")

    def test_add_and_get_node(self, graph_engine) -> None:
        graph_engine.create_node_label("Item", {"name": "STRING"})
        graph_engine.add_node("Item", "i1", {"name": "widget"})
        node = graph_engine.get_node("Item", "i1")
        assert node is not None
        assert node.node_id == "i1"
        assert node.label == "Item"
        assert node.properties["name"] == "widget"

    def test_get_node_nonexistent(self, graph_engine) -> None:
        graph_engine.create_node_label("Item2", {"name": "STRING"})
        assert graph_engine.get_node("Item2", "nonexistent") is None

    def test_node_exists(self, graph_engine) -> None:
        graph_engine.create_node_label("PresentNode", {"name": "STRING"})
        graph_engine.add_node("PresentNode", "e1", {"name": "x"})
        assert graph_engine.node_exists("PresentNode", "e1")
        assert not graph_engine.node_exists("PresentNode", "e2")

    def test_update_node_properties_merge(self, graph_engine) -> None:
        graph_engine.create_node_label("Merge", {"a": "STRING", "b": "STRING"})
        graph_engine.add_node("Merge", "m1", {"a": "1", "b": "2"})
        graph_engine.update_node_properties("Merge", "m1", {"a": "updated"})
        node = graph_engine.get_node("Merge", "m1")
        assert node.properties["a"] == "updated"
        assert node.properties["b"] == "2"  # merge preserves b

    def test_delete_node(self, graph_engine) -> None:
        graph_engine.create_node_label("Del", {"name": "STRING"})
        graph_engine.add_node("Del", "d1", {"name": "x"})
        assert graph_engine.delete_node("Del", "d1") is True
        assert not graph_engine.node_exists("Del", "d1")

    def test_delete_node_nonexistent(self, graph_engine) -> None:
        graph_engine.create_node_label("DelN", {"name": "STRING"})
        assert graph_engine.delete_node("DelN", "nope") is False

    def test_add_and_get_edge(self, graph_engine) -> None:
        graph_engine.create_node_label("Node", {"name": "STRING"})
        graph_engine.create_edge_type("REL", "Node", "Node", {"weight": "DOUBLE"})
        graph_engine.add_node("Node", "n1", {"name": "a"})
        graph_engine.add_node("Node", "n2", {"name": "b"})
        graph_engine.add_edge("Node", "n1", "Node", "n2", "REL", {"weight": 0.9})
        edge = graph_engine.get_edge("Node", "n1", "Node", "n2", "REL")
        assert edge is not None
        assert edge.properties["weight"] == 0.9

    def test_delete_edge(self, graph_engine) -> None:
        graph_engine.create_node_label("Node", {"name": "STRING"})
        graph_engine.create_edge_type("REL2", "Node", "Node")
        graph_engine.add_node("Node", "x1", {"name": "x"})
        graph_engine.add_node("Node", "x2", {"name": "y"})
        graph_engine.add_edge("Node", "x1", "Node", "x2", "REL2")
        assert graph_engine.delete_edge("Node", "x1", "Node", "x2", "REL2") is True
        assert graph_engine.get_edge("Node", "x1", "Node", "x2", "REL2") is None

    def test_count_nodes(self, graph_engine) -> None:
        graph_engine.create_node_label("Count", {"name": "STRING"})
        graph_engine.add_node("Count", "c1", {"name": "x"})
        graph_engine.add_node("Count", "c2", {"name": "y"})
        assert graph_engine.count_nodes("Count") == 2

    def test_execute_query(self, graph_engine) -> None:
        graph_engine.create_node_label("Query", {"name": "STRING"})
        graph_engine.add_node("Query", "q1", {"name": "alpha"})
        result = graph_engine.execute_query(
            "MATCH (n:Query) WHERE n.id = $id RETURN n.name AS name",
            {"id": "q1"},
        )
        assert isinstance(result, QueryResult)
        assert len(result) == 1
        assert result.first()["name"] == "alpha"

    def test_clear_all(self, graph_engine) -> None:
        graph_engine.create_node_label("Clear", {"name": "STRING"})
        graph_engine.add_node("Clear", "x", {"name": "data"})
        graph_engine.clear_all()
        assert graph_engine.count_nodes() == 0


# ═══════════════════════════════════════════════════════════════════════
# ChromaVectorEngine
# ═══════════════════════════════════════════════════════════════════════

class TestChromaVectorEngine:
    """Tests for the ChromaDB vector storage engine."""

    def test_initialize_and_close(self, vector_engine) -> None:
        assert vector_engine.is_ready
        vector_engine.close()
        assert vector_engine.state == "closed"

    def test_create_and_delete_collection(self, vector_engine) -> None:
        vector_engine.create_collection("test_col", dimension=4)
        assert vector_engine.collection_exists("test_col")
        assert vector_engine.delete_collection("test_col") is True
        assert not vector_engine.collection_exists("test_col")

    def test_delete_nonexistent_collection(self, vector_engine) -> None:
        assert vector_engine.delete_collection("nope") is False

    def test_list_collections(self, vector_engine) -> None:
        vector_engine.create_collection("list_a", dimension=2)
        vector_engine.create_collection("list_b", dimension=2)
        names = vector_engine.list_collections()
        assert "list_a" in names
        assert "list_b" in names

    def test_upsert_and_get(self, vector_engine) -> None:
        vector_engine.create_collection("crud", dimension=3)
        vector_engine.upsert(
            "crud",
            ids=["v1", "v2"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            documents=["doc1", "doc2"],
            metadatas=[{"k": "a"}, {"k": "b"}],
        )
        records = vector_engine.get("crud", ids=["v1"])
        assert len(records) == 1
        assert records[0].document == "doc1"

    def test_upsert_updates_existing(self, vector_engine) -> None:
        vector_engine.create_collection("upd", dimension=2)
        vector_engine.upsert("upd", ids=["x"], embeddings=[[0.1, 0.2]], documents=["old"], metadatas=[{"v": 1}])
        vector_engine.upsert("upd", ids=["x"], embeddings=[[0.3, 0.4]], documents=["new"], metadatas=[{"v": 2}])
        records = vector_engine.get("upd", ids=["x"])
        assert records[0].document == "new"
        assert vector_engine.count("upd") == 1

    def test_get_with_where_filter(self, vector_engine) -> None:
        vector_engine.create_collection("filter", dimension=2)
        vector_engine.upsert(
            "filter",
            ids=["a", "b"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"ns": "x"}, {"ns": "y"}],
        )
        records = vector_engine.get("filter", where={"ns": "x"})
        assert len(records) == 1
        assert records[0].id == "a"

    def test_similarity_search(self, vector_engine) -> None:
        vector_engine.create_collection("search", dimension=4, distance_metric="cosine")
        vector_engine.upsert(
            "search",
            ids=["s1", "s2", "s3"],
            embeddings=[[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6], [0.1, 0.1, 0.1, 0.1]],
            metadatas=[{"k": "1"}, {"k": "2"}, {"k": "3"}],
        )
        results = vector_engine.similarity_search("search", [0.1, 0.2, 0.3, 0.4], n_results=2)
        assert len(results) == 2
        # Most similar should be s1 (exact match, distance ~0)
        assert results[0].id == "s1"
        assert results[0].distance is not None
        # Results sorted by ascending distance
        assert results[0].distance <= results[1].distance

    def test_similarity_search_with_where(self, vector_engine) -> None:
        vector_engine.create_collection("where_search", dimension=2)
        vector_engine.upsert(
            "where_search",
            ids=["w1", "w2"],
            embeddings=[[0.1, 0.2], [0.1, 0.2]],
            metadatas=[{"cat": "a"}, {"cat": "b"}],
        )
        results = vector_engine.similarity_search(
            "where_search", [0.1, 0.2], n_results=5, where={"cat": "b"},
        )
        assert len(results) == 1
        assert results[0].id == "w2"

    def test_delete_by_ids(self, vector_engine) -> None:
        vector_engine.create_collection("del", dimension=2)
        vector_engine.upsert("del", ids=["d1", "d2"], embeddings=[[0.1, 0.2], [0.3, 0.4]], metadatas=[{"k": "1"}, {"k": "2"}])
        deleted = vector_engine.delete("del", ids=["d1"])
        assert deleted == 1
        assert vector_engine.count("del") == 1

    def test_delete_by_where(self, vector_engine) -> None:
        vector_engine.create_collection("delw", dimension=2)
        vector_engine.upsert(
            "delw",
            ids=["dw1", "dw2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"t": "temp"}, {"t": "keep"}],
        )
        deleted = vector_engine.delete("delw", where={"t": "temp"})
        assert deleted == 1

    def test_count(self, vector_engine) -> None:
        vector_engine.create_collection("cnt", dimension=2)
        vector_engine.upsert("cnt", ids=["c1", "c2"], embeddings=[[0.1, 0.2], [0.3, 0.4]], metadatas=[{"k": "1"}, {"k": "2"}])
        assert vector_engine.count("cnt") == 2

    def test_collection_not_found_raises(self, vector_engine) -> None:
        with pytest.raises(CollectionNotFoundError):
            vector_engine.get("nonexistent_col", ids=["x"])

    def test_dimension_validation_on_upsert(self, vector_engine) -> None:
        vector_engine.create_collection("dimval", dimension=4)
        with pytest.raises((EmbeddingDimensionError, Exception)):
            vector_engine.upsert("dimval", ids=["bad"], embeddings=[[0.1, 0.2, 0.3]])

    def test_get_collection_info(self, vector_engine) -> None:
        vector_engine.create_collection("info", dimension=4, distance_metric="cosine")
        info = vector_engine.get_collection_info("info")
        assert info["name"] == "info"
        assert info["dimension"] == 4
        assert info["distance_metric"] == "cosine"
        assert info["count"] == 0

    def test_clear_all(self, vector_engine) -> None:
        vector_engine.create_collection("ca1", dimension=2)
        vector_engine.create_collection("ca2", dimension=2)
        vector_engine.clear_all()
        assert len(vector_engine.list_collections()) == 0


# ═══════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════

class TestDataContainers:
    """Tests for the plain-data return types."""

    def test_node_record_immutable(self) -> None:
        node = NodeRecord(node_id="n1", label="Test", properties={"x": 1})
        with pytest.raises(Exception):
            node.node_id = "changed"  # frozen=True

    def test_vector_record_defaults(self) -> None:
        vr = VectorRecord(id="v1", embedding=[0.1, 0.2])
        assert vr.metadata == {}
        assert vr.document == ""
        assert vr.distance is None

    def test_query_result_first_empty(self) -> None:
        qr = QueryResult(records=[], columns=["x"])
        assert qr.first() is None
        assert bool(qr) is False
        assert len(qr) == 0

    def test_query_result_first_nonempty(self) -> None:
        qr = QueryResult(records=[{"x": 1}], columns=["x"])
        assert qr.first() == {"x": 1}
        assert bool(qr) is True
        assert len(qr) == 1

    def test_query_result_column_values(self) -> None:
        qr = QueryResult(
            records=[{"x": 1}, {"x": 2}],
            columns=["x"],
        )
        assert qr.column_values("x") == [1, 2]

    def test_query_result_column_values_missing(self) -> None:
        qr = QueryResult(records=[{"x": 1}], columns=["x"])
        with pytest.raises(KeyError):
            qr.column_values("nonexistent")


# ═══════════════════════════════════════════════════════════════════════
# BaseGraphEngine._GraphTransaction — no-op transaction context manager
# ═══════════════════════════════════════════════════════════════════════

class TestGraphTransaction:
    """Tests for the BaseGraphEngine.transaction() no-op context manager."""

    def test_transaction_success(self, graph_engine) -> None:
        """Transaction context manager yields itself and commits."""
        with graph_engine.transaction() as tx:
            assert tx is not None

    def test_transaction_rollback_propagates(self, graph_engine) -> None:
        """Exceptions inside transaction context are propagated."""
        with pytest.raises(RuntimeError, match="test error"):
            with graph_engine.transaction():
                raise RuntimeError("test error")


# ═══════════════════════════════════════════════════════════════════════
# ChromaVectorEngine error paths and edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestChromaVectorEngineErrorPaths:
    """Error-path and edge-case coverage for ChromaVectorEngine."""

    def test_normalize_embedding_none_raises(self) -> None:
        """_normalize_embedding(None) raises EmbeddingDimensionError."""
        from neuromem.storage.chroma_vector import _normalize_embedding
        with pytest.raises(EmbeddingDimensionError):
            _normalize_embedding(None)

    def test_extract_distance_metric_none_returns_cosine(self) -> None:
        from neuromem.storage.chroma_vector import _extract_distance_metric
        assert _extract_distance_metric(None) == "cosine"
        assert _extract_distance_metric({}) == "cosine"

    def test_records_from_query_result_out_of_range(self) -> None:
        """_records_from_query_result with query_index beyond data returns empty."""
        from neuromem.storage.chroma_vector import _records_from_query_result
        result = {
            "ids": [["a", "b"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.2]],
        }
        records = _records_from_query_result(result, query_index=5)
        assert records == []

    def test_delete_no_ids_no_where_raises(self, vector_engine) -> None:
        """delete() with neither ids nor where raises VectorEngineError."""
        vector_engine.create_collection("test_del")
        with pytest.raises(VectorEngineError, match="requires ids or where"):
            vector_engine.delete("test_del")

    def test_upsert_empty_ids_raises(self, vector_engine) -> None:
        vector_engine.create_collection("test_empty")
        with pytest.raises(VectorEngineError, match="at least one id"):
            vector_engine.upsert("test_empty", [], [[0.1, 0.2]])

    def test_upsert_length_mismatch_ids_embeddings(self, vector_engine) -> None:
        vector_engine.create_collection("test_mismatch")
        with pytest.raises(VectorEngineError, match="Length mismatch"):
            vector_engine.upsert("test_mismatch", ["a"], [[0.1, 0.2], [0.3, 0.4]])

    def test_upsert_length_mismatch_documents(self, vector_engine) -> None:
        vector_engine.create_collection("test_mismatch2")
        with pytest.raises(VectorEngineError, match="Length mismatch"):
            vector_engine.upsert("test_mismatch2", ["a"], [[0.1, 0.2]], documents=["d1", "d2"])

    def test_upsert_length_mismatch_metadatas(self, vector_engine) -> None:
        vector_engine.create_collection("test_mismatch3")
        with pytest.raises(VectorEngineError, match="Length mismatch"):
            vector_engine.upsert(
                "test_mismatch3", ["a"], [[0.1, 0.2]],
                metadatas=[{}, {}],
            )

    def test_collection_not_found_on_get(self, vector_engine) -> None:
        with pytest.raises(CollectionNotFoundError):
            vector_engine.get("nonexistent_collection")

    def test_collection_not_found_on_similarity_search(self, vector_engine) -> None:
        with pytest.raises(CollectionNotFoundError):
            vector_engine.similarity_search("nonexistent", [0.1, 0.2])

    def test_collection_not_found_on_count(self, vector_engine) -> None:
        with pytest.raises(CollectionNotFoundError):
            vector_engine.count("nonexistent")

    def test_collection_not_found_on_delete(self, vector_engine) -> None:
        with pytest.raises(CollectionNotFoundError):
            vector_engine.delete("never_existed")

    def test_dimension_mismatch_on_upsert(self, vector_engine) -> None:
        vector_engine.create_collection("test_dim", dimension=4)
        with pytest.raises(EmbeddingDimensionError):
            vector_engine.upsert("test_dim", ["a"], [[0.1, 0.2, 0.3]])  # 3-dim vs 4-dim

    def test_dimension_mismatch_on_search(self, vector_engine) -> None:
        vector_engine.create_collection("test_dim2", dimension=4)
        vector_engine.upsert(
            "test_dim2", ["a"], [[0.1, 0.2, 0.3, 0.4]], metadatas=[{"k": "v"}],
        )
        with pytest.raises(EmbeddingDimensionError):
            vector_engine.similarity_search("test_dim2", [0.1, 0.2, 0.3])  # wrong dim

    def test_require_ready_on_uninitialized(self) -> None:
        uninitialized = ChromaVectorEngine("./nonexistent")
        with pytest.raises(VectorEngineError, match="not ready"):
            uninitialized.similarity_search("any", [0.1])

    def test_require_ready_on_uninitialized_upsert(self) -> None:
        uninitialized = ChromaVectorEngine("./nonexistent")
        with pytest.raises(VectorEngineError, match="not ready"):
            uninitialized.upsert("any", ["a"], [[0.1]])

    def test_require_ready_on_uninitialized_create(self) -> None:
        uninitialized = ChromaVectorEngine("./nonexistent")
        with pytest.raises(VectorEngineError, match="not ready"):
            uninitialized.create_collection("any")

    def test_validate_storage_path_with_file(self, tmp_path) -> None:
        """If storage path is a file (not dir), it gets removed."""
        file_path = tmp_path / "vectors_file"
        file_path.write_text("junk")
        assert file_path.exists()
        from neuromem.storage.chroma_vector import _validate_storage_path
        _validate_storage_path(str(file_path))
        assert not file_path.exists()  # file was unlinked

    def test_list_collections(self, vector_engine) -> None:
        vector_engine.create_collection("lc_a")
        vector_engine.create_collection("lc_b")
        names = vector_engine.list_collections()
        assert "lc_a" in names
        assert "lc_b" in names

    def test_get_collection_info(self, vector_engine) -> None:
        vector_engine.create_collection("info_col", dimension=4)
        info = vector_engine.get_collection_info("info_col")
        assert info["name"] == "info_col"
        assert info["dimension"] == 4
        assert info["distance_metric"] == "cosine"

    def test_get_collection_info_nonexistent(self, vector_engine) -> None:
        with pytest.raises(CollectionNotFoundError):
            vector_engine.get_collection_info("never")

    def test_upsert_with_document_and_metadata(self, vector_engine) -> None:
        vector_engine.create_collection("doc_meta_col", dimension=2)
        vector_engine.upsert(
            "doc_meta_col", ["r1"], [[0.1, 0.2]],
            documents=["hello world"],
            metadatas=[{"source": "test"}],
        )
        records = vector_engine.get("doc_meta_col", ids=["r1"])
        assert len(records) == 1
        assert records[0].document == "hello world"
        assert records[0].metadata["source"] == "test"

    def test_similarity_search_with_where_filter(self, vector_engine) -> None:
        vector_engine.create_collection("where_col", dimension=2)
        vector_engine.upsert(
            "where_col", ["a", "b"], [[0.1, 0.2], [0.9, 0.8]],
            metadatas=[{"cat": "x"}, {"cat": "y"}],
        )
        hits = vector_engine.similarity_search(
            "where_col", [0.1, 0.2], n_results=5, where={"cat": "x"},
        )
        assert all(h.metadata.get("cat") == "x" for h in hits)

    def test_clear_all(self, vector_engine) -> None:
        vector_engine.create_collection("clear_a")
        vector_engine.create_collection("clear_b")
        vector_engine.clear_all()
        assert vector_engine.list_collections() == []

    def test_close_idempotent(self, vector_engine) -> None:
        vector_engine.close()
        vector_engine.close()  # second close is no-op
        assert vector_engine.state == "closed"


# ═══════════════════════════════════════════════════════════════════════
# KuzuGraphEngine error paths and edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestKuzuHelpers:
    """Unit tests for Kuzu module-level helpers."""

    def test_cypher_literal_none(self) -> None:
        from neuromem.storage.kuzu_graph import _cypher_literal
        assert _cypher_literal(None) == "null"

    def test_cypher_literal_bool(self) -> None:
        from neuromem.storage.kuzu_graph import _cypher_literal
        assert _cypher_literal(True) == "true"
        assert _cypher_literal(False) == "false"

    def test_cypher_literal_int_float(self) -> None:
        from neuromem.storage.kuzu_graph import _cypher_literal
        assert _cypher_literal(42) == "42"
        assert _cypher_literal(3.14) == "3.14"

    def test_cypher_literal_str(self) -> None:
        from neuromem.storage.kuzu_graph import _cypher_literal
        assert _cypher_literal("hello") == "'hello'"

    def test_cypher_literal_str_escape(self) -> None:
        from neuromem.storage.kuzu_graph import _cypher_literal
        # Backslashes and single quotes get escaped
        result = _cypher_literal("it's \\ here")
        assert "\\'" in result
        assert "\\\\" in result

    def test_cypher_literal_datetime(self) -> None:
        from neuromem.storage.kuzu_graph import _cypher_literal
        from datetime import datetime, timezone
        dt = datetime(2025, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        result = _cypher_literal(dt)
        assert "timestamp(" in result
        assert "2025" in result

    def test_cypher_literal_list(self) -> None:
        from neuromem.storage.kuzu_graph import _cypher_literal
        assert _cypher_literal([1, 2, 3]) == "[1, 2, 3]"

    def test_cypher_literal_object_fallback(self) -> None:
        from neuromem.storage.kuzu_graph import _cypher_literal

        class Foo:
            def __str__(self):
                return "foo_obj"

        result = _cypher_literal(Foo())
        assert "foo_obj" in result
        assert result.startswith("'") and result.endswith("'")

    def test_build_property_map_empty(self) -> None:
        from neuromem.storage.kuzu_graph import _build_property_map
        assert _build_property_map({}) == ""

    def test_build_property_map_with_values(self) -> None:
        from neuromem.storage.kuzu_graph import _build_property_map
        result = _build_property_map({"a": 1, "b": "x"})
        assert "a: 1" in result
        assert "b: 'x'" in result
        assert result.startswith("{")

    def test_infer_kuzu_type_none_type(self) -> None:
        from neuromem.storage.kuzu_graph import _infer_kuzu_type
        assert _infer_kuzu_type(type(None)) == "STRING"

    def test_infer_kuzu_type_list_generic(self) -> None:
        from neuromem.storage.kuzu_graph import _infer_kuzu_type
        assert _infer_kuzu_type(list[str]) == "STRING[]"

    def test_infer_kuzu_type_unrecognized(self) -> None:
        from neuromem.storage.kuzu_graph import _infer_kuzu_type
        assert _infer_kuzu_type(complex) == "STRING"

    def test_edge_type_filter(self) -> None:
        from neuromem.storage.kuzu_graph import edge_type_filter
        assert edge_type_filter(None) == ""
        assert edge_type_filter("SUPPORTS") == ":SUPPORTS"

    def test_validate_db_path_with_file(self, tmp_path) -> None:
        file_path = tmp_path / "db_file"
        file_path.write_text("junk")
        from neuromem.storage.kuzu_graph import _validate_db_path
        _validate_db_path(str(file_path))
        assert not file_path.exists()

    def test_strip_internal_node(self) -> None:
        from neuromem.storage.kuzu_graph import _strip_internal_node
        props = {"_id": 1, "name": "x", "_label": "L", "value": 42}
        clean = _strip_internal_node(props)
        assert "_id" not in clean
        assert "_label" not in clean
        assert clean["name"] == "x"
        assert clean["value"] == 42

    def test_strip_internal_edge(self) -> None:
        from neuromem.storage.kuzu_graph import _strip_internal_edge
        props = {"_src": 1, "_dst": 2, "weight": 0.5, "_label": "REL"}
        clean = _strip_internal_edge(props)
        assert "_src" not in clean
        assert "_dst" not in clean
        assert "_label" not in clean
        assert clean["weight"] == 0.5


class TestKuzuGraphEngineErrorPaths:
    """Error-path and edge-case coverage for KuzuGraphEngine."""

    def test_require_ready_on_uninitialized(self) -> None:
        uninitialized = KuzuGraphEngine("./nonexistent_kuzu")
        with pytest.raises(GraphEngineError, match="not ready"):
            uninitialized.execute_query("MATCH (n) RETURN n")

    def test_require_ready_on_uninitialized_add_node(self) -> None:
        uninitialized = KuzuGraphEngine("./nonexistent_kuzu")
        with pytest.raises(GraphEngineError, match="not ready"):
            uninitialized.add_node("L", "n1")

    def test_require_not_readonly_on_readonly_engine(self, graph_engine) -> None:
        """Mutating ops on a read-only-flagged engine raise GraphEngineError."""
        graph_engine._read_only = True
        try:
            with pytest.raises(GraphEngineError, match="read-only"):
                graph_engine.add_node("BeliefNode", "n1", {"x": 1})
        finally:
            graph_engine._read_only = False

    def test_readonly_missing_db_raises(self, tmp_path) -> None:
        engine = KuzuGraphEngine(str(tmp_path / "nonexistent"), read_only=True)
        with pytest.raises(GraphEngineError, match="does not exist"):
            engine.initialize()

    def test_get_edges_outgoing(self, graph_engine) -> None:
        graph_engine.create_node_label("Person", {"id": "STRING", "name": "STRING"}, primary_key="id")
        graph_engine.create_edge_type("KNOWS", "Person", "Person")
        graph_engine.add_node("Person", "alice", {"name": "Alice"})
        graph_engine.add_node("Person", "bob", {"name": "Bob"})
        graph_engine.add_edge("Person", "alice", "Person", "bob", "KNOWS")

        edges = graph_engine.get_edges("Person", "alice", direction="outgoing")
        assert len(edges) == 1
        assert edges[0].dst_id == "bob"

    def test_get_edges_incoming(self, graph_engine) -> None:
        graph_engine.create_node_label("Person", {"id": "STRING", "name": "STRING"}, primary_key="id")
        graph_engine.create_edge_type("KNOWS", "Person", "Person")
        graph_engine.add_node("Person", "alice", {"name": "Alice"})
        graph_engine.add_node("Person", "bob", {"name": "Bob"})
        graph_engine.add_edge("Person", "alice", "Person", "bob", "KNOWS")

        # Incoming edges to bob: alice→bob shows up
        edges = graph_engine.get_edges("Person", "bob", direction="incoming")
        assert len(edges) == 1
        # The connected node is alice (the other end of the edge)
        connected_ids = {edges[0].src_id, edges[0].dst_id}
        assert "alice" in connected_ids
        assert "bob" in connected_ids

    def test_get_edges_both(self, graph_engine) -> None:
        graph_engine.create_node_label("Person", {"id": "STRING", "name": "STRING"}, primary_key="id")
        graph_engine.create_edge_type("KNOWS", "Person", "Person")
        graph_engine.add_node("Person", "alice", {"name": "Alice"})
        graph_engine.add_node("Person", "bob", {"name": "Bob"})
        graph_engine.add_node("Person", "carol", {"name": "Carol"})
        graph_engine.add_edge("Person", "alice", "Person", "bob", "KNOWS")
        graph_engine.add_edge("Person", "carol", "Person", "alice", "KNOWS")

        edges = graph_engine.get_edges("Person", "alice", direction="both")
        assert len(edges) == 2

    def test_get_edges_with_type_filter(self, graph_engine) -> None:
        graph_engine.create_node_label("Person", {"id": "STRING", "name": "STRING"}, primary_key="id")
        graph_engine.create_edge_type("KNOWS", "Person", "Person")
        graph_engine.create_edge_type("LIKES", "Person", "Person")
        graph_engine.add_node("Person", "alice", {"name": "Alice"})
        graph_engine.add_node("Person", "bob", {"name": "Bob"})
        graph_engine.add_edge("Person", "alice", "Person", "bob", "KNOWS")
        graph_engine.add_edge("Person", "alice", "Person", "bob", "LIKES")

        edges = graph_engine.get_edges("Person", "alice", direction="outgoing", edge_type="KNOWS")
        assert len(edges) == 1

    def test_get_edges_empty(self, graph_engine) -> None:
        graph_engine.create_node_label("Person", {"id": "STRING", "name": "STRING"}, primary_key="id")
        graph_engine.add_node("Person", "alice", {"name": "Alice"})
        edges = graph_engine.get_edges("Person", "alice", direction="outgoing")
        assert edges == []

    def test_count_nodes_unfiltered(self, graph_engine) -> None:
        graph_engine.create_node_label("A", primary_key="id")
        graph_engine.create_node_label("B", primary_key="id")
        graph_engine.add_node("A", "a1")
        graph_engine.add_node("B", "b1")
        assert graph_engine.count_nodes() == 2

    def test_count_edges_unfiltered(self, graph_engine) -> None:
        graph_engine.create_node_label("A", primary_key="id")
        graph_engine.create_edge_type("REL", "A", "A")
        graph_engine.add_node("A", "a1")
        graph_engine.add_node("A", "a2")
        graph_engine.add_edge("A", "a1", "A", "a2", "REL")
        assert graph_engine.count_edges() == 1

    def test_count_edges_filtered(self, graph_engine) -> None:
        graph_engine.create_node_label("A", primary_key="id")
        graph_engine.create_edge_type("R1", "A", "A")
        graph_engine.create_edge_type("R2", "A", "A")
        graph_engine.add_node("A", "a1")
        graph_engine.add_node("A", "a2")
        graph_engine.add_edge("A", "a1", "A", "a2", "R1")
        graph_engine.add_edge("A", "a1", "A", "a2", "R2")
        assert graph_engine.count_edges("R1") == 1
        assert graph_engine.count_edges() == 2

    def test_update_node_properties_full_replace(self, graph_engine) -> None:
        """update_node_properties with merge=False does delete+recreate."""
        graph_engine.create_node_label("Item", {"id": "STRING", "name": "STRING", "price": "INT64"}, primary_key="id")
        graph_engine.add_node("Item", "i1", {"name": "old", "price": 10})
        graph_engine.update_node_properties("Item", "i1", {"name": "new", "price": 20}, merge=False)
        node = graph_engine.get_node("Item", "i1")
        assert node is not None
        assert node.properties["price"] == 20
        assert node.properties["name"] == "new"

    def test_update_node_nonexistent_raises(self, graph_engine) -> None:
        graph_engine.create_node_label("X", primary_key="id")
        with pytest.raises(NodeNotFoundError):
            graph_engine.update_node_properties("X", "ghost", {"v": 1})

    def test_add_edge_missing_src_raises(self, graph_engine) -> None:
        graph_engine.create_node_label("N", primary_key="id")
        graph_engine.create_edge_type("REL", "N", "N")
        graph_engine.add_node("N", "exists")
        with pytest.raises(NodeNotFoundError):
            graph_engine.add_edge("N", "ghost", "N", "exists", "REL")

    def test_add_edge_missing_dst_raises(self, graph_engine) -> None:
        graph_engine.create_node_label("N", primary_key="id")
        graph_engine.create_edge_type("REL", "N", "N")
        graph_engine.add_node("N", "exists")
        with pytest.raises(NodeNotFoundError):
            graph_engine.add_edge("N", "exists", "N", "ghost", "REL")

    def test_delete_edge_nonexistent_returns_false(self, graph_engine) -> None:
        graph_engine.create_node_label("N", primary_key="id")
        graph_engine.create_edge_type("REL", "N", "N")
        graph_engine.add_node("N", "a")
        graph_engine.add_node("N", "b")
        result = graph_engine.delete_edge("N", "a", "N", "b", "REL")
        assert result is False

    def test_drop_node_table_cascade(self, graph_engine) -> None:
        """drop_node_table with cascade=True drops related edge tables."""
        graph_engine.create_node_label("X", primary_key="id")
        graph_engine.create_edge_type("REL", "X", "X")
        graph_engine.drop_node_table("X", cascade=True)
        assert not graph_engine.label_exists("X")

    def test_drop_nonexistent_node_table_raises(self, graph_engine) -> None:
        with pytest.raises(SchemaViolationError):
            graph_engine.drop_node_table("NoSuchLabel")

    def test_drop_nonexistent_edge_table_raises(self, graph_engine) -> None:
        with pytest.raises(SchemaViolationError):
            graph_engine.drop_edge_table("NoSuchRel")

    def test_drop_edge_table_existing(self, graph_engine) -> None:
        graph_engine.create_node_label("N", primary_key="id")
        graph_engine.create_edge_type("RELABC", "N", "N")
        graph_engine.drop_edge_table("RELABC")
        assert not graph_engine.edge_type_exists("RELABC")

    def test_label_exists_fallback_catalog(self, graph_engine) -> None:
        graph_engine.create_node_label("CatalogTest", primary_key="id")
        assert graph_engine.label_exists("CatalogTest")
        assert not graph_engine.label_exists("DefinitelyMissing")

    def test_edge_type_exists_fallback_catalog(self, graph_engine) -> None:
        graph_engine.create_node_label("N", primary_key="id")
        graph_engine.create_edge_type("SOMEREL", "N", "N")
        assert graph_engine.edge_type_exists("SOMEREL")
        assert not graph_engine.edge_type_exists("MissingRel")

    def test_execute_query_with_retry_and_failure(self, tmp_path) -> None:
        """Engine with max_retries > 1 retries transient failures."""
        import unittest.mock

        engine = KuzuGraphEngine(str(tmp_path / "retry_db"), max_query_retries=3)
        engine.initialize()
        engine.create_node_label("N", primary_key="id")

        call_count = 0
        original_execute = engine._conn.execute

        def flaky_execute(cypher, params=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient lock")
            return original_execute(cypher, params)

        with unittest.mock.patch.object(engine._conn, "execute", side_effect=flaky_execute):
            result = engine.execute_query("MATCH (n:N) RETURN count(n) AS cnt")
            assert call_count == 3
        engine.close()

    def test_execute_query_all_retries_exhausted(self, tmp_path) -> None:
        """All retries exhausted raises GraphQueryError."""
        import unittest.mock

        engine = KuzuGraphEngine(str(tmp_path / "fail_db"), max_query_retries=2)
        engine.initialize()

        with unittest.mock.patch.object(
            engine._conn, "execute", side_effect=RuntimeError("permanent fail"),
        ):
            with pytest.raises(GraphQueryError):
                engine.execute_query("MATCH (n) RETURN n")
        engine.close()

    def test_ddl_failure_raises_schema_violation(self, graph_engine) -> None:
        """DDL execution failure is wrapped as SchemaViolationError."""
        import unittest.mock

        with unittest.mock.patch.object(
            graph_engine._conn, "execute", side_effect=RuntimeError("DDL fail"),
        ):
            with pytest.raises(SchemaViolationError):
                graph_engine.create_node_label("FailLabel", primary_key="id")

    def test_close_idempotent(self, graph_engine) -> None:
        graph_engine.close()
        graph_engine.close()  # no-op
        assert graph_engine.state == "closed"
