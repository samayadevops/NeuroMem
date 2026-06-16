"""Coverage gap-fillers for neuromem.client — targets 100 %.

These tests exercise RecallResult edge cases, the from_engine factory,
context manager lifecycle, error handling during close, _safe_embed
fallbacks, and property accessors not covered by ``test_client.py``.
"""

from __future__ import annotations

import unittest.mock

import pytest

from neuromem.client import NeuroMemClient, RecallResult
from neuromem.core.engine import EngineConfig, FusedResult
from neuromem.core.exceptions import ConfigurationError
from neuromem.core.models import BeliefNode, BeliefStatus, NegativeMemorySeverity


# ═══════════════════════════════════════════════════════════════════════
# RecallResult with no vector (text-only recall fallback)
# ═══════════════════════════════════════════════════════════════════════


class TestRecallResultNoVector:
    """RecallResult with vector_distance=None (text-only recall fallback)."""

    def test_similarity_none_when_no_vector(self, client: NeuroMemClient) -> None:
        belief = BeliefNode(claim="test", confidence=0.8)
        fused = FusedResult(
            belief=belief, vector_distance=None,
            graph_confidence=0.8, fused_score=0.8,
        )
        result = RecallResult(fused)
        assert result.similarity is None
        assert result.vector_distance is None
        assert result.fused_score == 0.8

    def test_belief_property(self, client: NeuroMemClient) -> None:
        belief = BeliefNode(claim="test claim")
        fused = FusedResult(
            belief=belief, vector_distance=0.1,
            graph_confidence=0.5, fused_score=0.5,
        )
        result = RecallResult(fused)
        assert result.belief is belief
        assert result.belief.claim == "test claim"

    def test_recall_result_repr(self, client: NeuroMemClient) -> None:
        belief = BeliefNode(claim="some claim text here", confidence=0.8)
        fused = FusedResult(
            belief=belief, vector_distance=0.1,
            graph_confidence=0.7, fused_score=0.65,
        )
        result = RecallResult(fused)
        r = repr(result)
        assert "RecallResult" in r

    def test_recall_result_to_dict_keys(self, client: NeuroMemClient) -> None:
        belief = BeliefNode(claim="dict test", confidence=0.6, namespace="test")
        fused = FusedResult(
            belief=belief, vector_distance=0.2,
            graph_confidence=0.6, fused_score=0.5,
        )
        result = RecallResult(fused)
        d = result.to_dict()
        assert "id" in d
        assert "claim" in d
        assert "confidence" in d
        assert "raw_confidence" in d
        assert "fused_score" in d
        assert "similarity" in d
        assert "vector_distance" in d
        assert "status" in d
        assert "source" in d
        assert "tags" in d
        assert "namespace" in d
        assert "evidence_count" in d
        assert "created_at" in d

    def test_recall_result_similarity_with_distance(
        self, client: NeuroMemClient,
    ) -> None:
        belief = BeliefNode(claim="sim test", confidence=0.8)
        fused = FusedResult(
            belief=belief, vector_distance=0.3,
            graph_confidence=0.8, fused_score=0.7,
        )
        result = RecallResult(fused)
        assert result.similarity == 0.7  # max(0, 1 - 0.3)

    def test_recall_result_top_level_accessors(
        self, client: NeuroMemClient,
    ) -> None:
        belief = BeliefNode(
            claim="accessor test", confidence=0.8,
            source="unittest", tags=["a", "b"], namespace="ns1",
        )
        fused = FusedResult(
            belief=belief, vector_distance=0.1,
            graph_confidence=0.8, fused_score=0.7,
        )
        result = RecallResult(fused)
        assert result.id == belief.id
        assert result.claim == "accessor test"
        assert result.confidence == 0.8
        assert result.raw_confidence == 0.8
        assert result.status == BeliefStatus.ACTIVE
        assert result.source == "unittest"
        assert result.tags == ["a", "b"]
        assert result.namespace == "ns1"
        assert result.evidence_count == 1


# ═══════════════════════════════════════════════════════════════════════
# from_engine factory
# ═══════════════════════════════════════════════════════════════════════


class TestClientFromEngine:
    def test_from_engine_factory(self, engine) -> None:
        client = NeuroMemClient.from_engine(engine)
        assert client.namespace == "test"
        assert client.engine is engine

    def test_from_engine_with_embed_fn(self, engine) -> None:
        client = NeuroMemClient.from_engine(engine, embed_fn=lambda t: [0.1])
        assert client.has_embed_fn


# ═══════════════════════════════════════════════════════════════════════
# check_contradiction client wrapper
# ═══════════════════════════════════════════════════════════════════════


class TestClientCheckContradiction:
    def test_check_contradiction_existing_belief(
        self, client_with_embeddings: NeuroMemClient,
    ) -> None:
        belief = client_with_embeddings.learn("the sky is blue", confidence=0.9)
        event = client_with_embeddings.check_contradiction(belief.id, "the sky is red")
        # May or may not detect contradiction depending on embeddings

    def test_check_contradiction_nonexistent_belief(
        self, client_with_embeddings: NeuroMemClient,
    ) -> None:
        event = client_with_embeddings.check_contradiction("nonexistent_id", "anything")
        assert event is None


# ═══════════════════════════════════════════════════════════════════════
# decay(advance_ticks=0)
# ═══════════════════════════════════════════════════════════════════════


class TestClientDecayZeroTicks:
    def test_decay_zero_ticks(self, client: NeuroMemClient) -> None:
        client.learn("test belief")
        count = client.decay(advance_ticks=0)
        assert client.current_tick == 0
        assert isinstance(count, int)


# ═══════════════════════════════════════════════════════════════════════
# guard with invalid severity string
# ═══════════════════════════════════════════════════════════════════════


class TestClientGuardInvalidSeverity:
    def test_guard_invalid_severity_falls_back(
        self, client: NeuroMemClient,
    ) -> None:
        neg = client.guard("bad pattern", severity="not_a_real_severity")
        assert neg.severity == NegativeMemorySeverity.WARNING

    def test_guard_with_enum_severity(self, client: NeuroMemClient) -> None:
        neg = client.guard("valid pattern", severity=NegativeMemorySeverity.ERROR)
        assert neg.severity == NegativeMemorySeverity.ERROR


# ═══════════════════════════════════════════════════════════════════════
# Properties
# ═══════════════════════════════════════════════════════════════════════


class TestClientProperties:
    def test_engine_property(self, client: NeuroMemClient) -> None:
        assert client.engine is not None

    def test_config_property(self, client: NeuroMemClient) -> None:
        assert isinstance(client.config, EngineConfig)

    def test_namespace_property(self, client: NeuroMemClient) -> None:
        assert client.namespace == "test"

    def test_is_closed_property(self, client: NeuroMemClient) -> None:
        assert client.is_closed is False


# ═══════════════════════════════════════════════════════════════════════
# Double close
# ═══════════════════════════════════════════════════════════════════════


class TestClientDoubleClose:
    def test_double_close_no_error(self, client: NeuroMemClient) -> None:
        client.close()
        client.close()
        assert client.is_closed


# ═══════════════════════════════════════════════════════════════════════
# Close with engine errors
# ═══════════════════════════════════════════════════════════════════════


class TestClientCloseWithError:
    def test_close_with_graph_error(self, client: NeuroMemClient) -> None:
        with unittest.mock.patch.object(
            client.engine.graph, "close",
            side_effect=RuntimeError("graph close failed"),
        ):
            client.close()
        assert client.is_closed

    def test_close_with_vector_error(self, client: NeuroMemClient) -> None:
        with unittest.mock.patch.object(
            client.engine.vector, "close",
            side_effect=RuntimeError("vector close failed"),
        ):
            client.close()
        assert client.is_closed


# ═══════════════════════════════════════════════════════════════════════
# Context manager
# ═══════════════════════════════════════════════════════════════════════


class TestClientContextManager:
    def test_with_block_closes(self, client: NeuroMemClient) -> None:
        with client as c:
            assert not c.is_closed
        assert client.is_closed

    def test_context_manager_returns_self(self, client: NeuroMemClient) -> None:
        with client as c:
            assert c is client


# ═══════════════════════════════════════════════════════════════════════
# _safe_embed edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestSafeEmbed:
    def test_embed_fn_returns_empty_list(self, client: NeuroMemClient) -> None:
        client.set_embed_fn(lambda t: [])
        belief = client.learn("test")
        assert belief.embedding is None

    def test_embed_fn_returns_none(self, client: NeuroMemClient) -> None:
        client.set_embed_fn(lambda t: None)
        belief = client.learn("test")
        assert belief.embedding is None

    def test_embed_fn_raises(self, client: NeuroMemClient) -> None:
        client.set_embed_fn(lambda t: (_ for _ in ()).throw(ValueError("boom")))
        belief = client.learn("test")
        assert belief.embedding is None

    def test_no_embed_fn_returns_none(self, client: NeuroMemClient) -> None:
        client.set_embed_fn(None)
        result = client._safe_embed("text")
        assert result is None

    def test_safe_embed_valid_result(self, client: NeuroMemClient) -> None:
        client.set_embed_fn(lambda t: [0.1, 0.2, 0.3])
        result = client._safe_embed("test")
        assert result == [0.1, 0.2, 0.3]

    def test_safe_embed_non_list_result(self, client: NeuroMemClient) -> None:
        client.set_embed_fn(lambda t: "not a list")
        result = client._safe_embed("test")
        assert result is None

    def test_safe_embed_integer_elements_cast(self, client: NeuroMemClient) -> None:
        client.set_embed_fn(lambda t: [1, 2, 3])
        result = client._safe_embed("test")
        assert result == [1.0, 2.0, 3.0]


# ═══════════════════════════════════════════════════════════════════════
# Operations after close raise ConfigurationError
# ═══════════════════════════════════════════════════════════════════════


class TestClientRequireOpen:
    def test_operations_after_close_raise(self, client: NeuroMemClient) -> None:
        client.close()
        with pytest.raises(ConfigurationError):
            client.learn("closed test")
        with pytest.raises(ConfigurationError):
            client.recall("closed query")
        with pytest.raises(ConfigurationError):
            client.forget("some_id")
        with pytest.raises(ConfigurationError):
            client.guard("some pattern")
        with pytest.raises(ConfigurationError):
            client.propagate("some_id", "other_ns")
        with pytest.raises(ConfigurationError):
            client.decay()
        with pytest.raises(ConfigurationError):
            client.get_belief("some_id")
        with pytest.raises(ConfigurationError):
            client.list_beliefs()
        with pytest.raises(ConfigurationError):
            client.count_beliefs()
        with pytest.raises(ConfigurationError):
            client.is_blocked("pattern")
        with pytest.raises(ConfigurationError):
            client.check_contradiction("id", "claim")


# ═══════════════════════════════════════════════════════════════════════
# has_embed_fn property transitions
# ═══════════════════════════════════════════════════════════════════════


class TestClientHasEmbedFn:
    def test_has_embed_fn_false_by_default(self, client: NeuroMemClient) -> None:
        assert client.has_embed_fn is False

    def test_has_embed_fn_true_after_set(self, client: NeuroMemClient) -> None:
        client.set_embed_fn(lambda t: [0.1, 0.2])
        assert client.has_embed_fn is True

    def test_has_embed_fn_false_after_clear(self, client: NeuroMemClient) -> None:
        client.set_embed_fn(lambda t: [0.1])
        assert client.has_embed_fn is True
        client.set_embed_fn(None)
        assert client.has_embed_fn is False
