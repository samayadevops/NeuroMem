"""Integration tests for the NeuroMemClient user-facing API.

These validate the high-level ``.learn()``, ``.recall()``,
``.forget()``, ``.guard()``, ``.propagate()``, and ``.decay()``
methods, including auto-embedding via an injected embedding function.
"""

from __future__ import annotations

import pytest

from neuromem.client import NeuroMemClient, RecallResult
from neuromem.core.exceptions import ConfigurationError
from neuromem.core.models import BeliefStatus, NegativeMemorySeverity


class TestClientLifecycle:
    """Tests for client creation and lifecycle."""

    def test_context_manager_closes(self, client) -> None:
        """Context manager __exit__ closes the client."""
        assert not client.is_closed
        client.close()
        assert client.is_closed

    def test_operations_after_close_raise(self, client) -> None:
        client.close()
        with pytest.raises(ConfigurationError):
            client.learn("x")

    def test_namespace_property(self, client) -> None:
        assert client.namespace == "test"

    def test_current_tick_property(self, client) -> None:
        assert client.current_tick == 0


class TestClientLearn:
    """Tests for client.learn()."""

    def test_learn_basic(self, client) -> None:
        belief = client.learn("The sky is blue", confidence=0.9)
        assert belief.claim == "The sky is blue"
        assert belief.confidence == 0.9

    def test_learn_with_explicit_embedding(self, client) -> None:
        belief = client.learn("test", confidence=0.7, embedding=[0.1, 0.2])
        assert belief.embedding is not None

    def test_learn_auto_embed(self, client_with_embeddings) -> None:
        """Auto-embedding triggers when embed_fn is set and no explicit embedding."""
        belief = client_with_embeddings.learn("auto embed me")
        assert belief.embedding is not None
        assert len(belief.embedding) == 8  # fake embed returns 8-dim

    def test_learn_auto_embed_disabled(self, client_with_embeddings) -> None:
        """auto_embed=False skips embedding."""
        belief = client_with_embeddings.learn("no embed", auto_embed=False)
        assert belief.embedding is None

    def test_learn_with_tags(self, client) -> None:
        belief = client.learn("tagged", tags=["a", "b"])
        assert "a" in belief.tags


class TestClientRecall:
    """Tests for client.recall()."""

    def test_recall_returns_recall_results(self, client) -> None:
        client.learn("recallable", confidence=0.9, embedding=[0.5, 0.5])
        results = client.recall(query_embedding=[0.5, 0.5])
        assert len(results) >= 1
        assert all(isinstance(r, RecallResult) for r in results)

    def test_recall_result_exposes_top_level_fields(self, client) -> None:
        client.learn("field test", confidence=0.8, embedding=[0.1, 0.2])
        results = client.recall(query_embedding=[0.1, 0.2])
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r.claim, str)
        assert isinstance(r.confidence, float)
        assert isinstance(r.fused_score, float)
        assert r.id.startswith("belief_")

    def test_recall_result_to_dict(self, client) -> None:
        client.learn("serialise me", confidence=0.7, embedding=[0.3, 0.4])
        results = client.recall(query_embedding=[0.3, 0.4])
        d = results[0].to_dict()
        assert "claim" in d
        assert "confidence" in d
        assert "fused_score" in d
        assert "id" in d

    def test_recall_auto_embed(self, client_with_embeddings) -> None:
        client_with_embeddings.learn("auto recall target", confidence=0.9)
        results = client_with_embeddings.recall("auto recall target")
        assert len(results) >= 1


class TestClientForget:
    """Tests for client.forget()."""

    def test_forget_deprecates_belief(self, client) -> None:
        belief = client.learn("forgettable", confidence=0.9, embedding=[0.1, 0.0])
        result = client.forget(belief.id)
        assert result is True
        reloaded = client.get_belief(belief.id)
        assert reloaded is not None
        assert reloaded.status == BeliefStatus.DEPRECATED
        assert reloaded.confidence == 0.0

    def test_forget_nonexistent_returns_false(self, client) -> None:
        assert client.forget("nonexistent_id") is False


class TestClientGuard:
    """Tests for client.guard() negative memory."""

    def test_guard_records_pattern(self, client) -> None:
        neg = client.guard("never_do_this")
        assert neg.pattern == "never_do_this"

    def test_guard_with_string_severity(self, client) -> None:
        neg = client.guard("bad", severity="error")
        assert neg.severity == NegativeMemorySeverity.ERROR

    def test_is_blocked_after_threshold(self, client) -> None:
        client.guard("blocked_pattern", block_threshold=2)
        assert not client.is_blocked("blocked_pattern")
        client.guard("blocked_pattern")
        assert client.is_blocked("blocked_pattern")


class TestClientPropagate:
    """Tests for client.propagate()."""

    def test_propagate_delivers(self, client) -> None:
        belief = client.learn("propagate me", confidence=0.9, embedding=[0.5, 0.5])
        record = client.propagate(belief.id, "other_agent")
        assert record.status.value == "delivered"

    def test_propagated_belief_in_target_namespace(self, client) -> None:
        belief = client.learn("cross ns", confidence=0.8, embedding=[0.6, 0.4])
        client.propagate(belief.id, "target_agent")
        target_beliefs = client.list_beliefs(namespace="target_agent")
        assert len(target_beliefs) >= 1


class TestClientDecay:
    """Tests for client.decay()."""

    def test_decay_advances_tick(self, client) -> None:
        client.decay(advance_ticks=3)
        assert client.current_tick == 3

    def test_decay_returns_decayed_count(self, client) -> None:
        client.learn("decaying", confidence=0.2, gamma=0.1, embedding=[0.1, 0.0])
        decayed = client.decay(advance_ticks=10)
        assert decayed >= 1


class TestClientInspection:
    """Tests for inspection helpers."""

    def test_get_belief_existing(self, client) -> None:
        belief = client.learn("inspect me", confidence=0.8, embedding=[0.2, 0.3])
        fetched = client.get_belief(belief.id)
        assert fetched is not None
        assert fetched.claim == "inspect me"

    def test_get_belief_nonexistent(self, client) -> None:
        assert client.get_belief("nonexistent") is None

    def test_list_beliefs(self, client) -> None:
        client.learn("list1", confidence=0.7, embedding=[0.1, 0.0])
        client.learn("list2", confidence=0.6, embedding=[0.2, 0.0])
        beliefs = client.list_beliefs()
        assert len(beliefs) >= 2

    def test_list_beliefs_filtered_by_status(self, client) -> None:
        b1 = client.learn("active one", confidence=0.9, embedding=[0.3, 0.0])
        client.forget(b1.id)
        active = client.list_beliefs(status=BeliefStatus.ACTIVE)
        deprecated = client.list_beliefs(status=BeliefStatus.DEPRECATED)
        assert all(b.status == BeliefStatus.ACTIVE for b in active)
        assert all(b.status == BeliefStatus.DEPRECATED for b in deprecated)

    def test_count_beliefs(self, client) -> None:
        client.learn("count1", confidence=0.7, embedding=[0.4, 0.0])
        client.learn("count2", confidence=0.6, embedding=[0.5, 0.0])
        total = client.count_beliefs()
        assert total >= 2

    def test_count_beliefs_active_only(self, client) -> None:
        client.learn("active count", confidence=0.7, embedding=[0.6, 0.0])
        active = client.count_beliefs(active_only=True)
        assert active >= 1


class TestClientEmbedFunction:
    """Tests for embedding function management."""

    def test_set_embed_fn(self, client) -> None:
        assert not client.has_embed_fn
        client.set_embed_fn(lambda text: [0.1, 0.2, 0.3])
        assert client.has_embed_fn

    def test_clear_embed_fn(self, client) -> None:
        client.set_embed_fn(lambda text: [0.1])
        client.set_embed_fn(None)
        assert not client.has_embed_fn
