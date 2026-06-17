"""Tests for Belief Reinforcement on learn() — Feature 1."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from neuromem.core.engine import EngineConfig, NeuroMemEngine
from neuromem.core.models import BeliefNode, BeliefStatus


# ── Fixtures ─────────────────────────────────────────────────────────


def make_engine(tmp_path, reinforce: bool = True) -> NeuroMemEngine:
    """Create a real engine backed by real Kuzu + ChromaDB in tmp_path."""
    from neuromem.storage.kuzu_graph import KuzuGraphEngine
    from neuromem.storage.chroma_vector import ChromaVectorEngine

    graph = KuzuGraphEngine(str(tmp_path / "graph"))
    vector = ChromaVectorEngine(str(tmp_path / "vectors"))
    graph.initialize()
    vector.initialize()
    cfg = EngineConfig(reinforce_on_duplicate=reinforce)
    return NeuroMemEngine(graph, vector, config=cfg, namespace="test")


# ── EngineConfig tests ────────────────────────────────────────────────


class TestEngineConfig:
    def test_reinforce_on_duplicate_default_true(self):
        cfg = EngineConfig()
        assert cfg.reinforce_on_duplicate is True

    def test_reinforce_on_duplicate_can_be_disabled(self):
        cfg = EngineConfig(reinforce_on_duplicate=False)
        assert cfg.reinforce_on_duplicate is False


# ── Integration: learn() reinforcement ───────────────────────────────


@pytest.mark.integration
class TestBeliefReinforcement:
    def test_duplicate_learn_reinforces_not_duplicates(self, tmp_path):
        engine = make_engine(tmp_path, reinforce=True)
        b1 = engine.learn("The sky is blue", confidence=0.5)
        b2 = engine.learn("The sky is blue", confidence=0.5)

        # Same belief returned
        assert b1.id == b2.id
        # evidence_count incremented
        assert b2.evidence_count == 2

    def test_duplicate_learn_increases_confidence(self, tmp_path):
        engine = make_engine(tmp_path, reinforce=True)
        b1 = engine.learn("The sky is blue", confidence=0.5)
        initial_conf = b1.confidence
        b2 = engine.learn("The sky is blue", confidence=0.9)

        assert b2.confidence > initial_conf

    def test_triple_learn_accumulates_evidence(self, tmp_path):
        engine = make_engine(tmp_path, reinforce=True)
        engine.learn("Paris is in France", confidence=0.5)
        engine.learn("Paris is in France", confidence=0.5)
        b = engine.learn("Paris is in France", confidence=0.5)
        assert b.evidence_count == 3

    def test_no_reinforce_creates_separate_beliefs(self, tmp_path):
        engine = make_engine(tmp_path, reinforce=False)
        b1 = engine.learn("The sky is blue", confidence=0.5)
        b2 = engine.learn("The sky is blue", confidence=0.5)
        # Different IDs — two separate nodes
        assert b1.id != b2.id

    def test_different_claims_always_create_new_beliefs(self, tmp_path):
        engine = make_engine(tmp_path, reinforce=True)
        b1 = engine.learn("The sky is blue", confidence=0.5)
        b2 = engine.learn("The grass is green", confidence=0.5)
        assert b1.id != b2.id

    def test_reinforced_belief_stays_active(self, tmp_path):
        engine = make_engine(tmp_path, reinforce=True)
        b = engine.learn("Water is wet", confidence=0.6)
        b2 = engine.learn("Water is wet", confidence=0.6)
        assert b2.status == BeliefStatus.ACTIVE
