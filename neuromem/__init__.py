"""NeuroMem — Unified hybrid memory engine for AI agents.

Top-level convenience imports.  Most users only need :class:`NeuroMemClient`::

    from neuromem import NeuroMemClient

    with NeuroMemClient.create("./my_data") as client:
        client.learn("The sky is blue", confidence=0.9)
        results = client.recall("sky colour")
"""

from neuromem.client import NeuroMemClient, RecallResult, SharedMemoryRecord
from neuromem.core.engine import EngineConfig, FusedResult, NeuroMemEngine
from neuromem.core.exceptions import NeuroMemError
from neuromem.core.models import (
    BeliefNode,
    BeliefStatus,
    ContradictionEvent,
    ContradictionResolution,
    NegativeMemory,
    NegativeMemorySeverity,
    PropagationRecord,
    PropagationStatus,
    ReasoningStep,
    ReasoningTrace,
    TraceStepType,
)
from neuromem.storage.base import (
    BaseGraphEngine,
    BaseVectorEngine,
)

__version__ = "0.1.0"

__all__: list[str] = [
    # Version
    "__version__",
    # Client (primary entry point)
    "NeuroMemClient",
    "RecallResult",
    "SharedMemoryRecord",
    # Engine
    "NeuroMemEngine",
    "EngineConfig",
    "FusedResult",
    # Models
    "BeliefNode",
    "BeliefStatus",
    "ContradictionEvent",
    "ContradictionResolution",
    "NegativeMemory",
    "NegativeMemorySeverity",
    "PropagationRecord",
    "PropagationStatus",
    "ReasoningStep",
    "ReasoningTrace",
    "TraceStepType",
    # Storage interfaces
    "BaseGraphEngine",
    "BaseVectorEngine",
    # Exceptions
    "NeuroMemError",
]
