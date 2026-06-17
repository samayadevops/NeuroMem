"""Built-in embedding providers for NeuroMem.

All providers implement :class:`BaseEmbedProvider` and are directly
callable as ``EmbedFn`` instances::

    from neuromem.providers import OpenAIEmbedProvider
    from neuromem import NeuroMemClient

    with NeuroMemClient.create(\"./data\", embed_fn=OpenAIEmbedProvider()) as client:
        client.learn(\"The sky is blue\")

Available providers
-------------------
:class:`OpenAIEmbedProvider`
    Uses the OpenAI Embeddings API.  Requires ``pip install openai``.

:class:`OllamaEmbedProvider`
    Uses a locally-running Ollama server.  No extra dependencies needed
    (uses stdlib ``urllib``).  Requires ``ollama serve`` to be running.

:class:`SentenceTransformerEmbedProvider`
    Runs fully offline using HuggingFace sentence-transformers.
    Requires ``pip install sentence-transformers``.
"""

from __future__ import annotations

from neuromem.providers.base import BaseEmbedProvider
from neuromem.providers.ollama import OllamaEmbedProvider
from neuromem.providers.openai import OpenAIEmbedProvider
from neuromem.providers.sentence_transformers import SentenceTransformerEmbedProvider

__all__: list[str] = [
    "BaseEmbedProvider",
    "OpenAIEmbedProvider",
    "OllamaEmbedProvider",
    "SentenceTransformerEmbedProvider",
]
