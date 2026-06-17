"""Ollama embedding provider for NeuroMem.

Uses the Ollama REST API (http://localhost:11434 by default) — no extra
Python dependencies needed.  Requires a locally-running Ollama server
with the desired model pulled::

    ollama pull nomic-embed-text
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error

from neuromem.providers.base import BaseEmbedProvider


class OllamaEmbedProvider(BaseEmbedProvider):
    """Embedding provider that uses a locally-running Ollama server.

    Communicates via the ``/api/embeddings`` endpoint using only
    Python's stdlib ``urllib`` — no extra dependencies required.

    Parameters
    ----------
    model:
        Ollama model tag.  Defaults to ``\"nomic-embed-text\"`` (768-dim,
        fast, suitable for most retrieval tasks).
    base_url:
        Ollama server base URL.  Defaults to ``\"http://localhost:11434\"``.
    timeout:
        HTTP request timeout in seconds.

    Example
    -------
    ::

        from neuromem.providers import OllamaEmbedProvider
        from neuromem import NeuroMemClient

        # Requires: ollama pull nomic-embed-text
        provider = OllamaEmbedProvider()
        with NeuroMemClient.create(\"./data\", embed_fn=provider) as client:
            client.learn(\"The sky is blue\", confidence=0.9)
    """

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._endpoint = f"{self._base_url}/api/embeddings"

    def embed(self, text: str) -> list[float]:
        """Embed *text* via the Ollama ``/api/embeddings`` endpoint."""
        payload = json.dumps({"model": self._model, "prompt": text}).encode()
        request = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                data: dict = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Failed to reach Ollama server at {self._endpoint}: {exc}. "
                "Make sure Ollama is running (ollama serve)."
            ) from exc

        embedding = data.get("embedding")
        if not embedding:
            raise ValueError(
                f"Ollama returned an empty embedding for model={self._model!r}. "
                "Check that the model is pulled: ollama pull " + self._model
            )
        return [float(x) for x in embedding]
