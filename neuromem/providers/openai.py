"""OpenAI embedding provider for NeuroMem."""

from __future__ import annotations

from neuromem.providers.base import BaseEmbedProvider


class OpenAIEmbedProvider(BaseEmbedProvider):
    """Embedding provider backed by the OpenAI Embeddings API.

    Requires the ``openai`` package::

        pip install openai

    Parameters
    ----------
    model:
        OpenAI embedding model name.  Defaults to
        ``\"text-embedding-3-small\"`` (1536-dim, cost-efficient).
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when ``None``.
    dimensions:
        Optional output-dimension override.  Supported by the ``v3``
        model family (e.g. ``text-embedding-3-small`` supports
        ``dimensions=512`` for a smaller representation).
    base_url:
        Optional custom base URL (useful for proxies / Azure OpenAI).

    Example
    -------
    ::

        from neuromem.providers import OpenAIEmbedProvider
        from neuromem import NeuroMemClient

        provider = OpenAIEmbedProvider(model=\"text-embedding-3-small\")
        with NeuroMemClient.create(\"./data\", embed_fn=provider) as client:
            client.learn(\"The Eiffel Tower is in Paris\", confidence=0.9)
            results = client.recall(\"French landmarks\")
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use OpenAIEmbedProvider. "
                "Install it with:  pip install openai"
            ) from exc

        kwargs: dict = {"api_key": api_key}
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self._dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        """Embed *text* using the OpenAI Embeddings API."""
        kwargs: dict = {"model": self._model, "input": text}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        response = self._client.embeddings.create(**kwargs)
        return response.data[0].embedding
