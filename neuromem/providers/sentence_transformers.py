"""sentence-transformers embedding provider for NeuroMem.

Runs entirely locally (no API key needed) and works offline.
Requires the ``sentence-transformers`` package::

    pip install sentence-transformers
"""

from __future__ import annotations

from neuromem.providers.base import BaseEmbedProvider


class SentenceTransformerEmbedProvider(BaseEmbedProvider):
    """Embedding provider using ``sentence-transformers`` (local, offline).

    Downloads the model from HuggingFace Hub on first use and caches it.
    No API key or network access is required after the initial download.

    Parameters
    ----------
    model:
        HuggingFace model name.  Defaults to
        ``\"all-MiniLM-L6-v2\"`` (384-dim, fast, good quality).
    device:
        PyTorch device string.  Defaults to ``\"cpu\"``.  Use ``\"cuda\"``
        or ``\"mps\"`` for GPU acceleration.
    normalize:
        If ``True`` (default), normalise embeddings to unit length before
        returning them.  Recommended for cosine-similarity search.

    Example
    -------
    ::

        from neuromem.providers import SentenceTransformerEmbedProvider
        from neuromem import NeuroMemClient

        provider = SentenceTransformerEmbedProvider()
        with NeuroMemClient.create(\"./data\", embed_fn=provider) as client:
            client.learn(\"The sky is blue\", confidence=0.9)
            results = client.recall(\"sky colour\")
    """

    def __init__(
        self,
        *,
        model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped, import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'sentence-transformers' package is required. "
                "Install it with:  pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model, device=device)
        self._normalize = normalize

    def embed(self, text: str) -> list[float]:
        """Embed *text* using the local sentence-transformers model."""
        vector = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return vector.tolist()
