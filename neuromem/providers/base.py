"""Abstract base class for all NeuroMem embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedProvider(ABC):
    """Contract for all built-in NeuroMem embedding providers.

    Every provider is also *callable* so it can be passed directly as
    an ``EmbedFn`` wherever NeuroMem expects one::

        from neuromem.providers import OpenAIEmbedProvider
        from neuromem import NeuroMemClient

        provider = OpenAIEmbedProvider()
        with NeuroMemClient.create(\"./data\", embed_fn=provider) as client:
            client.learn(\"Paris is the capital of France\")
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a dense float vector for *text*.

        Parameters
        ----------
        text:
            The input string to embed.

        Returns
        -------
        list[float]
            A non-empty list of floats representing the embedding.
        """

    def __call__(self, text: str) -> list[float]:
        """Allow providers to be used directly as ``EmbedFn`` callables."""
        return self.embed(text)
