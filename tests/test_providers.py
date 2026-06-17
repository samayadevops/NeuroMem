"""Tests for built-in embedding providers — Feature 3."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from neuromem.providers.base import BaseEmbedProvider


# ── Base provider contract ────────────────────────────────────────────


class TestBaseEmbedProvider:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseEmbedProvider()  # type: ignore[abstract]

    def test_concrete_subclass_is_callable(self):
        class MyProvider(BaseEmbedProvider):
            def embed(self, text: str) -> list[float]:
                return [1.0, 2.0, 3.0]

        p = MyProvider()
        result = p("hello")
        assert result == [1.0, 2.0, 3.0]


# ── OpenAIEmbedProvider ───────────────────────────────────────────────


class TestOpenAIEmbedProvider:
    def test_raises_importerror_without_openai(self):
        import sys
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(ImportError, match="pip install openai"):
                from neuromem.providers.openai import OpenAIEmbedProvider  # noqa: PLC0415
                OpenAIEmbedProvider()

    def test_embed_calls_openai_api(self):
        fake_embedding = [0.1, 0.2, 0.3]
        mock_data = MagicMock()
        mock_data.data = [MagicMock(embedding=fake_embedding)]

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.embeddings.create.return_value = mock_data

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from neuromem.providers.openai import OpenAIEmbedProvider  # noqa: PLC0415
            provider = OpenAIEmbedProvider(model="text-embedding-3-small")
            result = provider.embed("hello world")

        assert result == fake_embedding

    def test_dimensions_passed_to_api(self):
        fake_embedding = [0.1] * 512
        mock_data = MagicMock()
        mock_data.data = [MagicMock(embedding=fake_embedding)]

        mock_openai = MagicMock()
        mock_client = mock_openai.OpenAI.return_value
        mock_client.embeddings.create.return_value = mock_data

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from neuromem.providers.openai import OpenAIEmbedProvider  # noqa: PLC0415
            provider = OpenAIEmbedProvider(dimensions=512)
            result = provider.embed("test")

        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs.get("dimensions") == 512
        assert len(result) == 512

    def test_is_callable_as_embed_fn(self):
        fake_embedding = [0.5, 0.6]
        mock_data = MagicMock()
        mock_data.data = [MagicMock(embedding=fake_embedding)]

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.embeddings.create.return_value = mock_data

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from neuromem.providers.openai import OpenAIEmbedProvider  # noqa: PLC0415
            provider = OpenAIEmbedProvider()
            # Must be usable as a plain callable (EmbedFn signature)
            result = provider("hello")

        assert result == fake_embedding


# ── OllamaEmbedProvider ───────────────────────────────────────────────


class TestOllamaEmbedProvider:
    def test_embed_sends_correct_request(self):
        fake_embedding = [0.1, 0.2, 0.3]
        fake_response_body = json.dumps({"embedding": fake_embedding}).encode()

        mock_response = MagicMock()
        mock_response.read.return_value = fake_response_body
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            from neuromem.providers.ollama import OllamaEmbedProvider  # noqa: PLC0415
            provider = OllamaEmbedProvider(model="nomic-embed-text")
            result = provider.embed("hello")

        assert result == fake_embedding

    def test_connection_error_raises_friendly_message(self):
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            from neuromem.providers.ollama import OllamaEmbedProvider  # noqa: PLC0415
            provider = OllamaEmbedProvider()
            with pytest.raises(ConnectionError, match="Ollama"):
                provider.embed("hello")

    def test_empty_embedding_raises_valueerror(self):
        fake_response_body = json.dumps({"embedding": []}).encode()
        mock_response = MagicMock()
        mock_response.read.return_value = fake_response_body
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            from neuromem.providers.ollama import OllamaEmbedProvider  # noqa: PLC0415
            provider = OllamaEmbedProvider()
            with pytest.raises(ValueError, match="empty embedding"):
                provider.embed("hello")

    def test_is_callable(self):
        fake_embedding = [0.9]
        fake_response_body = json.dumps({"embedding": fake_embedding}).encode()
        mock_response = MagicMock()
        mock_response.read.return_value = fake_response_body
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            from neuromem.providers.ollama import OllamaEmbedProvider  # noqa: PLC0415
            provider = OllamaEmbedProvider()
            assert provider("hi") == fake_embedding


# ── SentenceTransformerEmbedProvider ─────────────────────────────────


class TestSentenceTransformerEmbedProvider:
    def test_raises_importerror_without_package(self):
        import sys
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                from neuromem.providers.sentence_transformers import SentenceTransformerEmbedProvider  # noqa: PLC0415
                SentenceTransformerEmbedProvider()

    def test_embed_returns_list_of_floats(self):
        import numpy as np

        fake_vector = np.array([0.1, 0.2, 0.3])
        mock_st_class = MagicMock()
        mock_st_class.return_value.encode.return_value = fake_vector
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer = mock_st_class

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            from neuromem.providers.sentence_transformers import SentenceTransformerEmbedProvider  # noqa: PLC0415
            provider = SentenceTransformerEmbedProvider()
            result = provider.embed("hello")

        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_is_subclass_of_base(self):
        from neuromem.providers.sentence_transformers import SentenceTransformerEmbedProvider  # noqa: PLC0415
        assert issubclass(SentenceTransformerEmbedProvider, BaseEmbedProvider)


# ── providers package __init__ ────────────────────────────────────────


class TestProvidersPackage:
    def test_all_providers_importable(self):
        from neuromem.providers import (
            BaseEmbedProvider,
            OpenAIEmbedProvider,
            OllamaEmbedProvider,
            SentenceTransformerEmbedProvider,
        )
        assert BaseEmbedProvider is not None
        assert OpenAIEmbedProvider is not None
        assert OllamaEmbedProvider is not None
        assert SentenceTransformerEmbedProvider is not None

    def test_providers_in_all(self):
        import neuromem.providers as pkg
        for name in ["BaseEmbedProvider", "OpenAIEmbedProvider",
                     "OllamaEmbedProvider", "SentenceTransformerEmbedProvider"]:
            assert name in pkg.__all__
