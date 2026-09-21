"""Embedding providers for Veritas.

A tiny abstraction over the two supported embedding backends so the rest of the
codebase depends on a single :class:`EmbeddingProvider` interface rather than a
specific SDK. The concrete backend is chosen via configuration.

* ``sentence_transformers`` — local, no API key, good for development.
* ``openai``                — hosted, higher quality, needs ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

from veritas.config import EmbeddingBackend, Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract embedding provider returning fixed-dimension float vectors."""

    #: Output dimension of this provider's vectors.
    dimension: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text (convenience wrapper around :meth:`embed_texts`)."""
        return self.embed_texts([text])[0]


class SentenceTransformerProvider(EmbeddingProvider):
    """Local embeddings via the ``sentence-transformers`` library."""

    def __init__(self, model_name: str, dimension: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required. Install with "
                "`pip install sentence-transformers`."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return [vec.tolist() for vec in vectors]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Hosted embeddings via the OpenAI API."""

    def __init__(self, model_name: str, dimension: int, api_key: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model_name
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


@lru_cache(maxsize=1)
def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Return the configured embedding provider (cached per process)."""
    settings = settings or get_settings()
    if settings.embedding_backend == EmbeddingBackend.OPENAI:
        logger.info("Using OpenAI embeddings (%s).", settings.openai_embedding_model)
        return OpenAIEmbeddingProvider(
            settings.openai_embedding_model,
            settings.embedding_dimension,
            settings.require_openai_key(),
        )
    logger.info(
        "Using SentenceTransformer embeddings (%s).",
        settings.sentence_transformers_model,
    )
    return SentenceTransformerProvider(
        settings.sentence_transformers_model, settings.embedding_dimension
    )
