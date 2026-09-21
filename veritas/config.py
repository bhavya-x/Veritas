"""Centralised, typed configuration for the Veritas platform.

Configuration is loaded from environment variables (optionally sourced from a
local ``.env`` file). Using a single typed settings object keeps every module
free of scattered ``os.getenv`` calls and gives us validation for free.

Example
-------
>>> from veritas.config import get_settings
>>> settings = get_settings()
>>> settings.neo4j.uri
'bolt://localhost:7687'
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractionBackend(str, Enum):
    """Supported entity/relationship extraction engines."""

    OPENAI = "openai"
    SPACY = "spacy"


class EmbeddingBackend(str, Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class Neo4jSettings(BaseSettings):
    """Connection settings for the Neo4j graph database."""

    model_config = SettingsConfigDict(env_prefix="NEO4J_", extra="ignore")

    uri: str = Field(default="bolt://localhost:7687")
    username: str = Field(default="neo4j")
    password: SecretStr = Field(default=SecretStr("please_change_me"))
    database: str = Field(default="neo4j")


class Settings(BaseSettings):
    """Top-level application settings.

    All values can be overridden via environment variables. A local ``.env``
    file is loaded automatically if present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Nested database settings ---
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)

    # --- Extraction (LLM / NLP) ---
    extraction_backend: ExtractionBackend = Field(
        default=ExtractionBackend.OPENAI, alias="VERITAS_EXTRACTION_BACKEND"
    )
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_llm_model: str = Field(default="gpt-4o-mini", alias="OPENAI_LLM_MODEL")

    # --- Embeddings ---
    embedding_backend: EmbeddingBackend = Field(
        default=EmbeddingBackend.SENTENCE_TRANSFORMERS,
        alias="VERITAS_EMBEDDING_BACKEND",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    sentence_transformers_model: str = Field(
        default="all-MiniLM-L6-v2", alias="SENTENCE_TRANSFORMERS_MODEL"
    )
    embedding_dimension: int = Field(default=384, alias="VERITAS_EMBEDDING_DIMENSION")

    # --- Chunking ---
    chunk_size: int = Field(default=800, alias="VERITAS_CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="VERITAS_CHUNK_OVERLAP")

    def require_openai_key(self) -> str:
        """Return the OpenAI API key or raise a clear error if it is missing."""
        if self.openai_api_key is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file or export it "
                "before using the OpenAI extraction/embedding backend."
            )
        return self.openai_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, singleton :class:`Settings` instance.

    The result is cached so that the ``.env`` file is parsed only once per
    process, and every module shares the same configuration object.
    """

    return Settings()
