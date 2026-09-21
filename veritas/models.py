"""Shared data models used across the ingestion and retrieval layers.

These lightweight dataclasses give the pipeline a typed, framework-agnostic
representation of documents, chunks, extracted entities, and triples. Keeping
them separate from any LLM/DB library keeps the core contracts stable even if
the extraction backend changes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def make_entity_id(label: str, name: str) -> str:
    """Build a deterministic, collision-resistant id for a domain entity.

    The id is a slug of ``label`` and ``name`` so that the same entity
    extracted from different documents MERGEs onto a single node.

    Examples
    --------
    >>> make_entity_id("Person", "Ada Lovelace")
    'person::ada-lovelace'
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return f"{label.strip().lower()}::{slug}"


def make_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    """Build a stable id for a chunk from its document, index, and content hash."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{document_id}::chunk-{chunk_index}::{digest}"


@dataclass
class SourceDocument:
    """A raw source document prior to chunking."""

    id: str
    name: str
    content: str
    source: str
    doc_type: str  # e.g. "email", "pdf", "markdown", "ticket", "txt"
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class Chunk:
    """A semantic text chunk belonging to a :class:`SourceDocument`."""

    id: str
    document_id: str
    chunk_index: int
    text: str
    embedding: list[float] | None = None
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class Entity:
    """A domain entity extracted from text (Person, Project, ...)."""

    label: str
    name: str
    properties: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Deterministic id derived from label + name."""
        return make_entity_id(self.label, self.name)


@dataclass
class Triple:
    """A directed (head)-[type]->(tail) relationship between two entities."""

    head: Entity
    relationship: str
    tail: Entity

    def as_display(self) -> str:
        """Return a readable ``Head -[REL]-> Tail`` string for logging/output."""
        return f"{self.head.name} -[{self.relationship}]-> {self.tail.name}"


@dataclass
class ExtractionResult:
    """The structured output of running extraction over a single chunk."""

    chunk: Chunk
    entities: list[Entity] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)
