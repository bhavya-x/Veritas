"""Veritas ontology definition and schema bootstrapping.

This module is the single source of truth for the Veritas graph ontology. It
declares:

* the canonical **node labels** and their expected properties,
* the canonical **relationship types** and their (head, tail) domain/range,
* helpers to **bootstrap** constraints and indexes into a live Neo4j database.

Keeping the ontology in Python (in addition to the raw Cypher in
``cypher/schema_setup.cypher``) lets the ingestion and extraction layers
validate extracted triples against an allow-list, which keeps the graph clean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from veritas.config import get_settings
from veritas.db import Neo4jConnection

logger = logging.getLogger(__name__)

# Path to the raw Cypher schema script shipped alongside the package.
SCHEMA_CYPHER_PATH = Path(__file__).resolve().parent.parent / "cypher" / "schema_setup.cypher"


# ===========================================================================
# Ontology declaration
# ===========================================================================
@dataclass(frozen=True)
class NodeType:
    """Describes a node label and the properties it is expected to carry."""

    label: str
    description: str
    properties: tuple[str, ...]


@dataclass(frozen=True)
class RelationshipType:
    """Describes a directed relationship and its permitted endpoints."""

    type: str
    description: str
    head_labels: tuple[str, ...]
    tail_labels: tuple[str, ...]


# --- Common properties every domain node carries ---------------------------
# id           : deterministic unique identifier (slug of type + name)
# name         : human-readable canonical name
# created_at   : ISO-8601 timestamp set at MERGE time
# source_doc   : id of the Document the entity was first extracted from
# description  : optional free-text description (used by full-text index)
_COMMON_PROPS = ("id", "name", "created_at", "source_doc", "description")

NODE_TYPES: dict[str, NodeType] = {
    "Person": NodeType(
        "Person", "An individual employee, stakeholder, or author.", _COMMON_PROPS + ("role", "email")
    ),
    "Project": NodeType(
        "Project", "A named initiative or product effort.", _COMMON_PROPS + ("status",)
    ),
    "Technology": NodeType(
        "Technology", "A tool, framework, language, or platform.", _COMMON_PROPS + ("category",)
    ),
    "Team": NodeType(
        "Team", "An organisational unit or working group.", _COMMON_PROPS
    ),
    "Decision": NodeType(
        "Decision", "A recorded choice affecting projects or architecture.", _COMMON_PROPS + ("decided_at",)
    ),
    "Issue": NodeType(
        "Issue", "A bug, blocker, or tracked work item.", _COMMON_PROPS + ("severity", "status")
    ),
    "Document": NodeType(
        "Document",
        "A source document (email, PDF, markdown, ticket).",
        ("id", "name", "created_at", "source", "doc_type", "embedding"),
    ),
    "Chunk": NodeType(
        "Chunk",
        "A semantic text chunk of a Document; carries the vector embedding.",
        ("id", "text", "chunk_index", "source_doc", "embedding", "created_at"),
    ),
}

RELATIONSHIP_TYPES: dict[str, RelationshipType] = {
    "WORKS_ON": RelationshipType(
        "WORKS_ON", "A person contributes to a project.", ("Person",), ("Project",)
    ),
    "HAS_SKILL": RelationshipType(
        "HAS_SKILL", "A person is skilled in a technology.", ("Person",), ("Technology",)
    ),
    "MEMBER_OF": RelationshipType(
        "MEMBER_OF", "A person belongs to a team.", ("Person",), ("Team",)
    ),
    "OWNS": RelationshipType(
        "OWNS", "A team owns a project.", ("Team",), ("Project",)
    ),
    "USES": RelationshipType(
        "USES", "A project uses a technology.", ("Project",), ("Technology",)
    ),
    "BLOCKS": RelationshipType(
        "BLOCKS", "An issue blocks a project.", ("Issue",), ("Project",)
    ),
    "AFFECTS": RelationshipType(
        "AFFECTS", "A decision affects a project.", ("Decision",), ("Project",)
    ),
    "MENTIONS": RelationshipType(
        "MENTIONS",
        "A document mentions a domain entity.",
        ("Document",),
        ("Person", "Project", "Technology", "Team", "Decision", "Issue"),
    ),
    "EXTRACTED_FROM": RelationshipType(
        "EXTRACTED_FROM", "A chunk was extracted from a document.", ("Chunk",), ("Document",)
    ),
}

# Labels that represent extractable domain entities (excludes Document/Chunk).
DOMAIN_ENTITY_LABELS: tuple[str, ...] = (
    "Person",
    "Project",
    "Technology",
    "Team",
    "Decision",
    "Issue",
)


@dataclass
class SchemaBootstrapResult:
    """Summary of statements executed during schema bootstrap."""

    statements_run: int = 0
    statements: list[str] = field(default_factory=list)


# ===========================================================================
# Bootstrapping helpers
# ===========================================================================
def load_schema_statements() -> list[str]:
    """Read ``schema_setup.cypher`` and split it into executable statements.

    Comments and blank lines are stripped, and statements are split on the
    semicolon terminator. This lets us run the script through the driver, which
    (unlike ``cypher-shell``) executes one statement per call.
    """
    if not SCHEMA_CYPHER_PATH.exists():
        raise FileNotFoundError(f"Schema Cypher not found at {SCHEMA_CYPHER_PATH}")

    raw = SCHEMA_CYPHER_PATH.read_text(encoding="utf-8")

    # Strip full-line comments while preserving statement structure.
    cleaned_lines = [
        line for line in raw.splitlines() if not line.strip().startswith("//")
    ]
    cleaned = "\n".join(cleaned_lines)

    statements = [stmt.strip() for stmt in cleaned.split(";") if stmt.strip()]
    return statements


def bootstrap_schema(connection: Neo4jConnection | None = None) -> SchemaBootstrapResult:
    """Apply all constraints and indexes to the connected Neo4j database.

    Parameters
    ----------
    connection:
        An optional live :class:`~veritas.db.Neo4jConnection`. When omitted, a
        new connection is opened (and closed) using global settings.

    Returns
    -------
    SchemaBootstrapResult
        A summary of the statements executed.
    """
    settings = get_settings()
    statements = load_schema_statements()
    result = SchemaBootstrapResult()

    owns_connection = connection is None
    conn = connection or Neo4jConnection()

    try:
        conn.connect()
        for stmt in statements:
            # The vector index statements reference $embedding_dimension.
            params = (
                {"embedding_dimension": settings.embedding_dimension}
                if "$embedding_dimension" in stmt
                else None
            )
            conn.execute_write(stmt, params)
            result.statements_run += 1
            result.statements.append(stmt.split("\n")[0][:80])
        logger.info("Schema bootstrap complete: %d statements applied.", result.statements_run)
    finally:
        if owns_connection:
            conn.close()

    return result


def describe_ontology() -> str:
    """Return a human-readable summary of the ontology (nodes + relationships).

    Useful both for documentation and for injecting an ontology hint into LLM
    extraction prompts.
    """
    lines = ["NODE TYPES:"]
    for node in NODE_TYPES.values():
        lines.append(f"  ({node.label}) — {node.description}")
    lines.append("\nRELATIONSHIP TYPES:")
    for rel in RELATIONSHIP_TYPES.values():
        heads = "|".join(rel.head_labels)
        tails = "|".join(rel.tail_labels)
        lines.append(f"  (:{heads})-[:{rel.type}]->(:{tails}) — {rel.description}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual bootstrap entrypoint
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(describe_ontology())
    print("\nBootstrapping schema into Neo4j...")
    summary = bootstrap_schema()
    print(f"Applied {summary.statements_run} schema statements.")
