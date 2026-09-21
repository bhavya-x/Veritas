#!/usr/bin/env python3
"""Offline knowledge-graph exporter for the Veritas visual dashboard.

This script runs the *same conceptual pipeline* as ``run_phase1_ingestion.py``
(load -> chunk -> extract -> build graph) but entirely in memory, with a
deterministic, ontology-aware rule-based extractor. That means the visual demo:

* needs **no Neo4j server**,
* needs **no OpenAI API key**,
* runs anywhere Python 3.10+ is installed, with **zero third-party deps**.

It reads ``data/sample_enterprise_docs.json``, extracts entities and typed
relationships, chunks the documents, records provenance (which document/chunk
each fact came from), and writes a single ``demo/graph_export.json`` that the
HTML dashboard loads directly.

The extraction ruleset is intentionally transparent (dictionary + pattern
based) so the graph is stable and explainable for a presentation. In the full
pipeline this stage is handled by the LLM ``ExtractionEngine``; here we mirror
its output shape exactly.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "sample_enterprise_docs.json"
EXPORT_PATH = ROOT / "demo" / "graph_export.json"

# Chunking parameters (mirror the defaults in veritas/config.py).
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


# ===========================================================================
# Deterministic ontology-aware extraction ruleset
# ===========================================================================
# Each domain entity is declared once with its canonical name, label, aliases
# (surface forms that appear in the text), and optional display properties.
@dataclass(frozen=True)
class EntitySpec:
    name: str
    label: str
    aliases: tuple[str, ...]
    properties: dict[str, str] = field(default_factory=dict)


ENTITY_SPECS: tuple[EntitySpec, ...] = (
    # --- People ---
    EntitySpec("Priya Nair", "Person", ("Priya Nair", "Priya"),
               {"role": "Principal Engineer"}),
    EntitySpec("Marcus Feld", "Person", ("Marcus Feld", "Marcus"),
               {"role": "Ingestion Service Owner"}),
    EntitySpec("Diana Okoro", "Person", ("Diana Okoro", "Diana"),
               {"role": "Retrieval Engineer"}),
    # --- Projects ---
    EntitySpec("Project Atlas", "Project", ("Project Atlas", "Atlas"),
               {"status": "Beta"}),
    # --- Technologies ---
    EntitySpec("Neo4j", "Technology", ("Neo4j",), {"category": "Graph Database"}),
    EntitySpec("LangChain", "Technology", ("LangChain",), {"category": "LLM Framework"}),
    EntitySpec("SentenceTransformers", "Technology", ("SentenceTransformers",),
               {"category": "Embeddings"}),
    EntitySpec("Python", "Technology", ("Python",), {"category": "Language"}),
    # --- Teams ---
    EntitySpec("Platform Team", "Team", ("Platform Team",)),
    EntitySpec("Data Platform Team", "Team", ("Data Platform Team",)),
    # --- Decisions ---
    EntitySpec("Decision D-114", "Decision", ("D-114",),
               {"summary": "Adopt Neo4j as system of record"}),
    EntitySpec("Decision D-118", "Decision", ("D-118",),
               {"summary": "Use SentenceTransformers for embeddings"}),
    # --- Issues ---
    EntitySpec("ATLAS-231", "Issue", ("ATLAS-231",),
               {"severity": "High", "status": "Open"}),
)

# Typed relationships. Each rule fires when BOTH endpoints are found in the same
# document, producing an ontology-valid triple with recorded provenance.
# (head_name, relationship, tail_name)
RELATIONSHIP_RULES: tuple[tuple[str, str, str], ...] = (
    # People -> Project
    ("Priya Nair", "WORKS_ON", "Project Atlas"),
    ("Marcus Feld", "WORKS_ON", "Project Atlas"),
    ("Diana Okoro", "WORKS_ON", "Project Atlas"),
    # People -> Team
    ("Priya Nair", "MEMBER_OF", "Platform Team"),
    ("Marcus Feld", "MEMBER_OF", "Platform Team"),
    # People -> Technology (skills)
    ("Diana Okoro", "HAS_SKILL", "SentenceTransformers"),
    ("Diana Okoro", "HAS_SKILL", "Python"),
    ("Marcus Feld", "HAS_SKILL", "LangChain"),
    ("Priya Nair", "HAS_SKILL", "Neo4j"),
    # Team -> Project
    ("Platform Team", "OWNS", "Project Atlas"),
    ("Data Platform Team", "OWNS", "Project Atlas"),
    # Project -> Technology
    ("Project Atlas", "USES", "Neo4j"),
    ("Project Atlas", "USES", "LangChain"),
    ("Project Atlas", "USES", "SentenceTransformers"),
    # Decisions -> Project
    ("Decision D-114", "AFFECTS", "Project Atlas"),
    ("Decision D-118", "AFFECTS", "Project Atlas"),
    # Issue -> Project
    ("ATLAS-231", "BLOCKS", "Project Atlas"),
)

# Colour + shape hints for the front-end legend, keyed by node label.
LABEL_STYLE = {
    "Person": {"color": "#4C9AFF", "shape": "dot"},
    "Project": {"color": "#F5A623", "shape": "diamond"},
    "Technology": {"color": "#36B37E", "shape": "square"},
    "Team": {"color": "#9F7AEA", "shape": "triangle"},
    "Decision": {"color": "#FF8B00", "shape": "star"},
    "Issue": {"color": "#FF5630", "shape": "hexagon"},
    "Document": {"color": "#8993A4", "shape": "database"},
}


def _entity_id(label: str, name: str) -> str:
    """Deterministic slug id (matches veritas.models.make_entity_id)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return f"{label.strip().lower()}::{slug}"


def _chunk_id(doc_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}::chunk-{index}::{digest}"


def _chunk_text(text: str) -> list[str]:
    """Sliding-window chunker (mirrors the pipeline fallback splitter)."""
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks = [text[i : i + CHUNK_SIZE].strip() for i in range(0, len(text), step)]
    return [c for c in chunks if c]


def _find_aliases(text: str, aliases: tuple[str, ...]) -> bool:
    """Return True if any alias appears in the text (word-boundary aware)."""
    for alias in aliases:
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text):
            return True
    return False


def build_export() -> dict:
    """Run the offline pipeline and return the export dictionary."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Sample dataset not found at {DATASET_PATH}")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    documents = dataset["documents"]

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    chunks: list[dict] = []
    edge_seen: set[tuple[str, str, str]] = set()

    def add_node(node_id: str, label: str, name: str, props: dict | None = None) -> None:
        if node_id not in nodes:
            style = LABEL_STYLE.get(label, {"color": "#8993A4", "shape": "dot"})
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "name": name,
                "properties": props or {},
                "color": style["color"],
                "shape": style["shape"],
                "source_docs": [],
            }

    def add_edge(head_id: str, rel: str, tail_id: str, source_doc: str) -> None:
        key = (head_id, rel, tail_id)
        if key in edge_seen:
            return
        edge_seen.add(key)
        edges.append(
            {"source": head_id, "relationship": rel, "target": tail_id, "source_doc": source_doc}
        )

    spec_by_name = {spec.name: spec for spec in ENTITY_SPECS}

    # --- Pass over each document ---
    for doc in documents:
        doc_id = doc["id"]
        add_node(doc_id, "Document", doc["name"],
                 {"source": doc["source"], "doc_type": doc["doc_type"]})

        content = doc["content"]

        # Chunk the document and record provenance.
        for idx, ctext in enumerate(_chunk_text(content)):
            chunks.append(
                {
                    "id": _chunk_id(doc_id, idx, ctext),
                    "document_id": doc_id,
                    "chunk_index": idx,
                    "text": ctext,
                }
            )

        # Which entities are mentioned in this document?
        present: set[str] = set()
        for spec in ENTITY_SPECS:
            if _find_aliases(content, spec.aliases):
                node_id = _entity_id(spec.label, spec.name)
                add_node(node_id, spec.label, spec.name, dict(spec.properties))
                nodes[node_id]["source_docs"].append(doc_id)
                # Document -[:MENTIONS]-> Entity provenance edge.
                add_edge(doc_id, "MENTIONS", node_id, doc_id)
                present.add(spec.name)

        # Fire relationship rules where both endpoints are present in this doc.
        for head_name, rel, tail_name in RELATIONSHIP_RULES:
            if head_name in present and tail_name in present:
                head_spec = spec_by_name[head_name]
                tail_spec = spec_by_name[tail_name]
                head_id = _entity_id(head_spec.label, head_name)
                tail_id = _entity_id(tail_spec.label, tail_name)
                add_edge(head_id, rel, tail_id, doc_id)

    # De-dupe per-node source doc lists.
    for node in nodes.values():
        node["source_docs"] = sorted(set(node["source_docs"]))

    domain_nodes = [n for n in nodes.values() if n["label"] != "Document"]
    domain_edges = [e for e in edges if e["relationship"] != "MENTIONS"]

    export = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset.get("dataset", "veritas_sample"),
        "legend": LABEL_STYLE,
        "documents": [
            {"id": d["id"], "name": d["name"], "source": d["source"],
             "doc_type": d["doc_type"], "content": d["content"]}
            for d in documents
        ],
        "nodes": list(nodes.values()),
        "edges": edges,
        "chunks": chunks,
        "stats": {
            "documents": len(documents),
            "chunks": len(chunks),
            "entities": len(domain_nodes),
            "relationships": len(domain_edges),
            "mentions": len(edges) - len(domain_edges),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        },
    }
    return export


def main() -> int:
    export = build_export()
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_PATH.write_text(json.dumps(export, indent=2), encoding="utf-8")

    s = export["stats"]
    print("Veritas graph export written to:", EXPORT_PATH)
    print("-" * 52)
    print(f"  Documents:       {s['documents']}")
    print(f"  Chunks:          {s['chunks']}")
    print(f"  Entities:        {s['entities']}")
    print(f"  Relationships:   {s['relationships']}")
    print(f"  Mentions edges:  {s['mentions']}")
    print(f"  Total nodes:     {s['total_nodes']}")
    print(f"  Total edges:     {s['total_edges']}")
    print("-" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
