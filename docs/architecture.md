# Veritas — System Architecture & Ontology (Section 1)

## 1. High-Level Architecture

```
                     ┌───────────────────────────────────────────┐
   Unstructured      │              INGESTION PIPELINE             │
   sources           │                                             │
  (txt/pdf/md,  ───▶ │  Loader ─▶ Chunker ─▶ LLM/NLP Extraction ─▶ │ ─┐
   emails,           │           (semantic)   (Nodes + Triples)    │  │
   tickets)          └───────────────────────────────────────────┘  │
                                                                     ▼
                                                        ┌──────────────────────┐
                                                        │        Neo4j          │
                                                        │  Knowledge Graph +    │
                                                        │  Vector + Full-text   │
                                                        │  indexes              │
                                                        └──────────────────────┘
                                                                     ▲
   User question  ─────────────────────────────────────────────────┘
        │              ┌───────────────────────────────────────────┐
        └────────────▶ │            RETRIEVAL LAYER                  │
                       │  Vector search (entry points)              │
                       │  + Cypher 1–2 hop neighbourhood expansion  │
                       │  ─▶ connected subgraph (triples + chunks)  │
                       └───────────────────────────────────────────┘
                                        │
                                        ▼
                              (Phase 2) RAG Generator
```

## 2. Ontology — Node Labels

| Label | Description | Key properties |
|-------|-------------|----------------|
| `Person` | Employee, stakeholder, author | `id, name, role, email, source_doc, created_at` |
| `Project` | Named initiative / product | `id, name, status, source_doc, created_at` |
| `Technology` | Tool, framework, language, platform | `id, name, category, source_doc, created_at` |
| `Team` | Organisational unit / working group | `id, name, source_doc, created_at` |
| `Decision` | Recorded architectural/business choice | `id, name, decided_at, source_doc, created_at` |
| `Issue` | Bug, blocker, tracked work item | `id, name, severity, status, source_doc, created_at` |
| `Document` | Source document | `id, name, source, doc_type, embedding, created_at` |
| `Chunk` | Semantic text chunk (carries embedding) | `id, text, chunk_index, source_doc, embedding, created_at` |

Every domain node also receives a generic `:Entity` label so that
document→entity `MENTIONS` links and cross-type lookups stay uniform.

## 3. Ontology — Relationship Types

| Pattern | Meaning |
|---------|---------|
| `(:Person)-[:WORKS_ON]->(:Project)` | Person contributes to a project |
| `(:Person)-[:HAS_SKILL]->(:Technology)` | Person is skilled in a technology |
| `(:Person)-[:MEMBER_OF]->(:Team)` | Person belongs to a team |
| `(:Team)-[:OWNS]->(:Project)` | Team owns a project |
| `(:Project)-[:USES]->(:Technology)` | Project uses a technology |
| `(:Issue)-[:BLOCKS]->(:Project)` | Issue blocks a project |
| `(:Decision)-[:AFFECTS]->(:Project)` | Decision affects a project |
| `(:Document)-[:MENTIONS]->(:Entity)` | Document references an entity |
| `(:Chunk)-[:EXTRACTED_FROM]->(:Document)` | Chunk originates from a document |

## 4. Constraints & Indexes

Defined in [`cypher/schema_setup.cypher`](../cypher/schema_setup.cypher) and
applied programmatically by [`veritas/schema.py`](../veritas/schema.py):

- **Uniqueness constraints** on `id` for every node label (dedupe + fast lookup).
- **Vector indexes** (`cosine`) on `Chunk.embedding` and `Document.embedding`.
- **Full-text indexes** on entity `name`/`description` and on `Chunk.text`.
- **Range indexes** on frequently filtered properties (`source`, `source_doc`).

The vector dimension is parameterised (`$embedding_dimension`) and must match
the configured embedding model.
