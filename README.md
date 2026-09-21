# Veritas: Enterprise Knowledge Graph & Graph RAG Platform

Veritas extracts entities and relationships from unstructured enterprise data
(emails, PDFs, docs, issue trackers, code), stores them in a **Neo4j** graph
using a structured ontology, and powers an explainable **Graph RAG**
(Retrieval-Augmented Generation) pipeline for natural-language Q&A.

> **Status:** Phase 1 — Core Foundation, Schema Design, Data Ingestion
> Pipeline, & Basic Cypher Query Layer.

---

## Phase 1 Scope

| Section | Deliverable |
|--------|-------------|
| 1 | System architecture + Neo4j ontology schema, constraints & indexes |
| 2 | Automated entity/relationship extraction ingestion pipeline |
| 3 | Vector + hybrid retrieval layer |
| 4 | Synthetic enterprise dataset + end-to-end runner |
| 5 | Verification Cypher queries + explainability output |

## Project Layout

```
Veritas/
├── veritas/                     # Python package
│   ├── config.py                # Typed settings (env-driven)
│   ├── db.py                    # Neo4j connection wrapper
│   ├── schema.py                # Ontology definition + schema bootstrap
│   ├── ingestion_pipeline.py    # Load → chunk → extract → build graph
│   └── retrieval.py             # Vector indexing + hybrid retrieval
├── cypher/                      # Raw Cypher scripts
│   ├── schema_setup.cypher      # Constraints + vector/full-text indexes
│   └── verification_queries.cypher
├── data/
│   └── sample_enterprise_docs.json
├── docs/
│   └── sample_hybrid_subgraph.json
├── run_phase1_ingestion.py      # End-to-end runner
├── requirements.txt
└── .env.example
```

## Quick Start

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env            # then edit credentials

# 3. Start Neo4j (Docker example)
docker run --name veritas-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/please_change_me neo4j:5

# 4. Bootstrap schema + run the ingestion pipeline
python run_phase1_ingestion.py
```

## Configuration

All configuration is environment-driven (see `.env.example`). Key toggles:

- `VERITAS_EXTRACTION_BACKEND` — `openai` or `spacy`
- `VERITAS_EMBEDDING_BACKEND` — `openai` or `sentence_transformers`
- `VERITAS_EMBEDDING_DIMENSION` — must match the embedding model output

## Requirements

- Python 3.10+
- Neo4j 5.x (vector index support required)
