# Veritas Visual Dashboard

An interactive, self-contained web dashboard that visually demonstrates the
Veritas Phase 1 pipeline output. It proves that unstructured enterprise
documents (a markdown architecture update, an email, and a Jira ticket) were
converted into a structured, interconnected knowledge graph — and it runs a
live Graph RAG retrieval demo on top of that graph.

**No Neo4j server and no API keys required.** It uses only the Python standard
library plus one CDN-hosted JavaScript library (`vis-network`) for the physics
graph.

## Run it (one command)

```bash
python3 demo/serve_dashboard.py
```

This regenerates the graph export, starts a local server, and opens the
dashboard at <http://127.0.0.1:8000/dashboard.html>.

Options:

```bash
python3 demo/serve_dashboard.py --port 9000     # use a different port
python3 demo/serve_dashboard.py --no-browser    # don't auto-open a browser
python3 demo/serve_dashboard.py --skip-export    # reuse existing export
```

## What you'll see

1. **Knowledge Graph** — a colour-coded, physics-based, draggable graph of
   every extracted entity (People, Projects, Technologies, Teams, Decisions,
   Issues) and their relationships. Click any node to inspect its properties
   and which documents it was extracted from. Toggle the dashed
   `Document -[:MENTIONS]-> Entity` provenance links on/off.
2. **Graph RAG Query** — type a natural-language question (or click a sample).
   The dashboard performs semantic entry-point retrieval over document chunks,
   expands the graph neighbourhood, and shows the exact subgraph payload (raw
   chunks + typed triples) that would be handed to the RAG generator in Phase 2.
3. **Source Documents** — the raw unstructured inputs, so viewers can see the
   before/after: messy text in, structured graph out.

## How it maps to the real pipeline

| Dashboard component | Backed by |
|---------------------|-----------|
| `build_graph_export.py` | `veritas/ingestion_pipeline.py` (load → chunk → extract → build) |
| Graph RAG query panel | `veritas/retrieval.py` `HybridRetriever` (vector entry points + 1–2 hop expansion) |
| Subgraph JSON output | `RetrievedSubgraph.to_dict()` shape |

The offline exporter uses a transparent, deterministic extraction ruleset so
the demo is stable and reproducible. In production this stage is performed by
the LLM `ExtractionEngine` against Neo4j; the data shapes are identical.

## Files

- `serve_dashboard.py` — launcher (generate export + serve).
- `build_graph_export.py` — offline pipeline → `graph_export.json`.
- `dashboard.html` — the single-page dashboard UI.
- `graph_export.json` — generated data (safe to delete; regenerated on run).
