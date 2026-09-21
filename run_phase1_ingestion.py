#!/usr/bin/env python3
"""Veritas Phase 1 end-to-end runner.

This script demonstrates the full Phase 1 flow against a live Neo4j instance:

    1. Bootstrap the schema (constraints + vector/full-text indexes).
    2. Load the synthetic sample dataset.
    3. Run the ingestion pipeline (chunk -> extract -> build graph).
    4. Generate and store chunk embeddings.
    5. Print an extracted-graph summary (node + relationship counts).
    6. Optionally run a demo hybrid-retrieval query.

Usage
-----
    python run_phase1_ingestion.py
    python run_phase1_ingestion.py --dataset data/sample_enterprise_docs.json
    python run_phase1_ingestion.py --query "Who is an expert in SentenceTransformers?"
    python run_phase1_ingestion.py --no-embeddings   # skip embedding generation

Configuration (Neo4j URI/credentials, extraction/embedding backends) is read
from environment variables / the local .env file. See .env.example.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from veritas.db import Neo4jConnection, Neo4jConnectionError
from veritas.ingestion_pipeline import IngestionPipeline
from veritas.retrieval import HybridRetriever, VectorIndexer
from veritas.schema import bootstrap_schema

logger = logging.getLogger("veritas.runner")

DEFAULT_DATASET = Path("data/sample_enterprise_docs.json")


def load_dataset(path: Path) -> list[dict]:
    """Load the sample dataset JSON and return the list of document records."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    if not documents:
        raise ValueError(f"No 'documents' array found in {path}")
    return documents


def print_graph_summary(conn: Neo4jConnection) -> None:
    """Query Neo4j for node/relationship counts and print a readable summary."""
    node_rows = conn.execute_read(
        """
        MATCH (n)
        WITH labels(n) AS ls
        UNWIND ls AS label
        RETURN label, count(*) AS count
        ORDER BY count DESC
        """
    )
    rel_rows = conn.execute_read(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
    )
    total_nodes = conn.execute_read("MATCH (n) RETURN count(n) AS c")[0]["c"]
    total_rels = conn.execute_read("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]

    print("\n" + "=" * 60)
    print("  EXTRACTED GRAPH SUMMARY")
    print("=" * 60)
    print(f"  Total nodes:         {total_nodes}")
    print(f"  Total relationships: {total_rels}")

    print("\n  Nodes by label:")
    for row in node_rows:
        print(f"    {row['label']:<14} {row['count']}")

    print("\n  Relationships by type:")
    for row in rel_rows:
        print(f"    {row['type']:<16} {row['count']}")
    print("=" * 60 + "\n")


def run_demo_query(conn: Neo4jConnection, query: str) -> None:
    """Run a single hybrid-retrieval query and pretty-print the subgraph."""
    print(f"\n>>> Hybrid retrieval for: {query!r}\n")
    retriever = HybridRetriever(conn)
    subgraph = retriever.retrieve(query, top_k=3, hops=2)
    print(json.dumps(subgraph.to_dict(), indent=2))


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Veritas Phase 1 ingestion runner.")
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="Path to dataset JSON."
    )
    parser.add_argument(
        "--query", type=str, default=None, help="Optional hybrid-retrieval demo query."
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding generation (schema + graph build only).",
    )
    parser.add_argument(
        "--skip-schema", action="store_true", help="Skip schema bootstrap step."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    try:
        records = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Could not load dataset: %s", exc)
        return 1

    try:
        with Neo4jConnection() as conn:
            # 1. Schema
            if not args.skip_schema:
                logger.info("Bootstrapping schema...")
                result = bootstrap_schema(conn)
                logger.info("Applied %d schema statements.", result.statements_run)

            # 2 + 3. Load + ingest
            logger.info("Ingesting %d documents...", len(records))
            pipeline = IngestionPipeline(conn)
            summary = pipeline.ingest_records(records)
            logger.info("Ingestion summary: %s", summary.as_dict())

            # 4. Embeddings
            if not args.no_embeddings:
                logger.info("Generating chunk embeddings...")
                indexed = VectorIndexer(conn).index_chunks()
                logger.info("Embedded %d chunks.", indexed)

            # 5. Graph summary
            print_graph_summary(conn)

            # 6. Optional demo query
            if args.query:
                run_demo_query(conn, args.query)

    except Neo4jConnectionError as exc:
        logger.error("Database error: %s", exc)
        logger.error("Is Neo4j running and are your .env credentials correct?")
        return 2

    logger.info("Phase 1 run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
