"""Vector + hybrid retrieval layer for Veritas Graph RAG.

Two responsibilities:

1. :class:`VectorIndexer` — generate embeddings for stored ``Chunk`` nodes and
   write them back into Neo4j so the vector index can serve similarity search.
2. :class:`HybridRetriever` — given a natural-language prompt, find semantic
   entry-point chunks (vector search), pull the entities they mention, expand
   1–2 hops through the graph, and return a connected subgraph of triples plus
   the supporting raw text chunks.

The returned :class:`RetrievedSubgraph` is a plain, serialisable structure ready
to be handed to a RAG generator in Phase 2.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from veritas.db import Neo4jConnection
from veritas.embeddings import EmbeddingProvider, get_embedding_provider

logger = logging.getLogger(__name__)


# ===========================================================================
# Vector indexing
# ===========================================================================
class VectorIndexer:
    """Compute and persist embeddings for Chunk (and optionally Document) nodes."""

    def __init__(
        self,
        connection: Neo4jConnection,
        provider: EmbeddingProvider | None = None,
        batch_size: int = 32,
    ) -> None:
        self.conn = connection
        self.provider = provider or get_embedding_provider()
        self.batch_size = batch_size

    def index_chunks(self) -> int:
        """Embed every Chunk lacking an embedding and store the vectors.

        Returns
        -------
        int
            The number of chunks that were (re-)embedded.
        """
        rows = self.conn.execute_read(
            """
            MATCH (c:Chunk)
            WHERE c.embedding IS NULL
            RETURN c.id AS id, c.text AS text
            """
        )
        if not rows:
            logger.info("No chunks require embedding.")
            return 0

        indexed = 0
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            vectors = self.provider.embed_texts([r["text"] for r in batch])
            for row, vector in zip(batch, vectors):
                self.conn.execute_write(
                    """
                    MATCH (c:Chunk {id: $id})
                    CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
                    """,
                    {"id": row["id"], "embedding": vector},
                )
                indexed += 1
        logger.info("Indexed %d chunk embeddings.", indexed)
        return indexed


# ===========================================================================
# Hybrid retrieval result structures
# ===========================================================================
@dataclass
class RetrievedChunk:
    """A semantically relevant chunk plus its similarity score."""

    id: str
    text: str
    score: float
    source_doc: str


@dataclass
class RetrievedTriple:
    """A triple discovered during neighbourhood expansion."""

    head: str
    head_label: str
    relationship: str
    tail: str
    tail_label: str
    source_doc: str | None = None


@dataclass
class RetrievedSubgraph:
    """The full hybrid-retrieval payload handed to the RAG generator."""

    query: str
    entry_chunks: list[RetrievedChunk] = field(default_factory=list)
    seed_entities: list[dict[str, str]] = field(default_factory=list)
    triples: list[RetrievedTriple] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of the subgraph."""
        return {
            "query": self.query,
            "entry_chunks": [asdict(c) for c in self.entry_chunks],
            "seed_entities": self.seed_entities,
            "triples": [asdict(t) for t in self.triples],
            "stats": {
                "entry_chunks": len(self.entry_chunks),
                "seed_entities": len(self.seed_entities),
                "triples": len(self.triples),
            },
        }


# ===========================================================================
# Hybrid retriever
# ===========================================================================
class HybridRetriever:
    """Vector entry-point search combined with graph neighbourhood expansion."""

    def __init__(
        self,
        connection: Neo4jConnection,
        provider: EmbeddingProvider | None = None,
        vector_index: str = "chunk_embedding_index",
    ) -> None:
        self.conn = connection
        self.provider = provider or get_embedding_provider()
        self.vector_index = vector_index

    def retrieve(
        self, prompt: str, top_k: int = 4, hops: int = 2
    ) -> RetrievedSubgraph:
        """Run hybrid retrieval for a user prompt.

        Steps
        -----
        1. Embed the prompt and run a vector similarity search over chunks.
        2. Collect the domain entities mentioned by the documents behind those
           chunks (the graph "entry points").
        3. Expand ``hops`` steps outward from those entities and collect the
           connecting triples.

        Parameters
        ----------
        prompt:
            The natural-language question.
        top_k:
            Number of entry-point chunks to retrieve via vector search.
        hops:
            Neighbourhood expansion depth (1 or 2 recommended).
        """
        if hops < 1:
            raise ValueError("hops must be >= 1")

        subgraph = RetrievedSubgraph(query=prompt)
        subgraph.entry_chunks = self._vector_search(prompt, top_k)
        if not subgraph.entry_chunks:
            logger.info("Vector search returned no chunks for prompt.")
            return subgraph

        seed_ids = self._seed_entity_ids(subgraph.entry_chunks)
        subgraph.seed_entities = self._describe_entities(seed_ids)
        subgraph.triples = self._expand_neighbourhood(seed_ids, hops)
        return subgraph

    # -- Step 1: vector search ---------------------------------------------
    def _vector_search(self, prompt: str, top_k: int) -> list[RetrievedChunk]:
        """Query the Neo4j vector index for the closest chunks to the prompt."""
        query_vector = self.provider.embed_text(prompt)
        rows = self.conn.execute_read(
            """
            CALL db.index.vector.queryNodes($index, $top_k, $vector)
            YIELD node, score
            RETURN node.id AS id,
                   node.text AS text,
                   node.source_doc AS source_doc,
                   score
            ORDER BY score DESC
            """,
            {"index": self.vector_index, "top_k": top_k, "vector": query_vector},
        )
        return [
            RetrievedChunk(
                id=r["id"],
                text=r["text"],
                score=float(r["score"]),
                source_doc=r["source_doc"],
            )
            for r in rows
        ]

    # -- Step 2: seed entities ---------------------------------------------
    def _seed_entity_ids(self, chunks: list[RetrievedChunk]) -> list[str]:
        """Find domain entities mentioned by the documents behind the chunks."""
        source_docs = list({c.source_doc for c in chunks if c.source_doc})
        if not source_docs:
            return []
        rows = self.conn.execute_read(
            """
            MATCH (d:Document)-[:MENTIONS]->(e:Entity)
            WHERE d.id IN $source_docs
            RETURN DISTINCT e.id AS id
            """,
            {"source_docs": source_docs},
        )
        return [r["id"] for r in rows]

    def _describe_entities(self, entity_ids: list[str]) -> list[dict[str, str]]:
        """Return name/label descriptors for the seed entities."""
        if not entity_ids:
            return []
        rows = self.conn.execute_read(
            """
            MATCH (e:Entity)
            WHERE e.id IN $ids
            RETURN e.id AS id,
                   e.name AS name,
                   [l IN labels(e) WHERE l <> 'Entity'][0] AS label
            """,
            {"ids": entity_ids},
        )
        return [{"id": r["id"], "name": r["name"], "label": r["label"]} for r in rows]

    # -- Step 3: neighbourhood expansion -----------------------------------
    def _expand_neighbourhood(
        self, seed_ids: list[str], hops: int
    ) -> list[RetrievedTriple]:
        """Collect every relationship within ``hops`` of the seed entities.

        The variable-length pattern is expanded to ``hops`` and each concrete
        relationship along the matched paths is returned as a flat triple. The
        hop count is validated and interpolated (not user-supplied text), while
        seed ids are passed as safe parameters.
        """
        rows = self.conn.execute_read(
            f"""
            MATCH (seed:Entity)
            WHERE seed.id IN $seed_ids
            MATCH path = (seed)-[*1..{int(hops)}]-(:Entity)
            UNWIND relationships(path) AS rel
            WITH DISTINCT rel, startNode(rel) AS h, endNode(rel) AS t
            RETURN h.name AS head,
                   [l IN labels(h) WHERE l <> 'Entity'][0] AS head_label,
                   type(rel) AS relationship,
                   t.name AS tail,
                   [l IN labels(t) WHERE l <> 'Entity'][0] AS tail_label,
                   rel.source_doc AS source_doc
            """,
            {"seed_ids": seed_ids},
        )
        return [
            RetrievedTriple(
                head=r["head"],
                head_label=r["head_label"],
                relationship=r["relationship"],
                tail=r["tail"],
                tail_label=r["tail_label"],
                source_doc=r.get("source_doc"),
            )
            for r in rows
        ]
