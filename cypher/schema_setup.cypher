// ===========================================================================
// Veritas :: Neo4j Schema Setup
// ---------------------------------------------------------------------------
// This script defines the structural guarantees for the Veritas ontology:
//   1. Uniqueness constraints on node identifiers.
//   2. A vector index on Chunk embeddings (for semantic retrieval).
//   3. Full-text indexes on entity names (for keyword / fuzzy entry points).
//   4. Supporting range indexes for frequently filtered properties.
//
// Requires Neo4j 5.x (vector index support).
// Safe to re-run: every statement uses IF NOT EXISTS.
//
// The `$embedding_dimension` parameter must match your embedding model output
// (e.g. 384 for all-MiniLM-L6-v2, 1536 for text-embedding-3-small).
// When running from the Veritas Python schema module the parameter is injected
// automatically. When running manually in cypher-shell / Browser, replace it.
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. UNIQUENESS CONSTRAINTS (also create backing range indexes on `id`)
// ---------------------------------------------------------------------------
CREATE CONSTRAINT person_id_unique IF NOT EXISTS
FOR (n:Person) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT project_id_unique IF NOT EXISTS
FOR (n:Project) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT technology_id_unique IF NOT EXISTS
FOR (n:Technology) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (n:Document) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT decision_id_unique IF NOT EXISTS
FOR (n:Decision) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT issue_id_unique IF NOT EXISTS
FOR (n:Issue) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT team_id_unique IF NOT EXISTS
FOR (n:Team) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (n:Chunk) REQUIRE n.id IS UNIQUE;

// A generic :Entity label is applied to every extracted domain node so that
// cross-type lookups and MENTIONS relationships stay uniform.
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (n:Entity) REQUIRE n.id IS UNIQUE;

// ---------------------------------------------------------------------------
// 2. VECTOR INDEX on Chunk embeddings
// ---------------------------------------------------------------------------
// Cosine similarity is the standard choice for normalised text embeddings.
CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: $embedding_dimension,
    `vector.similarity_function`: 'cosine'
  }
};

// Optional: index Document-level embeddings too (document summaries / titles).
CREATE VECTOR INDEX document_embedding_index IF NOT EXISTS
FOR (d:Document) ON (d.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: $embedding_dimension,
    `vector.similarity_function`: 'cosine'
  }
};

// ---------------------------------------------------------------------------
// 3. FULL-TEXT SEARCH INDEXES on entity names
// ---------------------------------------------------------------------------
// A single full-text index spanning all named domain entities gives us fast
// keyword/fuzzy entry points into the graph for hybrid retrieval.
CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
FOR (n:Person|Project|Technology|Decision|Issue|Team)
ON EACH [n.name, n.description];

// Full-text over raw chunk text supports keyword search alongside vectors.
CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS
FOR (c:Chunk) ON EACH [c.text];

// ---------------------------------------------------------------------------
// 4. SUPPORTING RANGE INDEXES (frequent filters / joins)
// ---------------------------------------------------------------------------
CREATE INDEX entity_name_index IF NOT EXISTS
FOR (n:Entity) ON (n.name);

CREATE INDEX document_source_index IF NOT EXISTS
FOR (d:Document) ON (d.source);

CREATE INDEX chunk_source_doc_index IF NOT EXISTS
FOR (c:Chunk) ON (c.source_doc);
