// ===========================================================================
// Veritas :: Verification & Explainability Queries (Section 5)
// ---------------------------------------------------------------------------
// Run these after `run_phase1_ingestion.py` to confirm graph connectivity and
// explainability. Each query includes an expectation against the sample data.
// Substitute the $parameters (or inline literals) as needed in cypher-shell /
// Neo4j Browser.
// ===========================================================================


// ---------------------------------------------------------------------------
// QUERY 1 — EXPERT DISCOVERY
// "Find all people connected to a specific Technology or Project."
// Answers questions like: "Who are the experts on SentenceTransformers?"
// ---------------------------------------------------------------------------
// Parameter: $target_name (e.g. "SentenceTransformers" or "Atlas")
MATCH (p:Person)-[rel:HAS_SKILL|WORKS_ON|USES|MEMBER_OF]->(target)
WHERE toLower(target.name) CONTAINS toLower($target_name)
RETURN p.name        AS person,
       type(rel)      AS relationship,
       labels(target) AS target_labels,
       target.name    AS target
ORDER BY person;

// Inline example (no parameters) — experts linked to anything named "Atlas":
// MATCH (p:Person)-[rel]->(target)
// WHERE toLower(target.name) CONTAINS 'atlas'
// RETURN p.name AS person, type(rel) AS relationship, target.name AS target;


// ---------------------------------------------------------------------------
// QUERY 2 — IMPACT / BLAST-RADIUS ANALYSIS
// "Trace all Projects affected by a specific Issue or Decision."
// Answers: "What does issue ATLAS-231 block?" / "What does decision D-114 affect?"
// ---------------------------------------------------------------------------
// Parameter: $source_name (e.g. "times out" for the issue, or "Neo4j" for a decision)
MATCH (source)-[impact:BLOCKS|AFFECTS]->(project:Project)
WHERE (source:Issue OR source:Decision)
  AND toLower(source.name) CONTAINS toLower($source_name)
RETURN labels(source)[0] AS source_type,
       source.name       AS source,
       type(impact)       AS impact,
       project.name       AS affected_project
ORDER BY affected_project;

// Broader 2-hop transitive impact (issue -> project -> downstream technology):
// MATCH path = (i:Issue)-[:BLOCKS]->(p:Project)-[:USES]->(t:Technology)
// RETURN i.name AS issue, p.name AS project, collect(t.name) AS impacted_tech;


// ---------------------------------------------------------------------------
// QUERY 3 — PROVENANCE / EXPLAINABILITY
// "Retrieve the exact source Document and Chunk for a given relationship."
// This is what makes Graph RAG answers explainable & auditable.
// ---------------------------------------------------------------------------
// Parameters: $head_name, $tail_name (endpoints of the relationship of interest)
MATCH (h:Entity)-[rel]->(t:Entity)
WHERE toLower(h.name) CONTAINS toLower($head_name)
  AND toLower(t.name) CONTAINS toLower($tail_name)
// Follow the relationship's recorded source_doc back to its Document + Chunks.
MATCH (doc:Document {id: rel.source_doc})
OPTIONAL MATCH (chunk:Chunk)-[:EXTRACTED_FROM]->(doc)
RETURN h.name        AS head,
       type(rel)      AS relationship,
       t.name         AS tail,
       doc.name       AS source_document,
       doc.source     AS document_uri,
       collect(DISTINCT {id: chunk.id, text: chunk.text})[..3] AS supporting_chunks;
