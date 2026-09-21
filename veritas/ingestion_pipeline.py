"""Veritas ingestion pipeline: load → chunk → extract → build graph.

This module is intentionally modular. Each stage is a class with a single
responsibility so it can be tested and swapped independently:

* :class:`DocumentLoader`  — load ``.txt``, ``.pdf``, ``.md`` into ``SourceDocument``.
* :class:`SemanticChunker` — split a document into overlapping semantic chunks.
* :class:`ExtractionEngine` — extract Entities + Triples from a chunk using an
  LLM (OpenAI via LangChain ``LLMGraphTransformer`` / custom prompt) or a spaCy
  fallback when no LLM is configured.
* :class:`GraphBuilder`    — MERGE entities, triples, chunks, and provenance
  links into Neo4j (idempotently, no duplicate nodes).
* :class:`IngestionPipeline` — orchestrates the stages end to end.

All external I/O (files, LLM calls, database) is wrapped with error handling so
a single bad document or transient failure does not abort the whole run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from veritas.config import ExtractionBackend, Settings, get_settings
from veritas.db import Neo4jConnection
from veritas.models import (
    Chunk,
    Entity,
    ExtractionResult,
    SourceDocument,
    Triple,
    make_chunk_id,
    utc_now_iso,
)
from veritas.schema import (
    DOMAIN_ENTITY_LABELS,
    RELATIONSHIP_TYPES,
    describe_ontology,
)

logger = logging.getLogger(__name__)

# File extension -> logical document type.
_EXT_TO_DOCTYPE = {".txt": "txt", ".md": "markdown", ".pdf": "pdf"}


# ===========================================================================
# Stage 1 — Document loading
# ===========================================================================
class DocumentLoader:
    """Load unstructured files into :class:`SourceDocument` objects."""

    SUPPORTED_EXTENSIONS = tuple(_EXT_TO_DOCTYPE.keys())

    def load_file(self, path: str | Path) -> SourceDocument:
        """Load a single file from disk.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        ValueError
            If the file extension is not supported.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Document not found: {p}")

        ext = p.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        content = self._read_pdf(p) if ext == ".pdf" else p.read_text(encoding="utf-8")
        return SourceDocument(
            id=f"doc::{p.stem.lower()}",
            name=p.name,
            content=content,
            source=str(p),
            doc_type=_EXT_TO_DOCTYPE[ext],
        )

    def load_directory(self, directory: str | Path) -> list[SourceDocument]:
        """Load every supported file in a directory (non-recursive)."""
        d = Path(directory)
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {d}")
        docs: list[SourceDocument] = []
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    docs.append(self.load_file(p))
                except Exception as exc:  # keep going on per-file failures
                    logger.warning("Skipping %s: %s", p, exc)
        return docs

    def load_records(self, records: Iterable[dict]) -> list[SourceDocument]:
        """Load documents from in-memory dict records (e.g. the sample JSON).

        Each record must provide ``id``/``name``/``content`` and optionally
        ``source`` and ``doc_type``.
        """
        docs: list[SourceDocument] = []
        for rec in records:
            docs.append(
                SourceDocument(
                    id=rec.get("id") or f"doc::{rec['name'].lower()}",
                    name=rec["name"],
                    content=rec["content"],
                    source=rec.get("source", rec["name"]),
                    doc_type=rec.get("doc_type", "txt"),
                )
            )
        return docs

    @staticmethod
    def _read_pdf(path: Path) -> str:
        """Extract text from a PDF using pypdf, with a helpful error message."""
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pypdf is required to load PDF files. Install with `pip install pypdf`."
            ) from exc
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)


# ===========================================================================
# Stage 2 — Chunking
# ===========================================================================
class SemanticChunker:
    """Split documents into overlapping chunks suitable for embedding.

    Uses LangChain's ``RecursiveCharacterTextSplitter`` when available (splits
    on paragraph/sentence boundaries first), and falls back to a simple
    sliding-window splitter otherwise so the pipeline never hard-depends on it.
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def split(self, document: SourceDocument) -> list[Chunk]:
        """Return the list of :class:`Chunk` objects for a document."""
        texts = self._split_text(document.content)
        chunks: list[Chunk] = []
        for idx, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=make_chunk_id(document.id, idx, text),
                    document_id=document.id,
                    chunk_index=idx,
                    text=text,
                )
            )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split raw text using LangChain if present, else a sliding window."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            return splitter.split_text(text)
        except ImportError:
            logger.debug("langchain-text-splitters unavailable; using fallback chunker.")
            return self._sliding_window(text)

    def _sliding_window(self, text: str) -> list[str]:
        """Deterministic fallback splitter with a fixed overlap."""
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]


# ===========================================================================
# Stage 3 — Extraction
# ===========================================================================
class ExtractionEngine:
    """Extract entities and triples from chunk text.

    Backend is selected via configuration:

    * ``openai``  — LangChain ``LLMGraphTransformer`` (with a custom-prompt
      fallback if the transformer package is unavailable).
    * ``spacy``   — offline named-entity extraction + co-occurrence heuristics,
      useful for development without an API key.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._allowed_nodes = list(DOMAIN_ENTITY_LABELS)
        self._allowed_rels = list(RELATIONSHIP_TYPES.keys())

    def extract(self, chunk: Chunk) -> ExtractionResult:
        """Extract structured graph data from a single chunk."""
        try:
            if self.settings.extraction_backend == ExtractionBackend.OPENAI:
                return self._extract_openai(chunk)
            return self._extract_spacy(chunk)
        except Exception as exc:  # never let one chunk kill the run
            logger.warning("Extraction failed for chunk %s: %s", chunk.id, exc)
            return ExtractionResult(chunk=chunk)

    # -- OpenAI / LangChain backend ----------------------------------------
    def _extract_openai(self, chunk: Chunk) -> ExtractionResult:
        """Use LangChain's LLMGraphTransformer, with a custom-prompt fallback."""
        api_key = self.settings.require_openai_key()
        try:
            from langchain_experimental.graph_transformers import LLMGraphTransformer
            from langchain_openai import ChatOpenAI
            from langchain_core.documents import Document as LCDocument

            llm = ChatOpenAI(
                model=self.settings.openai_llm_model, temperature=0, api_key=api_key
            )
            transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=self._allowed_nodes,
                allowed_relationships=self._allowed_rels,
            )
            graph_docs = transformer.convert_to_graph_documents(
                [LCDocument(page_content=chunk.text)]
            )
            return self._from_graph_documents(chunk, graph_docs)
        except ImportError:
            logger.debug("LLMGraphTransformer unavailable; using custom prompt.")
            return self._extract_custom_prompt(chunk, api_key)

    def _from_graph_documents(self, chunk: Chunk, graph_docs: list) -> ExtractionResult:
        """Convert LangChain GraphDocument output into Veritas models."""
        result = ExtractionResult(chunk=chunk)
        seen: dict[str, Entity] = {}

        def _entity(node) -> Entity:
            label = str(node.type).strip().capitalize()
            ent = Entity(label=label, name=str(node.id).strip())
            seen[ent.id] = ent
            return ent

        for gd in graph_docs:
            for node in gd.nodes:
                _entity(node)
            for rel in gd.relationships:
                head = _entity(rel.source)
                tail = _entity(rel.target)
                result.triples.append(
                    Triple(head=head, relationship=str(rel.type).upper(), tail=tail)
                )
        result.entities = list(seen.values())
        return result

    def _extract_custom_prompt(self, chunk: Chunk, api_key: str) -> ExtractionResult:
        """Direct OpenAI JSON-mode extraction using an ontology-aware prompt."""
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.settings.openai_llm_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(chunk.text)},
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        return self._from_json_payload(chunk, payload)

    def _system_prompt(self) -> str:
        """Build the ontology-constrained extraction instruction."""
        return (
            "You are an information-extraction engine for an enterprise "
            "knowledge graph. Extract entities and relationships STRICTLY "
            "following this ontology. Only use the listed node labels and "
            "relationship types. Return JSON.\n\n"
            f"{describe_ontology()}\n\n"
            'Output JSON shape: {"nodes":[{"label":"Person","name":"..."}],'
            '"triples":[{"head":"...","head_label":"Person",'
            '"relationship":"WORKS_ON","tail":"...","tail_label":"Project"}]}'
        )

    @staticmethod
    def _user_prompt(text: str) -> str:
        return f"Extract nodes and triples from this text:\n\n'''{text}'''"

    def _from_json_payload(self, chunk: Chunk, payload: dict) -> ExtractionResult:
        """Validate a JSON payload against the ontology allow-lists."""
        result = ExtractionResult(chunk=chunk)
        seen: dict[str, Entity] = {}

        def _register(name: str, label: str) -> Entity | None:
            label = label.strip().capitalize()
            if label not in self._allowed_nodes or not name.strip():
                return None
            ent = Entity(label=label, name=name.strip())
            seen[ent.id] = ent
            return ent

        for node in payload.get("nodes", []):
            _register(node.get("name", ""), node.get("label", ""))

        for tr in payload.get("triples", []):
            rel = str(tr.get("relationship", "")).upper()
            if rel not in self._allowed_rels:
                continue
            head = _register(tr.get("head", ""), tr.get("head_label", ""))
            tail = _register(tr.get("tail", ""), tr.get("tail_label", ""))
            if head and tail:
                result.triples.append(Triple(head=head, relationship=rel, tail=tail))

        result.entities = list(seen.values())
        return result

    # -- spaCy offline backend ---------------------------------------------
    def _extract_spacy(self, chunk: Chunk) -> ExtractionResult:
        """Offline extraction using spaCy NER + a co-occurrence heuristic.

        This backend does not infer typed relationships; it records the
        entities it finds and connects co-occurring Person/Project pairs with a
        best-effort ``WORKS_ON`` triple. It exists so the pipeline is runnable
        without any API key during development.
        """
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover
            raise ImportError("spaCy is required for the offline backend.") from exc

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. Install it with "
                "`python -m spacy download en_core_web_sm`."
            ) from exc

        doc = nlp(chunk.text)
        result = ExtractionResult(chunk=chunk)
        label_map = {"PERSON": "Person", "ORG": "Team", "PRODUCT": "Technology"}
        entities: dict[str, Entity] = {}
        for span in doc.ents:
            label = label_map.get(span.label_)
            if not label:
                continue
            ent = Entity(label=label, name=span.text.strip())
            entities[ent.id] = ent

        result.entities = list(entities.values())
        people = [e for e in result.entities if e.label == "Person"]
        teams = [e for e in result.entities if e.label == "Team"]
        for person in people:
            for team in teams:
                result.triples.append(Triple(person, "MEMBER_OF", team))
        return result


# ===========================================================================
# Stage 4 — Graph building
# ===========================================================================
class GraphBuilder:
    """Persist documents, chunks, entities, and triples into Neo4j.

    Every write uses ``MERGE`` keyed on the deterministic ``id`` so re-ingesting
    the same content never creates duplicate nodes.
    """

    def __init__(self, connection: Neo4jConnection) -> None:
        self.conn = connection

    def upsert_document(self, document: SourceDocument) -> None:
        """MERGE a Document node."""
        self.conn.execute_write(
            """
            MERGE (d:Document {id: $id})
            SET d.name = $name,
                d.source = $source,
                d.doc_type = $doc_type,
                d.created_at = coalesce(d.created_at, $created_at)
            """,
            {
                "id": document.id,
                "name": document.name,
                "source": document.source,
                "doc_type": document.doc_type,
                "created_at": document.created_at,
            },
        )

    def upsert_chunk(self, chunk: Chunk) -> None:
        """MERGE a Chunk node and link it to its source Document."""
        self.conn.execute_write(
            """
            MERGE (c:Chunk {id: $id})
            SET c.text = $text,
                c.chunk_index = $chunk_index,
                c.source_doc = $document_id,
                c.created_at = coalesce(c.created_at, $created_at)
            WITH c
            MATCH (d:Document {id: $document_id})
            MERGE (c)-[:EXTRACTED_FROM]->(d)
            """,
            {
                "id": chunk.id,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "document_id": chunk.document_id,
                "created_at": chunk.created_at,
            },
        )

    def upsert_entity(self, entity: Entity, source_doc: str) -> None:
        """MERGE a domain entity node with both its specific and :Entity label.

        The label is injected via string formatting (validated against the
        ontology allow-list upstream), while all values are passed as safe
        query parameters.
        """
        if entity.label not in DOMAIN_ENTITY_LABELS:
            logger.debug("Skipping entity with non-ontology label: %s", entity.label)
            return
        self.conn.execute_write(
            f"""
            MERGE (n:{entity.label} {{id: $id}})
            SET n:Entity,
                n.name = $name,
                n.source_doc = coalesce(n.source_doc, $source_doc),
                n.created_at = coalesce(n.created_at, $created_at)
            """,
            {
                "id": entity.id,
                "name": entity.name,
                "source_doc": source_doc,
                "created_at": utc_now_iso(),
            },
        )

    def upsert_triple(self, triple: Triple, source_doc: str) -> None:
        """MERGE both endpoints and the relationship between them."""
        rel = triple.relationship
        if rel not in RELATIONSHIP_TYPES:
            logger.debug("Skipping non-ontology relationship: %s", rel)
            return
        self.upsert_entity(triple.head, source_doc)
        self.upsert_entity(triple.tail, source_doc)
        self.conn.execute_write(
            f"""
            MATCH (h:Entity {{id: $head_id}})
            MATCH (t:Entity {{id: $tail_id}})
            MERGE (h)-[r:{rel}]->(t)
            SET r.source_doc = coalesce(r.source_doc, $source_doc)
            """,
            {"head_id": triple.head.id, "tail_id": triple.tail.id, "source_doc": source_doc},
        )

    def link_mention(self, document_id: str, entity: Entity) -> None:
        """Create a (:Document)-[:MENTIONS]->(:Entity) provenance edge."""
        self.conn.execute_write(
            """
            MATCH (d:Document {id: $document_id})
            MATCH (n:Entity {id: $entity_id})
            MERGE (d)-[:MENTIONS]->(n)
            """,
            {"document_id": document_id, "entity_id": entity.id},
        )


# ===========================================================================
# Orchestration
# ===========================================================================
@dataclass
class IngestionSummary:
    """Aggregate counts produced by an ingestion run."""

    documents: int = 0
    chunks: int = 0
    entities: int = 0
    triples: int = 0
    entity_ids: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, int]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "unique_entities": len(self.entity_ids),
            "triples": self.triples,
        }


class IngestionPipeline:
    """End-to-end orchestrator wiring all four stages together."""

    def __init__(
        self,
        connection: Neo4jConnection,
        loader: DocumentLoader | None = None,
        chunker: SemanticChunker | None = None,
        engine: ExtractionEngine | None = None,
    ) -> None:
        self.conn = connection
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or SemanticChunker()
        self.engine = engine or ExtractionEngine()
        self.builder = GraphBuilder(connection)

    def ingest_documents(self, documents: list[SourceDocument]) -> IngestionSummary:
        """Run every document through the full pipeline and persist to Neo4j."""
        summary = IngestionSummary()
        for document in documents:
            logger.info("Ingesting document: %s", document.name)
            self.builder.upsert_document(document)
            summary.documents += 1

            for chunk in self.chunker.split(document):
                self.builder.upsert_chunk(chunk)
                summary.chunks += 1

                result = self.engine.extract(chunk)
                for entity in result.entities:
                    self.builder.upsert_entity(entity, document.id)
                    self.builder.link_mention(document.id, entity)
                    summary.entity_ids.add(entity.id)
                    summary.entities += 1
                for triple in result.triples:
                    self.builder.upsert_triple(triple, document.id)
                    summary.triples += 1

        logger.info("Ingestion complete: %s", summary.as_dict())
        return summary

    def ingest_records(self, records: Iterable[dict]) -> IngestionSummary:
        """Convenience entrypoint for in-memory dict records (e.g. sample JSON)."""
        return self.ingest_documents(self.loader.load_records(records))
