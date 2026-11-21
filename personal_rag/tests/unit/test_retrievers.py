"""Unit tests for Dense, Lexical, and Hybrid (RRF) retrievers."""

from personal_rag.core.types import Chunk, ChunkType
from personal_rag.storage.sqlite_store import SQLiteStore
from personal_rag.storage.vector_store import InMemoryVectorStore
from personal_rag.embeddings.mock import MockEmbeddingProvider
from personal_rag.retrieval.dense import DenseRetriever
from personal_rag.retrieval.lexical import LexicalRetriever
from personal_rag.retrieval.hybrid import HybridRetriever
from personal_rag.retrieval.parent import ParentContextRetriever


def test_hybrid_rrf_retrieval(temp_dir):
    db_path = temp_dir / "retrieval_test.db"
    sqlite_store = SQLiteStore(db_path)
    vector_store = InMemoryVectorStore()
    embedder = MockEmbeddingProvider()

    # Create test chunks
    c1 = Chunk(
        chunk_id="c_term_01",
        document_id="doc_specs",
        source_path="specs/datasheet.pdf",
        text="Thermal throttle engages at junction temperature 94C with error E7731.",
        chunk_type=ChunkType.TEXT,
        embedding=embedder.embed_query("Thermal throttle junction temperature 94C E7731")
    )
    c2 = Chunk(
        chunk_id="c_fan_02",
        document_id="doc_parts",
        source_path="support/rma_parts.xlsx",
        text="Field-replaceable fan assembly part number is ORR-FAN-2214-B.",
        chunk_type=ChunkType.TEXT,
        embedding=embedder.embed_query("Field-replaceable fan assembly part number ORR-FAN-2214-B")
    )

    sqlite_store.index_chunks([c1, c2])
    vector_store.upsert_vectors([c1, c2])

    dense = DenseRetriever(vector_store, embedder)
    lexical = LexicalRetriever(sqlite_store)
    hybrid = HybridRetriever(dense, lexical)

    # 1. Search for thermal error
    results = hybrid.retrieve("What error code is logged when thermal throttle engages?", top_k=2)
    assert len(results) > 0
    assert results[0].chunk.chunk_id == "c_term_01"
    assert results[0].score > 0

    # 2. Search for fan part number
    fan_results = hybrid.retrieve("ORR-FAN-2214-B", top_k=2)
    assert len(fan_results) > 0
    assert fan_results[0].chunk.chunk_id == "c_fan_02"


def test_parent_context_expansion(temp_dir):
    db_path = temp_dir / "parent_test.db"
    sqlite_store = SQLiteStore(db_path)

    parent_chunk = Chunk(
        chunk_id="parent_doc_01",
        document_id="doc_01",
        source_path="doc.txt",
        text="Full Chapter 3: Overview of memory controllers and clock synchronization algorithms.",
        chunk_type=ChunkType.TEXT
    )
    child_chunk = Chunk(
        chunk_id="child_doc_01",
        document_id="doc_01",
        source_path="doc.txt",
        parent_id="parent_doc_01",
        text="Clock sync frequency is 2.4 GHz.",
        chunk_type=ChunkType.TEXT
    )
    sqlite_store.index_chunks([parent_chunk, child_chunk])

    # Simulate SearchResult pointing to child
    from personal_rag.core.types import SearchResult
    child_res = SearchResult(chunk=child_chunk, score=0.95, rank=1)

    expander = ParentContextRetriever(sqlite_store)
    expanded = expander.expand_context([child_res])

    assert len(expanded) == 1
    assert expanded[0].chunk.chunk_id == "parent_doc_01"
    assert "Full Chapter 3: Overview" in expanded[0].chunk.text

