"""Unit tests for the vector store abstraction (100% CPU deterministic)."""

from personal_rag.core.types import Chunk, ChunkType
from personal_rag.storage.vector_store import InMemoryVectorStore


def test_in_memory_vector_store_similarity():
    store = InMemoryVectorStore()

    # Three 4D dummy vectors
    chunk_a = Chunk(
        chunk_id="c_a",
        document_id="doc_1",
        source_path="a.txt",
        text="Thermal specifications",
        chunk_type=ChunkType.TEXT,
        embedding=[1.0, 0.0, 0.0, 0.0]
    )
    chunk_b = Chunk(
        chunk_id="c_b",
        document_id="doc_1",
        source_path="a.txt",
        text="Cooling fans",
        chunk_type=ChunkType.TEXT,
        embedding=[0.8, 0.2, 0.0, 0.0]
    )
    chunk_c = Chunk(
        chunk_id="c_c",
        document_id="doc_2",
        source_path="b.txt",
        text="Software scheduler",
        chunk_type=ChunkType.TEXT,
        embedding=[0.0, 0.0, 1.0, 0.0]
    )

    store.upsert_vectors([chunk_a, chunk_b, chunk_c])

    # Query vector pointing towards [1.0, 0.0, 0.0, 0.0]
    results = store.search_vector([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c_a"
    assert results[0].score > 0.99
    assert results[1].chunk.chunk_id == "c_b"


def test_vector_store_filtering():
    store = InMemoryVectorStore()

    c1 = Chunk(chunk_id="c1", document_id="doc_alpha", source_path="alpha.txt", text="Alpha", embedding=[1.0, 0.0])
    c2 = Chunk(chunk_id="c2", document_id="doc_beta", source_path="beta.txt", text="Beta", embedding=[1.0, 0.0])
    store.upsert_vectors([c1, c2])

    # Search with filter
    filtered = store.search_vector([1.0, 0.0], top_k=5, filters={"document_id": "doc_beta"})
    assert len(filtered) == 1
    assert filtered[0].chunk.chunk_id == "c2"


def test_vector_store_delete_document():
    store = InMemoryVectorStore()
    c1 = Chunk(chunk_id="c1", document_id="doc_alpha", source_path="alpha.txt", text="Alpha", embedding=[1.0, 0.0])
    c2 = Chunk(chunk_id="c2", document_id="doc_beta", source_path="beta.txt", text="Beta", embedding=[0.0, 1.0])
    store.upsert_vectors([c1, c2])

    store.delete_vectors_for_document("doc_alpha")
    remaining = store.search_vector([1.0, 0.0], top_k=5)
    assert len(remaining) == 1
    assert remaining[0].chunk.chunk_id == "c2"

