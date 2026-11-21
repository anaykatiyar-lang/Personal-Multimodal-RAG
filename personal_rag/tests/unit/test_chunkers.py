"""Unit tests for chunking strategies (Recursive, ParentChild, Table)."""

from personal_rag.core.types import ParsedDocument, ParsedPage, DocumentMetadata, ExtractedTable, ChunkType
from personal_rag.chunking.recursive import RecursiveChunker
from personal_rag.chunking.parent_child import ParentChildChunker
from personal_rag.chunking.table_chunker import TableChunker


def _create_dummy_doc(text: str) -> ParsedDocument:
    meta = DocumentMetadata(
        document_id="doc_test_123",
        title="Test Doc",
        source_path="test.txt",
        filename="test.txt",
        mime_type="text/plain",
        file_size_bytes=len(text),
        created_at="2026-11-14T00:00:00Z",
        modified_at="2026-11-14T00:00:00Z",
        parser_name="CodeTextParser"
    )
    return ParsedDocument(metadata=meta, pages=[ParsedPage(page_number=1, text=text)], raw_text=text)


def test_recursive_chunker():
    long_text = "\n\n".join([f"Paragraph {i}: Content about system specifications." for i in range(20)])
    doc = _create_dummy_doc(long_text)

    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    assert all(c.document_id == "doc_test_123" for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)


def test_parent_child_chunker():
    text = "Section 1: Hardware specs.\n\n" + ("Detailed descriptions of registers and bus timings. " * 30)
    doc = _create_dummy_doc(text)

    chunker = ParentChildChunker(parent_chunk_size=400, child_chunk_size=120, child_overlap=20)
    chunks = chunker.chunk(doc)

    parents = [c for c in chunks if c.metadata.get("is_parent") is True]
    children = [c for c in chunks if c.metadata.get("is_parent") is False]

    assert len(parents) > 0
    assert len(children) > len(parents)
    # Check that each child has a valid parent_id referencing an existing parent
    parent_ids = {p.chunk_id for p in parents}
    for ch in children:
        assert ch.parent_id in parent_ids


def test_table_chunker():
    doc = _create_dummy_doc("Table doc")
    table = ExtractedTable(
        table_id="t1",
        page_number=1,
        headers=["Part", "Price", "LeadTime"],
        rows=[[f"PART-{i}", f"${i*10}", f"{i}w"] for i in range(15)]
    )
    doc.tables.append(table)

    chunker = TableChunker(max_rows_per_table_chunk=5)
    chunks = chunker.chunk(doc)

    assert len(chunks) == 3  # 15 rows / 5 per chunk
    for c in chunks:
        assert c.chunk_type == ChunkType.TABLE
        assert "| Part | Price | LeadTime |" in c.text
        assert "| --- | --- | --- |" in c.text

