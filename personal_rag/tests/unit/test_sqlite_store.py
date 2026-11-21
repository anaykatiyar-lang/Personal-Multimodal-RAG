"""Unit tests for SQLite metadata, FTS5 lexical search, memory, and backup/restore."""

import pytest
from personal_rag.core.types import DocumentMetadata, Chunk, ChunkType


def test_sqlite_document_crud_and_modification_detection(sqlite_store):
    doc = DocumentMetadata(
        document_id="doc_abc123",
        title="Architecture Roadmap",
        source_path="planning/roadmap.docx",
        filename="roadmap.docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        file_size_bytes=1024,
        created_at="2026-11-14T10:00:00Z",
        modified_at="2026-11-14T10:00:00Z",
        parser_name="DocxParser",
        page_count=5,
        tags=["roadmap", "planning"]
    )

    # Initial state: document should be considered modified/unindexed
    assert sqlite_store.is_document_modified("planning/roadmap.docx", "doc_abc123") is True

    # Upsert
    sqlite_store.upsert_document(doc)

    # Now it should be recorded as matching
    assert sqlite_store.is_document_modified("planning/roadmap.docx", "doc_abc123") is False
    assert sqlite_store.is_document_modified("planning/roadmap.docx", "new_hash") is True

    fetched = sqlite_store.get_document("doc_abc123")
    assert fetched is not None
    assert fetched.title == "Architecture Roadmap"
    assert fetched.tags == ["roadmap", "planning"]


def test_sqlite_fts5_lexical_search_and_parent_child(sqlite_store):
    parent = Chunk(
        chunk_id="parent_001",
        document_id="doc_001",
        source_path="specs/datasheet.pdf",
        text="Section 4: Thermal specifications. Maximum junction temperature is 94 degrees Celsius.",
        chunk_type=ChunkType.TEXT,
        page_number=4
    )
    child = Chunk(
        chunk_id="child_001",
        document_id="doc_001",
        source_path="specs/datasheet.pdf",
        parent_id="parent_001",
        text="Maximum junction temperature of TQ-40 is 94C.",
        chunk_type=ChunkType.TEXT,
        page_number=4
    )

    sqlite_store.index_chunks([parent, child])

    # Test FTS5 search
    results = sqlite_store.search_lexical("junction temperature", top_k=5)
    assert len(results) > 0
    assert any(r.chunk.chunk_id == "child_001" for r in results)

    # Test Parent-Child retrieval
    fetched_parent = sqlite_store.get_parent_chunk("child_001")
    assert fetched_parent is not None
    assert fetched_parent.chunk_id == "parent_001"
    assert "Section 4: Thermal specifications" in fetched_parent.text


def test_conversational_history(sqlite_store):
    session_id = "session_research_01"
    sqlite_store.save_message(session_id, "user", "What is the timeout?")
    sqlite_store.save_message(session_id, "assistant", "The timeout is 180s.", [{"source": "ingest.py"}])

    history = sqlite_store.get_conversation_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["citations"] == [{"source": "ingest.py"}]


def test_user_memory(sqlite_store):
    sqlite_store.set_user_memory("preferred_model", "Qwen2.5-VL-7B", "preferences")
    sqlite_store.set_user_memory("research_topic", "RAG Systems", "project")

    memories = sqlite_store.get_all_user_memories()
    assert len(memories) == 2

    # Update memory
    sqlite_store.set_user_memory("preferred_model", "Qwen2.5-VL-32B", "preferences")
    updated = sqlite_store.get_all_user_memories()
    assert any(m["key"] == "preferred_model" and m["value"] == "Qwen2.5-VL-32B" for m in updated)

    # Delete
    sqlite_store.delete_user_memory("research_topic")
    remaining = sqlite_store.get_all_user_memories()
    assert len(remaining) == 1


def test_sqlite_backup_and_restore(sqlite_store, temp_dir):
    sqlite_store.set_user_memory("backup_check", "saved_value")
    backup_file = temp_dir / "backup.db"

    # Backup
    sqlite_store.backup(backup_file)
    assert backup_file.exists()

    # Create new store from backup
    from personal_rag.storage.sqlite_store import SQLiteStore
    restored_store = SQLiteStore(backup_file)
    memories = restored_store.get_all_user_memories()
    assert any(m["key"] == "backup_check" and m["value"] == "saved_value" for m in memories)

