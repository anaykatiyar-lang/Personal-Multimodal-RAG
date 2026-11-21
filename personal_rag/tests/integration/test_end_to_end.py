"""End-to-end integration test for Personal RAG Service on CPU."""

from pathlib import Path
from personal_rag.service import PersonalRAGService
from personal_rag.core.config import Settings


def test_end_to_end_service_flow(temp_dir):
    corpus_dir = temp_dir / "corpus"
    corpus_dir.mkdir()

    # 1. Create realistic sample files
    f1 = corpus_dir / "tq40_spec.txt"
    f1.write_text("The maximum junction temperature of the TQ-40 is 94 C.\n", encoding="utf-8")

    f2 = corpus_dir / "service.py"
    f2.write_text("DEFAULT_BATCH_TIMEOUT_S = 180\n", encoding="utf-8")

    settings = Settings(
        CPU_ONLY=True,
        GPU_EXECUTION_APPROVED=False,
        SQLITE_DB_PATH=temp_dir / "kb.db",
        QDRANT_STORAGE_PATH=temp_dir / "qdrant",
        DOCUMENTS_DIR=temp_dir / "docs",
        CACHE_DIR=temp_dir / "cache",
        LLM_PROVIDER="mock"
    )

    service = PersonalRAGService(settings=settings)

    # 2. Ingest
    rep = service.ingest_path(corpus_dir)
    assert rep.files_indexed == 2
    assert rep.total_chunks_created > 0

    # 3. Query 1: Fact lookup
    qr1 = service.answer_query("What is the maximum junction temperature of the TQ-40?", session_id="test_session")
    assert qr1.answer == "94"
    assert len(qr1.citations) > 0
    assert any("tq40_spec.txt" in c.source_path for c in qr1.citations)
    assert qr1.confidence > 0.9

    # 4. Query 2: Technical constant lookup
    qr2 = service.answer_query("What is the default batch timeout in the ingest service?", session_id="test_session")
    assert qr2.answer == "180"
    assert len(qr2.citations) > 0
    assert any("service.py" in c.source_path for c in qr2.citations)

    # 5. Query 3: Unanswerable refusal test (corpus does not contain unit price)
    qr3 = service.answer_query("What is the unit price at 10000 volume?", session_id="test_session")
    assert qr3.is_refusal is True
    assert qr3.answer == ""
    assert qr3.citations == []

    # 6. Verify Conversational Memory
    history = service.memory_manager.get_recent_dialogue("test_session")
    assert len(history) == 6  # 3 user turns + 3 assistant turns

