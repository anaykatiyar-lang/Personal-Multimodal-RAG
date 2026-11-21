"""Unit tests for the incremental ingestion pipeline."""

from pathlib import Path
from personal_rag.ingestion.pipeline import IngestionPipeline


def test_ingestion_pipeline_incremental_change_detection(temp_dir, sqlite_store, vector_store):
    corpus_dir = temp_dir / "corpus"
    corpus_dir.mkdir()

    file_a = corpus_dir / "note.txt"
    file_a.write_text("Initial release note.", encoding="utf-8")

    pipeline = IngestionPipeline(sqlite_store=sqlite_store, vector_store=vector_store)

    # First pass: Ingests 1 file
    rep1 = pipeline.ingest_directory(corpus_dir)
    assert rep1.files_indexed == 1
    assert rep1.files_skipped_unchanged == 0
    assert rep1.total_chunks_created > 0

    # Second pass: File unchanged -> skipped!
    rep2 = pipeline.ingest_directory(corpus_dir)
    assert rep2.files_indexed == 0
    assert rep2.files_skipped_unchanged == 1

    # Modify file: Triggers re-indexing
    file_a.write_text("Updated release note with defect fix.", encoding="utf-8")
    rep3 = pipeline.ingest_directory(corpus_dir)
    assert rep3.files_indexed == 1
    assert rep3.files_skipped_unchanged == 0


def test_ingestion_pipeline_pruning_deleted_files(temp_dir, sqlite_store, vector_store):
    corpus_dir = temp_dir / "corpus"
    corpus_dir.mkdir()

    f1 = corpus_dir / "keep.txt"
    f1.write_text("Permanent document.", encoding="utf-8")
    f2 = corpus_dir / "temporary.txt"
    f2.write_text("Temporary draft.", encoding="utf-8")

    pipeline = IngestionPipeline(sqlite_store=sqlite_store, vector_store=vector_store)
    pipeline.ingest_directory(corpus_dir)

    docs = sqlite_store.list_documents()
    assert len(docs) == 2

    # Remove temporary.txt and run with prune_deleted=True
    f2.unlink()
    pipeline.ingest_directory(corpus_dir, prune_deleted=True)

    remaining = sqlite_store.list_documents()
    assert len(remaining) == 1
    assert remaining[0].filename == "keep.txt"

