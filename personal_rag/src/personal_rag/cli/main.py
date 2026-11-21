"""Main CLI entrypoint for Personal RAG operations."""

import argparse
import sys
import json
from pathlib import Path
from personal_rag.service import PersonalRAGService
from personal_rag.core.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="personal-rag",
        description="Personal Advanced RAG Platform — Local, Multimodal Research Assistant"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Ingest command
    ingest_p = subparsers.add_parser("ingest", help="Ingest a directory of documents into knowledge base")
    ingest_p.add_argument("path", type=Path, help="Directory or file path to ingest")
    ingest_p.add_argument("--prune", action="store_true", help="Remove deleted documents from the index")

    # Ask command
    ask_p = subparsers.add_parser("ask", help="Query the personal knowledge base")
    ask_p.add_argument("query", type=str, help="Question to ask")
    ask_p.add_argument("--session", type=str, default="default_cli_session", help="Conversation session ID")
    ask_p.add_argument("--top-k", type=int, default=10, help="Number of retrieved candidates")

    # List documents command
    subparsers.add_parser("docs", help="List all indexed documents in the knowledge base")

    # Hardware inspect command
    subparsers.add_parser("hw", help="Inspect available compute hardware without allocating GPU memory")

    # Backup / Restore
    backup_p = subparsers.add_parser("backup", help="Create a backup snapshot of the database")
    backup_p.add_argument("dest", type=Path, help="Target backup file path (.db)")

    restore_p = subparsers.add_parser("restore", help="Restore database from a backup file")
    restore_p.add_argument("src", type=Path, help="Source backup file path (.db)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    service = PersonalRAGService()

    if args.command == "ingest":
        print(f"[personal-rag] Ingesting path: {args.path} ...")
        report = service.ingest_path(args.path, prune_deleted=args.prune)
        print("════════════════════════════════════════════════════════════════════")
        print(f" Ingestion Summary:")
        print(f"   Files Scanned:        {report.total_files_scanned}")
        print(f"   Files Indexed:        {report.files_indexed}")
        print(f"   Skipped (Unchanged):  {report.files_skipped_unchanged}")
        print(f"   Failed / Unreadable:  {report.files_failed}")
        print(f"   Total Chunks Created: {report.total_chunks_created}")
        if report.errors:
            print("   Warnings / Errors:")
            for err in report.errors[:5]:
                print(f"     - {err}")
        print("════════════════════════════════════════════════════════════════════")
        return 0

    elif args.command == "ask":
        result = service.answer_query(args.query, session_id=args.session, top_k=args.top_k)
        print("════════════════════════════════════════════════════════════════════")
        print(f" Question: {result.query}")
        print(f" Strategy: {result.strategy_used} (Total: {result.latency_breakdown_ms.get('total_ms', 0):.1f} ms)")
        print("────────────────────────────────────────────────────────────────────")
        if result.is_refusal:
            print(" Answer: [Insufficient evidence in knowledge base]")
        else:
            print(f" Answer: {result.answer}")
            print(f" Confidence: {result.confidence:.2f}")
            print(" Citations:")
            for cit in result.citations:
                print(f"   - {cit.source_path} (Page {cit.page_number})")
                print(f"     Excerpt: {cit.excerpt[:120]}...")
        print("════════════════════════════════════════════════════════════════════")
        return 0

    elif args.command == "docs":
        docs = service.sqlite_store.list_documents(limit=100)
        print(f"Total Registered Documents: {len(docs)}")
        print(f"{'Filename':<35} {'Size (KB)':<12} {'Pages':<6} {'MIME Type':<30}")
        print("-" * 85)
        for d in docs:
            size_kb = d.file_size_bytes / 1024.0
            print(f"{d.filename[:33]:<35} {size_kb:<12.1f} {d.page_count:<6} {d.mime_type:<30}")
        return 0

    elif args.command == "hw":
        info = service.hw_manager.get_device_info()
        print("════════════════════════════════════════════════════════════════════")
        print(" Hardware Inspection & Credit Protection Policy:")
        print(f"   Active Mode:         {info.device_type.upper()}")
        print(f"   Device Name:         {info.device_name}")
        print(f"   GPU Available:       {info.is_gpu_available}")
        print(f"   ROCm Detected:       {info.is_rocm}")
        print(f"   CPU Cores:           {info.cpu_count}")
        print(f"   Policy Enforcement:  CPU_ONLY={service.settings.cpu_only}, GPU_APPROVED={service.settings.gpu_execution_approved}")
        print("════════════════════════════════════════════════════════════════════")
        return 0

    elif args.command == "backup":
        service.sqlite_store.backup(args.dest)
        print(f"[OK] Database backed up to {args.dest}")
        return 0

    elif args.command == "restore":
        service.sqlite_store.restore(args.src)
        print(f"[OK] Database restored from {args.src}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

