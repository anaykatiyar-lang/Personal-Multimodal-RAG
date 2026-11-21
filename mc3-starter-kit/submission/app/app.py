#!/usr/bin/env python3
"""MC3 RAG submission — client entry point.

Follows the challenge execution contract:
  1. Index pass:
       python3 /app/app.py --index /app/corpus
  2. Query pass (once per question, 10 times):
       python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."
     Writing output to:
       /app/output/<query-id>_output.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
SOCKET_PATH = "/tmp/mc3_server.sock"
SOCKET_TIMEOUT = 25  # must comfortably fit inside the 30s budget


# ── Socket Client ────────────────────────────────────────────────────────────

def _send_msg(sock, data: dict):
    payload = json.dumps(data).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_msg(sock) -> dict:
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("Server connection terminated prematurely.")
        header += chunk
    length = struct.unpack(">I", header)[0]
    body = b""
    while len(body) < length:
        chunk = sock.recv(min(length - len(body), 65536))
        if not chunk:
            raise ConnectionError("Incomplete payload received from server.")
        body += chunk
    return json.loads(body.decode("utf-8"))


def call_server(request: dict, timeout: int = SOCKET_TIMEOUT) -> dict:
    """Send request to the resident GPU server."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(SOCKET_PATH)
    try:
        _send_msg(sock, request)
        return _recv_msg(sock)
    finally:
        sock.close()


def wait_for_server(timeout: int = 300):
    """Wait until the GPU server is responsive."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = call_server({"action": "health"}, timeout=3)
            if resp.get("status") == "ok":
                return
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError("GPU server failed to initialize within timeout window.")


# ── Document Parsers (Defensive & Robust) ────────────────────────────────────

def parse_pdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF; catch encryption and corruptions."""
    import fitz
    text_parts = []
    try:
        doc = fitz.open(str(path))
        if doc.is_encrypted:
            # Cannot be opened without password; intentionally skip
            doc.close()
            return ""
        for page in doc:
            t = page.get_text()
            if t:
                text_parts.append(t)
        doc.close()
    except Exception as e:
        print(f"[parser] Warning on PDF {path.name}: {e}", file=sys.stderr)
        return ""
    return "\n".join(text_parts).strip()


def parse_docx(path: Path) -> str:
    """Extract paragraphs and tables from DOCX."""
    from docx import Document
    text_parts = []
    try:
        doc = Document(str(path))
        for p in doc.paragraphs:
            if p.text.strip():
                text_parts.append(p.text.strip())
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    text_parts.append(" | ".join(cells))
    except Exception as e:
        print(f"[parser] Warning on DOCX {path.name}: {e}", file=sys.stderr)
        return ""
    return "\n".join(text_parts).strip()


def parse_xlsx(path: Path) -> str:
    """Extract sheet contents and tables from XLSX."""
    import openpyxl
    text_parts = []
    try:
        wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            text_parts.append(f"[Sheet: {sheet}]")
            for row in ws.iter_rows(values_only=True):
                vals = [str(v).strip() for v in row if v is not None and str(v).strip()]
                if vals:
                    text_parts.append(" | ".join(vals))
        wb.close()
    except Exception as e:
        print(f"[parser] Warning on XLSX {path.name}: {e}", file=sys.stderr)
        return ""
    return "\n".join(text_parts).strip()


def parse_text_file(path: Path) -> str:
    """Parse plain text, code, or logs with fault-tolerant utf-8 decoding."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as e:
        print(f"[parser] Warning on text file {path.name}: {e}", file=sys.stderr)
        return ""


def chunk_text(text: str, max_chars: int = 600, overlap: int = 100) -> list[str]:
    """Chunk text respecting line breaks and table rows."""
    if not text:
        return []

    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        line_len = len(line_str)
        if current_len + line_len > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = current_chunk[-2:] if len(current_chunk) >= 2 else []
            current_len = sum(len(x) for x in current_chunk)

        current_chunk.append(line_str)
        current_len += line_len

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks if chunks else [text[:max_chars]]


# ── Index Phase ──────────────────────────────────────────────────────────────

def index(corpus: Path) -> None:
    """Build and persist the document index."""
    wait_for_server(timeout=300)

    docs_to_index = []
    skipped_count = 0

    for root, dirs, files in os.walk(corpus):
        for fname in sorted(files):
            fpath = Path(root) / fname
            rel_path = str(fpath.relative_to(corpus)).replace("\\", "/")

            try:
                # Catch unreadable permissions (e.g. chmod 000 with dropped DAC_OVERRIDE)
                with open(fpath, "rb") as test_f:
                    _ = test_f.read(8)
            except (PermissionError, OSError) as e:
                print(f"[index] Skipping unreadable file {rel_path}: {e}", flush=True)
                skipped_count += 1
                continue

            ext = fpath.suffix.lower()

            try:
                if ext == ".pdf":
                    text = parse_pdf(fpath)
                    if text:
                        docs_to_index.append({
                            "rel_path": rel_path,
                            "text": text,
                            "chunks": chunk_text(text),
                            "is_image": False
                        })
                elif ext == ".docx":
                    text = parse_docx(fpath)
                    if text:
                        docs_to_index.append({
                            "rel_path": rel_path,
                            "text": text,
                            "chunks": chunk_text(text),
                            "is_image": False
                        })
                elif ext == ".xlsx":
                    text = parse_xlsx(fpath)
                    if text:
                        docs_to_index.append({
                            "rel_path": rel_path,
                            "text": text,
                            "chunks": chunk_text(text),
                            "is_image": False
                        })
                elif ext in {".txt", ".log", ".py", ".csv"}:
                    text = parse_text_file(fpath)
                    if text:
                        docs_to_index.append({
                            "rel_path": rel_path,
                            "text": text,
                            "chunks": chunk_text(text),
                            "is_image": False
                        })
                elif ext in {".png", ".jpg", ".jpeg"}:
                    raw_bytes = fpath.read_bytes()
                    b64_str = base64.b64encode(raw_bytes).decode("ascii")
                    # Send to VLM server to pre-extract high-fidelity text transcription
                    resp = call_server({"action": "extract_image", "image_b64": b64_str}, timeout=60)
                    extracted_text = resp.get("text", "")
                    docs_to_index.append({
                        "rel_path": rel_path,
                        "text": extracted_text,
                        "chunks": [f"[IMAGE: {rel_path}]\n{extracted_text}"],
                        "is_image": True,
                        "image_b64": b64_str
                    })
                else:
                    # Unknown type (e.g. .dat) - ignore defensively
                    skipped_count += 1
            except Exception as e:
                print(f"[index] Handled error on {rel_path}: {e}", file=sys.stderr)
                skipped_count += 1

    print(f"[index] Successfully parsed {len(docs_to_index)} documents ({skipped_count} skipped).", flush=True)

    # Send to server for in-memory indexing, GPU embedding computation, and persistence
    resp = call_server({"action": "build_index", "documents": docs_to_index}, timeout=300)
    if resp.get("status") != "ok":
        raise RuntimeError(f"Indexing failed: {resp.get('msg')}")

    print("[index] Indexing phase completed successfully.", flush=True)


# ── Query Phase ──────────────────────────────────────────────────────────────

def answer(corpus: Path, query: str) -> tuple[str, list[str], float]:
    """Execute query via server (zero disk re-reading overhead)."""
    resp = call_server({"action": "query", "query": query}, timeout=SOCKET_TIMEOUT)
    if resp.get("status") != "ok":
        return "", [], 0.0

    res = resp.get("result", {})
    return (
        res.get("answer", ""),
        res.get("citations", []),
        res.get("confidence", 0.0)
    )


# ── CLI Harness Contract ─────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        index(args.index)
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    text, citations, confidence = answer(args.corpus, args.query)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (args.query_id + "_output.json")
    out.write_text(
        json.dumps(
            {"answer": text, "citations": list(citations), "confidence": confidence},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
