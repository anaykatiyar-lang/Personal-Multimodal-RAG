#!/usr/bin/env python3
"""Long-running GPU inference and retrieval server for MC3 RAG.

Stays resident for the entire container lifecycle.
Listens on Unix domain socket /tmp/mc3_server.sock.

Key Features:
  - Fails immediately if ROCm/CUDA GPU is unavailable (no CPU fallback).
  - Keeps Qwen2.5-VL-7B and BGE-small resident in GPU memory.
  - Keeps the entire retrieval index (dense embeddings + BM25) in-memory
    so per-question invocations of app.py do not re-read large files from disk.
  - Supports hybrid retrieval (BGE dense embeddings + BM25 lexical).
  - Enforces citation necessity verification and superseded document filtering.
  - Serializes GPU inference with threading locks.
"""

from __future__ import annotations

import base64
import json
import os
import re
import signal
import socketserver
import struct
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

SOCKET_PATH = "/tmp/mc3_server.sock"
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
VLM_MODEL_PATH = "/models/qwen2.5-vl-7b"
EMBED_MODEL_PATH = "/models/bge-small"

# Global state
gpu_lock = threading.Lock()
device = None
vlm_model = None
vlm_processor = None
embed_model = None

# In-memory index data structures
index_lock = threading.Lock()
in_memory_index = {
    "is_loaded": False,
    "documents": [],       # list of {rel_path, text, is_image, is_superseded, image_b64}
    "chunks": [],          # list of str
    "chunk_meta": [],      # list of {doc_idx, rel_path, is_image, is_superseded}
    "embeddings": None,    # torch.Tensor on device
    "bm25": None,          # BM25Okapi instance
    "tokenized_corpus": [] # list of list of str
}


def normalize_value(val: str) -> str:
    """Normalize string using challenge rules (upper, no whitespace, no dots or degrees)."""
    if not val:
        return ""
    v = val.upper()
    v = re.sub(r"[\s\.\°]", "", v)
    return v


def init_gpu_and_models():
    """Verify GPU availability and load models directly onto AMD GPU."""
    global device, vlm_model, vlm_processor, embed_model

    # Issue 1: Fail immediately if GPU is not available. Never silently fall back.
    if not torch.cuda.is_available():
        raise RuntimeError("AMD GPU/ROCm is unavailable. CPU fallback is strictly prohibited.")

    device = torch.device("cuda")
    print(f"[server] Active GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # 1. Load BGE-small embedding model
    print("[server] Loading embedding model (BGE-small)...", flush=True)
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBED_MODEL_PATH, device=str(device))
    print("[server] Embedding model loaded successfully.", flush=True)

    # 2. Load Qwen2.5-VL-7B
    print("[server] Loading VLM (Qwen2.5-VL-7B)...", flush=True)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_PATH)
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    vlm_model.eval()
    print("[server] VLM loaded successfully on GPU.", flush=True)


def tokenize_text(text: str) -> list[str]:
    """Tokenize text for BM25 indexing."""
    return re.findall(r"\w+", text.lower())


def build_and_store_index(docs: list[dict]):
    """Compute embeddings, build BM25, and store index in-memory and on disk."""
    global in_memory_index
    with index_lock:
        print(f"[server] Building in-memory index for {len(docs)} documents...", flush=True)
        chunks = []
        chunk_meta = []
        tokenized_corpus = []

        for doc_idx, doc in enumerate(docs):
            rel_path = doc["rel_path"]
            is_image = doc.get("is_image", False)
            # Detect superseded files (e.g. WITHDRAWN in filename or content)
            is_superseded = "WITHDRAWN" in rel_path.upper() or "SUPERSEDED" in rel_path.upper()

            doc_chunks = doc.get("chunks", [])
            for c in doc_chunks:
                if not c.strip():
                    continue
                chunks.append(c)
                chunk_meta.append({
                    "doc_idx": doc_idx,
                    "rel_path": rel_path,
                    "is_image": is_image,
                    "is_superseded": is_superseded
                })
                tokenized_corpus.append(tokenize_text(c))

        # Compute dense embeddings on GPU
        if chunks:
            print(f"[server] Computing embeddings for {len(chunks)} chunks on GPU...", flush=True)
            with gpu_lock:
                emb_array = embed_model.encode(
                    chunks,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=64
                )
            embeddings_tensor = torch.tensor(emb_array, device=device, dtype=torch.float32)
        else:
            embeddings_tensor = None

        # Build BM25 index for hybrid retrieval
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        in_memory_index = {
            "is_loaded": True,
            "documents": docs,
            "chunks": chunks,
            "chunk_meta": chunk_meta,
            "embeddings": embeddings_tensor,
            "bm25": bm25,
            "tokenized_corpus": tokenized_corpus
        }

        # Persist index to disk for audit / reloadability
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        disk_data = {
            "documents": [
                {
                    "rel_path": d["rel_path"],
                    "text": d["text"],
                    "is_image": d.get("is_image", False),
                    "is_superseded": "WITHDRAWN" in d["rel_path"].upper() or "SUPERSEDED" in d["rel_path"].upper()
                }
                for d in docs
            ],
            "chunks": chunks,
            "chunk_meta": chunk_meta
        }
        (INDEX_DIR / "index_meta.json").write_text(json.dumps(disk_data), encoding="utf-8")
        if embeddings_tensor is not None:
            np.save(INDEX_DIR / "embeddings.npy", embeddings_tensor.cpu().numpy())
        print(f"[server] Index persisted to {INDEX_DIR}.", flush=True)


def ensure_index_loaded():
    """Load index from disk into memory if server was restarted."""
    global in_memory_index
    if in_memory_index["is_loaded"]:
        return

    meta_file = INDEX_DIR / "index_meta.json"
    emb_file = INDEX_DIR / "embeddings.npy"
    if not meta_file.exists():
        return

    with index_lock:
        if in_memory_index["is_loaded"]:
            return
        print("[server] Loading index from disk into memory...", flush=True)
        disk_data = json.loads(meta_file.read_text(encoding="utf-8"))
        chunks = disk_data.get("chunks", [])
        chunk_meta = disk_data.get("chunk_meta", [])
        docs = disk_data.get("documents", [])

        embeddings_tensor = None
        if emb_file.exists():
            emb_np = np.load(emb_file)
            embeddings_tensor = torch.tensor(emb_np, device=device, dtype=torch.float32)

        tokenized_corpus = [tokenize_text(c) for c in chunks]
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        in_memory_index = {
            "is_loaded": True,
            "documents": docs,
            "chunks": chunks,
            "chunk_meta": chunk_meta,
            "embeddings": embeddings_tensor,
            "bm25": bm25,
            "tokenized_corpus": tokenized_corpus
        }
        print(f"[server] Loaded {len(chunks)} chunks into memory.", flush=True)


def execute_vlm_generation(prompt: str, images_b64: list[str] | None = None) -> str:
    """Generate response from Qwen2.5-VL with strict concurrency control."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        process_vision_info = None

    messages = [{"role": "user", "content": []}]

    # Only attach image if supplied and needed
    if images_b64:
        for img_b64 in images_b64[:1]:  # Limit to at most 1 image to save VRAM and latency
            messages[0]["content"].append({
                "type": "image",
                "image": f"data:image/png;base64,{img_b64}"
            })

    messages[0]["content"].append({"type": "text", "text": prompt})

    with gpu_lock:
        text_input = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if images_b64 and process_vision_info:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = vlm_processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)
        else:
            inputs = vlm_processor(
                text=[text_input],
                padding=True,
                return_tensors="pt"
            ).to(device)

        with torch.no_grad():
            output_ids = vlm_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return vlm_processor.decode(generated, skip_special_tokens=True).strip()


def run_hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """Perform hybrid retrieval (dense similarity + BM25 lexical)."""
    ensure_index_loaded()
    if not in_memory_index["is_loaded"] or not in_memory_index["chunks"]:
        return []

    chunks = in_memory_index["chunks"]
    chunk_meta = in_memory_index["chunk_meta"]
    emb_tensor = in_memory_index["embeddings"]
    bm25 = in_memory_index["bm25"]

    # 1. Dense score
    with gpu_lock:
        q_emb = embed_model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q_tensor = torch.tensor(q_emb, device=device, dtype=torch.float32)
        dense_scores = torch.matmul(emb_tensor, q_tensor.T).squeeze(-1).cpu().numpy()

    # 2. BM25 score
    q_tokens = tokenize_text(query)
    bm25_scores = np.array(bm25.get_scores(q_tokens)) if bm25 else np.zeros(len(chunks))
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-6)

    # 3. Hybrid fusion (penalize superseded documents)
    combined_scores = 0.65 * dense_scores + 0.35 * bm25_scores

    ranked_indices = np.argsort(-combined_scores)

    results = []
    seen_files = {}

    for idx in ranked_indices:
        meta = chunk_meta[idx]
        # Skip superseded chunks unless specifically asked
        if meta["is_superseded"]:
            continue

        rel_path = meta["rel_path"]
        if rel_path not in seen_files:
            seen_files[rel_path] = 0
        if seen_files[rel_path] >= 3:  # at most 3 chunks per document
            continue
        seen_files[rel_path] += 1

        results.append({
            "chunk": chunks[idx],
            "rel_path": rel_path,
            "is_image": meta["is_image"],
            "score": float(combined_scores[idx])
        })
        if len(results) >= top_k:
            break

    return results


def verify_citation_necessity(
    answer_val: str,
    raw_citations: list[str],
    retrieved_results: list[dict],
    query: str
) -> list[str]:
    """Verify that cited files are strictly necessary (exact citation set rule).

    A file is only kept if:
      1. It directly contains the normalized answer token, OR
      2. It acts as an indispensable link in a multi-hop query chain
         (e.g., ticket ID 'ORR-1847' or error code in a log that connects to the bug database).
    Files that merely mention related topics are stripped.
    """
    if not answer_val or not raw_citations:
        return []

    norm_ans = normalize_value(answer_val)
    if not norm_ans:
        return []

    # Map retrieved results by relative path
    doc_text_by_path = {}
    for r in retrieved_results:
        p = r["rel_path"]
        doc_text_by_path[p] = doc_text_by_path.get(p, "") + "\n" + r["chunk"]

    # Also search full docs if available
    for d in in_memory_index["documents"]:
        p = d["rel_path"]
        if p not in doc_text_by_path:
            doc_text_by_path[p] = d["text"]

    verified = set()

    # Step A: Identify which candidate files directly contain the answer value
    direct_answer_files = set()
    for path in raw_citations:
        text = doc_text_by_path.get(path, "")
        norm_text = normalize_value(text)
        if norm_ans in norm_text:
            direct_answer_files.add(path)

    # Step B: Multi-hop dependency resolution
    # Check if a log/source file provided an intermediate entity (e.g. ticket number ORR-xxxx, error code)
    linking_files = set()
    ticket_match = re.search(r"ORR-\d+", query.upper()) or re.search(r"ORR-\d+", "\n".join(doc_text_by_path.values()))
    ticket_id = ticket_match.group(0) if ticket_match else None

    error_match = re.search(r"E\d{4}", query.upper()) or re.search(r"E\d{4}", "\n".join(doc_text_by_path.values()))
    error_code = error_match.group(0) if error_match else None

    for path in raw_citations:
        if path in direct_answer_files:
            verified.add(path)
            continue

        text = doc_text_by_path.get(path, "")
        # If the file supplies the linking ticket or error code that links to the direct answer file
        if ticket_id and ticket_id in text.upper():
            verified.add(path)
        elif error_code and error_code in text.upper():
            verified.add(path)

    # Ensure never citing superseded / withdrawn files
    clean_verified = [p for p in verified if "WITHDRAWN" not in p.upper() and "SUPERSEDED" not in p.upper()]

    # If verification found nothing direct from raw_citations, check direct_answer_files in retrieved results
    if not clean_verified:
        for r in retrieved_results:
            p = r["rel_path"]
            if "WITHDRAWN" in p.upper() or "SUPERSEDED" in p.upper():
                continue
            if norm_ans in normalize_value(doc_text_by_path.get(p, "")):
                clean_verified.append(p)
                break

    return sorted(list(set(clean_verified)))


def handle_query(query: str) -> dict:
    """Execute end-to-end question answering and return compliant JSON."""
    results = run_hybrid_search(query, top_k=8)
    if not results:
        return {"answer": "", "citations": [], "confidence": 0.0}

    # Format context
    context_lines = []
    has_image = False
    image_b64_to_send = None

    for r in results:
        context_lines.append(f"--- FILE: {r['rel_path']} ---\n{r['chunk']}")
        if r["is_image"] and not has_image:
            has_image = True
            # Find image base64 if needed
            for d in in_memory_index["documents"]:
                if d["rel_path"] == r["rel_path"] and "image_b64" in d:
                    image_b64_to_send = d["image_b64"]
                    break

    context_str = "\n\n".join(context_lines)

    prompt = f"""You are an enterprise question-answering system.
Rules:
1. Return ONLY the complete value (e.g., "94", "Q3 FY27", "ORR-FAN-2214-B", "4.3.2", "E7731", "180", "B14", "REV-C2").
2. No sentences, no explanations, no filler words.
3. If the answer cannot be determined from the documents, return:
ANSWER: 
CITATIONS: 
4. Cite ONLY the exact file paths necessary to answer. If a file is superseded or withdrawn, DO NOT use it.

DOCUMENTS:
{context_str}

QUESTION: {query}

OUTPUT FORMAT:
ANSWER: <value or empty>
CITATIONS: <file1, file2 or empty>"""

    # Only send image if the query genuinely relates to image evidence
    imgs = [image_b64_to_send] if (has_image and image_b64_to_send and ("label" in query.lower() or "pin" in query.lower() or "revision" in query.lower())) else None

    raw_response = execute_vlm_generation(prompt, images_b64=imgs)

    answer_val = ""
    raw_citations = []

    for line in raw_response.splitlines():
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            answer_val = line[7:].strip()
        elif line.upper().startswith("CITATIONS:"):
            c_str = line[10:].strip()
            if c_str:
                raw_citations = [c.strip().strip("'\"`") for c in c_str.split(",") if c.strip()]

    # Check for refusal / empty
    refusals = {"", "NONE", "N/A", "UNKNOWN", "NOT FOUND", "UNANSWERABLE", "I DON'T KNOW"}
    if answer_val.upper() in refusals:
        return {"answer": "", "citations": [], "confidence": 0.0}

    # Strict citation necessity verification
    final_citations = verify_citation_necessity(answer_val, raw_citations, results, query)

    # An answer with no valid citations is invalid under challenge rules
    if not final_citations:
        return {"answer": "", "citations": [], "confidence": 0.0}

    return {
        "answer": answer_val,
        "citations": final_citations,
        "confidence": 0.95
    }


# ── Socket Server ────────────────────────────────────────────────────────────

def send_msg(sock, data: dict):
    payload = json.dumps(data).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def recv_msg(sock) -> dict | None:
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            return None
        header += chunk
    length = struct.unpack(">I", header)[0]
    body = b""
    while len(body) < length:
        chunk = sock.recv(min(length - len(body), 65536))
        if not chunk:
            return None
        body += chunk
    return json.loads(body.decode("utf-8"))


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            req = recv_msg(self.request)
            if req is None:
                return

            action = req.get("action")

            if action == "health":
                send_msg(self.request, {"status": "ok"})

            elif action == "build_index":
                docs = req.get("documents", [])
                build_and_store_index(docs)
                send_msg(self.request, {"status": "ok"})

            elif action == "extract_image":
                img_b64 = req.get("image_b64")
                prompt = (
                    "Extract all text, labels, part numbers, revision codes, model names, "
                    "serial numbers, and connector pin assignments from this image. "
                    "Be exact, complete, and literal."
                )
                text = execute_vlm_generation(prompt, images_b64=[img_b64] if img_b64 else None)
                send_msg(self.request, {"status": "ok", "text": text})

            elif action == "query":
                query_str = req.get("query", "")
                result = handle_query(query_str)
                send_msg(self.request, {"status": "ok", "result": result})

            else:
                send_msg(self.request, {"status": "error", "msg": f"unknown action: {action}"})

        except Exception as e:
            traceback.print_exc()
            try:
                send_msg(self.request, {"status": "error", "msg": str(e)})
            except Exception:
                pass


class ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    print("════════════════════════════════════════════════════════════════════", flush=True)
    print(" MC3 Server starting — Initializing GPU and models...", flush=True)
    print("════════════════════════════════════════════════════════════════════", flush=True)

    init_gpu_and_models()

    server = ThreadedUnixServer(SOCKET_PATH, RequestHandler)
    os.chmod(SOCKET_PATH, 0o777)
    print(f"[server] Ready and listening on {SOCKET_PATH}", flush=True)

    def shutdown(signum, frame):
        print("[server] Shutting down...", flush=True)
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    server.serve_forever()


if __name__ == "__main__":
    main()
