# Existing Project Audit: Multimodal-RAG to Personal RAG Platform

**Document Version:** 1.0.0  
**Date:** 2026-11-14  
**Author:** Principal AI Infrastructure Architect & Lead Engineer  
**Status:** Complete — Phase 0 Repository Audit  
**Upstream Repository:** `https://github.com/anaykatiyar-lang/Multimodal-RAG`  
**Local Workspace:** `c:\Users\HP\Downloads\mc3-starter-kit`  

---

## 1. Executive Summary & Context

This audit evaluates the codebase originally developed for the **AMD Mini-Challenge 3 (Multimodal Retrieval-Augmented Generation)** to establish a solid foundation for building a **personal, production-grade, multimodal knowledge and research assistant (`personal_rag/`)**.

### Key Context & Constraints
* **GPU Cost & Credit Protection:** Exactly **$8.71** of AMD GPU credits have been consumed to date. GPU resources are treated as expensive. The personal RAG platform must enforce a strict **`CPU_ONLY=true`** default development and testing policy. No AMD cloud instances, SSH tunnels, or multi-gigabyte weight downloads will be initiated without explicit prior authorization.
* **Preservation of Challenge Integrity:** The existing challenge directories (`submission/`, `starter/`), evaluation harnesses (`selfcheck.py`, `run_all_questions.sh`, `build_and_test.sh`), contracts (`CONTRACT.md`), and sample questions (`sample-questions.json`) must remain **100% intact, runnable, and isolated**.
* **Hardware Independence:** While the challenge submission was specialized for AMD ROCm / MI300X execution within a container, the personal RAG platform must be fully decoupled from underlying hardware, running seamlessly on CPU-only local environments, Apple Silicon, NVIDIA CUDA, AMD ROCm, or local/remote OpenAI-compatible model endpoints.

---

## 2. Workspace & File Inventory

The local development directory is structured as follows:

```
c:\Users\HP\Downloads\mc3-starter-kit\
└── mc3-starter-kit\
    ├── CONTRACT.md                     # Official AMD Challenge specification & scoring rules
    ├── README.txt                      # Official starter kit instructions
    ├── sample-questions.json           # 10 machine-readable benchmark questions with ground truth
    ├── selfcheck.py                    # Official AMD pre-submission contract validation script
    ├── setup.sh                        # Shell script to recreate zip-uncarryable edge-cases
    ├── mc3-corpus/                     # 13 sample documents across 6 subdirectories
    │   ├── archive/                    # Edge case: empty directory
    │   ├── engineering/                # ingest_service.py, meridian_release_notes.txt
    │   ├── logs/                       # prod_inference_2026-09-02.log
    │   ├── planning/                   # roadmap_fy27.docx
    │   ├── specs/                      # backplane_pinout.png, tq40_datasheet_r1_WITHDRAWN.pdf, r2.pdf
    │   ├── support/                    # asset_label.jpg, bug_database.csv, rma_parts.xlsx
    │   └── vendor/                     # internal_audit.txt (chmod 000), supplier_agreement_ENCRYPTED.pdf, telemetry_capture.dat
    ├── starter/                        # AMD-supplied initial skeleton
    │   ├── Dockerfile
    │   └── app/
    │       ├── app.py                  # Blank reference skeleton
    │       └── requirements.txt
    └── submission/                     # Audited AMD challenge solution
        ├── Dockerfile                  # ROCm base image + dependencies + weight pre-download
        ├── build_and_test.sh           # Automated build, selfcheck, and evaluation script
        ├── run_all_questions.sh        # Rigorous 200-point exact-match evaluation harness
        └── app/
            ├── app.py                  # CLI thin client handling --index and --query
            ├── server.py               # Resident GPU daemon with in-memory index & VLM
            └── requirements.txt        # Pinned dependencies (no-deps for PyTorch)
```

### Git Repository State
* **Local State:** The current directory is an extracted standalone distribution; no `.git` metadata directory was initialized in this folder.
* **Remote State:** Upstream GitHub repository is registered at `https://github.com/anaykatiyar-lang/Multimodal-RAG`.
* **Action Required:** When tracking versions, personal platform development will be managed cleanly with proper `.gitignore` rules preventing any personal data, model weights, vector databases, or secrets from being staged.

---

## 3. Analysis of Existing Challenge Architecture

### 3.1 Dual-Process Client-Server Pattern
The current implementation in `submission/` cleanly splits the challenge requirements across two components:
1. **`submission/app/server.py` (Daemon):**
   * Starts on container initialization via `CMD`.
   * Listens on Unix domain socket `/tmp/mc3_server.sock`.
   * Loads **Qwen2.5-VL-7B-Instruct** (via Hugging Face `transformers` with `bfloat16`) and **BAAI/bge-small-en-v1.5** (via `sentence-transformers`) onto GPU memory once.
   * Holds the document index, embeddings tensor, and `rank-bm25` index in Python RAM/VRAM.
2. **`submission/app/app.py` (CLI Client):**
   * Invoked via CLI for `--index /app/corpus` and ten separate `--query` invocations.
   * Traverses files defensively and sends parsed structures to `server.py` via framed socket messages.
   * For queries, delegates retrieval and generation to `server.py` and immediately writes `/app/output/<query-id>_output.json`.

### 3.2 Strengths & What Works Well
* **Sub-Second Per-Question Latency:** Bypasses the 30-second query timeout trap by maintaining resident models and avoiding JSON index re-parsing.
* **Defensive Corpus Ingestion:** Gracefully survives `chmod 000` permission errors, encrypted PDFs, empty directories, and unknown binary formats (`.dat`).
* **Multimodal Extraction:** Solves image-only questions (`backplane_pinout.png` and `asset_label.jpg`) by pre-extracting exhaustive visual text transcriptions at indexing time.
* **Exact Citation Necessity Filtering:** Strips irrelevant retrievals by verifying that candidate citations directly match normalized answer tokens or provide linking identifiers (e.g. ticket numbers `ORR-1847` connecting logs to bug databases).
* **Superseded Document Protection:** Actively penalizes and removes withdrawn specifications (`tq40_datasheet_r1_WITHDRAWN.pdf`).

---

## 4. Technical Audit: Vulnerabilities, Limitations & Risks

While `submission/` is tailored to score 200/200 on the fixed challenge dataset, it has fundamental architectural limitations that prevent it from being used as a personal, general-purpose RAG platform:

### 4.1 Critical GPU Coupling & Hardware Rigidity
* **Problem:** `server.py` line 74 explicitly executes:
  ```python
  if not torch.cuda.is_available():
      raise RuntimeError("AMD GPU/ROCm is unavailable. CPU fallback is strictly prohibited.")
  ```
* **Impact:** The code cannot start on standard developer laptops, CPU test runners, or non-AMD workstations.
* **Remediation for `personal_rag`:** Decouple hardware completely. Build a pluggable device manager supporting `cpu`, `cuda`, `rocm`, `mps`, and remote API providers.

### 4.2 Monolithic In-Memory Index (Scalability Ceiling)
* **Problem:** Documents, raw strings, chunk lists, and dense embeddings are stored as a global Python dictionary `in_memory_index` in `server.py`.
* **Impact:** Works for 13 files and 50 chunks. On a personal knowledge base of 5,000 documents (hundreds of thousands of chunks), storing unindexed arrays in memory will trigger Out-Of-Memory (OOM) crashes.
* **Remediation for `personal_rag`:** Integrate a persistent, local-first vector store (embedded Qdrant) combined with a local SQLite database for document metadata and full-text search (SQLite FTS5).

### 4.3 Brute-Force Vector Search
* **Problem:** Dense search computes an exhaustive matrix multiplication across all indexed chunks on the GPU:
  ```python
  dense_scores = torch.matmul(emb_tensor, q_tensor.T).squeeze(-1).cpu().numpy()
  ```
* **Impact:** $O(N)$ linear complexity. Has no indexing graph (HNSW) and cannot perform fast filtered similarity search over document tags, dates, or collections.
* **Remediation for `personal_rag`:** Use indexed ANN retrieval with payload metadata filters.

### 4.4 Fragile Character-Based Chunking
* **Problem:** `chunk_text()` splits text purely by character counts (default 600 chars with 100 char overlap).
* **Impact:** Destroys table layouts, splits code blocks across boundary lines, isolates headers from body content, and loses hierarchical context.
* **Remediation for `personal_rag`:** Implement layout-aware and structure-aware chunking (Markdown, table-preserving, parent-child contextual chunking).

### 4.5 Static Linear Retrieval (No Query Routing)
* **Problem:** Every user question receives the exact same fixed hybrid formula (`0.65 * dense + 0.35 * bm25`) with a static `top_k=8`.
* **Impact:** Factual identifier queries do not get 100% lexical focus; complex multi-part queries cannot decompose into sub-queries; broad thematic research queries retrieve shallow localized fragments.
* **Remediation for `personal_rag`:** Implement an adaptive Query Router that inspects question complexity and selects optimal retrieval workflows (Lexical, Dense, Hybrid, Multi-Query, Parent-Child).

### 4.6 Lack of Incremental Ingestion & Change Detection
* **Problem:** Ingesting a corpus requires re-parsing every single file from scratch.
* **Impact:** Adding one new PDF requires re-indexing the entire knowledge base.
* **Remediation for `personal_rag`:** SHA-256 fingerprinting and an ingestion registry to detect added, modified, and deleted documents.

### 4.7 Absence of State, Chat History & Conversational Memory
* **Problem:** The challenge harness is purely single-turn and stateless.
* **Impact:** Cannot follow up on previous answers, refine searches, or maintain user preferences.
* **Remediation for `personal_rag`:** Explicit user-controlled conversational memory partitioned into short-term dialogue context and long-term searchable memory.

### 4.8 Security & File System Exposure
* **Problem:** `server.py` creates a socket at `/tmp/mc3_server.sock` with permissive `0o777` permissions; file paths from queries are evaluated without traversal safeguards.
* **Impact:** Unsafe for multi-user or network-accessible deployments.
* **Remediation for `personal_rag`:** Strict path validation, sandboxed file storage, and untrusted-content prompt isolation.

---

## 5. Reusable Components vs. New Modules

| Component | Status in `submission/` | Reusability Plan for `personal_rag/` |
| :--- | :--- | :--- |
| **Normalization Logic** | High | Extract challenge string normalization (`normalize_value`) into `core.utils.text_norm`. |
| **BM25 Lexical Scoring** | Medium | Upgrade from memory-only `BM25Okapi` to SQLite FTS5 for persistent, zero-RAM keyword search. |
| **Multi-hop Linking Logic** | Prototype | Refactor regex entity bridging into an extensible Graph & Entity Extractor. |
| **Defensive File Readers** | Working | Refactor PDF, DOCX, XLSX, CSV, and code parsers into formal parser classes with unified output schemas. |
| **Image Vision Extraction** | Working | Refactor into an optional async background pipeline that preserves page numbers and bounding coordinates. |
| **Hardware Management** | Needs complete rewrite | Build a provider-agnostic device manager supporting CPU, ROCm, CUDA, and API providers. |
| **Storage & Indexing** | Needs complete rewrite | Replace in-memory dictionaries with SQLite (metadata/FTS5) and Qdrant (vectors). |
| **User Interface** | Non-existent | Build a modern Streamlit / FastAPI personal dashboard. |

---

## 6. Proposed Separation: Challenge vs. Personal Platform

To guarantee that the original challenge submission remains intact and functional, the directory layout will strictly isolate the two codebases:

```
c:\Users\HP\Downloads\mc3-starter-kit\
├── mc3-starter-kit/
│   ├── CONTRACT.md                     [FROZEN - Original Challenge Contract]
│   ├── sample-questions.json           [FROZEN - Original Benchmark Suite]
│   ├── selfcheck.py                    [FROZEN - AMD Validation Tool]
│   ├── mc3-corpus/                     [FROZEN - Challenge Sample Corpus]
│   ├── starter/                        [FROZEN - Original Starter]
│   └── submission/                     [FROZEN - Audited 200/200 Challenge Code]
│
├── docs/
│   ├── EXISTING_PROJECT_AUDIT.md       [THIS AUDIT]
│   └── ARCHITECTURE.md                 [Personal Platform Architecture Plan]
│
└── personal_rag/                       [NEW - Personal Advanced RAG Platform]
    ├── pyproject.toml                  # Python package configuration
    ├── README.md                       # Platform documentation
    ├── .env.example                    # Configuration template (CPU default)
    ├── config/                         # YAML/Pydantic configurations
    ├── src/personal_rag/               # Core application package
    │   ├── api/                        # FastAPI REST API endpoints
    │   ├── core/                       # Interfaces, config, hardware detection
    │   ├── parsers/                    # Multimodal document parsers
    │   ├── chunking/                   # Structure-aware & parent-child chunkers
    │   ├── embeddings/                 # Provider-independent embedding interfaces
    │   ├── storage/                    # SQLite metadata/FTS5 + Qdrant vector store
    │   ├── retrieval/                  # Hybrid, BM25, Dense, RRF, MMR, Parent-Child
    │   ├── reranking/                  # Cross-encoder & score fusion
    │   ├── generation/                 # LLM providers (CPU/Local/ROCm/API)
    │   ├── citations/                  # Grounding, attribution, & verification
    │   ├── memory/                     # User-controlled conversational memory
    │   └── ui/                         # Streamlit research assistant frontend
    └── tests/                          # 100% CPU-executable unit & integration tests
```

---

## 7. Next Actions & Staged Implementation Roadmap

Following the non-negotiable rules, development will proceed in clear, independently testable phases:

1. **Phase 1: Foundation & Core Interfaces**
   * Package structure (`personal_rag/`), Pydantic settings, hardware detection (`CPU_ONLY=true`), and SQLite schema.
2. **Phase 2: Ingestion & Document Intelligence**
   * Pluggable parsers (PDF, DOCX, XLSX, CSV, Code, Markdown) with table preservation and hashing.
3. **Phase 3: Chunking & Storage Engine**
   * Recursive, table-aware, and parent-child chunking stored in SQLite + embedded local Qdrant.
4. **Phase 4: Retrieval Engine & Query Router**
   * Dense search, BM25 FTS5 lexical search, Reciprocal Rank Fusion (RRF), and Query Router.
5. **Phase 5: Grounded Generation & Citation Verifier**
   * Provider abstraction (Mock/Local/API/Qwen), strict abstention on insufficient context, and chunk-level citation validation.
6. **Phase 6: Conversational Memory**
   * Explicit session management and inspectable user memory.
7. **Phase 7: Local User Interface**
   * Streamlit dashboard with document uploader, chat interface, passage inspector, and retrieval controls.
8. **Phase 8: Benchmarking & CPU Evaluation**
   * Automated benchmark suite comparing dense vs. lexical vs. hybrid vs. reranked retrieval.
9. **Phase 9: Optional AMD ROCm / GPU Acceleration**
   * Prepared deployment configs for MI300X, strictly dormant until explicitly activated.

*All development in Phases 1 through 8 will run deterministically on CPU with zero GPU cost.*

