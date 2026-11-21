# Multimodal-RAG: Advanced Personal Knowledge & Research Platform 🧠🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Fast & Zero-Cost](https://img.shields.io/badge/CPU--First-Zero--Cost-brightgreen.svg)](#zero-cost-credit-protection)
[![Tests Passing](https://img.shields.io/badge/tests-35%20passed-success.svg)](#testing--verification)
[![Streamlit UI](https://img.shields.io/badge/UI-Streamlit%20Dashboard-FF4B4B.svg)](#-local-web-dashboard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, local-first **Multimodal Retrieval-Augmented Generation (RAG) Platform** designed for personal knowledge synthesis, deep technical document analysis, and zero-hallucination research. 

Evolved from an AMD ROCm Mini-Challenge 3 multimodal submission into an extensible, production-minded research assistant that runs **100% locally on CPU at zero compute cost**, with seamless plug-in support for custom LLMs (including models trained from scratch, e.g. 500M parameters) and optional AMD ROCm / MI300X acceleration.

---

## 🏗️ Repository Architecture

This repository maintains a strict separation between the production platform and benchmark artifacts:

```
Multimodal-RAG/
├── personal_rag/               # 🧠 Production Personal RAG Platform
│   ├── src/personal_rag/       # Modular application core
│   │   ├── parsers/            # PDF, DOCX, XLSX/CSV, Markdown, Code, Image
│   │   ├── chunking/           # Contextual Parent-Child, Table-aware, Recursive
│   │   ├── storage/            # SQLite metadata/FTS5 & Qdrant vector store
│   │   ├── embeddings/         # BAAI/bge-small & Deterministic Mock Embeddings
│   │   ├── retrieval/          # Dense, BM25 Lexical, Hybrid RRF, Parent Expansion
│   │   ├── generation/         # Extractive Synthesizer, OpenAI/Ollama, Custom Scratch 500M
│   │   ├── citations/          # Strict token-overlap & necessity verifier
│   │   ├── memory/             # Multi-turn dialogue context & explicit user memory
│   │   ├── ui/                 # Interactive Streamlit Web Dashboard
│   │   ├── cli/                # Terminal management interface
│   │   └── service.py          # Unified service orchestrator
│   ├── tests/                  # 35 deterministic unit & integration tests
│   └── README.md               # Detailed personal platform documentation
│
├── mc3-starter-kit/            # 🏆 AMD Mini-Challenge 3 Benchmark Suite
│   ├── submission/             # Evaluated challenge submission (200/200 sample score)
│   ├── starter/                # Evaluation harness & baseline scripts
│   ├── CONTRACT.md             # Benchmark specification contract
│   └── selfcheck.py            # Local contract validation script
│
└── docs/                       # Architectural audits & GPU deployment plans
    ├── EXISTING_PROJECT_AUDIT.md
    └── AMD_MI300X_EXECUTION_PLAN.md
```

---

## ✨ Core Features & Innovations

### 1. 🔄 Adaptive Hybrid Search with Reciprocal Rank Fusion (RRF)
Combines dense vector semantics with exact lexical keyword matching (SQLite FTS5 BM25) dynamically weighted according to query intent:
$$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{lexical}\}} \frac{w_m}{k + \text{rank}_m(d)}$$

### 2. 🎯 Intent Routing & Conversational Intelligence
An adaptive query router classifies user intent into:
* **Conversational Intent:** Greetings ("hi", "hello", "who are you") receive immediate, helpful conversational replies in under 1ms without false document refusals.
* **Exact Lookup:** Part numbers, error codes, ticket IDs (boosts lexical search).
* **Semantic Discovery:** Broad conceptual explanations (boosts dense retrieval).
* **Multi-Hop Synthesis:** Traverses related files across incidents, bug databases, and release notes.
* **Table & Numerical:** Preserves row/column matrices for pricing and specifications.
* **Multimodal:** Pinouts, schematics, asset labels, and visual diagrams.

### 3. 🛡️ Strict Citation Necessity & Anti-Hallucination
Never hallucinates citations. Every cited source is verified with:
* Exact entity/value substring matching against retrieved chunks.
* Token-overlap verification for natural language answers.
* Zero-refusal policy: cleanly abstains when the corpus lacks verified proof.

### 4. 🧩 Contextual Parent-Child & Table Chunking
* **Parent-Child Chunking:** Child chunks (~400 chars) are indexed for precision retrieval; parent chunks (~1500 chars) are injected into the prompt for rich context.
* **Table Chunking:** Slices tabular rows while preserving markdown column headers across all segments.

### 5. 🤖 Bring-Your-Own-LLM (From Scratch 500M to Ollama & Cloud)
* **Local Extractive Synthesizer (Default):** Instant, grounded answer synthesis directly on CPU with 0 tokens and 0 API calls.
* **Custom Scratch LLM Provider:** Plug in your own custom-trained models (e.g. 500M parameters) via PyTorch `.pt` or Hugging Face directory.
* **Local Ollama / vLLM:** Connect to local endpoints (`http://127.0.0.1:11434/v1`).
* **OpenAI Compatible:** Connect to any custom or OpenAI API endpoint.

### 6. 🧠 User-Controlled Memory
Explicit separation between short-term dialogue history and permanent, inspectable user preferences. Users can inspect, save, and delete memories directly in the UI.

---

## ⚡ Quickstart Guide

### Prerequisites
* Python 3.10 or higher (Tested on Python 3.14 on Windows & Linux)
* Git

### 1. Installation
```powershell
# Navigate into the personal platform directory
cd personal_rag

# Install editable package with all capabilities
py -m pip install -e ".[dev,documents,search,embeddings,ui]"
```

### 2. Ingest Your Documents
Ingest any directory of PDFs, Word files, spreadsheets, markdown, code, and logs:
```powershell
py -m personal_rag ingest ../mc3-starter-kit/mc3-corpus
```

### 3. Ask Questions via CLI
```powershell
# Technical factual query with verified citations
py -m personal_rag ask "What is the maximum junction temperature of the TQ-40?"

# Multi-hop bug and log correlation
py -m personal_rag ask "Which firmware version fixed ticket ORR-1847?"

# Natural conversational query
py -m personal_rag ask "hii"
```

### 4. Launch Interactive Web Dashboard
```powershell
py -m streamlit run src/personal_rag/ui/app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 🤖 Using Your Own 500M Parameter LLM

Have a model you trained from scratch? Connect it in two lines:

```python
from personal_rag.service import PersonalRAGService
from personal_rag.generation.custom_llm_provider import CustomScratchLLMProvider

service = PersonalRAGService()

# Load your custom 500M checkpoint (runs on CPU using ~1 GB RAM!)
service.model_provider = CustomScratchLLMProvider(
    model_path="./models/my_500m_model",
    device="cpu"
)
service.generation_engine.provider = service.model_provider

# Query grounded across your documents
result = service.answer_query("What is the default batch timeout?")
print("Answer:", result.answer)
print("Citations:", [c.source_path for c in result.citations])
```

---

## 🧪 Testing & Verification

Run the comprehensive 35-test deterministic test suite:
```powershell
cd personal_rag
py -m pytest tests/unit tests/integration -v
```
Output:
```text
======================== 35 passed in 2.76s ========================
```

---

## 🛡️ Zero-Cost Credit Protection Policy

By default, the platform runs strictly in **Zero-Cost CPU Mode**:
```ini
CPU_ONLY=true
GPU_EXECUTION_APPROVED=false
CLOUD_INSTANCE_AUTOSTART=false
MODEL_DOWNLOAD_AUTOSTART=false
```
* **No accidental cloud compute spend**
* **No unauthorized model downloads**
* Optional AMD ROCm / MI300X execution path prepared for heavy batch workloads when explicitly authorized.

---

## 📜 License
MIT License. Built for personal research and production knowledge retrieval.

