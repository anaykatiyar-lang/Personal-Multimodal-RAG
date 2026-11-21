# Personal Advanced RAG Platform 🧠

A modular, production-minded personal multimodal knowledge and research assistant platform.

Built with **local data sovereignty**, **zero-cost CPU-default execution**, **adaptive hybrid search**, **parent-child retrieval**, and **strict citation necessity verification**.

---

## 🌟 Key Architecture & Capabilities

* **Multimodal Layout-Aware Ingestion:** Pluggable, fault-tolerant parsers for **PDF, DOCX, XLSX, CSV, Markdown, Code (.py, .log, .txt, .json), and Images (.png, .jpg)**.
* **Contextual Parent-Child Chunking:** Child chunks capture granular facts for precision retrieval; parent chunks expand context envelopes for grounded generation.
* **Table-Aware Processing:** Preserves tabular matrix structure and repeats headers across row segments without cutting through cell values.
* **Hybrid Search with Reciprocal Rank Fusion (RRF):** Fuses dense vector similarity with SQLite FTS5 BM25 keyword rankings:
  $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{lexical}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
* **Adaptive Query Routing & Conversational Intent:** Classifies user query complexity (Exact Lookup, Semantic Discovery, Multi-Hop Synthesis, Table Analysis, Multimodal, Conversational). Natural greetings ("hi", "hello", "who are you") receive immediate, helpful conversational replies without hallucinating or falsely reporting "not in documents".
* **Custom Scratch LLM Support (e.g. 500M parameters):** Plug-and-play provider interface (`CustomScratchLLMProvider`) to plug in models trained from scratch using PyTorch or HuggingFace checkpoints. Runs 100% locally on CPU with zero GPU cost.
* **Strict Citation Necessity & Grounding:** Eliminates hallucinated attributions. Every cited document must verifiably contain the claimed factual entity or linking identifier. Refuses cleanly when evidence is absent.
* **User-Controlled Conversational Memory:** Explicit session tracking and user-inspectable permanent memory, strictly separated from document chunks.
* **Local Web Dashboard & CLI:** Fast desktop interface via Streamlit and CLI commands bound exclusively to `127.0.0.1`.
* **Zero-Cost Credit Protection:** Default `CPU_ONLY=true`. Zero GPU spend, zero external cloud downloads without explicit approval. Optional AMD ROCm / MI300X execution path.

---

## 📁 Repository Layout

```
personal_rag/
├── pyproject.toml              # Modern Python build & package specification
├── README.md                   # This documentation
├── .env.example                # Safe environment template (CPU-only default)
├── .gitignore                  # Strict rules excluding personal data, caches, models
├── config/
│   └── default_config.yaml     # Application defaults
├── docs/
│   └── AMD_MI300X_EXECUTION_PLAN.md # 7-point GPU impact proposal (Dormant)
├── src/personal_rag/
│   ├── core/                   # Hardware policy, Pydantic types, hashing, text norm
│   ├── parsers/                # PDF, DOCX, Spreadsheet, Code/Text, Image parsers
│   ├── chunking/               # Recursive, Parent-Child, and Table chunkers
│   ├── storage/                # SQLite metadata/FTS5 & embedded vector store
│   ├── embeddings/             # Embedding provider abstractions & Mock provider
│   ├── retrieval/              # Dense, Lexical, Hybrid RRF, Parent expansion, Router
│   ├── reranking/              # Cross-Encoder reranker & score fusion
│   ├── generation/             # LLM provider abstractions & grounded engine
│   ├── citations/              # Strict citation necessity verifier
│   ├── memory/                 # Session history & persistent user memory
│   ├── evaluation/             # Retrieval metrics & benchmark runner
│   ├── cli/                    # Command-line interface
│   ├── ui/                     # Streamlit web dashboard
│   └── service.py              # Central orchestrator service
└── tests/
    ├── conftest.py             # Deterministic CPU test fixtures
    ├── unit/                   # 100% CPU unit tests
    └── integration/            # End-to-end service integration tests
```

---

## 🚀 Quickstart & Setup

### 1. Installation

Create a virtual environment and install in editable mode:

```bash
cd personal_rag
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev,documents,search,embeddings,ui]"
```

Copy the environment template:
```bash
cp .env.example .env
```

---

## 🛠️ CLI Commands

### Ingest Documents
Ingest a directory of mixed files incrementally (skips unchanged files via SHA-256):
```bash
personal-rag ingest ../mc3-corpus
```
To remove deleted files from the catalog:
```bash
personal-rag ingest ../mc3-corpus --prune
```

### Ask Questions
Ask a question against your local documents:
```bash
personal-rag ask "What is the maximum junction temperature of the TQ-40?"
```

Output:
```text
════════════════════════════════════════════════════════════════════
 Question: What is the maximum junction temperature of the TQ-40?
 Strategy: adaptive_hybrid_exact_lookup (Total: 4.2 ms)
────────────────────────────────────────────────────────────────────
 Answer: 94
 Confidence: 0.95
 Citations:
   - specs/tq40_datasheet_r2.pdf (Page 1)
     Excerpt: The maximum junction temperature of the TQ-40 is 94 C...
════════════════════════════════════════════════════════════════════
```

### Inspect Hardware & Credit Policy
Verify that GPU credits are protected and execution is running safely on CPU:
```bash
personal-rag hw
```

### List Indexed Documents
```bash
personal-rag docs
```

### Database Backup & Restore
Create an atomic online backup snapshot:
```bash
personal-rag backup ./backup_knowledge_base.db
personal-rag restore ./backup_knowledge_base.db
```

---

## 🤖 Using Your Own 500M Parameter LLM (From Scratch)

You can easily connect your own model trained from scratch (e.g. 500M parameters) to serve as the generation engine:

### Option A: In the Streamlit Web UI
1. Launch the dashboard (`streamlit run src/personal_rag/ui/app.py`).
2. In the sidebar under **Generation Engine**, select **"Custom Scratch LLM (500M / PyTorch / HF)"**.
3. Enter your model path (e.g. `./models/my_500m_model` or `path/to/checkpoint.pt`).
4. Click **Load Custom Model**. A 500M model uses only ~1 GB of RAM and executes cleanly on CPU with zero GPU cost!

### Option B: In Python Code / Service
```python
from personal_rag.service import PersonalRAGService
from personal_rag.generation.custom_llm_provider import CustomScratchLLMProvider

service = PersonalRAGService()

# Point to your custom PyTorch checkpoint or Hugging Face directory
service.model_provider = CustomScratchLLMProvider(
    model_path="./models/my_500m_model",
    device="cpu"  # or "cuda" / "rocm" if approved
)
service.generation_engine.provider = service.model_provider

# Ask questions with your custom LLM grounded on your documents
result = service.answer_query("What is the maximum junction temperature of the TQ-40?")
print(result.answer)
print(result.citations)
```

---

## 🖥️ Local Web UI (Streamlit)

Launch the interactive dashboard on `http://127.0.0.1:8501`:

```bash
streamlit run src/personal_rag/ui/app.py
```

### UI Features:
1. **Interactive Research Chat:** Conversational turns with expandable citation drawers displaying exact excerpts and page numbers.
2. **Document Drag-and-Drop:** Ingest new PDFs, spreadsheets, and notes right from the browser.
3. **Knowledge Base Browser:** View document sizes, page counts, and indexing timestamps.
4. **Retrieval Controls:** Live sliders for dense vs. lexical weights, candidate Top-K, and parent expansion.
5. **Persistent User Memory:** Inspect, add, and delete personal preferences and research topics.
6. **Hardware Telemetry:** Live monitor of compute mode and zero-cost policy enforcement.

---

## 🧪 Testing & Evaluation

### Run Fast Deterministic CPU Tests
```bash
pytest tests/unit tests/integration -v
```

### Run Benchmark Suite
Evaluate retrieval strategies (Dense vs. Lexical vs. Hybrid RRF):
```bash
python -c "
from personal_rag.service import PersonalRAGService
from personal_rag.evaluation.benchmarker import RAGBenchmarker
import json

service = PersonalRAGService()
with open('../sample-questions.json') as f:
    suite = json.load(f)['queries']

bench = RAGBenchmarker(service)
report = bench.evaluate_suite(suite)
print(report['summary_markdown'])
"
```

---

## 🛡️ Privacy & Non-Destructive Guarantee

* **Preservation of Challenge Files:** The original AMD challenge submission in `submission/` and starter in `starter/` remain completely untouched, isolated, and runnable.
* **Localhost Only:** Services bind strictly to `127.0.0.1`.
* **Zero Network Leaks:** Documents and embeddings are stored locally in SQLite (`./data/knowledge_base.db`) and embedded Qdrant (`./data/qdrant`). No third-party network requests are made unless explicitly configured.

