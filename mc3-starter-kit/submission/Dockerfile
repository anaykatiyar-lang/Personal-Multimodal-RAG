# ── MC3 RAG Submission ────────────────────────────────────────────────────────
# Built on the mandated ROCm base image. Layer identity is strictly checked.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# ── System dependencies for document parsing ─────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────────────────────────
# CRITICAL: Packages touching torch MUST use --no-deps to prevent pip from
# overwriting the base image's ROCm PyTorch 2.13.0 with a CUDA build.
# The base image already ships: torch (ROCm), torchvision (ROCm), PIL.

# Step 1: Install packages that DO NOT depend on torch
RUN pip install --no-cache-dir \
        python-docx \
        openpyxl \
        pymupdf \
        numpy \
        scikit-learn \
        rank-bm25 \
        pillow

# Step 2: Install torch-dependent packages with --no-deps
RUN pip install --no-cache-dir --no-deps \
        transformers==4.52.4 \
        tokenizers \
        safetensors \
        accelerate \
        huggingface-hub \
        qwen-vl-utils \
        sentence-transformers

# Step 3: Supporting packages that don't pull torch
RUN pip install --no-cache-dir \
        regex \
        requests \
        tqdm \
        pyyaml \
        packaging \
        filelock \
        psutil \
        jinja2

# ── Verify ROCm torch is intact ─────────────────────────────────────────────
RUN python3 -c "\
import torch; \
v = torch.__version__; \
assert 'rocm' in v, f'FATAL: CUDA torch detected after pip installs: {v}'; \
print(f'VERIFIED: torch {v}')"

# ── Download model weights INTO the image ────────────────────────────────────
# No network at evaluation time (--network none). Weights must be baked in.
# Qwen2.5-VL-7B-Instruct: ~15 GiB (bf16) fits well within the 30 GiB limit.
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='/models/qwen2.5-vl-7b')"

# BGE-small-en-v1.5: ~130 MB embedding model
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-small-en-v1.5', local_dir='/models/bge-small')"

# ── Copy application code ───────────────────────────────────────────────────
COPY app/ /app/

RUN mkdir -p /app/corpus /app/output /app/index

# ── Final environment verification ──────────────────────────────────────────
RUN python3 -c "\
import torch; \
assert 'rocm' in torch.__version__, 'Not ROCm build'; \
import transformers; \
import sentence_transformers; \
import rank_bm25; \
import docx; \
import openpyxl; \
import fitz; \
print('All critical imports verified successfully.')"

# ── Startup ─────────────────────────────────────────────────────────────────
# Start server.py as the primary process. If the server crashes or fails on
# startup (e.g. missing GPU), the container exits immediately rather than
# falsely hanging alive.
CMD ["/bin/bash", "-c", "python3 /app/server.py & SERVER_PID=$!; wait $SERVER_PID"]
