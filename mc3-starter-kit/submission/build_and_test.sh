#!/bin/bash
# ── Build and Verification Script for MC3 RAG Submission ────────────────────
set -e

IMAGE_NAME="${1:-mc3-rag:latest}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORPUS_DIR="$SCRIPT_DIR/../mc3-corpus"

echo "════════════════════════════════════════════════════════════════════"
echo " MC3 RAG — Build & Verification Pipeline"
echo " Target Image: $IMAGE_NAME"
echo "════════════════════════════════════════════════════════════════════"

# 1. Build Docker image
echo ""
echo "▶ 1. Building Docker image..."
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

# 2. Verify ROCm torch build inside image
echo ""
echo "▶ 2. Verifying ROCm PyTorch installation..."
docker run --rm "$IMAGE_NAME" python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
assert 'rocm' in torch.__version__, 'FATAL: Non-ROCm PyTorch detected!'
print('[OK] Verified ROCm PyTorch build.')
"

# 3. Setup zip-proof corpus cases
echo ""
echo "▶ 3. Re-creating empty directory and unreadable file..."
mkdir -p "$CORPUS_DIR/archive"
chmod 000 "$CORPUS_DIR/vendor/internal_audit.txt" 2>/dev/null || true

# 4. Run official selfcheck
echo ""
echo "▶ 4. Running official selfcheck.py..."
python3 "$SCRIPT_DIR/../selfcheck.py" "$IMAGE_NAME" "$CORPUS_DIR"

# 5. Run full rigorous 200-point evaluator
echo ""
echo "▶ 5. Running 200-point evaluation harness..."
"$SCRIPT_DIR/run_all_questions.sh" "$IMAGE_NAME"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " ALL CHECKS & EVALUATION PASSED! READY FOR SUBMISSION."
echo "════════════════════════════════════════════════════════════════════"
