#!/bin/bash
# ── Comprehensive Evaluation Test for MC3 RAG Submission ────────────────────
# Evaluates BOTH answers and exact citation sets as mandated by the challenge contract.
set -e

IMAGE_NAME="${1:-mc3-rag:latest}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORPUS_DIR="$SCRIPT_DIR/../mc3-corpus"
QUESTIONS_FILE="$SCRIPT_DIR/../sample-questions.json"
CONTAINER_NAME="mc3-eval-$(head -c 8 /dev/urandom | xxd -p 2>/dev/null || echo $RANDOM)"

echo "════════════════════════════════════════════════════════════════════"
echo " AMD Mini-Challenge 3 — Rigorous Evaluation Harness"
echo " Image: $IMAGE_NAME"
echo "════════════════════════════════════════════════════════════════════"

# Clean up container on exit
trap 'docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true' EXIT

# 1. Start container with --network none (strictly enforced by competition)
echo ""
echo "▶ 1. Starting container with --network none..."
docker run -d --network none --name "$CONTAINER_NAME" "$IMAGE_NAME"

# 2. Wait for GPU server to become healthy
echo "▶ 2. Waiting for resident GPU server to initialize..."
HEALTH_CHECK=0
for attempt in $(seq 1 45); do
    if docker exec "$CONTAINER_NAME" python3 -c "
import socket, json, struct
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect('/tmp/mc3_server.sock')
req = json.dumps({'action': 'health'}).encode('utf-8')
s.sendall(struct.pack('>I', len(req)) + req)
header = s.recv(4)
length = struct.unpack('>I', header)[0]
res = json.loads(s.recv(length).decode('utf-8'))
assert res.get('status') == 'ok'
" 2>/dev/null; then
        echo "  [OK] Resident GPU server is healthy and responding on /tmp/mc3_server.sock"
        HEALTH_CHECK=1
        break
    fi
    sleep 2
done

if [ "$HEALTH_CHECK" -ne 1 ]; then
    echo "  [FAIL] Server failed to start or died. Check container logs:"
    docker logs "$CONTAINER_NAME"
    exit 1
fi

# 3. Setup corpus and run indexing pass
echo ""
echo "▶ 3. Copying corpus and running --index phase..."
docker exec "$CONTAINER_NAME" mkdir -p /app/corpus /app/output
docker cp "$CORPUS_DIR/." "$CONTAINER_NAME:/app/corpus"

t0=$(date +%s)
docker exec "$CONTAINER_NAME" python3 /app/app.py --index /app/corpus
t1=$(date +%s)
echo "  [OK] Indexing completed in $((t1 - t0))s (Startup budget: 600s)"

# 4. Evaluate each query with exact answer and citation scoring
echo ""
echo "▶ 4. Evaluating questions against sample-questions.json..."
echo ""

python3 - << 'EOF'
import json
import re
import subprocess
import sys
import time

QUESTIONS_FILE = "sample-questions.json"
CONTAINER_NAME = sys.argv[1] if len(sys.argv) > 1 else ""

def normalize(val: str) -> str:
    if not val:
        return ""
    v = val.upper()
    v = re.sub(r"[\s\.\°]", "", v)
    return v

with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    suite = json.load(f)

queries = suite.get("queries", [])
total_score = 0
max_score = len(queries) * 20

print(f"{'#':<3} {'Query':<55} {'Ans':<5} {'Cit':<5} {'Points':<6}")
print("-" * 75)

for q in queries:
    n = q["n"]
    qid = f"query_{n:02d}"
    query_text = q["query"]
    expected_ans = q["expected_answer"]
    aliases = [expected_ans] + q.get("answer_aliases", [])
    expected_cits = sorted(q.get("expected_citations", []))

    t_start = time.time()
    proc = subprocess.run([
        "docker", "exec", CONTAINER_NAME,
        "python3", "/app/app.py",
        "--corpus", "/app/corpus",
        "--query-id", qid,
        "--query", query_text
    ], capture_output=True, text=True, timeout=30)
    elapsed = time.time() - t_start

    # Read output json
    out_proc = subprocess.run([
        "docker", "exec", CONTAINER_NAME,
        "cat", f"/app/output/{qid}_output.json"
    ], capture_output=True, text=True)

    ans_ok = False
    cit_ok = False
    actual_ans = ""
    actual_cits = []

    if out_proc.returncode == 0:
        try:
            payload = json.loads(out_proc.stdout)
            actual_ans = payload.get("answer", "")
            actual_cits = sorted(payload.get("citations", []))

            # Match normalized answer
            norm_actual = normalize(actual_ans)
            for target in aliases:
                if norm_actual == normalize(target):
                    ans_ok = True
                    break

            # Exact citation set match
            if [c.strip("/") for c in actual_cits] == [c.strip("/") for c in expected_cits]:
                cit_ok = True
        except Exception as e:
            pass

    points = 20 if (ans_ok and cit_ok) else 0
    total_score += points

    ans_mark = "PASS" if ans_ok else "FAIL"
    cit_mark = "PASS" if cit_ok else "FAIL"
    print(f"{n:<3} {query_text[:53]:<55} {ans_mark:<5} {cit_mark:<5} {points}/20 ({elapsed:.1f}s)")
    if points == 0:
        print(f"    Expected Ans: {expected_ans!r} | Got: {actual_ans!r}")
        print(f"    Expected Cit: {expected_cits} | Got: {actual_cits}")

print("=" * 75)
print(f"Final Score: {total_score} / {max_score}")
if total_score == max_score:
    print("SUCCESS: Perfect score achieved on sample set.")
    sys.exit(0)
else:
    print("WARNING: Some questions failed evaluation.")
    sys.exit(1)
EOF "$CONTAINER_NAME"
