#!/usr/bin/env bash
#
# dev-reset.sh — automates the full local dev cycle for PromptOps:
#   1. Kill any existing uvicorn process on port 8000
#   2. Wipe and rebuild the vector store (chroma_db) from docs/
#   3. Restart the FastAPI server in the background
#   4. Wait for it to be healthy
#   5. Fire a smoke-test curl against /ask
#
# Usage:
#   chmod +x dev-reset.sh   (one-time)
#   ./dev-reset.sh
#
# Run this from the promptops/ project root.

set -e  # exit immediately if any command fails

QUESTION="${1:-What is Phase 1 of PromptOps?}"  # optional: pass a custom question as arg 1
PORT=8000

echo "==> Step 1: Killing any process on port $PORT"
PID=$(lsof -ti :$PORT || true)
if [ -n "$PID" ]; then
    kill -9 $PID
    echo "    Killed PID $PID"
else
    echo "    Nothing running on port $PORT"
fi

echo "==> Step 2: Rebuilding vector store"
rm -rf rag/chroma_db
(cd rag && python3 ingestor.py)

echo "==> Step 3: Starting FastAPI server in background"
uvicorn main:app --reload > server.log 2>&1 &
SERVER_PID=$!
echo "    Server starting (PID $SERVER_PID), logs -> server.log"

echo "==> Step 4: Waiting for server to become healthy"
for i in $(seq 1 20); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "    Server is up"
        break
    fi
    sleep 0.5
done

echo "==> Step 5: Smoke-test curl"
echo "    Question: $QUESTION"
curl -s -X POST http://localhost:$PORT/ask \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"$QUESTION\"}" | python3 -m json.tool

echo ""
echo "==> Step 6: Running eval suite"
set +e  # don't let a failing eval kill the script before printing the summary below
python3 evals/run_evals.py
EVAL_EXIT_CODE=$?
set -e

echo ""
echo "==> Done. Server is running in the background (PID $SERVER_PID)."
echo "    Tail logs with: tail -f server.log"
echo "    Stop it with:   kill $SERVER_PID"