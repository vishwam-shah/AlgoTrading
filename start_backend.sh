#!/bin/bash
WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKSPACE"

# Kill any other running instances of this script
MY_PID=$$
pgrep -f "bash.*start_backend.sh" 2>/dev/null | grep -v "$MY_PID" | xargs kill 2>/dev/null
sleep 0.5

echo "Starting AlgoTrading backend (auto-restart enabled)..."

while true; do
  # Kill anything holding port 8000 before each attempt
  lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null
  sleep 0.5

  echo "[$(date '+%H:%M:%S')] Backend starting..."
  PYTHONPATH=. venv/bin/uvicorn backend.main:app \
    --reload \
    --reload-dir backend \
    --host 0.0.0.0 \
    --port 8000 \
    --reload-delay 2 \
    --log-level info

  EXIT_CODE=$?
  echo "[$(date '+%H:%M:%S')] Backend exited (code $EXIT_CODE). Restarting in 3s..."
  sleep 3
done
