#!/usr/bin/env bash
# Watchdog: runs pipeline until all 100 stocks complete, auto-resumes on crash.
set -e

RUN_ID="20260307_141956"
WORKERS=2
PIPELINE="V3/07_pipeline/run_pipeline.py"
LOG="V3/06_results/pipeline_run.log"
RUN_DIR="V3/06_results/runs/$RUN_ID"
TOTAL=100

cd /c/Users/Home/Documents/AI_IN_STOCK_V2

get_completed() {
    python -c "
from pathlib import Path
run_dir = Path('$RUN_DIR')
completed = list(run_dir.rglob('window_results.csv'))
print(len(completed))
" 2>/dev/null || echo 0
}

attempt=0
while true; do
    completed=$(get_completed)
    echo "[watchdog] Attempt $((++attempt)) — $completed/$TOTAL completed" | tee -a "$LOG"

    if [ "$completed" -ge "$TOTAL" ]; then
        echo "[watchdog] All $TOTAL stocks complete!" | tee -a "$LOG"
        break
    fi

    echo "[watchdog] Launching pipeline (--resume $RUN_ID --workers $WORKERS) ..." | tee -a "$LOG"
    python "$PIPELINE" --resume "$RUN_ID" --workers "$WORKERS" >> "$LOG" 2>&1 || true

    completed=$(get_completed)
    echo "[watchdog] After run: $completed/$TOTAL completed" | tee -a "$LOG"

    if [ "$completed" -ge "$TOTAL" ]; then
        echo "[watchdog] All $TOTAL stocks complete!" | tee -a "$LOG"
        break
    fi

    echo "[watchdog] Pipeline exited before finishing. Waiting 10s then retrying..." | tee -a "$LOG"
    sleep 10
done
