#!/bin/bash
# =============================================================================
# setup_cron.sh — Install daily algo-trading cron jobs
# =============================================================================
# Run once:  bash V3/05_live_trading/setup_cron.sh
#
# Installs 3 cron jobs (IST = UTC+5:30):
#   18:00 IST Mon–Fri → evening run  (predictions + orders)
#   09:00 IST Mon–Fri → morning run  (place orders on Angel One)
#   15:45 IST Mon–Fri → reconcile    (verify fills, update history)
# =============================================================================

WORKSPACE="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$WORKSPACE/venv/bin/python"
RUNNER="$WORKSPACE/V3/05_live_trading/daily_runner.py"
LOG_DIR="$WORKSPACE/V3/05_live_trading/logs"
mkdir -p "$LOG_DIR"

# Verify runner exists
if [ ! -f "$RUNNER" ]; then
  echo "ERROR: daily_runner.py not found at $RUNNER"
  exit 1
fi
if [ ! -f "$PYTHON" ]; then
  echo "ERROR: Python venv not found at $PYTHON"
  exit 1
fi

echo "Workspace : $WORKSPACE"
echo "Python    : $PYTHON"
echo "Runner    : $RUNNER"
echo ""

# Build cron entries
# IST = UTC+5:30 → 18:00 IST = 12:30 UTC | 09:00 IST = 03:30 UTC | 15:45 IST = 10:15 UTC
EVENING_JOB="30 12 * * 1-5 $PYTHON $RUNNER --mode evening --capital 500000 >> $LOG_DIR/evening.log 2>&1"
MORNING_JOB="30 3  * * 1-5 $PYTHON $RUNNER --mode morning --capital 500000 --paper >> $LOG_DIR/morning.log 2>&1"
RECONCILE_JOB="15 10 * * 1-5 $PYTHON $RUNNER --mode reconcile >> $LOG_DIR/reconcile.log 2>&1"

# Remove old entries, add new ones
(crontab -l 2>/dev/null | grep -v "daily_runner.py"; \
  echo "$EVENING_JOB"; \
  echo "$MORNING_JOB"; \
  echo "$RECONCILE_JOB") | crontab -

echo "Cron jobs installed:"
crontab -l | grep "daily_runner"
echo ""
echo "NOTE: Morning run uses --paper by default."
echo "      Remove --paper from morning job when ready for live trading."
echo ""
echo "Log files:"
echo "  $LOG_DIR/evening.log"
echo "  $LOG_DIR/morning.log"
echo "  $LOG_DIR/reconcile.log"
