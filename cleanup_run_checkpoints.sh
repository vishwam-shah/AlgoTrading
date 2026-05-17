#!/bin/bash
RUN_ID="20260516_091908"
RESULTS="/c/Users/Home/Documents/AI_IN_STOCK_V2/V3/06_results/runs/$RUN_ID"
MODELS="/c/Users/Home/Documents/AI_IN_STOCK_V2/V3/02_models/runs/$RUN_ID"
CLEANED=0
while true; do
  for sym_dir in "$RESULTS"/*/; do
    sym=$(basename "$sym_dir")
    [ "$sym" = "plots" ] && continue
    ckpt="$MODELS/$sym"
    if [ -d "$ckpt" ] && [ -f "$sym_dir/window_results.csv" ]; then
      rm -rf "$ckpt"
      echo "[cleanup] deleted checkpoints for $sym"
      CLEANED=$((CLEANED+1))
    fi
  done
  # Stop when runs dir is empty or results show 100 symbols
  n_results=$(ls "$RESULTS" | grep -v plots | wc -l)
  [ "$n_results" -ge 100 ] && { echo "All 100 done. Total cleaned: $CLEANED"; break; }
  [ ! -d "$MODELS" ] && { echo "Runs dir gone. Done."; break; }
  sleep 30
done
