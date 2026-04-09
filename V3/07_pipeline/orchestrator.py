"""
================================================================================
V3 ALGO TRADING PIPELINE — ORCHESTRATOR
================================================================================
Entry point for the full pipeline. Coordinates all steps autonomously:

  Step 1 — Download      : NSE OHLCV + USD/INR + Global cues (incremental)
  Step 2 — Features      : 260+ features (cached, stale-check)
  Step 3 — Train         : Walk-forward ensemble (LightGBM + XGBoost + 7 DL)
  Step 4 — Evaluate      : Per-symbol metrics + plots + production model save
  Step 5 — Predict       : Next-day directional signals for all symbols
  Step 6 — Backtest      : HRP portfolio simulation (optional)

Usage:
  python orchestrator.py                              # all 100 stocks
  python orchestrator.py --symbols SBIN HDFCBANK     # specific stocks
  python orchestrator.py --serial                     # single-process debug
  python orchestrator.py --resume 20260409_120000     # continue crashed run
  python orchestrator.py --backtest                   # run HRP backtest after training
================================================================================
"""

from __future__ import annotations

import argparse
import concurrent.futures as _cf
import gc
import json
import logging
import multiprocessing as _mp
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Suppress TF/protobuf noise before any imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── V3 root + config ──────────────────────────────────────────────────────────
_PIPE_DIR = Path(__file__).resolve().parent
_V3_ROOT  = _PIPE_DIR.parent
sys.path.insert(0, str(_V3_ROOT))
sys.path.insert(0, str(_V3_ROOT / "02_models"))

from config_v3 import (  # type: ignore  # noqa: E402
    SYMBOLS, RESULTS_RUNS_DIR, MODELS_RUNS_DIR,
    DATA_START_DATE, MIN_MOVE, CONFIDENCE_THRESHOLD,
    INITIAL_TRAIN_RATIO, EXPANSION_STEP, MAX_TRAIN_RATIO,
    N_TOP_FEATURES, LOG_DIR,
)

# ── Step modules ──────────────────────────────────────────────────────────────
from steps.download import (  # type: ignore  # noqa: E402
    download_all_symbols, download_usdinr, download_global_cues,
)
from steps.evaluate import (  # type: ignore  # noqa: E402
    run_symbol, run_symbol_worker, worker_init, flush_aggregate_csvs,
    plot_cross_stock_comparison, plot_signal_heatmap,
)
from steps.predict import predict_next_day, predict_worker  # type: ignore  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_loguru(run_id: str) -> None:
    """Configure loguru: file-only (errors still go to stderr)."""
    try:
        from loguru import logger
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.remove()
        logger.add(
            str(LOG_DIR / f"run_{run_id}.log"),
            level="INFO", rotation="50 MB", retention="30 days",
            compression="gz", enqueue=True,
        )
        logger.add(sys.stderr, level="ERROR")
    except ImportError:
        pass   # loguru optional; fall back to print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # ── CLI ───────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="V3 AlgoTrading Pipeline Orchestrator")
    parser.add_argument("--workers",  type=int, default=None,
                        help="Parallel workers (default: auto = min(cpu, 3))")
    parser.add_argument("--symbols",  nargs="+", default=None,
                        help="Override stock list (default: all 100)")
    parser.add_argument("--serial",   action="store_true",
                        help="Force serial mode (single process, easier debugging)")
    parser.add_argument("--resume",   metavar="RUN_ID", default=None,
                        help="Resume a crashed run — skips completed symbols")
    parser.add_argument("--backtest", action="store_true",
                        help="Run HRP portfolio backtest after training")
    parser.add_argument("--force-features", action="store_true",
                        help="Recompute features even if cache is fresh")
    args = parser.parse_args()

    symbols_to_run: List[str] = args.symbols if args.symbols else list(SYMBOLS)
    run_id = args.resume or datetime.now().strftime("%Y%m%d_%H%M%S")
    t0     = time.time()

    setup_loguru(run_id)

    # ── Worker / CPU math ─────────────────────────────────────────────────────
    n_cpu = os.cpu_count() or 4
    if args.serial:
        n_workers          = 1
        n_jobs_per_worker  = -1
    else:
        n_workers         = args.workers or min(n_cpu, 3, len(symbols_to_run))
        n_jobs_per_worker = max(1, n_cpu // n_workers)

    print("=" * 70)
    print("  AI STOCK PREDICTION — PRODUCTION PIPELINE V3")
    print(f"  Run ID    : {run_id}")
    print(f"  Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Stocks    : {len(symbols_to_run)}  "
          f"({', '.join(symbols_to_run[:10])}{'...' if len(symbols_to_run) > 10 else ''})")
    print(f"  Workers   : {n_workers}  (n_jobs/model={n_jobs_per_worker},  cpu_count={n_cpu})")
    print(f"  MIN_MOVE  : {MIN_MOVE*100:.1f}%  |  CONF≥{CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  Data from : {DATA_START_DATE} → today  (incremental)")
    print(f"  Windows   : {INITIAL_TRAIN_RATIO:.0%} → {MAX_TRAIN_RATIO:.0%} (step {EXPANSION_STEP:.0%})")
    print(f"  Log       : {LOG_DIR.relative_to(_V3_ROOT)}/run_{run_id}.log")
    print("=" * 70)

    model_run_path  = MODELS_RUNS_DIR  / run_id
    result_run_path = RESULTS_RUNS_DIR / run_id
    run_plot_path   = result_run_path  / "plots"
    for p in [model_run_path, result_run_path, run_plot_path]:
        p.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1 — DOWNLOAD
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  STEP 1 — DATA DOWNLOAD  (incremental)")
    print(f"{'='*70}")

    raw_data = download_all_symbols(symbols_to_run)
    if not raw_data:
        print("\n✗ No data downloaded. Exiting.")
        return

    print(f"\n  Downloading auxiliary data ...")
    usdinr_df      = download_usdinr()
    global_cues_df = download_global_cues()
    if usdinr_df      is not None: print(f"  USD/INR     : {len(usdinr_df)} rows")
    if global_cues_df is not None: print(f"  Global cues : {len(global_cues_df)} rows  "
                                         f"cols={list(global_cues_df.columns[:6])}...")

    market_df: Optional[pd.DataFrame] = None

    # Build peer return series for sector cross-sectional features
    peer_returns: Dict[str, pd.Series] = {
        sym: df.set_index("date")["close"].pct_change()
        for sym, df in raw_data.items()
    }

    # ══════════════════════════════════════════════════════════════════════════
    #  STEPS 2–4 — FEATURES + TRAINING + EVALUATION
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STEPS 2–4 — FEATURES + TRAINING + EVALUATION  ({n_workers} workers)")
    print(f"  Per-symbol logs → {result_run_path.relative_to(_V3_ROOT)}/<symbol>/run.log")
    print(f"{'='*70}")

    summary_rows: List[Dict] = []
    symbols_with_data = [s for s in symbols_to_run if s in raw_data]

    # Resume: skip already-completed symbols
    if args.resume:
        pending = []
        for sym in symbols_with_data:
            done_marker = result_run_path / sym / "window_results.csv"
            if done_marker.exists():
                print(f"  [skip] {sym} — already completed")
                sr_path = result_run_path / sym / "summary_row.json"
                if sr_path.exists():
                    try:
                        with open(sr_path) as _f:
                            summary_rows.append(json.load(_f))
                    except Exception:
                        summary_rows.append({"symbol": sym, "status": "skipped_resume"})
                else:
                    summary_rows.append({"symbol": sym, "status": "skipped_resume"})
            else:
                pending.append(sym)
        symbols_with_data = pending
        print(f"  [RESUME] {len(pending)} symbols remaining")

    n_total = len(symbols_with_data)
    n_done  = 0

    if n_workers <= 1 or args.serial:
        # ── Serial mode ───────────────────────────────────────────────────────
        for symbol in symbols_with_data:
            try:
                result = run_symbol(
                    symbol=symbol, raw_df=raw_data[symbol], run_id=run_id,
                    model_run_path=model_run_path, result_run_path=result_run_path,
                    peer_returns=peer_returns, market_df=market_df,
                    usdinr_df=usdinr_df, global_cues_df=global_cues_df,
                    force_recompute_features=args.force_features,
                )
                summary_rows.append(result)
            except Exception as exc:
                print(f"\n  {symbol}: ✗ ERROR — {exc}"); traceback.print_exc()
                summary_rows.append({"symbol": symbol, "status": "error", "reason": str(exc)})
            n_done += 1
            print(f"  [{n_done}/{n_total}] {symbol} done")
            flush_aggregate_csvs(result_run_path)

    else:
        # ── Parallel batched mode — fresh pool per batch avoids TF memory leak ─
        BATCH_SIZE = n_workers * 3
        mp_ctx     = _mp.get_context("spawn")
        batches    = [symbols_with_data[i:i+BATCH_SIZE]
                      for i in range(0, len(symbols_with_data), BATCH_SIZE)]
        print(f"  [BATCH] {len(batches)} batches of ≤{BATCH_SIZE} "
              f"({n_workers} workers, fresh pool per batch)")

        for b_idx, batch in enumerate(batches, 1):
            print(f"\n  ── Batch {b_idx}/{len(batches)}: {', '.join(batch)} ──")
            with _cf.ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=mp_ctx,
                initializer=worker_init,
                initargs=(n_jobs_per_worker,),
            ) as pool:
                future_to_sym = {
                    pool.submit(
                        run_symbol_worker,
                        sym, raw_data[sym], run_id,
                        model_run_path, result_run_path,
                        peer_returns, market_df, usdinr_df, global_cues_df,
                    ): sym
                    for sym in batch
                }
                for fut in _cf.as_completed(future_to_sym):
                    sym    = future_to_sym[fut]
                    n_done += 1
                    try:
                        result = fut.result()
                        summary_rows.append(result)
                        status = result.get("status", "?")
                        acc    = f"acc={result['oos_accuracy']:.2%}" if status == "ok" else status
                        print(f"  [{n_done:>3}/{n_total}] ✓ {sym:<14} {acc}  "
                              f"(log → {result_run_path.relative_to(_V3_ROOT)}/{sym}/run.log)")
                    except Exception as exc:
                        print(f"  [{n_done:>3}/{n_total}] ✗ {sym}: {exc}")
                        summary_rows.append({"symbol": sym, "status": "error", "reason": str(exc)})
                    flush_aggregate_csvs(result_run_path)
            gc.collect()
            print(f"  [BATCH {b_idx}] Done — workers killed, memory reclaimed")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 5 — NEXT-DAY PREDICTIONS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  STEP 5 — NEXT-DAY PREDICTIONS  (parallel threads)")
    print(f"{'='*70}")

    predictions: List[Dict] = []
    with _cf.ThreadPoolExecutor(max_workers=min(n_workers * 2, 16)) as tpool:
        pred_futures = {
            tpool.submit(predict_worker, sym, df, peer_returns, market_df, usdinr_df, global_cues_df): sym
            for sym, df in raw_data.items()
        }
        for fut in _cf.as_completed(pred_futures):
            pred = fut.result()
            if pred and pred.get("symbol"):
                predictions.append(pred)

    predictions.sort(key=lambda p: p.get("confidence", 0), reverse=True)
    active_signals = [p for p in predictions if p.get("signal_active")]

    print(f"\n  {'Dir':<5} {'Symbol':<14} {'Action':<5} {'Conf':>6} {'Regime':<9} {'T':>5}  Close")
    print(f"  {'─'*5} {'─'*14} {'─'*5} {'─'*6} {'─'*9} {'─'*5}  {'─'*8}")
    for pred in predictions[:30]:   # top 30 by confidence
        arrow = "UP  " if pred["direction"] == "UP" else "DOWN"
        gate  = "" if pred["signal_active"] else "  [HOLD]"
        print(f"  {arrow} {pred['symbol']:<14} {pred['action']:<5} "
              f"{pred['confidence']:>6.1%} {pred['regime_label']:<9} "
              f"{pred.get('temperature',1.0):>5.2f}  ₹{pred['last_close']:.2f}  "
              f"{pred['last_date']}{gate}")
    if len(predictions) > 30:
        print(f"  ... {len(predictions)-30} more symbols")

    print(f"\n  Active signals (conf≥{CONFIDENCE_THRESHOLD:.0%}): {len(active_signals)} / {len(predictions)}")

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    elapsed  = time.time() - t0
    ok_rows  = [r for r in summary_rows if r.get("status") == "ok"]

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY  [MIN_MOVE={MIN_MOVE*100:.1f}%, CONF≥{CONFIDENCE_THRESHOLD*100:.0f}%]")
    print(f"{'='*70}")

    if ok_rows:
        summary_df = pd.DataFrame([{
            "symbol": r["symbol"], "oos_accuracy": r["oos_accuracy"],
            "oos_f1": r["oos_f1"], "n_windows": r["n_windows"],
            "n_predictions": r["n_predictions"],
            "n_features": r["n_features"], "n_rows": r["n_rows"],
        } for r in ok_rows])

        print(f"\n {'Symbol':<12} {'OOS Acc':>8} {'F1':>7} {'Wins':>5} {'Preds':>6} {'Rows':>6}")
        print(f" {'─'*12} {'─'*8} {'─'*7} {'─'*5} {'─'*6} {'─'*6}")
        for _, row in summary_df.iterrows():
            tag = "✅" if row["oos_accuracy"] >= 0.58 else ("⚠️ " if row["oos_accuracy"] >= 0.50 else "❌")
            print(f" {tag} {row['symbol']:<10} {row['oos_accuracy']:>8.2%}"
                  f" {row['oos_f1']:>7.4f} {int(row['n_windows']):>5}"
                  f" {int(row['n_predictions']):>6} {int(row['n_rows']):>6}")

        avg_acc = summary_df["oos_accuracy"].mean()
        avg_f1  = summary_df["oos_f1"].mean()
        best    = summary_df.loc[summary_df["oos_accuracy"].idxmax()]
        print(f"\n  Avg OOS Accuracy : {avg_acc:.2%}")
        print(f"  Avg F1 Score     : {avg_f1:.4f}")
        print(f"  Best Stock       : {best['symbol']} ({best['oos_accuracy']:.2%})")
        print(f"  ≥58% stocks      : {(summary_df['oos_accuracy'] >= 0.58).sum()}/{len(summary_df)}")

        # Per-model comparison
        all_win_rows = [w for r in ok_rows for w in r.get("window_rows", [])]
        if all_win_rows:
            aw_df = pd.DataFrame(all_win_rows)
            model_cols = {
                "lgbm_acc": "LightGBM", "xgb_acc": "XGBoost",
                "lstm_acc": "LSTM", "bilstm_acc": "BiLSTM", "gru_acc": "GRU",
                "cnn_lstm_acc": "CNN_LSTM", "cnn_gru_acc": "CNN_GRU",
                "tcn_gru_acc": "TCN_GRU", "tcn_transformer_acc": "TCN_Transformer",
                "nbeats_acc": "NBEATS",
            }
            model_comp_rows = []
            _all_avgs = {}
            print(f"\n  ── Per-Model Comparison (all stocks × windows) ──")
            print(f"  {'Model':<16} {'Avg':>8} {'Med':>8} {'Best':>8} {'Worst':>8} {'Std':>7}")
            print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
            for col in model_cols:
                if col in aw_df.columns:
                    vals = aw_df[col].dropna(); vals = vals[vals > 0]
                    if len(vals) > 0:
                        _all_avgs[col] = vals.mean()
            for col, name in model_cols.items():
                if col in aw_df.columns:
                    vals = aw_df[col].dropna(); vals = vals[vals > 0]
                    if len(vals) > 0:
                        tag = "🏆" if vals.mean() == max(_all_avgs.values(), default=0) else "  "
                        print(f"  {tag}{name:<14} {vals.mean():>8.2%} {vals.median():>8.2%}"
                              f" {vals.max():>8.2%} {vals.min():>8.2%} {vals.std():>7.3f}")
                        model_comp_rows.append({
                            "model": name, "avg_accuracy": vals.mean(),
                            "median_accuracy": vals.median(), "max_accuracy": vals.max(),
                            "min_accuracy": vals.min(), "std_accuracy": vals.std(),
                            "n_datapoints": len(vals),
                        })
            if model_comp_rows:
                mc_df = pd.DataFrame(model_comp_rows).sort_values("avg_accuracy", ascending=False)
                mc_df.to_csv(result_run_path / "model_comparison.csv", index=False)
                best_m = mc_df.iloc[0]
                print(f"\n  🏆 Best: {best_m['model']} "
                      f"(avg={best_m['avg_accuracy']:.2%}, std={best_m['std_accuracy']:.3f})")
            aw_df.to_csv(result_run_path / "all_windows_detail.csv", index=False)

        # Full summary CSV
        avg_row = pd.DataFrame([{
            "symbol": "AVERAGE", "oos_accuracy": avg_acc, "oos_f1": avg_f1,
            "n_windows": summary_df["n_windows"].mean(),
            "n_predictions": summary_df["n_predictions"].sum(),
            "n_features": summary_df["n_features"].mean(),
            "n_rows": summary_df["n_rows"].mean(),
        }])
        pd.concat([summary_df, avg_row], ignore_index=True).to_csv(
            result_run_path / "summary.csv", index=False
        )
        try:
            plot_cross_stock_comparison(
                pd.concat([summary_df, avg_row], ignore_index=True), run_plot_path
            )
        except Exception:
            pass

    if predictions:
        pd.DataFrame(predictions).to_csv(result_run_path / "next_day_predictions.csv", index=False)
        try:
            plot_signal_heatmap(predictions, run_plot_path)
        except Exception:
            pass

    # Run metadata
    with open(result_run_path / "run_metadata.json", "w") as f:
        json.dump({
            "run_id": run_id, "symbols": symbols_to_run,
            "data_start": DATA_START_DATE,
            "min_move": MIN_MOVE, "confidence_threshold": CONFIDENCE_THRESHOLD,
            "initial_train": INITIAL_TRAIN_RATIO, "expansion_step": EXPANSION_STEP,
            "max_train": MAX_TRAIN_RATIO, "n_top_features": N_TOP_FEATURES,
            "global_cues_enabled": global_cues_df is not None,
            "nse_calendar_enabled": True,
            "calibration": "temperature_scaling",
            "n_workers": n_workers, "n_jobs_per_worker": n_jobs_per_worker,
            "elapsed_sec": round(elapsed, 1), "n_symbols_ok": len(ok_rows),
            "trained_at": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n  Results  → {result_run_path.relative_to(_V3_ROOT)}")
    print(f"  Models   → {(MODELS_RUNS_DIR/run_id).relative_to(_V3_ROOT)}")
    print(f"  Prod     → {(MODELS_RUNS_DIR.parent/'production').relative_to(_V3_ROOT)}/{{symbol}}/")
    print(f"  Elapsed  : {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 6 — OPTIONAL BACKTEST
    # ══════════════════════════════════════════════════════════════════════════
    if args.backtest:
        print(f"\n{'='*70}")
        print("  STEP 6 — HRP PORTFOLIO BACKTEST")
        print(f"{'='*70}")
        try:
            sys.path.insert(0, str(_V3_ROOT / "04_backtesting"))
            from backtest_runner import BacktestRunner  # type: ignore
            runner = BacktestRunner(run_id=run_id, initial_capital=100_000)
            metrics = runner.run()
            print(f"  Sharpe: {metrics.get('sharpe_ratio',0):.3f}  "
                  f"CAGR: {metrics.get('cagr',0):.2%}  "
                  f"MaxDD: {metrics.get('max_drawdown',0):.2%}")
        except Exception as bt_exc:
            print(f"  ⚠ Backtest error: {bt_exc}")

    print("=" * 70)


if __name__ == "__main__":
    main()
