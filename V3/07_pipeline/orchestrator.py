"""
================================================================================
V3 ALGO TRADING PIPELINE — ORCHESTRATOR
================================================================================
Entry point for the full pipeline. Coordinates all steps autonomously:

  Step 1 — Download      : NSE OHLCV + USD/INR + Global cues (incremental)
  Step 2 — Features      : 260+ features (cached, stale-check)
  Step 3 — Train         : Walk-forward ensemble (LightGBM + XGBoost + CatBoost + 3 DL)
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

# ── Backend: TensorFlow CPU (stable on Python 3.13 + M1) ─────────────────────
# GPU notes: tensorflow-metal (Py≤3.11 only), jax-metal (StableHLO incompat
# with Keras 3.14), torch backend (Keras LSTM/GRU broken). TF CPU is stable.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
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
    plot_cross_stock_comparison, plot_model_comparison_heatmap, plot_feature_importance_top20,
)
from steps.predict import (  # type: ignore  # noqa: E402
    predict_next_day, predict_worker,
    predict_with_news, _print_prediction, _download_fresh,
)


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

def _run_predict_mode(symbols: List[str], verbose: bool = False,
                      as_json: bool = False) -> None:
    """
    --predict mode: full inference with FinBERT news + earnings proximity.
    Replaces the old V3/scripts/predict_today.py.

    Usage:
        python orchestrator.py --predict HDFCBANK
        python orchestrator.py --predict HDFCBANK SBIN --verbose
        python orchestrator.py --predict HDFCBANK --json
    """
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    symbols = [s.upper() for s in symbols]

    print(f"\n  AI Stock Prediction  —  V3 with News + Earnings Sentiment")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Symbols : {', '.join(symbols)}\n")

    # Load auxiliary data once (shared across symbols)
    global_cues_df: Optional[pd.DataFrame] = None
    usdinr_df:      Optional[pd.DataFrame] = None
    try:
        from config_v3 import RAW_DATA_DIR  # type: ignore
        p = RAW_DATA_DIR / "global_cues.parquet"
        if p.exists():
            global_cues_df = pd.read_parquet(p)
            global_cues_df["date"] = pd.to_datetime(global_cues_df["date"]).astype("datetime64[us]")
    except Exception:
        pass
    try:
        from config_v3 import RAW_DATA_DIR  # type: ignore
        p = RAW_DATA_DIR / "usdinr.parquet"
        if p.exists():
            usdinr_df = pd.read_parquet(p)
            usdinr_df["date"] = pd.to_datetime(usdinr_df["date"]).astype("datetime64[us]")
    except Exception:
        pass

    results = []
    for sym in symbols:
        r = predict_with_news(
            sym,
            usdinr_df=usdinr_df,
            global_cues_df=global_cues_df,
            verbose=verbose,
        )
        if r is None:
            continue
        results.append(r)
        if as_json:
            print(json.dumps(r, indent=2))
        else:
            _print_prediction(r)

    # Multi-symbol summary
    if len(results) > 1 and not as_json:
        print("\n  ── Multi-symbol Summary ──")
        print(f"  {'Symbol':<14} {'Dir':<5} {'Conf':>6} {'News':>7} {'Regime':<10} {'Signal':<6}")
        print(f"  {'──────':<14} {'───':<5} {'────':>6} {'─────':>7} {'──────':<10} {'──────':<6}")
        for r in results:
            dirn = "▲ UP" if r["direction"] == "UP" else "▼ DN"
            ns   = r["news"]["raw_score"]
            print(f"  {r['symbol']:<14} {dirn:<5} {r['confidence']*100:>5.1f}%"
                  f" {ns:>+6.3f} {r['regime']:<10} {r['signal']:<6}")
        print()


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
    parser.add_argument("--predict", nargs="+", metavar="SYMBOL",
                        help="Predict-only mode: skip training, run full news+earnings "
                             "inference for listed symbols. "
                             "Example: --predict HDFCBANK SBIN")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output for --predict mode")
    parser.add_argument("--json",    action="store_true",
                        help="Output JSON instead of display for --predict mode")
    parser.add_argument("--fast",    action="store_true",
                        help="Fast mode: skip DL models, trees only (LightGBM + XGBoost + CatBoost). "
                             "~3× faster per stock. Use for production / quick iteration.")
    args = parser.parse_args()

    # ── --predict SYMBOL [SYMBOL …]  (fast path — no training) ──────────────
    if args.predict:
        _run_predict_mode(args.predict, verbose=args.verbose, as_json=args.json)
        return

    symbols_to_run: List[str] = args.symbols if args.symbols else list(SYMBOLS)
    run_id = args.resume or datetime.now().strftime("%Y%m%d_%H%M%S")
    t0     = time.time()

    setup_loguru(run_id)

    # ── Fast mode: disable DL, trees only ────────────────────────────────────
    if args.fast:
        from steps.train import set_fast_mode
        set_fast_mode(True)

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

    # Save symbols attempted so dashboard can show correct n_total
    (result_run_path / "symbols.txt").write_text("\n".join(symbols_to_run))

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

    # Load sentiment history (accumulated daily by sentiment_history.py)
    sentiment_df: Optional[pd.DataFrame] = None
    try:
        _sent_path = _V3_ROOT / "01_data" / "news" / "sentiment_history.parquet"
        if _sent_path.exists():
            sentiment_df = pd.read_parquet(_sent_path)
            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).astype("datetime64[us]")
            n_sym  = sentiment_df["symbol"].nunique()
            n_days = sentiment_df["date"].nunique()
            print(f"  Sentiment   : {len(sentiment_df)} rows  "
                  f"({n_sym} symbols × {n_days} dates)")
        else:
            print("  Sentiment   : no history yet — run sentiment_history.py daily to build it")
    except Exception as _e:
        print(f"  Sentiment   : skipped ({_e})")

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
                    sentiment_df=sentiment_df,
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
                initargs=(n_jobs_per_worker, args.fast),
            ) as pool:
                future_to_sym = {
                    pool.submit(
                        run_symbol_worker,
                        sym, raw_data[sym], run_id,
                        model_run_path, result_run_path,
                        peer_returns, market_df, usdinr_df, global_cues_df,
                        sentiment_df,
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
    elapsed       = time.time() - t0
    ok_rows       = [r for r in summary_rows if r.get("status") == "ok"]
    skipped_rows  = [r for r in summary_rows if str(r.get("status", "")).startswith("skipped")]
    error_rows    = [r for r in summary_rows if r.get("status") == "error"]

    # Persist skipped/error stocks so the dashboard can show "100/100 — N skipped (reason)"
    # instead of silently dropping them and reporting a misleading "99/100".
    try:
        with open(result_run_path / "skipped_symbols.json", "w") as _sf:
            json.dump({
                "universe_size":   len(symbols_to_run),
                "trained":         len(ok_rows),
                "skipped":         [{"symbol": r.get("symbol"), "reason": r.get("reason", "unknown"),
                                     "status": r.get("status")} for r in skipped_rows],
                "errors":          [{"symbol": r.get("symbol"), "reason": r.get("reason", "unknown")}
                                    for r in error_rows],
            }, _sf, indent=2, default=str)
    except Exception:
        pass

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY  [MIN_MOVE={MIN_MOVE*100:.1f}%, CONF≥{CONFIDENCE_THRESHOLD*100:.0f}%]")
    print(f"  Universe: {len(symbols_to_run)}  |  trained: {len(ok_rows)}  "
          f"|  skipped: {len(skipped_rows)}  |  errors: {len(error_rows)}")
    for r in skipped_rows:
        print(f"   ↪ skipped {r.get('symbol','?'):<12} ({r.get('reason','no_reason')})")
    for r in error_rows:
        print(f"   ↪ error   {r.get('symbol','?'):<12} ({r.get('reason','no_reason')})")
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
                "lgbm_acc": "LightGBM", "xgb_acc": "XGBoost", "catboost_acc": "CatBoost",
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

        # Full summary CSV — include trained, skipped, and error rows with a
        # `status` column so the dashboard can render "100/100 (N skipped)" instead
        # of silently dropping incomplete stocks.
        summary_df["status"] = "ok"
        skip_err = []
        for r in (skipped_rows + error_rows):
            skip_err.append({
                "symbol":        r.get("symbol", ""),
                "oos_accuracy":  None, "oos_f1": None,
                "n_windows":     0, "n_predictions": 0,
                "n_features":    0, "n_rows": int(r.get("n_rows", 0) or 0),
                "status":        r.get("status", "skipped"),
            })
        avg_row = pd.DataFrame([{
            "symbol": "AVERAGE", "oos_accuracy": avg_acc, "oos_f1": avg_f1,
            "n_windows": summary_df["n_windows"].mean(),
            "n_predictions": summary_df["n_predictions"].sum(),
            "n_features": summary_df["n_features"].mean(),
            "n_rows": summary_df["n_rows"].mean(),
            "status": "ok",
        }])
        full_df = pd.concat(
            [summary_df, pd.DataFrame(skip_err), avg_row],
            ignore_index=True
        )
        full_df.to_csv(result_run_path / "summary.csv", index=False)
        try:
            plot_cross_stock_comparison(
                pd.concat([summary_df, avg_row], ignore_index=True), run_plot_path
            )
        except Exception:
            pass

    if predictions:
        pred_df = pd.DataFrame(predictions)
        # Join OOS accuracy from summary; tradeable will be updated after backtest
        if ok_rows:
            acc_map = {r["symbol"]: r["oos_accuracy"] for r in ok_rows}
            up_acc_map = {r["symbol"]: r.get("avg_dir_acc_up", 0.0) for r in ok_rows}
            down_acc_map = {r["symbol"]: r.get("avg_dir_acc_down", 0.0) for r in ok_rows}
            best_model_map = {r["symbol"]: r.get("best_model", "") for r in ok_rows}
            best_model_acc_map = {r["symbol"]: r.get("best_model_acc", 0.0) for r in ok_rows}
            pred_df["oos_accuracy"] = pred_df["symbol"].map(acc_map).fillna(0.0)
            pred_df["ensemble_accuracy"] = pred_df["oos_accuracy"]
            pred_df["up_signal_accuracy"] = pred_df["symbol"].map(up_acc_map).fillna(0.0)
            pred_df["down_signal_accuracy"] = pred_df["symbol"].map(down_acc_map).fillna(0.0)
            pred_df["directional_accuracy_for_signal"] = np.where(
                pred_df["direction"].eq("UP"),
                pred_df["up_signal_accuracy"],
                pred_df["down_signal_accuracy"],
            )
            pred_df["best_model"] = pred_df["symbol"].map(best_model_map).fillna("")
            pred_df["best_model_accuracy"] = pred_df["symbol"].map(best_model_acc_map).fillna(0.0)
            pred_df["tradeable"]    = pred_df["oos_accuracy"] >= 0.52  # preliminary
        pred_df.to_csv(result_run_path / "next_day_predictions.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 6 — TRADE SIMULATION BACKTEST  (always runs, uses predictions.csv)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  STEP 6 — TRADE SIMULATION BACKTEST")
    print(f"{'='*70}")
    try:
        from steps.backtest import run_trade_backtest  # type: ignore
        run_trade_backtest(result_run_path, min_confidence=CONFIDENCE_THRESHOLD)

        # Update next_day_predictions.csv tradeable + cross_sectional_top15 flags
        bt_path   = result_run_path / "backtest_results.csv"
        pred_path = result_run_path / "next_day_predictions.csv"
        if bt_path.exists() and pred_path.exists():
            bt_df   = pd.read_csv(bt_path)
            pred_df = pd.read_csv(pred_path)
            t_map   = dict(zip(bt_df["symbol"], bt_df["tradeable"]))
            pred_df["tradeable"] = pred_df["symbol"].map(t_map).fillna(False)
            if "cross_sectional_top15" in bt_df.columns:
                c_map = dict(zip(bt_df["symbol"], bt_df["cross_sectional_top15"]))
                pred_df["cross_sectional_top15"] = pred_df["symbol"].map(c_map).fillna(False)
            if "sharpe_rank" in bt_df.columns:
                r_map = dict(zip(bt_df["symbol"], bt_df["sharpe_rank"]))
                pred_df["sharpe_rank"] = pred_df["symbol"].map(r_map)
            pred_df.to_csv(pred_path, index=False)
            n_tradeable = int(pred_df["tradeable"].sum())
            n_cs = int(pred_df.get("cross_sectional_top15", pd.Series(dtype=bool)).sum())
            print(f"  ✓ Updated flags: {n_tradeable} strictly tradeable, "
                  f"{n_cs} in cross-sectional top15")
    except Exception as bt_exc:
        print(f"  ⚠ Backtest error: {bt_exc}")
        import traceback; traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 7 — DIAGNOSTICS  (regime replay + Diebold-Mariano)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  STEP 7 — DIAGNOSTICS (regime replay + DM tests)")
    print(f"{'='*70}")
    try:
        from steps.diagnostics import run_diagnostics  # type: ignore
        run_diagnostics(result_run_path)
    except Exception as diag_exc:
        print(f"  ⚠ Diagnostics error: {diag_exc}")

    # Run-level journal plots
    try:
        plot_model_comparison_heatmap(result_run_path, run_plot_path)
    except Exception:
        pass
    try:
        plot_feature_importance_top20(result_run_path, run_plot_path)
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
