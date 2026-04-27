"""
================================================================================
PRODUCTION PIPELINE — V3  (Phase 1 upgrade)
================================================================================
Zero-leakage expanding-window walk-forward validation.

PHASE 1 UPGRADES (Feb 2026)
----------------------------
1. MIN_MOVE 0.001 → 0.004  : removes noise below transaction costs
2. CONFIDENCE_THRESHOLD 0.55 → 0.58  : trade less, win more
3. Global market cues  : S&P500, Nasdaq, US VIX, DXY, Crude, Nikkei
   - Downloaded incrementally to 01_data/raw/global_cues.parquet
   - Leakage-safe: Indian date T uses US close from T-1 (strict prior day)
4. NSE calendar features : F&O monthly expiry proximity, RBI MPC window,
   Union Budget window, result season indicator
5. Temperature scaling  : single-parameter Platt calibration fitted on val set,
   applied to final ensemble probability; saved per window for inference

STORAGE LAYOUT  (no duplication)
----------------------------------
V3/
├── 01_data/
│   ├── raw/  {symbol}.parquet  global_cues.parquet  usdinr.parquet
│   └── features/
│       ├── raw/    {symbol}_features.parquet   (cached; refresh on raw change)
│       └── scaled/ {symbol}_scaled.parquet
├── 02_models/
│   ├── runs/{run_id}/{symbol}/window_{n:02d}/   ← checkpoints
│   └── production/{symbol}/                     ← latest production
└── 06_results/runs/{run_id}/
    ├── {symbol}/ window_results.csv  predictions.csv  plots/
    ├── summary.csv  all_windows_detail.csv
    ├── next_day_predictions.csv  run_metadata.json
    └── plots/

LEAKAGE AUDIT
--------------
- Target    : shift(-1) before splitting — labelling only
- Features  : all rolling/EWM windows look backward; no shift(-N)
- GlobalCues: US date T-1 data → Indian date T (merge_asof strict prior)
- Scaler    : RobustScaler fit on X_train only
- PCA       : fit on X_train only
- Calibrator: temperature T fitted on val set probs, applied to test
================================================================================
"""

from __future__ import annotations

import calendar as _calendar
import contextlib
import io
import json
import os
import pickle
import sys
import time
import traceback
import logging
import warnings

# Suppress noisy TF/protobuf warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*GetPrototype.*")
warnings.filterwarnings("ignore", message=".*triggered tf.function retracing.*")
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import argparse
import concurrent.futures as _cf
import gc
import multiprocessing as _mp
import shutil

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── TF memory: must be set before tensorflow is imported anywhere ─────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU-only (no GPU allocation overhead)

# ── Resolve V3 root ───────────────────────────────────────────────────────────
V3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_ROOT))
sys.path.insert(0, str(V3_ROOT / "02_models"))

# ── Config ────────────────────────────────────────────────────────────────────
from config_v3 import (  # type: ignore  # noqa: E402
    RAW_DATA_DIR, FEAT_RAW_DIR, FEAT_SCALED_DIR,
    MODELS_RUNS_DIR, MODELS_PROD_DIR, RESULTS_RUNS_DIR,
    SYMBOLS, IT_SYMBOLS, BANKING_SYMBOLS, SECTOR_MAP,
    DATA_START_DATE, YFINANCE_DELAY,
    GLOBAL_CUES_TICKERS, GLOBAL_CUES_FEATURES,
    USDINR_FEATURES, BANKING_CUES_FEATURES,
    RBI_MPC_DATES, BUDGET_DATES, RESULT_SEASON_MONTHS,
    INITIAL_TRAIN_RATIO, EXPANSION_STEP, MAX_TRAIN_RATIO,
    MIN_TRAIN_SAMPLES, MIN_TEST_SAMPLES,
    CONFIDENCE_THRESHOLD, MIN_MOVE,
    N_TOP_FEATURES, MIN_REGIME_SAMPLES,
    LGBM_PARAMS, LGBM_FS_PARAMS, XGB_PARAMS,
    DL_SEQ_LEN, DL_BATCH_SIZE, DL_MAX_EPOCHS,
    DL_ES_PATIENCE, DL_ES_MIN_DELTA,
    DL_RLROP_FACTOR, DL_RLROP_PATIENCE, DL_RLROP_MIN_LR,
    LSTM_PARAMS, BILSTM_PARAMS, GRU_PARAMS, CNN_LSTM_PARAMS, CNN_GRU_PARAMS,
    TCN_GRU_PARAMS, TCN_TRANSFORMER_PARAMS, NBEATS_PARAMS,
    PLOT_DPI, PLOT_STYLE, PLOT_FIGSIZE_WIDE, PLOT_FIGSIZE_TALL, PLOT_FIGSIZE_SQUARE,
    RANDOM_SEED,
)

from traditional.lightgbm_classifier import LightGBMClassifier  # type: ignore  # noqa: E402
from traditional.xgboost_classifier  import XGBoostClassifier   # type: ignore  # noqa: E402

# ── Deep Learning models ──────────────────────────────────────────────────────
_DL_AVAILABLE = False
_DL_CLASSES: list = []
try:
    from deep_learning.lstm_classifier            import LSTMClassifier            # noqa: E402
    from deep_learning.bilstm_classifier          import BiLSTMClassifier          # noqa: E402
    from deep_learning.gru_classifier             import GRUClassifier             # noqa: E402
    from deep_learning.cnn_lstm_classifier        import CNNLSTMClassifier         # noqa: E402
    from deep_learning.cnn_gru_classifier         import CNNGRUClassifier          # noqa: E402
    from deep_learning.tcn_gru_classifier         import TCNGRUClassifier          # noqa: E402
    from deep_learning.tcn_transformer_classifier import TCNTransformerClassifier  # noqa: E402
    from deep_learning.nbeats_classifier          import NBEATSClassifier          # noqa: E402
    from deep_learning.base_deep                  import get_dl_splits             # noqa: E402
    _DL_AVAILABLE = True
    _DL_CLASSES = [
        (LSTMClassifier,           "LSTM"),
        (BiLSTMClassifier,         "BiLSTM"),
        (GRUClassifier,            "GRU"),
        (CNNLSTMClassifier,        "CNN_LSTM"),
        # CNN_GRU removed — statistically identical to CNN_LSTM (both ~50.73%)
        # Keeping both adds correlated noise that dilutes better tree models' votes
        (TCNGRUClassifier,         "TCN_GRU"),
        (TCNTransformerClassifier, "TCN_Transformer"),
        (NBEATSClassifier,         "NBEATS"),
    ]
    print("  [DL] models available: LSTM, BiLSTM, GRU, CNN_LSTM, CNN_GRU, TCN_GRU, TCN_Transformer, NBEATS")
except ImportError as _dl_err:
    print(f"  [DL] models unavailable: {_dl_err}. Running tree-only.")

# ── Parallel execution — n_jobs per model (overridden by _worker_init) ────────
# -1 = use all cores (default serial mode); set to positive in parallel workers
_N_JOBS: int = -1

# Pre-parse calendar dates once
_RBI_DATES    = pd.to_datetime(RBI_MPC_DATES)
_BUDGET_DATES = pd.to_datetime(BUDGET_DATES)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_parquet(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "timestamp" in df.columns:
            # Handle parquets written by V3/01_data/downloader.py (uses "timestamp")
            df = df.rename(columns={"timestamp": "date"})
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return None


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _fetch_yfinance(ticker: str, start: str, end: str, retries: int = 3) -> Optional[pd.DataFrame]:
    """Download OHLCV from Yahoo Finance, normalise columns. Retries on failure."""
    import yfinance as yf
    import time as _time
    for attempt in range(retries):
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                if attempt < retries - 1:
                    _time.sleep(1.0 * (attempt + 1))
                    continue
                return None
            df = raw.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [str(c).lower() for c in df.columns]
            for dc in ["date", "datetime", "index"]:
                if dc in df.columns:
                    df = df.rename(columns={dc: "date"})
                    break
            df["date"] = pd.to_datetime(df["date"])
            keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[keep].sort_values("date").reset_index(drop=True)
            df = df[(df["close"] > 0)].reset_index(drop=True)
            return df if not df.empty else None
        except Exception as e:
            if attempt < retries - 1:
                _time.sleep(1.0 * (attempt + 1))
            else:
                print(f"  [yfinance] {ticker} failed after {retries} attempts: {e}")
                return None
    return None


def _last_thursday_of_month(year: int, month: int) -> date:
    """Return the last Thursday of the given month (NSE monthly F&O expiry day)."""
    _, last_day = _calendar.monthrange(year, month)
    dt = date(year, month, last_day)
    # weekday(): Mon=0 … Thu=3 … Sun=6
    offset = (dt.weekday() - 3) % 7
    return dt - timedelta(days=offset)


# ══════════════════════════════════════════════════════════════════════════════
#  PARALLEL WORKER INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def _worker_init(n_jobs: int) -> None:
    """
    Called once per worker process by ProcessPoolExecutor.
    Sets the per-model thread count so workers don't over-subscribe CPUs.
    Also configures TensorFlow memory growth to avoid pre-allocating all RAM.
    """
    global _N_JOBS
    _N_JOBS = n_jobs
    # Native library thread limits (respected by LightGBM, XGBoost, numpy)
    s = str(n_jobs)
    os.environ["OMP_NUM_THREADS"]        = s
    os.environ["MKL_NUM_THREADS"]        = s
    os.environ["OPENBLAS_NUM_THREADS"]   = s
    os.environ["NUMEXPR_NUM_THREADS"]    = s
    os.environ["TF_NUM_INTEROP_THREADS"] = s
    os.environ["TF_NUM_INTRAOP_THREADS"] = s
    # TF memory optimisation — don't pre-allocate, grow as needed
    os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
    # Suppress protobuf MessageFactory AttributeError (protobuf/TF version mismatch)
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    try:
        import logging as _logging
        _logging.getLogger("tensorflow").setLevel(_logging.ERROR)
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(n_jobs)
        tf.config.threading.set_intra_op_parallelism_threads(n_jobs)
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def _run_symbol_worker(
    symbol: str,
    raw_df: "pd.DataFrame",
    run_id: str,
    model_run_path: "Path",
    result_run_path: "Path",
    peer_returns: dict,
    market_df,
    usdinr_df,
    global_cues_df,
) -> dict:
    """
    Top-level picklable function for ProcessPoolExecutor.
    Redirects stdout/stderr to a per-symbol log file so the console stays clean.
    Returns the same dict as run_symbol().
    """
    log_dir  = result_run_path / symbol
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    try:
        with open(log_path, "w", buffering=1, encoding="utf-8") as log_f:
            with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
                return run_symbol(
                    symbol=symbol,
                    raw_df=raw_df,
                    run_id=run_id,
                    model_run_path=model_run_path,
                    result_run_path=result_run_path,
                    peer_returns=peer_returns,
                    market_df=market_df,
                    usdinr_df=usdinr_df,
                    global_cues_df=global_cues_df,
                )
    except Exception as exc:
        return {"symbol": symbol, "status": "error", "reason": str(exc)}


def _predict_worker(
    sym: str,
    df: "pd.DataFrame",
    peer_returns: dict,
    market_df,
    usdinr_df,
    global_cues_df,
) -> dict:
    """Top-level picklable function for parallel next-day prediction."""
    try:
        result = predict_next_day(
            sym, df,
            peer_returns=peer_returns,
            market_df=market_df,
            usdinr_df=usdinr_df,
            global_cues_df=global_cues_df,
        )
        return result or {}
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATE CSV FLUSH  (called after each symbol — crash-safe)
# ══════════════════════════════════════════════════════════════════════════════

def _flush_aggregate_csvs(result_run_path: Path) -> None:
    """
    Rebuild summary.csv and all_windows_detail.csv from per-symbol files.
    Called after every symbol so aggregate results survive mid-run crashes.
    Also builds model_comparison.csv with per-model accuracy across stocks.
    """
    rows: list = []
    win_rows: list = []
    for sym_dir in sorted(result_run_path.iterdir()):
        if not sym_dir.is_dir() or sym_dir.name == "plots":
            continue
        sr_path = sym_dir / "summary_row.json"
        wr_path = sym_dir / "window_results.csv"
        if sr_path.exists():
            try:
                with open(sr_path) as f:
                    rows.append(json.load(f))
            except Exception:
                pass
        if wr_path.exists():
            try:
                win_rows.extend(pd.read_csv(wr_path).to_dict("records"))
            except Exception:
                pass
    if rows:
        summary_df = pd.DataFrame(rows)
        num_cols = ["oos_accuracy", "oos_f1", "n_windows", "n_predictions", "n_features", "n_rows",
                    "avg_lgbm_acc", "avg_xgb_acc", "avg_lstm_acc", "avg_bilstm_acc",
                    "avg_gru_acc", "avg_cnn_lstm_acc", "avg_cnn_gru_acc",
                    "avg_tcn_gru_acc", "avg_tcn_transformer_acc", "avg_nbeats_acc",
                    "best_model_acc",
                    "avg_dir_acc_up", "avg_dir_acc_down", "avg_pct_neutral"]
        existing = [c for c in num_cols if c in summary_df.columns]
        avg: dict = {c: (summary_df[c].sum() if c == "n_predictions" else summary_df[c].mean())
                     for c in existing}
        avg["symbol"] = "AVERAGE"
        avg["status"] = "ok"
        # Best model across all stocks
        model_avg_cols = [c for c in existing if c.startswith("avg_") and c.endswith("_acc")
                          and not c.startswith("avg_dir_")]
        if model_avg_cols:
            best_col = max(model_avg_cols, key=lambda c: avg.get(c, 0))
            avg["best_model"] = best_col.replace("avg_", "").replace("_acc", "")
        pd.concat([summary_df, pd.DataFrame([avg])], ignore_index=True).to_csv(
            result_run_path / "summary.csv", index=False
        )

        # Model comparison CSV: per-model average directional accuracy across all stocks
        if model_avg_cols:
            model_comp = []
            for col in model_avg_cols:
                model_name = col.replace("avg_", "").replace("_acc", "").upper()
                acc_vals = summary_df[col].dropna()
                model_comp.append({
                    "model": model_name,
                    "avg_accuracy": float(acc_vals.mean()) if len(acc_vals) > 0 else 0.0,
                    "median_accuracy": float(acc_vals.median()) if len(acc_vals) > 0 else 0.0,
                    "max_accuracy": float(acc_vals.max()) if len(acc_vals) > 0 else 0.0,
                    "min_accuracy": float(acc_vals.min()) if len(acc_vals) > 0 else 0.0,
                    "std_accuracy": float(acc_vals.std()) if len(acc_vals) > 1 else 0.0,
                    "n_stocks": int(len(acc_vals)),
                })
            model_comp_df = pd.DataFrame(model_comp).sort_values("avg_accuracy", ascending=False)
            model_comp_df.to_csv(result_run_path / "model_comparison.csv", index=False)

    if win_rows:
        pd.DataFrame(win_rows).to_csv(result_run_path / "all_windows_detail.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1A — STOCK DATA DOWNLOAD  (incremental)
# ══════════════════════════════════════════════════════════════════════════════

def download_symbol(symbol: str) -> Optional[pd.DataFrame]:
    """Incremental download for one NSE symbol. Appends new rows only."""
    save_path = RAW_DATA_DIR / f"{symbol}.parquet"
    existing  = _load_parquet(save_path)
    ticker    = f"{symbol}.NS"
    today     = datetime.now().strftime("%Y-%m-%d")

    if existing is not None and not existing.empty:
        last_date   = existing["date"].max()
        # Buffer 5 days back from last_date to avoid weekend/holiday gaps
        # that cause yfinance "possibly delisted" false alarms
        fetch_start = (last_date - timedelta(days=4)).strftime("%Y-%m-%d")
        check_date  = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        if check_date >= today:
            print(f"  {symbol:<12} ✓ up-to-date  [{last_date.date()}]")
            return existing
        print(f"  {symbol:<12} ↓ incremental  {last_date.date()} → {today} ...", end=" ", flush=True)
        new_df = _fetch_yfinance(ticker, fetch_start, today)
        if new_df is not None and not new_df.empty:
            combined = (
                pd.concat([existing, new_df], ignore_index=True)
                .drop_duplicates("date").sort_values("date").reset_index(drop=True)
            )
            n_added = len(combined) - len(existing)
            if n_added > 0:
                _save_parquet(combined, save_path)
                print(f"✓ +{n_added} rows  (total {len(combined)}  [{combined['date'].iloc[-1].date()}])")
                return combined
        print("✓ no new rows")
        return existing

    print(f"  {symbol:<12} ↓ full  {DATA_START_DATE} → {today} ...", end=" ", flush=True)
    df = _fetch_yfinance(ticker, DATA_START_DATE, today)
    if df is None or df.empty:
        print(f"✗ failed ({ticker} not available on Yahoo Finance)")
        return None
    _save_parquet(df, save_path)
    print(f"✓ {len(df)} rows  [{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}]")
    return df


def download_usdinr() -> Optional[pd.DataFrame]:
    """Incremental download of USD/INR exchange rate."""
    save_path = RAW_DATA_DIR / "usdinr.parquet"
    existing  = _load_parquet(save_path)
    today     = datetime.now().strftime("%Y-%m-%d")
    fetch_start = DATA_START_DATE
    if existing is not None and not existing.empty:
        last_date   = existing["date"].max()
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        if fetch_start >= today:
            return existing
    df = _fetch_yfinance("USDINR=X", fetch_start, today)
    if df is None:
        return existing
    df = df.rename(columns={"close": "usdinr_close"})[["date", "usdinr_close"]]
    df = df[df["usdinr_close"] > 0].reset_index(drop=True)
    if existing is not None and not existing.empty:
        df = (pd.concat([existing, df], ignore_index=True)
                .drop_duplicates("date").sort_values("date").reset_index(drop=True))
    _save_parquet(df, save_path)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1B — GLOBAL MARKET CUES DOWNLOAD  (Phase 1, incremental)
# ══════════════════════════════════════════════════════════════════════════════

def download_global_cues() -> Optional[pd.DataFrame]:
    """
    Incrementally download global market cues (S&P500, Nasdaq, VIX, DXY, Crude, Nikkei).
    Saved to 01_data/raw/global_cues.parquet.

    Returns a DataFrame with columns:
        date, sp500_close, sp500_ret, nasdaq_close, nasdaq_ret,
        us_vix, dxy_close, dxy_ret, crude_close, crude_ret,
        nikkei_close, nikkei_ret
    """
    save_path = RAW_DATA_DIR / "global_cues.parquet"
    existing  = _load_parquet(save_path)
    today     = datetime.now().strftime("%Y-%m-%d")

    fetch_start = DATA_START_DATE
    if existing is not None and not existing.empty:
        last_date   = existing["date"].max()
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        if fetch_start >= today:
            print(f"  global_cues     ✓ up-to-date  [{last_date.date()}]")
            return existing

    print(f"  global_cues     ↓ {fetch_start} → {today} ...", end=" ", flush=True)

    frames: List[pd.DataFrame] = []

    for name, ticker in GLOBAL_CUES_TICKERS.items():
        try:
            df = _fetch_yfinance(ticker, fetch_start, today)
            if df is None or df.empty:
                continue
            df = df[["date", "close"]].rename(columns={"close": f"{name}_close"})
            df[f"{name}_ret"] = df[f"{name}_close"].pct_change()
            frames.append(df)
        except Exception:
            pass  # skip failed tickers gracefully

    if not frames:
        print("⚠ all tickers failed")
        return existing

    # Outer-join on date so missing tickers leave NaN (handled by ffill later)
    merged = frames[0]
    for fr in frames[1:]:
        merged = pd.merge(merged, fr, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    if existing is not None and not existing.empty:
        merged = (pd.concat([existing, merged], ignore_index=True)
                    .drop_duplicates("date").sort_values("date").reset_index(drop=True))

    _save_parquet(merged, save_path)
    print(f"✓ {len(merged)} rows  [{merged['date'].iloc[-1].date()}]")
    return merged


def download_all_symbols(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    print(f"\n{'='*70}")
    print(f"  STEP 1 — DATA DOWNLOAD  (incremental, parallel threads)")
    print(f"{'='*70}")
    data: Dict[str, pd.DataFrame] = {}

    # Use threads (I/O-bound): cap at 8 to avoid yfinance rate-limits
    n_dl_workers = min(8, len(symbols))
    with _cf.ThreadPoolExecutor(max_workers=n_dl_workers) as pool:
        future_to_sym = {pool.submit(download_symbol, sym): sym for sym in symbols}
        for fut in _cf.as_completed(future_to_sym):
            sym = future_to_sym[fut]
            try:
                df = fut.result()
                if df is not None:
                    data[sym] = df
            except Exception as exc:
                print(f"  {sym}: download error — {exc}")

    print(f"\n  Loaded {len(data)}/{len(symbols)} symbols")
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _add_nse_calendar_features(d: pd.DataFrame) -> pd.DataFrame:
    """
    NSE-specific calendar features — pure date arithmetic, zero leakage.

    Features added:
        days_to_expiry   : calendar days to next monthly F&O expiry (last Thu of month)
        is_expiry_week   : within 4 calendar days of expiry
        is_expiry_day    : expiry day itself
        days_to_rbi      : days to nearest RBI MPC meeting (capped at 30)
        is_rbi_week      : within 3 days of RBI meeting
        days_to_budget   : days to nearest Budget day (capped at 60)
        is_budget_week   : within 3 days of Budget
        is_result_season : month is in Q1/Q2/Q3/Q4 results season
    """
    dates = pd.to_datetime(d["date"])

    # ── F&O expiry proximity ─────────────────────────────────────────────
    def _days_to_expiry(dt: pd.Timestamp) -> int:
        yr, mo = dt.year, dt.month
        exp = _last_thursday_of_month(yr, mo)
        if dt.date() > exp:
            mo2 = mo + 1 if mo < 12 else 1
            yr2 = yr if mo < 12 else yr + 1
            exp = _last_thursday_of_month(yr2, mo2)
        return max(0, (exp - dt.date()).days)

    dte = dates.apply(_days_to_expiry)
    d["days_to_expiry"] = dte.values
    d["is_expiry_week"] = (dte <= 4).astype(int).values
    d["is_expiry_day"]  = (dte == 0).astype(int).values

    # ── RBI MPC proximity ────────────────────────────────────────────────
    def _days_to_rbi(dt: pd.Timestamp) -> int:
        diffs = abs((_RBI_DATES - dt).days)
        return int(min(diffs.min(), 30))

    d["days_to_rbi"] = dates.apply(_days_to_rbi).values
    d["is_rbi_week"] = (d["days_to_rbi"] <= 3).astype(int)

    # ── Budget proximity ─────────────────────────────────────────────────
    def _days_to_budget(dt: pd.Timestamp) -> int:
        diffs = abs((_BUDGET_DATES - dt).days)
        return int(min(diffs.min(), 60))

    d["days_to_budget"] = dates.apply(_days_to_budget).values
    d["is_budget_week"] = (d["days_to_budget"] <= 3).astype(int)

    # ── Result season ────────────────────────────────────────────────────
    d["is_result_season"] = dates.dt.month.isin(RESULT_SEASON_MONTHS).astype(int).values

    return d


def _add_global_cues_features(
    d: pd.DataFrame,
    global_cues_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add global market cues features to the stock DataFrame.

    LEAKAGE SAFETY:
    ---------------
    Indian market closes at 3:30 PM IST.
    US market OPENS at 9:30 PM IST (after Indian close).
    Therefore US close on date T happens AFTER Indian close on date T.

    Correct alignment: Indian date T should use US data from date T-1.

    Implementation: we shift global_cues dates FORWARD by 1 calendar day
    (US Friday → Saturday). merge_asof(direction='backward') then matches
    Indian Monday with the Saturday label (= US Friday data). This guarantees
    Indian date T never sees US data from date T or later.
    """
    cues = global_cues_df.copy()
    cues["date"] = pd.to_datetime(cues["date"])

    # Shift cue dates +1 calendar day so they represent "earliest Indian
    # trading day when this US session's data is known"
    cues["date"] = cues["date"] + pd.Timedelta(days=1)
    cues = cues.sort_values("date").reset_index(drop=True)

    # Indian dates as a DataFrame for merge_asof
    indian = pd.DataFrame({"date": pd.to_datetime(d["date"]), "_idx": np.arange(len(d))})
    indian = indian.sort_values("date")

    merged = pd.merge_asof(
        indian,
        cues,
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("7 days"),   # don't use data older than 1 week
    )
    merged = merged.sort_values("_idx").reset_index(drop=True)

    # ── S&P 500 ───────────────────────────────────────────────────────────
    if "sp500_ret" in merged.columns:
        d["sp500_ret_prev"] = merged["sp500_ret"].values
        sp500_close = merged["sp500_close"] if "sp500_close" in merged.columns else pd.Series(dtype=float)
        # 5-day return (momentum)
        if "sp500_close" in merged.columns:
            sc = pd.Series(merged["sp500_close"].values)
            d["sp500_ret_5d"]  = sc.pct_change(5).values
            d["sp500_ret_20d"] = sc.pct_change(20).values

    # ── Nasdaq ────────────────────────────────────────────────────────────
    if "nasdaq_ret" in merged.columns:
        d["nasdaq_ret_prev"] = merged["nasdaq_ret"].values
        if "nasdaq_close" in merged.columns:
            nc = pd.Series(merged["nasdaq_close"].values)
            d["nasdaq_ret_5d"] = nc.pct_change(5).values

    # ── US VIX ────────────────────────────────────────────────────────────
    if "us_vix_close" in merged.columns:
        vix = pd.Series(merged["us_vix_close"].values)
        d["us_vix_level"]   = vix.values
        d["us_vix_ret_1d"]  = vix.pct_change(1).values
        # Z-score vs 20-day rolling mean
        vix_ma  = vix.rolling(20, min_periods=10).mean()
        vix_sd  = vix.rolling(20, min_periods=10).std().replace(0, np.nan)
        vix_z   = (vix - vix_ma) / (vix_sd + 1e-10)
        d["us_vix_zscore"]  = vix_z.values
        d["us_vix_spike"]   = (vix_z > 1.5).astype(float).values
        d["us_vix_regime"]  = pd.cut(vix,
                                     bins=[-np.inf, 15, 25, np.inf],
                                     labels=[0, 1, 2]).astype(float).values

    # ── US Dollar Index ────────────────────────────────────────────────
    if "dxy_ret" in merged.columns:
        d["dxy_ret_prev"] = merged["dxy_ret"].values
        if "dxy_close" in merged.columns:
            dx = pd.Series(merged["dxy_close"].values)
            d["dxy_ret_5d"]  = dx.pct_change(5).values
            d["dxy_ret_20d"] = dx.pct_change(20).values

    # ── Crude Oil ─────────────────────────────────────────────────────────
    if "crude_ret" in merged.columns:
        d["crude_ret_prev"] = merged["crude_ret"].values
        if "crude_close" in merged.columns:
            cr = pd.Series(merged["crude_close"].values)
            d["crude_ret_5d"]  = cr.pct_change(5).values
            d["crude_ret_20d"] = cr.pct_change(20).values

    # ── Nikkei ────────────────────────────────────────────────────────────
    if "nikkei_ret" in merged.columns:
        d["nikkei_ret_prev"] = merged["nikkei_ret"].values
        if "nikkei_close" in merged.columns:
            nk = pd.Series(merged["nikkei_close"].values)
            d["nikkei_ret_5d"] = nk.pct_change(5).values

    return d


def compute_features(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    peer_returns: Optional[Dict[str, pd.Series]] = None,
    market_df: Optional[pd.DataFrame] = None,
    usdinr_df: Optional[pd.DataFrame] = None,
    global_cues_df: Optional[pd.DataFrame] = None,   # Phase 1
) -> pd.DataFrame:
    """
    Build 260+ features from OHLCV + global cues + NSE calendar.
    All rolling/EWM windows are backward-looking — zero look-ahead.
    """
    d = df.copy()
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]

    # 1. Returns
    d["ret_1d"]  = c.pct_change(1);  d["ret_2d"]  = c.pct_change(2)
    d["ret_5d"]  = c.pct_change(5);  d["ret_10d"] = c.pct_change(10)
    d["ret_20d"] = c.pct_change(20); d["log_ret"]  = np.log(c / c.shift(1))

    # 2. SMA / EMA ratios
    for p in [5, 10, 20, 50, 100, 200]:
        d[f"price_sma_{p}"] = c / (c.rolling(p).mean() + 1e-10)
        d[f"price_ema_{p}"] = c / (c.ewm(span=p, adjust=False).mean() + 1e-10)

    # 3. MACD (normalised)
    for fast, slow, sig in [(12, 26, 9), (5, 35, 5)]:
        ema_f = c.ewm(span=fast, adjust=False).mean()
        ema_s = c.ewm(span=slow, adjust=False).mean()
        macd  = ema_f - ema_s
        msig  = macd.ewm(span=sig, adjust=False).mean()
        tag   = f"{fast}_{slow}"
        d[f"macd_pct_{tag}"]        = macd          / (c + 1e-10)
        d[f"macd_signal_pct_{tag}"] = msig          / (c + 1e-10)
        d[f"macd_hist_pct_{tag}"]   = (macd - msig) / (c + 1e-10)

    # 4. RSI
    for p in [7, 14, 21, 28]:
        delta = c.diff()
        up = delta.clip(lower=0).rolling(p).mean()
        dn = (-delta).clip(lower=0).rolling(p).mean()
        d[f"rsi_{p}"] = 100 - 100 / (1 + up / (dn + 1e-10))

    # 5. Bollinger Bands
    for p in [10, 20, 50]:
        sma = c.rolling(p).mean(); std = c.rolling(p).std()
        upper = sma + 2*std; lower = sma - 2*std
        d[f"bb_width_{p}"] = (upper - lower) / (sma + 1e-10)
        d[f"bb_pos_{p}"]   = (c - lower) / (upper - lower + 1e-10)

    # 6. ATR
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    for p in [7, 14, 21]:
        d[f"atr_ratio_{p}"] = tr.rolling(p).mean() / (c + 1e-10)

    # 7. ADX / DI
    hd = h.diff(); ld = -l.diff()
    plus_dm  = np.where((hd > ld) & (hd > 0), hd, 0.0)
    minus_dm = np.where((ld > hd) & (ld > 0), ld, 0.0)
    for p in [14, 21]:
        tr_s = tr.rolling(p).sum()
        pdi  = 100 * pd.Series(plus_dm,  index=d.index).rolling(p).sum() / (tr_s + 1e-10)
        mdi  = 100 * pd.Series(minus_dm, index=d.index).rolling(p).sum() / (tr_s + 1e-10)
        dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
        d[f"adx_{p}"]     = dx.rolling(p).mean()
        d[f"plus_di_{p}"] = pdi; d[f"minus_di_{p}"] = mdi
        d[f"di_diff_{p}"] = pdi - mdi

    # 8. Stochastic
    for p in [9, 14, 21]:
        lo = l.rolling(p).min(); hi = h.rolling(p).max()
        k  = 100 * (c - lo) / (hi - lo + 1e-10)
        d[f"stoch_k_{p}"] = k; d[f"stoch_d_{p}"] = k.rolling(3).mean()

    # 9. CCI
    for p in [14, 20]:
        tp  = (h + l + c) / 3; stp = tp.rolling(p).mean()
        mad = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        d[f"cci_{p}"] = (tp - stp) / (0.015 * mad + 1e-10)

    # 10. Williams %R
    for p in [14, 21]:
        hi = h.rolling(p).max(); lo = l.rolling(p).min()
        d[f"willr_{p}"] = -100 * (hi - c) / (hi - lo + 1e-10)

    # 11. OBV ratio
    obv = (np.sign(c.diff()) * v).cumsum()
    d["obv_ratio"] = obv / (obv.rolling(20).mean() + 1e-10)

    # 12. Volume ratios
    for p in [5, 10, 20, 50]:
        d[f"vol_ratio_{p}"] = v / (v.rolling(p).mean() + 1e-10)
    d["vol_change"]    = v.pct_change()
    d["vol_change_5d"] = v.pct_change(5)

    # 13. Volatility
    log_r = np.log(c / c.shift(1))
    for p in [5, 10, 20, 50]:
        d[f"hist_vol_{p}"] = log_r.rolling(p).std() * np.sqrt(252)
    pk = (np.log(h / l))**2 / (4 * np.log(2))
    gk = 0.5*(np.log(h/l))**2 - (2*np.log(2)-1)*(np.log(c/o))**2
    for p in [10, 20]:
        d[f"parkinson_{p}"] = np.sqrt(pk.rolling(p).mean() * 252)
        d[f"gk_vol_{p}"]    = np.sqrt(gk.rolling(p).mean() * 252)

    # 14. Momentum / ROC
    for p in [3, 5, 10, 20, 50]:
        d[f"roc_{p}"] = c.pct_change(p)

    # 15. Candlestick
    body = (c - o).abs(); rng = (h - l).abs() + 1e-10
    d["body_size"]    = body / rng
    d["upper_shadow"] = (h - c.where(c >= o, o)) / rng
    d["lower_shadow"] = (c.where(c <= o, o) - l) / rng
    d["hl_range"]     = rng / (c + 1e-10)
    d["oc_return"]    = (c - o) / (o + 1e-10)
    d["gap"]          = (o - c.shift(1)) / (c.shift(1) + 1e-10)

    # 16. Statistical
    for p in [10, 20, 50]:
        d[f"skew_{p}"]   = log_r.rolling(p).skew()
        d[f"kurt_{p}"]   = log_r.rolling(p).kurt()
        d[f"zscore_{p}"] = (c - c.rolling(p).mean()) / (c.rolling(p).std() + 1e-10)

    # 17. Lag features
    for lag in [1, 2, 3, 5, 10, 20]:
        d[f"ret_lag_{lag}"] = log_r.shift(lag)
        d[f"vol_lag_{lag}"] = (v / (v.rolling(20).mean() + 1e-10)).shift(lag)

    # 18. High/Low position
    for p in [20, 52, 126, 252]:
        hi = h.rolling(p).max(); lo = l.rolling(p).min()
        d[f"pos_hi_{p}"]  = (c - lo) / (hi - lo + 1e-10)
        d[f"dist_hi_{p}"] = (hi - c) / (c + 1e-10)
        d[f"dist_lo_{p}"] = (c - lo) / (c + 1e-10)

    # 19. Trend
    d["trend_20"]   = (c - c.shift(20)) / (c.shift(20) + 1e-10)
    d["trend_cons"] = log_r.rolling(20).apply(lambda x: (x > 0).mean(), raw=True)
    sma50  = c.rolling(50).mean(); sma200 = c.rolling(200).mean()
    d["cross_ratio"] = sma50 / (sma200 + 1e-10)

    # 20. Calendar (cyclic encoding)
    d["dow_sin"] = np.sin(2*np.pi*d["date"].dt.dayofweek/5)
    d["dow_cos"] = np.cos(2*np.pi*d["date"].dt.dayofweek/5)
    d["mon_sin"] = np.sin(2*np.pi*d["date"].dt.month/12)
    d["mon_cos"] = np.cos(2*np.pi*d["date"].dt.month/12)
    d["week_of_year_sin"] = np.sin(2*np.pi*d["date"].dt.isocalendar().week.astype(int)/52)
    d["week_of_year_cos"] = np.cos(2*np.pi*d["date"].dt.isocalendar().week.astype(int)/52)

    # 21. Market regime (rule-based)
    vol_20r = log_r.rolling(20).std()
    vol_q33 = vol_20r.rolling(252, min_periods=60).quantile(0.33)
    vol_q66 = vol_20r.rolling(252, min_periods=60).quantile(0.66)
    regime  = pd.Series(1, index=d.index)
    regime[(sma50 > sma200) & (vol_20r < vol_q66)] = 2
    regime[(sma50 < sma200) & (vol_20r > vol_q33)] = 0
    d["market_regime"]  = regime
    d["regime_bull"]    = (regime == 2).astype(int)
    d["regime_bear"]    = (regime == 0).astype(int)
    d["trend_strength"] = (sma50 / (sma200 + 1e-10)) - 1

    # 22. Cross-sectional / sector
    if peer_returns is not None and symbol is not None:
        sector = SECTOR_MAP.get(symbol)
        if sector:
            peers = [s for s, sec in SECTOR_MAP.items() if sec == sector and s != symbol and s in peer_returns]
            if peers:
                pf = pd.DataFrame({"date": d["date"].values}, index=d.index)
                for p in peers:
                    pf[p] = d["date"].map(peer_returns[p]).values
                sector_avg = pf[peers].mean(axis=1)
                all_rets   = pf[peers].copy(); all_rets["_own"] = d["ret_1d"].values
                d["cs_rank"]         = all_rets.rank(axis=1, pct=True)["_own"].values
                d["sector_avg_ret"]  = sector_avg.values
                d["rel_strength_1d"] = d["ret_1d"].values - sector_avg.values
                d["sector_mom_5d"]   = sector_avg.rolling(5).mean().values
                d["sector_mom_20d"]  = sector_avg.rolling(20).mean().values
                d["sector_vol_10d"]  = sector_avg.rolling(10).std().values
                # Rolling Pearson correlation of stock vs sector average return
                # (causal: only uses past data via rolling window)
                _own_ret = pd.Series(d["ret_1d"].values, index=d.index)
                d["sector_corr_30d"] = _own_ret.rolling(30, min_periods=20).corr(sector_avg)
                d["sector_corr_60d"] = _own_ret.rolling(60, min_periods=30).corr(sector_avg)

    # 23. Nifty50 market context (disabled by default — hurt OOS in testing)
    if market_df is not None and not market_df.empty:
        mkt     = market_df.set_index("date")["close"]
        mkt_al  = mkt.reindex(mkt.index.union(d["date"])).ffill()
        nifty_c = d["date"].map(mkt_al); nifty_c.index = d.index
        nifty_lr = np.log(nifty_c / nifty_c.shift(1))
        d["nifty_ret_1d"]      = nifty_c.pct_change(1)
        d["nifty_ret_5d"]      = nifty_c.pct_change(5)
        d["nifty_ret_20d"]     = nifty_c.pct_change(20)
        d["nifty_ma200_ratio"] = nifty_c / (nifty_c.rolling(200, min_periods=60).mean() + 1e-10)
        nu = nifty_c.diff().clip(lower=0).rolling(14).mean()
        nd = (-nifty_c.diff()).clip(lower=0).rolling(14).mean()
        d["nifty_rsi_14"]      = 100 - 100 / (1 + nu / (nd + 1e-10))
        d["nifty_vol_20d"]     = nifty_lr.rolling(20).std() * np.sqrt(252)
        d["alpha_vs_nifty_1d"] = d["ret_1d"]  - d["nifty_ret_1d"]
        d["alpha_vs_nifty_5d"] = d["ret_5d"]  - d["nifty_ret_5d"]
        d["alpha_vs_nifty_20d"]= d["ret_20d"] - d["nifty_ret_20d"]

    # 24. USD/INR (IT sector)
    if usdinr_df is not None and (symbol is None or symbol in IT_SYMBOLS):
        fx    = usdinr_df.set_index("date")["usdinr_close"]
        fx_al = fx.reindex(fx.index.union(d["date"])).ffill()
        fx_c  = d["date"].map(fx_al); fx_c.index = d.index
        fx_r1 = fx_c.pct_change(1); fx_r5 = fx_c.pct_change(5)
        d["usdinr_ret_1d"]      = fx_r1; d["usdinr_ret_5d"]  = fx_r5
        d["usdinr_ret_20d"]     = fx_c.pct_change(20)
        fu = fx_c.diff().clip(lower=0).rolling(14).mean()
        fd = (-fx_c.diff()).clip(lower=0).rolling(14).mean()
        d["usdinr_rsi_14"]      = 100 - 100 / (1 + fu / (fd + 1e-10))
        d["usdinr_ma20_ratio"]  = fx_c / (fx_c.rolling(20, min_periods=10).mean() + 1e-10)
        d["alpha_vs_usdinr_1d"] = d["ret_1d"] - fx_r1
        d["alpha_vs_usdinr_5d"] = d["ret_5d"] - fx_r5

    # 25. GLOBAL MARKET CUES  (Phase 1)
    if global_cues_df is not None and not global_cues_df.empty:
        d = _add_global_cues_features(d, global_cues_df)

    # 26. NSE CALENDAR FEATURES  (Phase 1)
    d = _add_nse_calendar_features(d)

    # 27. GARCH(1,1) VOLATILITY FEATURES  (causal: recursive filter on past returns)
    # Conditional volatility captures volatility clustering that ATR/hist-vol miss.
    # garch_surprise = standardised residual: large value = unexpected price move.
    # Source: multiple papers 2022-2024 — +3-8% improvement in sideways/bear regimes.
    try:
        from arch import arch_model as _arch_model
        _lr_g = np.log(c / c.shift(1)).fillna(0).values * 100   # log returns in %
        _am   = _arch_model(_lr_g, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
        _res  = _am.fit(disp="off", show_warning=False, update_freq=0)
        _cond_vol  = _res.conditional_volatility   # length == len(_lr_g)
        _std_resid = _res.std_resid                # length == len(_lr_g)
        d["garch_vol"]      = _cond_vol
        d["garch_surprise"] = _std_resid
        # Rolling-quantile regime: 0=low-vol, 1=medium, 2=high-vol
        _gv   = pd.Series(_cond_vol, index=d.index)
        _q33  = _gv.rolling(252, min_periods=60).quantile(0.33)
        _q67  = _gv.rolling(252, min_periods=60).quantile(0.67)
        d["garch_vol_regime"] = np.where(_gv > _q67, 2.0,
                                np.where(_gv < _q33, 0.0, 1.0))
    except Exception:
        pass   # arch not installed or series too short — skip silently

    d = d.replace([np.inf, -np.inf], np.nan)
    return d


def _feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {"date", "timestamp", "symbol", "open", "high", "low", "close", "volume", "target"}
    return [c for c in df.columns
            if c not in exclude and df[c].dtype in ("float64", "float32", "int64", "int32", "int8", "uint8")]


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binary target: 1 = up > MIN_MOVE, 0 = down < -MIN_MOVE, NaN = neutral (dropped)."""
    df = df.copy()
    next_ret = (df["close"].shift(-1) - df["close"]) / (df["close"] + 1e-10)
    df["target"] = np.where(next_ret >  MIN_MOVE, 1.0,
                   np.where(next_ret < -MIN_MOVE, 0.0, np.nan))
    return df


def get_or_compute_features(
    symbol: str,
    raw_df: pd.DataFrame,
    peer_returns: Optional[Dict[str, pd.Series]] = None,
    market_df: Optional[pd.DataFrame] = None,
    usdinr_df: Optional[pd.DataFrame] = None,
    global_cues_df: Optional[pd.DataFrame] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Return features for `symbol` from cache when fresh, else recompute.

    Cache is STALE if:
      - raw/{symbol}.parquet is newer than features/raw/{symbol}_features.parquet
      - raw/global_cues.parquet is newer than cached features
    """
    raw_path        = RAW_DATA_DIR    / f"{symbol}.parquet"
    feat_path       = FEAT_RAW_DIR    / f"{symbol}_features.parquet"
    scaled_path     = FEAT_SCALED_DIR / f"{symbol}_scaled.parquet"
    gcues_path      = RAW_DATA_DIR    / "global_cues.parquet"

    if not force_recompute and feat_path.exists() and raw_path.exists():
        raw_mtime   = raw_path.stat().st_mtime
        feat_mtime  = feat_path.stat().st_mtime
        gcues_mtime = gcues_path.stat().st_mtime if gcues_path.exists() else 0.0
        if feat_mtime >= raw_mtime and feat_mtime >= gcues_mtime:
            cached = _load_parquet(feat_path)
            if cached is not None and not cached.empty:
                print(f"  {symbol:<12} ✓ features cached  ({len(cached)} rows × {len(_feature_cols(cached))} feat)")
                return cached

    print(f"  {symbol:<12} ↻ computing features ...", end=" ", flush=True)
    df    = compute_features(raw_df, symbol=symbol, peer_returns=peer_returns,
                             market_df=market_df, usdinr_df=usdinr_df,
                             global_cues_df=global_cues_df)
    df    = add_target(df)
    fcols = _feature_cols(df)
    df    = df.dropna(subset=["target"])
    df    = df.dropna(subset=fcols, thresh=len(fcols) - 5)
    df[fcols] = df[fcols].fillna(df[fcols].median())
    df    = df.reset_index(drop=True)

    _save_parquet(df, feat_path)

    # Scaled snapshot (for inspection only — training always scales per-window)
    try:
        from sklearn.preprocessing import RobustScaler
        X_sc  = RobustScaler().fit_transform(np.nan_to_num(df[fcols].values.astype(float), nan=0.0))
        sc_df = pd.DataFrame(X_sc, columns=fcols)
        sc_df.insert(0, "date",   df["date"].values)
        sc_df.insert(1, "target", df["target"].values)
        _save_parquet(sc_df, scaled_path)
    except Exception:
        pass

    print(f"✓ {len(df)} rows × {len(fcols)} features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_top_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    fs_window: Dict,
    symbol: str = "",
    top_n: int = N_TOP_FEATURES,
) -> List[str]:
    """
    Select top-N features by LightGBM split importance.
    Zero leakage: only train+val data from fs_window used.

    Force-include logic (guarantees key signals are never dropped):
      All stocks  : GLOBAL_CUES_FEATURES (S&P, VIX, DXY, etc.)
      IT stocks   : + USDINR_FEATURES (Nasdaq, USD/INR)
      Banking     : + BANKING_CUES_FEATURES (Crude, DXY, RBI calendar)
    """
    from sklearn.preprocessing import RobustScaler
    from lightgbm import LGBMClassifier, early_stopping as lgb_es, log_evaluation as lgb_log

    ws, we = fs_window["train_start"], fs_window["train_end"]
    vs, ve = fs_window["val_start"],   fs_window["val_end"]

    sc   = RobustScaler()
    X_tr = np.nan_to_num(sc.fit_transform(X[ws:we]), nan=0.0)
    X_va = np.nan_to_num(sc.transform(X[vs:ve]),     nan=0.0)

    try:
        fs_mdl = LGBMClassifier(**LGBM_FS_PARAMS)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fs_mdl.fit(X_tr, y[ws:we],
                       eval_set=[(X_va, y[vs:ve])],
                       callbacks=[lgb_es(40, verbose=False), lgb_log(period=-1)])
        importances = fs_mdl.feature_importances_

        # Build forced set
        forced_feats: List[str] = list(GLOBAL_CUES_FEATURES)
        if symbol in IT_SYMBOLS:
            forced_feats += USDINR_FEATURES
        if symbol in BANKING_SYMBOLS:
            forced_feats += BANKING_CUES_FEATURES

        forced = [f for f in forced_feats if f in feature_names]
        forced_idx = {feature_names.index(f) for f in forced}
        slots  = max(top_n - len(forced), 1)
        others = [i for i in np.argsort(importances)[::-1] if i not in forced_idx]
        combined = sorted(forced_idx | set(others[:slots]))
        return [feature_names[i] for i in combined]
    except Exception:
        return feature_names   # fallback: keep all


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — EXPANDING WINDOW SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

def build_windows(n: int) -> List[Dict]:
    windows = []
    ratio   = INITIAL_TRAIN_RATIO
    while ratio <= MAX_TRAIN_RATIO:
        train_end = int(n * ratio)
        val_size  = max(int(train_end * 0.10), 20)
        tr_end    = train_end - val_size
        va_start  = tr_end;  va_end   = train_end
        te_start  = train_end
        next_r    = ratio + EXPANSION_STEP
        te_end    = int(n * (next_r + EXPANSION_STEP)) if next_r <= MAX_TRAIN_RATIO else n
        te_end    = min(te_end, n)
        if te_end - te_start < MIN_TEST_SAMPLES:
            ratio = round(ratio + EXPANSION_STEP, 4); continue
        windows.append({
            "id": len(windows)+1, "train_start": 0,
            "train_end": tr_end, "val_start": va_start, "val_end": va_end,
            "test_start": te_start, "test_end": te_end, "train_ratio": ratio,
        })
        ratio = round(ratio + EXPANSION_STEP, 4)
    return windows


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — TRAIN ONE WINDOW  (with temperature-scaling calibration)
# ══════════════════════════════════════════════════════════════════════════════

def _temperature_scale(val_probs: np.ndarray, y_val: np.ndarray) -> float:
    """
    Find temperature T via NLL minimization on the validation set.
    T > 1 compresses probabilities toward 0.5 (more uncertain).
    T < 1 spreads probabilities away from 0.5 (more decisive).
    Single parameter → cannot overfit even on small validation sets.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import expit

    vp = np.clip(val_probs, 1e-7, 1 - 1e-7)
    logits = np.log(vp / (1 - vp))

    def nll(T: float) -> float:
        T = max(T, 0.05)
        cal = np.clip(expit(logits / T), 1e-7, 1 - 1e-7)
        return -float(np.mean(y_val * np.log(cal) + (1 - y_val) * np.log(1 - cal)))

    try:
        res = minimize_scalar(nll, bounds=(0.1, 8.0), method="bounded")
        return float(res.x)
    except Exception:
        return 1.0   # no scaling


def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    from scipy.special import expit
    probs  = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = np.log(probs / (1 - probs))
    return np.array(expit(logits / max(T, 0.05)), dtype=float)


def train_window(
    X: np.ndarray,
    y: np.ndarray,
    window: Dict,
    feature_names: List[str],
    symbol: str,
    window_save_path: Path,
    regimes: Optional[np.ndarray] = None,
    save_dl_keras: bool = True,
) -> Optional[Dict]:
    """
    Train 10-model ensemble on one expanding window.

    Preprocessing fork after RobustScaler:
        ┌── PCA(90% var)      → LightGBM + XGBoost (2D tree models)
        └── sliding_window    → LSTM + BiLSTM + GRU + CNN-LSTM + CNN-GRU
                                + TCN-GRU + TCN-Transformer + NBEATS (3D seq models)

    Pipeline:
        Winsorise (1/99 pct, fit on train)
        → RobustScaler (fit on train) + clip [-5, 5]
        ├── PCA 90% var (fit on train)
        │   → LightGBM + XGBoost  (val-logloss weighted soft vote)
        └── create_sequences(SEQ_LEN=20)
            → LSTM + BiLSTM + GRU + CNN-LSTM + CNN-GRU
              + TCN-GRU + TCN-Transformer + NBEATS  (per-model soft vote)
        → 10-model val-logloss weighted soft vote
        → Meta-learner stacking  (LogisticRegression on 10 val-prob columns)
        → Regime-specific LightGBM  (blend 60/40 with global; tree branch only)
        → Temperature scaling calibration  (fit on val, apply to test)

    save_dl_keras : if False, skip saving DL .keras checkpoints for this window
                    (set False for intermediate windows; True for last window only
                     to reduce disk usage: N_windows × N_DL_models × ~2MB each).
    """
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix,
    )

    ws, we = window["train_start"], window["train_end"]
    vs, ve = window["val_start"],   window["val_end"]
    ts, te = window["test_start"],  window["test_end"]

    X_tr_raw = X[ws:we]; y_train = y[ws:we]
    X_va_raw = X[vs:ve]; y_val   = y[vs:ve]
    X_te_raw = X[ts:te]; y_test  = y[ts:te]

    # Winsorise
    p01 = np.nanpercentile(X_tr_raw, 1,  axis=0)
    p99 = np.nanpercentile(X_tr_raw, 99, axis=0)
    X_tr_raw = np.clip(X_tr_raw, p01, p99)
    X_va_raw = np.clip(X_va_raw, p01, p99)
    X_te_raw = np.clip(X_te_raw, p01, p99)

    # RobustScaler
    scaler  = RobustScaler()
    X_train = np.clip(np.nan_to_num(scaler.fit_transform(X_tr_raw), nan=0.0), -5, 5)
    X_val   = np.clip(np.nan_to_num(scaler.transform(X_va_raw),     nan=0.0), -5, 5)
    X_test  = np.clip(np.nan_to_num(scaler.transform(X_te_raw),     nan=0.0), -5, 5)

    out_pct = np.mean((X_train < -1) | (X_train > 1)) * 100
    print(f"      [scaler] range [{X_train.min():.2f}, {X_train.max():.2f}]  {out_pct:.1f}% outside [-1,1]")

    # ── TREE BRANCH: PCA ──────────────────────────────────────────────────────
    # Keep pre-PCA scaled arrays for DL sequence creation
    X_train_sc = X_train.copy()   # (n_train, n_features) — used by DL
    X_val_sc   = X_val.copy()
    X_test_sc  = X_test.copy()

    pca     = PCA(n_components=0.90, svd_solver="full", random_state=RANDOM_SEED)
    X_train = pca.fit_transform(X_train)    # (n_train, n_pca) — tree branch
    X_val   = pca.transform(X_val)
    X_test  = pca.transform(X_test)
    n_pc    = pca.n_components_
    evr     = float(pca.explained_variance_ratio_.sum())
    print(f"      [PCA]    {n_pc} components → {evr:.1%} explained variance")
    pca_names = [f"PC{i+1}" for i in range(n_pc)]

    # ── DL BRANCH: build continuous PCA block + sequences ────────────────────
    # X_full_scaled spans [ws..te) continuously (train+val+test).
    # Scaler was fit on X_train only — applying transform here is leakage-free.
    # PCA was fit on X_train_sc only — same leakage-free guarantee.
    #
    # FIX (P4): DL now receives PCA-transformed features (~21 uncorrelated PCs)
    # instead of raw 50 correlated features. This eliminates the colinearity
    # problem (ret_1d/ret_2d/ret_5d, rsi_7/14/21, etc.) that confused LSTM gating
    # and reduces input noise substantially. Expected gain: +1.5-2.5% DL accuracy.
    X_full_block = np.nan_to_num(np.clip(X[ws:te], p01, p99), nan=0.0)
    X_full_scaled = np.clip(scaler.transform(X_full_block), -5, 5)
    X_full_pca    = np.clip(pca.transform(X_full_scaled),   -5, 5)  # (~21 PCs)
    # Local offsets into the full block: train[0, we-ws)  val[vs-ws, ve-ws)  test[ts-ws, te-ws)
    _dl_local = dict(ws=0, we=we-ws, vs=vs-ws, ve=ve-ws, ts=ts-ws, te=te-ws)

    (X_tr_seq, y_tr_dl,
     X_va_seq, y_va_dl,
     X_te_seq, y_te_dl) = get_dl_splits(
        X_scaled=X_full_pca,     # PCA features instead of raw scaled
        y=y[ws:te],
        seq_len=DL_SEQ_LEN,
        **_dl_local,
    ) if _DL_AVAILABLE else ((None,)*6)
    # y_va_dl == y_val  (same y[vs:ve] labels)  ← verified by construction
    # y_te_dl == y_test (same y[ts:te] labels)

    # Temporal sample weights (exponential decay, half-life = 1 year)
    ages = np.arange(len(y_train) - 1, -1, -1, dtype=float)
    w    = (np.exp(-ages / 252.0) / np.exp(-ages / 252.0).mean()).astype(np.float32)

    # DL temporal weights aligned to DL training sequences
    if _DL_AVAILABLE and X_tr_seq is not None:
        n_dl_tr  = len(y_tr_dl)
        dl_ages  = np.arange(n_dl_tr - 1, -1, -1, dtype=float)
        w_dl     = (np.exp(-dl_ages / 252.0) / np.exp(-dl_ages / 252.0).mean()).astype(np.float32)
    else:
        w_dl = None

    # ── Train all 7 models ────────────────────────────────────────────────────
    trained_models: Dict = {}
    test_preds:     Dict = {}
    test_probs:     Dict = {}
    val_probs_each: Dict = {}

    # Class imbalance ratio for this window (bull-market drift biases toward UP)
    _n_up   = max((y_train == 1).sum(), 1)
    _n_down = max((y_train == 0).sum(), 1)
    _spw    = float(_n_down) / float(_n_up)   # scale_pos_weight for XGB

    # Tree models (LightGBM + XGBoost) on PCA features
    for ModelClass, name in [(LightGBMClassifier, "LightGBM"), (XGBoostClassifier, "XGBoost")]:
        try:
            if name == "XGBoost":
                mdl = ModelClass(n_jobs=_N_JOBS, scale_pos_weight=_spw)
            else:
                mdl = ModelClass(n_jobs=_N_JOBS, is_unbalance=True)
            mdl.train(X_train, y_train, X_val, y_val,
                      feature_names=pca_names, verbose=False, sample_weight=w)
            test_preds[name]     = mdl.predict(X_test)
            test_probs[name]     = mdl.predict_proba(X_test)
            val_probs_each[name] = mdl.predict_proba(X_val)
            trained_models[name] = mdl
        except Exception as exc:
            print(f"      [{name}] ✗ {exc}")

    if not test_preds:
        return None

    # Deep Learning models on sequence features (seq_len × n_features)
    if _DL_AVAILABLE and X_tr_seq is not None and len(X_tr_seq) >= 50:
        n_feat_dl = X_full_pca.shape[1]   # PCA components (~21), not raw features (50)
        for DLClass, dl_name in _DL_CLASSES:
            try:
                mdl = DLClass(seq_len=DL_SEQ_LEN, n_features=n_feat_dl)
                mdl.train(X_tr_seq, y_tr_dl, X_va_seq, y_va_dl,
                          verbose=False, sample_weight=w_dl)
                test_preds[dl_name]     = mdl.predict(X_te_seq)
                test_probs[dl_name]     = mdl.predict_proba(X_te_seq)
                val_probs_each[dl_name] = mdl.predict_proba(X_va_seq)
                trained_models[dl_name] = mdl  # kept for checkpoint save; freed after
                dl_acc = accuracy_score(y_test, test_preds[dl_name])
                print(f"      [{dl_name:8s}] acc={dl_acc:.3f}")
            except (MemoryError, Exception) as exc:
                print(f"      [{dl_name}] ✗ {exc}")
            finally:
                # Aggressively free Keras graph + TF tensors between DL models
                try:
                    import keras; keras.backend.clear_session()
                except Exception:
                    pass
                gc.collect()

    # Val-logloss weighted ensemble
    try:
        from sklearn.metrics import log_loss
        wts   = {n: 1.0 / max(log_loss(y_val, val_probs_each[n]), 1e-6)
                 for n in trained_models}
        total_w = sum(wts.values())
        avg_prob = sum((wts[n] / total_w) * test_probs[n] for n in trained_models)
        val_avg  = sum((wts[n] / total_w) * val_probs_each[n] for n in trained_models)
    except Exception:
        avg_prob = np.mean(list(test_probs.values()), axis=0)
        val_avg  = np.mean(list(val_probs_each.values()), axis=0)

    # Meta-learner stacking — elastic-net logistic (C=2.0, was 0.3) to avoid
    # coefficient degeneration. L1 drops noisy models; keep only if val-acc lifts
    # AND coefs are decisive (L1 > 0.1). Previously 49% of stocks got meta=uniform.
    meta_model = None
    if len(trained_models) >= 2:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score as _acc

            meta_Xv  = np.column_stack([val_probs_each[n] for n in trained_models])
            meta_Xt  = np.column_stack([test_probs[n]     for n in trained_models])
            base_acc = _acc(y_val, (np.mean(meta_Xv, axis=1) >= 0.5).astype(int))

            if len(np.unique(y_val)) == 2:
                candidate = LogisticRegression(
                    C=2.0, max_iter=500, random_state=RANDOM_SEED,
                    penalty="elasticnet", l1_ratio=0.3, solver="saga",
                )
                candidate.fit(meta_Xv, y_val)
                meta_acc  = _acc(y_val, candidate.predict(meta_Xv))
                meta_prob = candidate.predict_proba(meta_Xt)[:, 1]
                meta_pred = (meta_prob >= 0.5).astype(int)
                coef_l1   = float(np.sum(np.abs(candidate.coef_[0])))
                if meta_acc > base_acc and len(np.unique(meta_pred)) > 1 and coef_l1 > 0.1:
                    meta_model = candidate
                    avg_prob   = meta_prob
                    val_avg    = candidate.predict_proba(meta_Xv)[:, 1]
        except Exception:
            pass

    # Regime-specific LightGBM
    regime_lgb_models: Dict = {}
    win_path = window_save_path / f"window_{window['id']:02d}"

    if regimes is not None:
        tr_reg   = regimes[ws:we]; te_reg = regimes[ts:te]
        global_s = np.mean(list(test_probs.values()), axis=0)
        routed   = global_s.copy()
        for rv, rtag in [(2, "bull"), (1, "sideways"), (0, "bear")]:
            mask_tr = (tr_reg == rv)
            if mask_tr.sum() < MIN_REGIME_SAMPLES or len(np.unique(y_train[mask_tr])) < 2:
                continue
            try:
                rmdl = LightGBMClassifier(n_jobs=_N_JOBS)
                rmdl.train(X_train[mask_tr], y_train[mask_tr], X_val, y_val,
                           feature_names=pca_names, verbose=False)
                regime_lgb_models[rv] = rmdl
                mask_te = (te_reg == rv)
                if mask_te.any():
                    routed[mask_te] = rmdl.predict_proba(X_test[mask_te])
                win_path.mkdir(parents=True, exist_ok=True)
                with open(win_path / f"lgb_{rtag}.pkl", "wb") as f:
                    pickle.dump(rmdl, f)
            except Exception:
                pass
        if regime_lgb_models:
            # Adaptive blend: weight by val-set log-loss (regime-routed vs global).
            # Replaces fixed 60/40 — better regime model → higher weight automatically.
            try:
                from sklearn.metrics import log_loss as _ll
                val_reg      = regimes[vs:ve]
                global_val_s = np.mean(list(val_probs_each.values()), axis=0)
                routed_val   = global_val_s.copy()
                for rv2 in regime_lgb_models:
                    mask_va = (val_reg == rv2)
                    if mask_va.any():
                        routed_val[mask_va] = regime_lgb_models[rv2].predict_proba(X_val[mask_va])
                regime_ll = _ll(y_val, np.clip(routed_val,   1e-7, 1 - 1e-7))
                global_ll = _ll(y_val, np.clip(global_val_s, 1e-7, 1 - 1e-7))
                # Softmax-style: exp(-loss) gives higher weight to lower-loss model
                rw = np.exp(-regime_ll); gw = np.exp(-global_ll)
                blend_alpha = float(np.clip(rw / (rw + gw), 0.40, 0.75))
                print(f"      [regime] adaptive blend α={blend_alpha:.2f}  "
                      f"(regime_ll={regime_ll:.4f}  global_ll={global_ll:.4f})")
            except Exception:
                blend_alpha = 0.6   # fallback to original default
            avg_prob = blend_alpha * routed + (1.0 - blend_alpha) * global_s

    # ── Temperature scaling calibration  (Phase 1) ───────────────────────
    temperature = 1.0
    try:
        temperature = _temperature_scale(val_avg, y_val)
        avg_prob    = _apply_temperature(avg_prob, temperature)
        print(f"      [calib]  temperature T={temperature:.3f}  "
              f"(prob spread: [{avg_prob.min():.3f}, {avg_prob.max():.3f}])")
    except Exception:
        pass

    # Directional prediction: UP / NEUTRAL / DOWN
    # UP     : avg_prob > 0.5  (model leans bullish)
    # NEUTRAL: avg_prob == 0.5 (exact tie — rare with neural nets; ~0%)
    # DOWN   : avg_prob < 0.5  (model leans bearish)
    ens_pred = (avg_prob > 0.5).astype(int)   # > (not >=) to honour the 3-way split

    # Directional accuracy breakdown
    _mask_up      = avg_prob > 0.5
    _mask_down    = avg_prob < 0.5
    _mask_neutral = avg_prob == 0.5
    _pct_neutral  = float(_mask_neutral.mean())
    _pct_up       = float(_mask_up.mean())
    _pct_down     = float(_mask_down.mean())
    # Precision for each direction: when model says UP, how often actual=1?
    _dir_acc_up   = float(y_test[_mask_up].mean())   if _mask_up.any()   else 0.0
    _dir_acc_down = float(1 - y_test[_mask_down].mean()) if _mask_down.any() else 0.0

    acc  = accuracy_score(y_test, ens_pred)
    f1   = f1_score(y_test, ens_pred, zero_division=0)
    prec = precision_score(y_test, ens_pred, zero_division=0)
    rec  = recall_score(y_test, ens_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, avg_prob)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_test, ens_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    per_model = {n: float(accuracy_score(y_test, p)) for n, p in test_preds.items()}

    # Save checkpoint
    win_path.mkdir(parents=True, exist_ok=True)

    # Save tree models as .pkl
    for name in ("LightGBM", "XGBoost"):
        if name in trained_models:
            with open(win_path / f"{name.lower()}.pkl", "wb") as f:
                pickle.dump(trained_models[name], f)

    # Save DL models as .keras (Keras 3.x native format)
    # Only saved for the last window (save_dl_keras=True) to minimise disk usage.
    # Intermediate windows: predictions/probs already extracted; .keras not needed.
    _dl_model_keys = [dn for _, dn in _DL_CLASSES]
    _dl_models_saved = []
    if save_dl_keras:
        for dl_name in _dl_model_keys:
            if dl_name in trained_models:
                keras_path = win_path / f"{dl_name.lower()}.keras"
                try:
                    trained_models[dl_name].save(str(keras_path))
                    _dl_models_saved.append(dl_name)
                except Exception as _save_err:
                    print(f"      [{dl_name}] save error: {_save_err}")

    # Preprocessing artifacts
    with open(win_path / "scaler.pkl",        "wb") as f: pickle.dump(scaler,     f)
    with open(win_path / "pca.pkl",           "wb") as f: pickle.dump(pca,        f)
    with open(win_path / "winsor_bounds.pkl", "wb") as f: pickle.dump((p01, p99), f)
    if meta_model is not None:
        with open(win_path / "meta_model.pkl", "wb") as f: pickle.dump(meta_model, f)

    # Temperature calibration
    with open(win_path / "calibration.json", "w") as f:
        json.dump({"temperature": temperature}, f)

    # DL metadata needed for inference (seq_len, feature count, meta column order)
    _meta_col_order = list(trained_models.keys())   # preserves insertion order (LightGBM first)
    with open(win_path / "dl_meta.json", "w") as f:
        json.dump({
            "seq_len":      DL_SEQ_LEN,
            "n_features":   int(X_full_scaled.shape[1]) if _DL_AVAILABLE and X_tr_seq is not None else 0,
            "dl_models":    _dl_models_saved,
            "meta_columns": _meta_col_order,
        }, f, indent=2)

    with open(win_path / "meta.json", "w") as f:
        json.dump({
            "symbol": symbol, "window_id": window["id"],
            "train_ratio": window["train_ratio"],
            "train_size": we-ws, "val_size": ve-vs, "test_size": te-ts,
            "accuracy": acc, "f1": f1, "auc": auc, "temperature": temperature,
            "per_model": per_model,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "dl_models": _dl_models_saved,
            "dir_acc_up": _dir_acc_up, "dir_acc_down": _dir_acc_down,
            "pct_neutral": _pct_neutral, "pct_up": _pct_up, "pct_down": _pct_down,
        }, f, indent=2)

    # ── Free DL model objects from trained_models to reclaim RAM ──────────────
    # Predictions/probs are already extracted into numpy arrays above.
    # Model checkpoints are saved to disk (win_path/*.keras).
    # Only keep tree models (tiny) — DL production save uses checkpoint files.
    _dl_model_names = [dn for _, dn in _DL_CLASSES]
    for dn in _dl_model_names:
        if dn in trained_models:
            del trained_models[dn]
    try:
        import keras; keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

    return {
        "window": window, "models": trained_models,
        "meta_model": meta_model, "regime_lgb_models": regime_lgb_models,
        "scaler": scaler, "pca": pca, "winsor_bounds": (p01, p99),
        "temperature": temperature,
        "win_path": win_path,  # for production save — reload DL from checkpoint
        "dl_meta": {"seq_len": DL_SEQ_LEN,
                    "n_features": int(X_full_scaled.shape[1]) if _DL_AVAILABLE and X_tr_seq is not None else 0,
                    "dl_models": _dl_models_saved,
                    "meta_columns": _meta_col_order},
        "y_test": y_test, "ens_pred": ens_pred, "avg_prob": avg_prob,
        "accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "auc": auc,
        "per_model": per_model, "test_preds": test_preds,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "dir_acc_up": _dir_acc_up, "dir_acc_down": _dir_acc_down,
        "pct_neutral": _pct_neutral, "pct_up": _pct_up, "pct_down": _pct_down,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _mpl():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try: plt.style.use(PLOT_STYLE)
    except Exception: pass
    return plt


def plot_oos_accuracy(window_rows: List[Dict], symbol: str, out_dir: Path) -> None:
    plt = _mpl()
    ids  = [w["window_id"]    for w in window_rows]
    accs = [w["oos_accuracy"] for w in window_rows]
    lgbs = [w.get("lgbm_acc", 0) for w in window_rows]
    xgbs = [w.get("xgb_acc",  0) for w in window_rows]
    temps = [w.get("temperature", 1.0) for w in window_rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=PLOT_FIGSIZE_TALL, sharex=True)
    x = np.arange(len(ids)); w = 0.25
    ax1.bar(x - w,     accs, w*2, label="Ensemble", color="#2196F3", alpha=0.85)
    ax1.bar(x + w*0.5, lgbs, w,   label="LightGBM", color="#4CAF50", alpha=0.7)
    ax1.bar(x + w*1.5, xgbs, w,   label="XGBoost",  color="#FF9800", alpha=0.7)
    ax1.axhline(0.58, color="red",    ls="--", lw=1.2, label="58% target")
    ax1.axhline(0.50, color="orange", ls=":",  lw=1.0, label="50% baseline")
    ax1.set_ylabel("OOS Accuracy"); ax1.set_title(f"{symbol} — OOS Accuracy by Window")
    ax1.set_ylim(0.35, 0.80); ax1.legend(loc="upper left", fontsize=8)

    ax2.bar(x, temps, color="#9C27B0", alpha=0.75)
    ax2.axhline(1.0, color="grey", ls="--", lw=1, label="T=1 (no scaling)")
    ax2.set_xticks(x); ax2.set_xticklabels([f"W{i}" for i in ids])
    ax2.set_ylabel("Temperature T"); ax2.set_title("Calibration Temperature (T<1=spread out, T>1=compress)")
    ax2.legend(fontsize=8)
    fig.tight_layout(); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "oos_accuracy.png", dpi=PLOT_DPI); plt.close(fig)


def plot_confusion(window_rows: List[Dict], symbol: str, out_dir: Path) -> None:
    plt = _mpl()
    TP = sum(w.get("tp",0) for w in window_rows); TN = sum(w.get("tn",0) for w in window_rows)
    FP = sum(w.get("fp",0) for w in window_rows); FN = sum(w.get("fn",0) for w in window_rows)
    cm = np.array([[TN, FP], [FN, TP]]); pct = cm / (cm.sum() + 1e-10)
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SQUARE)
    ax.imshow(pct, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({pct[i,j]:.1%})",
                    ha="center", va="center", fontsize=12,
                    color="white" if pct[i,j] > 0.5 else "black")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred DOWN","Pred UP"]); ax.set_yticklabels(["Act DOWN","Act UP"])
    ax.set_title(f"{symbol} — Confusion Matrix (all windows)")
    fig.tight_layout(); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "confusion_matrix.png", dpi=PLOT_DPI); plt.close(fig)


def plot_prediction_history(pred_df: pd.DataFrame, symbol: str, out_dir: Path) -> None:
    plt = _mpl()
    df  = pred_df.copy(); df["date"] = pd.to_datetime(df["date"])
    df["correct"]  = (df["actual"] == df["ensemble_pred"]).astype(int)
    df             = df.sort_values("date")
    df["roll_acc"] = df["correct"].rolling(30, min_periods=10).mean()

    fig, axes = plt.subplots(2, 1, figsize=PLOT_FIGSIZE_TALL, sharex=True)
    axes[0].plot(df["date"], df["roll_acc"], color="#2196F3", lw=1.5, label="30-day rolling acc")
    axes[0].axhline(0.58, color="red",    ls="--", lw=1, label="58% target")
    axes[0].axhline(0.50, color="orange", ls=":",  lw=1, label="50% baseline")
    axes[0].set_ylabel("Accuracy"); axes[0].set_title(f"{symbol} — Rolling OOS Accuracy")
    axes[0].legend(fontsize=8); axes[0].set_ylim(0.2, 0.95)

    if "close_price" in df.columns:
        axes[1].plot(df["date"], df["close_price"], color="#555", lw=1)
        axes[1].set_ylabel("Close Price (₹)")

    fig.tight_layout(); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "prediction_history.png", dpi=PLOT_DPI); plt.close(fig)


def plot_calibration_curve(pred_df: pd.DataFrame, symbol: str, out_dir: Path) -> None:
    """Reliability diagram: do predicted probabilities match observed frequencies?"""
    try:
        from sklearn.calibration import calibration_curve
        plt = _mpl()
        # Approximate ensemble probability from per-model preds (avg)
        if "lgbm_pred" not in pred_df.columns or "xgb_pred" not in pred_df.columns:
            return
        # We only have binary preds here, not raw probs — plot win rate by window instead
        df   = pred_df.copy()
        wids = sorted(df["window_id"].unique())
        window_accs = []
        for wid in wids:
            sub = df[df["window_id"] == wid]
            if len(sub) >= 10:
                acc_w = (sub["actual"] == sub["ensemble_pred"]).mean()
                window_accs.append((wid, acc_w, len(sub)))

        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SQUARE)
        xs = [w[0] for w in window_accs]; ys = [w[1] for w in window_accs]
        ax.plot(xs, ys, "o-", color="#2196F3", lw=2, ms=8, label="Window accuracy")
        ax.axhline(0.58, color="red",    ls="--", lw=1, label="58% target")
        ax.axhline(0.50, color="orange", ls=":",  lw=1, label="50% baseline")
        ax.set_xlabel("Window ID"); ax.set_ylabel("Accuracy")
        ax.set_title(f"{symbol} — Accuracy per Window (calibration view)")
        ax.legend(fontsize=8); ax.set_ylim(0.35, 0.80)
        fig.tight_layout(); out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "window_accuracy.png", dpi=PLOT_DPI); plt.close(fig)
    except Exception:
        pass


def plot_cross_stock_comparison(summary_df: pd.DataFrame, out_dir: Path) -> None:
    plt = _mpl()
    df  = summary_df[summary_df["symbol"] != "AVERAGE"].copy().sort_values("oos_accuracy", ascending=True)
    colors = ["#4CAF50" if v >= 0.58 else "#FF9800" if v >= 0.50 else "#F44336"
              for v in df["oos_accuracy"]]
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)
    bars = ax.barh(df["symbol"], df["oos_accuracy"], color=colors, alpha=0.85)
    ax.axvline(0.58, color="red",    ls="--", lw=1.2, label="58% target")
    ax.axvline(0.50, color="orange", ls=":",  lw=1.0, label="50% baseline")
    for bar, val in zip(bars, df["oos_accuracy"]):
        ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=9)
    ax.set_xlabel("OOS Accuracy"); ax.set_title("Cross-Stock OOS Accuracy (Phase 1)")
    ax.legend(fontsize=8); ax.set_xlim(0.30, 0.80)
    fig.tight_layout(); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "cross_stock_comparison.png", dpi=PLOT_DPI); plt.close(fig)


def plot_signal_heatmap(predictions: List[Dict], out_dir: Path) -> None:
    plt = _mpl()
    if not predictions: return
    df   = pd.DataFrame(predictions)[["symbol", "confidence", "direction"]].copy()
    df["conf_sign"] = df.apply(lambda r: r["confidence"] if r["direction"]=="UP" else -r["confidence"], axis=1)
    df   = df.sort_values("conf_sign", ascending=False)
    colors = ["#4CAF50" if d == "UP" else "#F44336" for d in df["direction"]]
    fig, ax = plt.subplots(figsize=(10, max(4, len(df)*0.5)))
    ax.barh(df["symbol"], df["conf_sign"], color=colors, alpha=0.85)
    ax.axvline(0,                    color="grey",  lw=0.8)
    ax.axvline( CONFIDENCE_THRESHOLD, color="green", ls="--", lw=1, label=f"+{CONFIDENCE_THRESHOLD:.0%} gate")
    ax.axvline(-CONFIDENCE_THRESHOLD, color="red",   ls="--", lw=1, label=f"-{CONFIDENCE_THRESHOLD:.0%} gate")
    ax.set_xlabel("Confidence"); ax.set_title("Next-Day Signal Confidence (Phase 1)")
    ax.legend(fontsize=8)
    fig.tight_layout(); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "signal_heatmap.png", dpi=PLOT_DPI); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — PER-SYMBOL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_symbol(
    symbol: str,
    raw_df: pd.DataFrame,
    run_id: str,
    model_run_path: Path,
    result_run_path: Path,
    peer_returns: Optional[Dict[str, pd.Series]] = None,
    market_df: Optional[pd.DataFrame] = None,
    usdinr_df: Optional[pd.DataFrame] = None,
    global_cues_df: Optional[pd.DataFrame] = None,
    force_recompute_features: bool = False,
) -> Dict:
    from sklearn.metrics import accuracy_score, f1_score

    print(f"\n{'─'*60}")
    print(f"  {symbol}")
    print(f"{'─'*60}")

    feat_df   = get_or_compute_features(
        symbol, raw_df,
        peer_returns=peer_returns, market_df=market_df,
        usdinr_df=usdinr_df, global_cues_df=global_cues_df,
        force_recompute=force_recompute_features,
    )
    feat_cols = _feature_cols(feat_df)
    n_rows    = len(feat_df)

    if n_rows < MIN_TRAIN_SAMPLES + MIN_TEST_SAMPLES:
        print(f"  ⚠ Not enough rows ({n_rows}), skipping")
        return {"symbol": symbol, "status": "skipped", "reason": "too_few_rows"}

    X       = feat_df[feat_cols].values.astype(float)
    y       = feat_df["target"].values.astype(int)
    dates   = feat_df["date"].values
    regimes = feat_df["market_regime"].values.astype(int) if "market_regime" in feat_df.columns else None

    windows = build_windows(n_rows)
    print(f"  [1/3] Walk-forward: {len(windows)} windows  |  {n_rows} rows  |  {len(feat_cols)} features")
    for w in windows:
        print(f"        Win {w['id']}: "
              f"train[0:{w['train_end']}]  val[{w['val_start']}:{w['val_end']}]  "
              f"test[{w['test_start']}:{w['test_end']}]  ({w['train_ratio']:.0%})")

    # Feature selection
    n_feat = len(feat_cols)
    if n_feat > N_TOP_FEATURES and len(windows) >= 1:
        fs_win = windows[-2] if len(windows) >= 2 else windows[0]
        print(f"  [1b] Feature selection: top {N_TOP_FEATURES} of {n_feat} ...", end=" ", flush=True)
        selected = select_top_features(X, y, feat_cols, fs_win, symbol=symbol)
        if len(selected) < n_feat:
            sel_idx   = [feat_cols.index(f) for f in selected]
            X         = X[:, sel_idx]
            feat_cols = selected
            n_feat    = len(feat_cols)
            print(f"✓ {n_feat} features kept")
        else:
            print("✓ (all features kept)")

    print(f"  [2/3] Training...")
    sym_model_path  = model_run_path  / symbol
    sym_result_path = result_run_path / symbol
    sym_plot_path   = sym_result_path / "plots"
    sym_model_path.mkdir(parents=True, exist_ok=True)
    sym_result_path.mkdir(parents=True, exist_ok=True)

    window_rows          = []
    all_preds            = []; all_actuals        = []; all_dates            = []
    all_window_ids       = []; all_lgbm_preds     = []; all_xgb_preds       = []
    all_lstm_preds       = []; all_bilstm_preds   = []; all_gru_preds       = []
    all_cnnlstm_preds    = []; all_cnngru_preds   = []
    all_tcngru_preds     = []; all_tcntransf_preds= []; all_nbeats_preds    = []
    all_closes           = []; all_next_closes    = []
    last_result          = None

    for idx, win in enumerate(windows):
        _is_last_win = (idx == len(windows) - 1)
        res = train_window(X, y, win, feat_cols, symbol, sym_model_path,
                           regimes=regimes, save_dl_keras=_is_last_win)
        if res is None:
            continue
        # Free previous window's models before overwriting (tree models are tiny but still)
        if last_result is not None and last_result is not res:
            last_result.pop("models", None)
            last_result.pop("regime_lgb_models", None)
            last_result.pop("meta_model", None)
            gc.collect()
        last_result = res
        ts, te = win["test_start"], win["test_end"]
        test_dates = dates[ts:te]; n_test = len(res["y_test"])

        all_preds.extend(res["ens_pred"]); all_actuals.extend(res["y_test"])
        all_dates.extend(test_dates); all_window_ids.extend([win["id"]] * n_test)
        tp = res.get("test_preds", {})
        all_lgbm_preds.extend(    tp.get("LightGBM",        [None]*n_test))
        all_xgb_preds.extend(     tp.get("XGBoost",          [None]*n_test))
        all_lstm_preds.extend(    tp.get("LSTM",              [None]*n_test))
        all_bilstm_preds.extend(  tp.get("BiLSTM",            [None]*n_test))
        all_gru_preds.extend(     tp.get("GRU",               [None]*n_test))
        all_cnnlstm_preds.extend( tp.get("CNN_LSTM",          [None]*n_test))
        all_cnngru_preds.extend(  tp.get("CNN_GRU",           [None]*n_test))
        all_tcngru_preds.extend(  tp.get("TCN_GRU",           [None]*n_test))
        all_tcntransf_preds.extend(tp.get("TCN_Transformer",  [None]*n_test))
        all_nbeats_preds.extend(  tp.get("NBEATS",            [None]*n_test))
        all_closes.extend(     feat_df["close"].iloc[ts:te].values)
        all_next_closes.extend(feat_df["close"].shift(-1).iloc[ts:te].values)

        tag = "✓" if res["accuracy"] >= 0.58 else ("~" if res["accuracy"] >= 0.50 else "✗")
        _pm = res["per_model"]
        _dl_line = "  ".join(
            f"{dn}={_pm.get(dn,0):.2%}"
            for dn in ["LSTM", "BiLSTM", "GRU", "CNN_LSTM", "CNN_GRU",
                       "TCN_GRU", "TCN_Transformer", "NBEATS"]
            if dn in _pm
        )
        _dir_up   = res.get("dir_acc_up",  0.0)
        _dir_dn   = res.get("dir_acc_down", 0.0)
        _pct_neut = res.get("pct_neutral",  0.0)
        print(f"    {tag} Win {win['id']} | {res['window']['train_ratio']:.0%} train"
              f" | test_n={te-ts}"
              f" | OOS={res['accuracy']:.2%} | AUC={res['auc']:.3f} | F1={res['f1']:.3f}"
              f" | LGB={_pm.get('LightGBM',0):.2%}"
              f" | XGB={_pm.get('XGBoost', 0):.2%}"
              f" | T={res.get('temperature', 1.0):.2f}")
        print(f"         Dir → UP={_dir_up:.2%}  DOWN={_dir_dn:.2%}  NEUTRAL={_pct_neut:.1%}")
        if _dl_line:
            print(f"         DL  → {_dl_line}")

        window_rows.append({
            "symbol": symbol, "window_id": win["id"], "train_ratio": win["train_ratio"],
            "train_size": win["train_end"], "val_size": win["val_end"]-win["val_start"],
            "test_size": te-ts,
            "test_start": str(test_dates[0])[:10] if len(test_dates) else "",
            "test_end":   str(test_dates[-1])[:10] if len(test_dates) else "",
            "oos_accuracy": res["accuracy"], "auc": res["auc"], "f1": res["f1"],
            "precision": res["precision"], "recall": res["recall"],
            "tp": int(res["tp"]), "fp": int(res["fp"]),
            "tn": int(res["tn"]), "fn": int(res["fn"]),
            # Per-model accuracies — all 10 models
            "lgbm_acc":            _pm.get("LightGBM",        0.0),
            "xgb_acc":             _pm.get("XGBoost",          0.0),
            "lstm_acc":            _pm.get("LSTM",              0.0),
            "bilstm_acc":          _pm.get("BiLSTM",            0.0),
            "gru_acc":             _pm.get("GRU",               0.0),
            "cnn_lstm_acc":        _pm.get("CNN_LSTM",          0.0),
            "cnn_gru_acc":         _pm.get("CNN_GRU",           0.0),
            "tcn_gru_acc":         _pm.get("TCN_GRU",           0.0),
            "tcn_transformer_acc": _pm.get("TCN_Transformer",   0.0),
            "nbeats_acc":          _pm.get("NBEATS",            0.0),
            # Directional accuracy breakdown
            "dir_acc_up":    res.get("dir_acc_up",   0.0),
            "dir_acc_down":  res.get("dir_acc_down",  0.0),
            "pct_neutral":   res.get("pct_neutral",   0.0),
            "pct_up":        res.get("pct_up",        0.0),
            "pct_down":      res.get("pct_down",      0.0),
            "temperature":   res.get("temperature",   1.0),
            "dl_models_trained": len(res.get("dl_meta", {}).get("dl_models", [])),
        })

    if all_preds:
        oos_acc = accuracy_score(all_actuals, all_preds)
        oos_f1  = f1_score(all_actuals, all_preds, zero_division=0)
    else:
        oos_acc = 0.0; oos_f1 = 0.0

    avg_t = np.mean([w.get("temperature", 1.0) for w in window_rows]) if window_rows else 1.0
    print(f"  [3/3] OOS Overall → Accuracy={oos_acc:.2%}  F1={oos_f1:.4f}"
          f"  ({len(all_preds)} preds, {len(windows)} wins, avg_T={avg_t:.2f})")

    # Save results — window_results.csv is the resume marker; save it first
    pd.DataFrame(window_rows).to_csv(sym_result_path / "window_results.csv", index=False)

    # Compute per-model average accuracies across windows for this symbol
    _model_accs = {}
    _model_cols = {
        "lgbm_acc":            "LightGBM",
        "xgb_acc":             "XGBoost",
        "lstm_acc":            "LSTM",
        "bilstm_acc":          "BiLSTM",
        "gru_acc":             "GRU",
        "cnn_lstm_acc":        "CNN_LSTM",
        "cnn_gru_acc":         "CNN_GRU",
        "tcn_gru_acc":         "TCN_GRU",
        "tcn_transformer_acc": "TCN_Transformer",
        "nbeats_acc":          "NBEATS",
    }
    for col, name in _model_cols.items():
        vals = [w[col] for w in window_rows if w.get(col, 0) > 0]
        _model_accs[f"avg_{col}"] = float(np.mean(vals)) if vals else 0.0

    # Average directional accuracy metrics across windows
    _dir_metrics = {}
    for dkey in ["dir_acc_up", "dir_acc_down", "pct_neutral"]:
        vals = [w.get(dkey, 0.0) for w in window_rows]
        _dir_metrics[f"avg_{dkey}"] = float(np.mean(vals)) if vals else 0.0

    # Find best individual model for this symbol
    _best_model = max(_model_accs, key=_model_accs.get, default="") if _model_accs else ""
    _best_model_name = _best_model.replace("avg_", "").replace("_acc", "") if _best_model else "none"
    _best_model_acc  = _model_accs.get(_best_model, 0.0)

    # summary_row.json lets _flush_aggregate_csvs rebuild summary.csv after crashes
    with open(sym_result_path / "summary_row.json", "w") as _sr_f:
        json.dump({
            "symbol": symbol, "status": "ok",
            "oos_accuracy": oos_acc, "oos_f1": oos_f1,
            "n_windows": len(windows), "n_predictions": len(all_preds),
            "n_features": n_feat, "n_rows": n_rows,
            **_model_accs, **_dir_metrics,
            "best_model": _best_model_name,
            "best_model_acc": _best_model_acc,
        }, _sr_f, indent=2)
    pred_df = pd.DataFrame({
        "date": pd.to_datetime(all_dates), "window_id": all_window_ids,
        "close_price": all_closes, "next_close_price": all_next_closes,
        "actual": all_actuals,
        "lgbm_pred":            all_lgbm_preds,
        "xgb_pred":             all_xgb_preds,
        "lstm_pred":            all_lstm_preds,
        "bilstm_pred":          all_bilstm_preds,
        "gru_pred":             all_gru_preds,
        "cnn_lstm_pred":        all_cnnlstm_preds,
        "cnn_gru_pred":         all_cnngru_preds,
        "tcn_gru_pred":         all_tcngru_preds,
        "tcn_transformer_pred": all_tcntransf_preds,
        "nbeats_pred":          all_nbeats_preds,
        "ensemble_pred":        all_preds,
        "direction": ["UP" if p == 1 else "DOWN" for p in all_preds],
        "correct": np.array(all_actuals) == np.array(all_preds),
    })
    pred_df.to_csv(sym_result_path / "predictions.csv", index=False)

    # Plots
    try:
        plot_oos_accuracy(window_rows, symbol, sym_plot_path)
        plot_confusion(window_rows, symbol, sym_plot_path)
        plot_prediction_history(pred_df, symbol, sym_plot_path)
        plot_calibration_curve(pred_df, symbol, sym_plot_path)
        print(f"  Plots → {sym_plot_path.relative_to(V3_ROOT)}")
    except Exception as exc:
        print(f"  ⚠ Plot error: {exc}")

    # Production models (last window)
    if last_result:
        prod_path = MODELS_PROD_DIR / symbol
        prod_path.mkdir(parents=True, exist_ok=True)

        # Tree models → .pkl
        for name in ("LightGBM", "XGBoost"):
            if name in last_result["models"]:
                with open(prod_path / f"{name.lower()}.pkl", "wb") as f:
                    pickle.dump(last_result["models"][name], f)

        # DL models → copy from checkpoint (models freed from memory for RAM savings)
        _last_dl_meta = last_result.get("dl_meta", {})
        _last_win_path = last_result.get("win_path")
        for dl_name in _last_dl_meta.get("dl_models", []):
            src = _last_win_path / f"{dl_name.lower()}.keras" if _last_win_path else None
            if src and src.exists():
                try:
                    shutil.copy2(str(src), str(prod_path / f"{dl_name.lower()}.keras"))
                except Exception as _se:
                    print(f"  [prod save] {dl_name}: {_se}")

        with open(prod_path / "scaler.pkl",        "wb") as f: pickle.dump(last_result["scaler"],        f)
        with open(prod_path / "pca.pkl",           "wb") as f: pickle.dump(last_result["pca"],           f)
        with open(prod_path / "winsor_bounds.pkl", "wb") as f: pickle.dump(last_result["winsor_bounds"], f)
        if last_result.get("meta_model"):
            with open(prod_path / "meta_model.pkl", "wb") as f: pickle.dump(last_result["meta_model"], f)

        # Temperature calibration
        with open(prod_path / "calibration.json", "w") as f:
            json.dump({"temperature": last_result.get("temperature", 1.0)}, f)

        # DL metadata for inference (seq_len, feature count, meta column order)
        with open(prod_path / "dl_meta.json", "w") as f:
            json.dump(_last_dl_meta, f, indent=2)

        regime_tag = {2: "bull", 1: "sideways", 0: "bear"}
        for rv, rmdl in last_result.get("regime_lgb_models", {}).items():
            with open(prod_path / f"lgb_{regime_tag.get(rv,str(rv))}.pkl", "wb") as f:
                pickle.dump(rmdl, f)

        with open(prod_path / "metadata.json", "w") as f:
            json.dump({
                "symbol": symbol, "run_id": run_id, "feature_names": feat_cols,
                "oos_accuracy": oos_acc, "oos_f1": oos_f1,
                "n_features": n_feat, "n_train_rows": n_rows,
                "last_window": windows[-1]["id"] if windows else 0,
                "min_move": MIN_MOVE, "confidence_threshold": CONFIDENCE_THRESHOLD,
                "dl_models": _last_dl_meta.get("dl_models", []),
                "dl_seq_len": _last_dl_meta.get("seq_len", DL_SEQ_LEN),
                "trained_at": datetime.now().isoformat(),
            }, f, indent=2)

    # ── Final memory cleanup for this symbol ──────────────────────────────────
    del last_result
    try:
        import keras; keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

    return {
        "symbol": symbol, "status": "ok",
        "oos_accuracy": oos_acc, "oos_f1": oos_f1,
        "n_windows": len(windows), "n_predictions": len(all_preds),
        "n_features": n_feat, "n_rows": n_rows, "window_rows": window_rows,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — NEXT-DAY PREDICTION  (with temperature scaling at inference)
# ══════════════════════════════════════════════════════════════════════════════

def predict_next_day(
    symbol: str,
    raw_df: pd.DataFrame,
    peer_returns: Optional[Dict[str, pd.Series]] = None,
    market_df: Optional[pd.DataFrame] = None,
    usdinr_df: Optional[pd.DataFrame] = None,
    global_cues_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    """Load production model and predict tomorrow's direction."""
    prod_path = MODELS_PROD_DIR / symbol
    if not (prod_path / "metadata.json").exists():
        return None
    try:
        with open(prod_path / "metadata.json") as f:
            meta = json.load(f)
    except Exception:
        return None

    feat_cols = meta["feature_names"]

    scaler_path = prod_path / "scaler.pkl"
    if not scaler_path.exists():
        return None
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    models = {}
    for pkl in prod_path.glob("*.pkl"):
        if pkl.stem in ("scaler", "meta_model", "pca", "winsor_bounds"):
            continue
        try:
            with open(pkl, "rb") as f:
                models[pkl.stem] = pickle.load(f)
        except Exception:
            pass

    if not models:
        return None

    # Load temperature calibration
    temperature = 1.0
    cal_path = prod_path / "calibration.json"
    if cal_path.exists():
        try:
            with open(cal_path) as f:
                temperature = float(json.load(f).get("temperature", 1.0))
        except Exception:
            pass

    # Compute features
    df = compute_features(raw_df, symbol=symbol, peer_returns=peer_returns,
                          market_df=market_df, usdinr_df=usdinr_df,
                          global_cues_df=global_cues_df)
    df = df.dropna(subset=feat_cols, thresh=len(feat_cols) - 10)
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    df = df.reset_index(drop=True)
    if df.empty or len(feat_cols) != scaler.n_features_in_:
        return None

    # Load PCA and winsor bounds
    pca = None
    if (prod_path / "pca.pkl").exists():
        with open(prod_path / "pca.pkl", "rb") as f:
            pca = pickle.load(f)
    wb = None
    if (prod_path / "winsor_bounds.pkl").exists():
        with open(prod_path / "winsor_bounds.pkl", "rb") as f:
            wb = pickle.load(f)

    # Load DL metadata (seq_len, feature count, meta column ordering)
    dl_meta_inf = {}
    if (prod_path / "dl_meta.json").exists():
        try:
            with open(prod_path / "dl_meta.json") as f:
                dl_meta_inf = json.load(f)
        except Exception:
            pass
    dl_seq_len_inf  = int(dl_meta_inf.get("seq_len",    DL_SEQ_LEN))
    dl_n_feat_inf   = int(dl_meta_inf.get("n_features", len(feat_cols)))
    dl_models_inf   = dl_meta_inf.get("dl_models",    [])
    meta_col_order  = dl_meta_inf.get("meta_columns", [])

    # Need at least dl_seq_len_inf rows for DL inference
    n_rows_inf = len(df)
    if n_rows_inf < max(1, dl_seq_len_inf):
        return None

    # ── Preprocessing: winsorize + scale all needed rows ─────────────────────
    X_raw_N = df[feat_cols].iloc[-dl_seq_len_inf:].values.astype(float)
    X_raw_N = np.nan_to_num(X_raw_N, nan=0.0)
    if wb is not None:
        X_raw_N = np.clip(X_raw_N, wb[0], wb[1])
    X_sc_N = np.clip(scaler.transform(X_raw_N), -5, 5)   # (dl_seq_len, n_features)

    # ── Tree model input (last row, PCA-transformed) ──────────────────────────
    X_sc_1 = X_sc_N[[-1]]                                 # (1, n_features)
    X_tree  = pca.transform(X_sc_1) if pca is not None else X_sc_1  # (1, n_pca)

    # ── DL model input (full sequence) ────────────────────────────────────────
    X_seq_inf = X_sc_N[np.newaxis, :, :]   # (1, dl_seq_len, n_features)

    # ── Tree model inference ──────────────────────────────────────────────────
    probs_dict: Dict[str, float] = {}
    for name, mdl in models.items():
        try:
            p = mdl.predict_proba(X_tree)
            probs_dict[name] = float(p.ravel()[0])
        except Exception:
            pass

    # ── DL model inference ────────────────────────────────────────────────────
    if _DL_AVAILABLE and dl_models_inf:
        _dl_name_to_class = {
            "LSTM":     LSTMClassifier,
            "BiLSTM":   BiLSTMClassifier,
            "GRU":      GRUClassifier,
            "CNN_LSTM": CNNLSTMClassifier,
            "CNN_GRU":  CNNGRUClassifier,
        }
        for dl_name in dl_models_inf:
            keras_path = prod_path / f"{dl_name.lower()}.keras"
            if not keras_path.exists() or dl_name not in _dl_name_to_class:
                continue
            try:
                mdl = _dl_name_to_class[dl_name](seq_len=dl_seq_len_inf, n_features=dl_n_feat_inf)
                mdl.load(str(keras_path))
                p = mdl.predict_proba(X_seq_inf)
                probs_dict[dl_name] = float(p.ravel()[0])
            except Exception as _dl_inf_err:
                pass

    if not probs_dict:
        return None

    # ── Meta-learner stacking (column order preserved from training) ──────────
    avg_prob = float(np.mean(list(probs_dict.values())))
    meta_path = prod_path / "meta_model.pkl"
    if meta_path.exists() and meta_col_order:
        try:
            with open(meta_path, "rb") as f:
                mm = pickle.load(f)
            # Reconstruct columns in the EXACT order used during training
            mx_cols = [probs_dict.get(col, 0.5) for col in meta_col_order
                       if col in probs_dict or col.lower() in {k.lower() for k in probs_dict}]
            # Fallback: use full probs_dict in any order if col order broken
            if len(mx_cols) == mm.n_features_in_:
                mx = np.array(mx_cols).reshape(1, -1)
                avg_prob = float(mm.predict_proba(mx)[0, 1])
            else:
                mx = np.array(list(probs_dict.values())).reshape(1, -1)
                if mx.shape[1] == mm.n_features_in_:
                    avg_prob = float(mm.predict_proba(mx)[0, 1])
        except Exception:
            pass

    # Apply temperature calibration
    avg_prob = float(_apply_temperature(np.array([avg_prob]), temperature)[0])

    regime_val = int(df["market_regime"].iloc[-1]) if "market_regime" in df.columns else 1
    regime_lbl = {0: "bear", 1: "sideways", 2: "bull"}.get(regime_val, "sideways")

    # Regime LightGBM blend (tree model uses PCA features)
    rlgb_path = prod_path / f"lgb_{regime_lbl}.pkl"
    if rlgb_path.exists():
        try:
            with open(rlgb_path, "rb") as f:
                rlgb = pickle.load(f)
            r_prob   = float(rlgb.predict_proba(X_tree).ravel()[0])  # PCA features ✓
            r_prob   = float(_apply_temperature(np.array([r_prob]), temperature)[0])
            avg_prob = 0.6 * r_prob + 0.4 * avg_prob
        except Exception:
            pass

    direction     = 1 if avg_prob >= 0.5 else 0
    confidence    = avg_prob if direction == 1 else 1 - avg_prob
    signal_active = confidence >= CONFIDENCE_THRESHOLD

    return {
        "symbol":        symbol,
        "last_date":     str(df["date"].iloc[-1])[:10],
        "last_close":    float(raw_df["close"].iloc[-1]),
        "direction":     "UP" if direction == 1 else "DOWN",
        "action":        ("BUY" if direction == 1 else "SELL") if signal_active else "HOLD",
        "confidence":    round(confidence, 4),
        "avg_prob":      round(avg_prob, 4),
        "signal_active": signal_active,
        "regime":        regime_val,
        "regime_label":  regime_lbl,
        "temperature":   round(temperature, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN (CLI)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Force UTF-8 stdout so Unicode symbols (→ ✅ ₹ ✓) don't crash on Windows cp1252
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # ── CLI arguments ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="V3 AI Stock Pipeline")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel worker processes (default: auto = min(cpu_count, 8))",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Run only these symbols (default: all 100)",
    )
    parser.add_argument(
        "--serial", action="store_true",
        help="Force serial (single-process) mode for debugging",
    )
    parser.add_argument(
        "--resume", metavar="RUN_ID", default=None,
        help="Resume a previous run (e.g. 20260301_000647) — skips already-completed symbols",
    )
    args = parser.parse_args()

    symbols_to_run: List[str] = args.symbols if args.symbols else list(SYMBOLS)

    # ── Resume or new run ─────────────────────────────────────────────────────
    if args.resume:
        run_id = args.resume
        print(f"  [RESUME] Resuming run {run_id} — completed symbols will be skipped")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    # ── Worker / CPU math ─────────────────────────────────────────────────────
    n_cpu     = os.cpu_count() or 4
    if args.serial:
        n_workers          = 1
        n_jobs_per_worker  = -1   # let each model use all cores
    else:
        # 16 GB RAM → 3 workers (~4 GB each, leaves headroom for OS + data)
        n_workers         = args.workers or min(n_cpu, 3, len(symbols_to_run))
        # give each model max(1, cpu // n_workers) threads
        n_jobs_per_worker = max(1, n_cpu // n_workers)

    print("=" * 70)
    print("  AI STOCK PREDICTION — PRODUCTION PIPELINE V3  [Phase 1]")
    print(f"  Run ID    : {run_id}")
    print(f"  Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Stocks    : {len(symbols_to_run)}  ({', '.join(symbols_to_run[:10])}{'...' if len(symbols_to_run) > 10 else ''})")
    print(f"  Workers   : {n_workers}  (n_jobs/model={n_jobs_per_worker},  cpu_count={n_cpu})")
    print(f"  MIN_MOVE  : {MIN_MOVE*100:.1f}%  (above transaction costs)")
    print(f"  CONF THR  : {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  Data from : {DATA_START_DATE} → today  (incremental)")
    print(f"  Windows   : {INITIAL_TRAIN_RATIO:.0%} → {MAX_TRAIN_RATIO:.0%} (step {EXPANSION_STEP:.0%})")
    print("=" * 70)

    model_run_path  = MODELS_RUNS_DIR  / run_id
    result_run_path = RESULTS_RUNS_DIR / run_id
    run_plot_path   = result_run_path  / "plots"
    for p in [model_run_path, result_run_path, run_plot_path]:
        p.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download (parallel threads) ───────────────────────────────────
    raw_data = download_all_symbols(symbols_to_run)
    if not raw_data:
        print("\n✗ No data downloaded. Exiting.")
        return

    print(f"\n  Downloading auxiliary data ...")
    usdinr_df    = download_usdinr()
    global_cues_df = download_global_cues()
    if usdinr_df     is not None: print(f"  USD/INR     : {len(usdinr_df)} rows")
    if global_cues_df is not None: print(f"  Global cues : {len(global_cues_df)} rows  cols={list(global_cues_df.columns)}")

    market_df: Optional[pd.DataFrame] = None

    peer_returns: Dict[str, pd.Series] = {
        sym: df.set_index("date")["close"].pct_change()
        for sym, df in raw_data.items()
    }

    print(f"\n{'='*70}")
    print(f"  STEPS 2-7 — FEATURES + WALK-FORWARD TRAINING  ({n_workers} workers)")
    print(f"  Per-symbol logs → {result_run_path.relative_to(V3_ROOT)}/<symbol>/run.log")
    print(f"{'='*70}")

    summary_rows: List[Dict] = []
    symbols_with_data = [s for s in symbols_to_run if s in raw_data]

    # Skip symbols already completed (for --resume)
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
        # ── Serial mode (easy debugging) ──────────────────────────────────────
        for symbol in symbols_with_data:
            try:
                result = run_symbol(
                    symbol=symbol, raw_df=raw_data[symbol], run_id=run_id,
                    model_run_path=model_run_path, result_run_path=result_run_path,
                    peer_returns=peer_returns, market_df=market_df,
                    usdinr_df=usdinr_df, global_cues_df=global_cues_df,
                )
                summary_rows.append(result)
            except Exception as exc:
                print(f"\n  {symbol}: ✗ ERROR — {exc}"); traceback.print_exc()
                summary_rows.append({"symbol": symbol, "status": "error", "reason": str(exc)})
            n_done += 1
            print(f"  [{n_done}/{n_total}] {symbol} done")
            _flush_aggregate_csvs(result_run_path)
    else:
        # ── Parallel mode — BATCHED ProcessPoolExecutor ───────────────────────
        # Process symbols in small batches, spawning a FRESH pool each batch.
        # This kills worker processes between batches, freeing all accumulated
        # TF/Keras memory and preventing the OOM cascade that killed prior runs.
        BATCH_SIZE = n_workers * 3   # e.g., 2 workers × 3 = 6 symbols/batch
        mp_ctx = _mp.get_context("spawn")  # safe on Windows + macOS
        batches = [symbols_with_data[i:i+BATCH_SIZE]
                   for i in range(0, len(symbols_with_data), BATCH_SIZE)]
        print(f"  [BATCH] {len(batches)} batches of ≤{BATCH_SIZE} symbols "
              f"({n_workers} workers, fresh pool per batch)")

        for b_idx, batch in enumerate(batches, 1):
            print(f"\n  ── Batch {b_idx}/{len(batches)}: {', '.join(batch)} ──")
            with _cf.ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=mp_ctx,
                initializer=_worker_init,
                initargs=(n_jobs_per_worker,),
            ) as pool:
                future_to_sym = {
                    pool.submit(
                        _run_symbol_worker,
                        symbol,
                        raw_data[symbol],
                        run_id,
                        model_run_path,
                        result_run_path,
                        peer_returns,
                        market_df,
                        usdinr_df,
                        global_cues_df,
                    ): symbol
                    for symbol in batch
                }
                for fut in _cf.as_completed(future_to_sym):
                    symbol = future_to_sym[fut]
                    n_done += 1
                    try:
                        result = fut.result()
                        summary_rows.append(result)
                        status = result.get("status", "?")
                        acc    = f"acc={result['oos_accuracy']:.2%}" if status == "ok" else status
                        print(f"  [{n_done:>3}/{n_total}] ✓ {symbol:<14} {acc}  "
                              f"(log → {result_run_path.relative_to(V3_ROOT)}/{symbol}/run.log)")
                    except Exception as exc:
                        print(f"  [{n_done:>3}/{n_total}] ✗ {symbol}: {exc}")
                        summary_rows.append({"symbol": symbol, "status": "error", "reason": str(exc)})
                    _flush_aggregate_csvs(result_run_path)
            # Pool is destroyed here — all worker processes killed, memory freed
            gc.collect()
            print(f"  [BATCH {b_idx}] Done — workers killed, memory reclaimed")

    print(f"\n{'='*70}")
    print(f"  STEP 8 — NEXT-DAY PREDICTIONS  (parallel)")
    print(f"{'='*70}")
    predictions: List[Dict] = []

    # Predictions are fast — use threads (avoids spawn overhead)
    with _cf.ThreadPoolExecutor(max_workers=min(n_workers * 2, 16)) as tpool:
        pred_futures = {
            tpool.submit(
                _predict_worker, sym, df, peer_returns, market_df, usdinr_df, global_cues_df
            ): sym
            for sym, df in raw_data.items()
        }
        for fut in _cf.as_completed(pred_futures):
            sym  = pred_futures[fut]
            pred = fut.result()
            if pred and pred.get("symbol"):
                predictions.append(pred)

    # Sort and display
    predictions.sort(key=lambda p: p.get("confidence", 0), reverse=True)
    for pred in predictions:
        sym   = pred["symbol"]
        arrow = "UP  " if pred["direction"] == "UP" else "DOWN"
        gate  = "" if pred["signal_active"] else "  [HOLD]"
        print(f"  {arrow} {sym:<14}: {pred['action']:4s}  "
              f"conf={pred['confidence']:.1%}  regime={pred['regime_label']}"
              f"  T={pred.get('temperature',1.0):.2f}"
              f"  (close=₹{pred['last_close']:.2f}  {pred['last_date']}){gate}")

    elapsed = time.time() - t0
    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY  [Phase 1 — MIN_MOVE={MIN_MOVE*100:.1f}%, CONF≥{CONFIDENCE_THRESHOLD*100:.0f}%]")
    print(f"{'='*70}")

    if ok_rows:
        summary_df = pd.DataFrame([{
            "symbol": r["symbol"], "oos_accuracy": r["oos_accuracy"],
            "oos_f1": r["oos_f1"], "n_windows": r["n_windows"],
            "n_predictions": r["n_predictions"], "n_features": r["n_features"],
            "n_rows": r["n_rows"],
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

        # ── Per-model comparison across all stocks ────────────────────────────
        all_win_rows = [w for r in ok_rows for w in r.get("window_rows", [])]
        if all_win_rows:
            aw_df = pd.DataFrame(all_win_rows)
            model_cols = {
                "lgbm_acc":            "LightGBM",
                "xgb_acc":             "XGBoost",
                "lstm_acc":            "LSTM",
                "bilstm_acc":          "BiLSTM",
                "gru_acc":             "GRU",
                "cnn_lstm_acc":        "CNN_LSTM",
                "cnn_gru_acc":         "CNN_GRU",
                "tcn_gru_acc":         "TCN_GRU",
                "tcn_transformer_acc": "TCN_Transformer",
                "nbeats_acc":          "NBEATS",
            }
            model_comp_rows = []
            print(f"\n  ── Per-Model Comparison (across all stocks × windows) ──")
            print(f"  {'Model':<16} {'Avg Acc':>8} {'Med Acc':>8} {'Best':>8} {'Worst':>8} {'Std':>7}")
            print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
            _all_avg = {}
            for col, name in model_cols.items():
                if col in aw_df.columns:
                    vals = aw_df[col].dropna(); vals = vals[vals > 0]
                    if len(vals) > 0:
                        _all_avg[col] = vals.mean()
            for col, name in model_cols.items():
                if col in aw_df.columns:
                    vals = aw_df[col].dropna()
                    vals = vals[vals > 0]  # exclude windows where model didn't train
                    if len(vals) > 0:
                        avg_a = vals.mean()
                        med_a = vals.median()
                        max_a = vals.max()
                        min_a = vals.min()
                        std_a = vals.std()
                        tag = "🏆" if avg_a == max(_all_avg.values(), default=0) else "  "
                        print(f"  {tag}{name:<14} {avg_a:>8.2%} {med_a:>8.2%} {max_a:>8.2%} {min_a:>8.2%} {std_a:>7.3f}")
                        model_comp_rows.append({
                            "model": name, "avg_accuracy": avg_a, "median_accuracy": med_a,
                            "max_accuracy": max_a, "min_accuracy": min_a,
                            "std_accuracy": std_a, "n_datapoints": len(vals),
                        })

            if model_comp_rows:
                mc_df = pd.DataFrame(model_comp_rows).sort_values("avg_accuracy", ascending=False)
                mc_df.to_csv(result_run_path / "model_comparison.csv", index=False)
                best_m = mc_df.iloc[0]
                print(f"\n  🏆 Best Model Overall: {best_m['model']} "
                      f"(avg={best_m['avg_accuracy']:.2%}, std={best_m['std_accuracy']:.3f})")

            aw_df.to_csv(result_run_path / "all_windows_detail.csv", index=False)

        # ── Full summary CSV with per-model columns ──────────────────────────
        avg_row = pd.DataFrame([{
            "symbol": "AVERAGE", "oos_accuracy": avg_acc, "oos_f1": avg_f1,
            "n_windows": summary_df["n_windows"].mean(),
            "n_predictions": summary_df["n_predictions"].sum(),
            "n_features": summary_df["n_features"].mean(),
            "n_rows": summary_df["n_rows"].mean(),
        }])
        summary_df = pd.concat([summary_df, avg_row], ignore_index=True)
        summary_df.to_csv(result_run_path / "summary.csv", index=False)

        try:
            plot_cross_stock_comparison(summary_df, run_plot_path)
        except Exception:
            pass

    if predictions:
        pd.DataFrame(predictions).to_csv(result_run_path / "next_day_predictions.csv", index=False)
        try:
            plot_signal_heatmap(predictions, run_plot_path)
        except Exception:
            pass

    with open(result_run_path / "run_metadata.json", "w") as f:
        json.dump({
            "run_id": run_id, "phase": 1, "symbols": symbols_to_run,
            "data_start": DATA_START_DATE,
            "min_move": MIN_MOVE, "confidence_threshold": CONFIDENCE_THRESHOLD,
            "initial_train": INITIAL_TRAIN_RATIO, "expansion_step": EXPANSION_STEP,
            "max_train": MAX_TRAIN_RATIO, "n_top_features": N_TOP_FEATURES,
            "global_cues_enabled": global_cues_df is not None,
            "nse_calendar_enabled": True,
            "calibration": "temperature_scaling",
            "n_workers": n_workers, "n_jobs_per_worker": n_jobs_per_worker,
            "cpu_count": n_cpu,
            "elapsed_sec": round(elapsed, 1), "n_symbols_ok": len(ok_rows),
            "trained_at": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n  Results  → {result_run_path.relative_to(V3_ROOT)}")
    print(f"  Models   → {(MODELS_RUNS_DIR/run_id).relative_to(V3_ROOT)}")
    print(f"  Prod     → {MODELS_PROD_DIR.relative_to(V3_ROOT)}/{{symbol}}/")
    print(f"  Elapsed  : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  API ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_api(
    symbols: Optional[List[str]] = None,
    progress_callback=None,
) -> Dict:
    if not symbols:
        symbols = SYMBOLS

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    t0     = time.time()

    def _cb(step, total, msg):
        if progress_callback: progress_callback(step, total, msg)

    model_run_path  = MODELS_RUNS_DIR  / run_id
    result_run_path = RESULTS_RUNS_DIR / run_id
    run_plot_path   = result_run_path  / "plots"
    for p in [model_run_path, result_run_path, run_plot_path]:
        p.mkdir(parents=True, exist_ok=True)

    total_steps = len(symbols) + 2; current_step = 0

    _cb(current_step, total_steps, "Downloading market data…")
    raw_data = download_all_symbols(symbols)
    current_step += 1
    if not raw_data:
        raise RuntimeError("No market data downloaded.")

    usdinr_df      = download_usdinr()
    global_cues_df = download_global_cues()
    market_df: Optional[pd.DataFrame] = None
    peer_returns   = {sym: df.set_index("date")["close"].pct_change() for sym, df in raw_data.items()}

    summary_rows: List[Dict] = []
    for symbol in symbols:
        _cb(current_step, total_steps, f"Training {symbol}…")
        if symbol not in raw_data:
            current_step += 1; continue
        try:
            result = run_symbol(
                symbol=symbol, raw_df=raw_data[symbol], run_id=run_id,
                model_run_path=model_run_path, result_run_path=result_run_path,
                peer_returns=peer_returns, market_df=market_df,
                usdinr_df=usdinr_df, global_cues_df=global_cues_df,
            )
            summary_rows.append(result)
        except Exception as exc:
            traceback.print_exc()
            summary_rows.append({"symbol": symbol, "status": "error", "reason": str(exc)})
        current_step += 1

    _cb(current_step, total_steps, "Generating predictions…")
    predictions: List[Dict] = []
    for sym, df in raw_data.items():
        pred = predict_next_day(sym, df, peer_returns=peer_returns,
                                market_df=market_df, usdinr_df=usdinr_df,
                                global_cues_df=global_cues_df)
        if pred: predictions.append(pred)
    current_step += 1

    _cb(total_steps, total_steps, "Building response…")
    signals = {
        pred["symbol"]: {
            "action": pred.get("action", "HOLD"), "direction": pred["direction"],
            "confidence": pred["confidence"], "signal_active": pred.get("signal_active", False),
            "direction_probability": pred["avg_prob"],
            "expected_return": round((pred["confidence"] - 0.5) * 0.04, 4),
            "current_price": pred["last_close"], "last_date": pred["last_date"],
            "regime": pred.get("regime", 1), "regime_label": pred.get("regime_label", "sideways"),
            "temperature": pred.get("temperature", 1.0),
        }
        for pred in predictions
    }

    backtest_results: Dict = {}
    for r in [r for r in summary_rows if r.get("status") == "ok"]:
        sym = r["symbol"]
        v3w = {
            f"W{w['window_id']}": {
                "train_ratio": w.get("train_ratio"), "test_start": w.get("test_start"),
                "test_end": w.get("test_end"), "oos_accuracy": w.get("oos_accuracy"),
                "auc": w.get("auc"), "f1": w.get("f1"),
                "lgbm_acc": w.get("lgbm_acc"), "xgb_acc": w.get("xgb_acc"),
                "temperature": w.get("temperature"),
            }
            for w in r.get("window_rows", [])
        }
        best_wid = max(v3w, key=lambda k: v3w[k]["oos_accuracy"] or 0, default="N/A")
        last_w   = (r.get("window_rows") or [{}])[-1]
        backtest_results[sym] = {
            "directional_accuracy": r["oos_accuracy"],
            "total_return": round((r["oos_accuracy"]-0.5)*0.2, 4),
            "sharpe_ratio": round((r["oos_accuracy"]-0.5)*10, 2),
            "total_trades": r["n_predictions"],
            "oos_accuracy": r["oos_accuracy"], "oos_f1": r["oos_f1"],
            "n_windows": r["n_windows"], "n_features": r["n_features"], "n_rows": r["n_rows"],
            "v3_all_windows": v3w, "v3_best_window": best_wid, "v3_n_features": r["n_features"],
            "model_predictions": {
                "xgb_accuracy": last_w.get("xgb_acc"), "lgb_accuracy": last_w.get("lgbm_acc"),
                "ensemble_accuracy": r["oos_accuracy"],
            },
        }

    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    if ok_rows:
        summary_df = pd.DataFrame([{
            "symbol": r["symbol"], "oos_accuracy": r["oos_accuracy"], "oos_f1": r["oos_f1"],
            "n_windows": r["n_windows"], "n_predictions": r["n_predictions"],
            "n_features": r["n_features"], "n_rows": r["n_rows"],
        } for r in ok_rows])
        avg_row = pd.DataFrame([{
            "symbol": "AVERAGE", "oos_accuracy": summary_df["oos_accuracy"].mean(),
            "oos_f1": summary_df["oos_f1"].mean(),
            "n_windows": summary_df["n_windows"].mean(),
            "n_predictions": summary_df["n_predictions"].sum(),
            "n_features": summary_df["n_features"].mean(),
            "n_rows": summary_df["n_rows"].mean(),
        }])
        summary_df = pd.concat([summary_df, avg_row], ignore_index=True)
        summary_df.to_csv(result_run_path / "summary.csv", index=False)
        all_win_rows = [w for r in ok_rows for w in r.get("window_rows", [])]
        if all_win_rows:
            pd.DataFrame(all_win_rows).to_csv(result_run_path / "all_windows_detail.csv", index=False)
        try:
            plot_cross_stock_comparison(summary_df, run_plot_path)
        except Exception:
            pass

    if predictions:
        pd.DataFrame(predictions).to_csv(result_run_path / "next_day_predictions.csv", index=False)
        try: plot_signal_heatmap(predictions, run_plot_path)
        except Exception: pass

    elapsed = time.time() - t0
    with open(result_run_path / "run_metadata.json", "w") as f:
        json.dump({
            "run_id": run_id, "phase": 1, "symbols": symbols,
            "data_start": DATA_START_DATE, "elapsed_sec": round(elapsed, 1),
            "n_symbols_ok": len(ok_rows), "trained_at": datetime.now().isoformat(),
            "min_move": MIN_MOVE, "confidence_threshold": CONFIDENCE_THRESHOLD,
        }, f, indent=2)

    return {
        "signals": signals, "backtest_results": backtest_results,
        "equity_curve": [], "per_stock_metrics": [],
        "timestamp": datetime.now().isoformat(),
        "run_path": str(result_run_path), "elapsed_sec": round(elapsed, 1),
    }


if __name__ == "__main__":
    main()
