"""
daily_runner.py — Master daily automation for the AlgoTrading system
=====================================================================
Two modes:

  EVENING (T-1, ~6 PM IST):
    1. Fetch today's news sentiment (all 100 symbols)
    2. Run V3 pipeline (fast mode — generates predictions + next_day_predictions.csv)
    3. Build + validate orders → approved_orders.json
    Cron: 0 18 * * 1-5  (Mon–Fri 6 PM IST)

  MORNING (T, 9:00 AM IST):
    1. Angel One login
    2. Fetch live prices for approved symbols
    3. Check slippage guard
    4. Place LIMIT orders
    5. Wait for fills (up to 30 min)
    6. Save execution log
    Cron: 0 9 * * 1-5  (Mon–Fri 9 AM IST)

  EVENING RECONCILE (T, 3:45 PM IST):
    1. Fetch final holdings from Angel One
    2. Compare with execution log
    3. Append to trade_history.parquet
    Cron: 45 15 * * 1-5

Usage:
    python V3/05_live_trading/daily_runner.py --mode evening
    python V3/05_live_trading/daily_runner.py --mode morning --capital 500000
    python V3/05_live_trading/daily_runner.py --mode reconcile
    python V3/05_live_trading/daily_runner.py --mode morning --paper   # dry run
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

_LIVE_DIR = Path(__file__).resolve().parent
_V3_ROOT  = _LIVE_DIR.parent

# Load .env so TRADING_MODE / ANGEL_* are available when cron invokes this script.
_ENV_PATH = _V3_ROOT.parent / ".env"
if _ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH)
    except Exception:
        pass


def _resolve_paper_mode(cli_paper: bool) -> bool:
    """
    Precedence: CLI --paper flag (if set True) > TRADING_MODE env var > default paper.

    TRADING_MODE=live  → real orders on Angel One
    TRADING_MODE=paper → simulated fills only (DEFAULT for safety)
    """
    if cli_paper:
        return True
    mode = os.getenv("TRADING_MODE", "paper").strip().lower()
    return mode != "live"
_PIPELINE_DIR = _V3_ROOT / "07_pipeline"
_NEWS_DIR     = _V3_ROOT / "01_data" / "news"
_ORDERS_DIR   = _LIVE_DIR / "orders"
_RESULTS_DIR  = _V3_ROOT / "06_results" / "runs"
_HISTORY_PATH = _LIVE_DIR / "trade_history.parquet"

# Incremental retrain constants
_RAW_DIR      = _V3_ROOT / "01_data" / "raw"
_FEAT_RAW_DIR = _V3_ROOT / "01_data" / "features" / "raw"
_PROD_DIR     = _V3_ROOT / "02_models" / "production"
_FINETUNE_ROWS       = 252   # last 1 trading year
_FINETUNE_ROUNDS_LGB = 20    # warm-start additional rounds for LightGBM
_FINETUNE_ROUNDS_XGB = 20    # warm-start additional rounds for XGBoost
_MIN_MOVE            = 0.004 # must match config.py MIN_MOVE

# Python executable (venv)
_PYTHON = str(Path(sys.executable))

# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str) -> int:
    print(f"\n  ┌─ {label}")
    result = subprocess.run(cmd, cwd=str(_V3_ROOT.parent))
    print(f"  └─ exit={result.returncode}")
    return result.returncode


def _latest_run_id() -> Optional[str]:
    runs = sorted(_RESULTS_DIR.glob("20*"), reverse=True)
    return runs[0].name if runs else None


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

# ══════════════════════════════════════════════════════════════════════════════
#  INCREMENTAL RETRAIN — Fast daily fine-tune of LightGBM + XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def _load_parquet_safe(path: Path) -> Optional[pd.DataFrame]:
    """Load a parquet file, normalising the date column name. Returns None on error."""
    try:
        df = pd.read_parquet(path)
        if "timestamp" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"timestamp": "date"})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return None


def _feature_cols_inc(df: pd.DataFrame) -> List[str]:
    """Return numeric feature column names, excluding OHLCV/date/target metadata."""
    exclude = {"date", "timestamp", "symbol", "open", "high", "low", "close", "volume", "target"}
    return [c for c in df.columns
            if c not in exclude and df[c].dtype in ("float64", "float32", "int64", "int32", "int8", "uint8")]


def _add_target_inc(df: pd.DataFrame) -> pd.DataFrame:
    """Binary target matching run_pipeline.add_target: 1=up>MIN_MOVE, 0=down<-MIN_MOVE."""
    df = df.copy()
    next_ret = (df["close"].shift(-1) - df["close"]) / (df["close"] + 1e-10)
    df["target"] = np.where(next_ret >  _MIN_MOVE, 1.0,
                   np.where(next_ret < -_MIN_MOVE, 0.0, np.nan))
    return df


def _compute_incremental_features(
    symbol: str,
    raw_df: pd.DataFrame,
    feat_path: Path,
    global_cues_df: Optional[pd.DataFrame],
    usdinr_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """
    Return up-to-date feature DataFrame for `symbol`.

    Strategy:
    1. If features parquet is fresh (newer than raw + global_cues), load and return it.
    2. Otherwise import compute_features from run_pipeline, compute only the new rows
       (rows with date > last cached date), append to cache, and save.

    Falls back to a full recompute on the last 2 years if the cache is empty or missing.
    """
    raw_path    = _RAW_DIR / f"{symbol}.parquet"
    gcues_path  = _RAW_DIR / "global_cues.parquet"

    # ── Freshness check ───────────────────────────────────────────────────────
    if feat_path.exists() and raw_path.exists():
        raw_mtime   = raw_path.stat().st_mtime
        feat_mtime  = feat_path.stat().st_mtime
        gcues_mtime = gcues_path.stat().st_mtime if gcues_path.exists() else 0.0
        if feat_mtime >= raw_mtime and feat_mtime >= gcues_mtime:
            cached = _load_parquet_safe(feat_path)
            if cached is not None and not cached.empty:
                return cached

    # ── Need to (re)compute features ─────────────────────────────────────────
    # Import compute_features lazily to avoid heavy imports at module load time.
    try:
        import importlib.util, importlib
        _pipeline_py = _PIPELINE_DIR / "run_pipeline.py"
        spec = importlib.util.spec_from_file_location("run_pipeline_inc", str(_pipeline_py))
        rp = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(rp)  # type: ignore[union-attr]
        compute_features = rp.compute_features
        add_target       = rp.add_target
    except Exception as _imp_err:
        _log(f"    [inc] Cannot import compute_features: {_imp_err}")
        return None

    # Load existing cache to determine the last computed date
    cached = _load_parquet_safe(feat_path) if feat_path.exists() else None
    last_cached_date = None
    if cached is not None and not cached.empty and "date" in cached.columns:
        last_cached_date = pd.to_datetime(cached["date"]).max()

    # Determine rows to compute: if cache exists, only new rows; else all
    if last_cached_date is not None:
        # Need extra history rows for rolling windows (max window = 200 days)
        lookback_start = last_cached_date - pd.Timedelta(days=300)
        raw_slice = raw_df[raw_df["date"] >= lookback_start].copy()
    else:
        raw_slice = raw_df.copy()

    if raw_slice.empty or len(raw_slice) < 50:
        _log(f"    [inc] {symbol}: insufficient raw rows for feature computation")
        return cached  # return stale cache rather than nothing

    try:
        feat_slice = compute_features(
            raw_slice, symbol=symbol,
            global_cues_df=global_cues_df,
            usdinr_df=usdinr_df,
        )
        feat_slice = add_target(feat_slice)
        feat_slice = feat_slice.dropna(subset=["target"])
        fcols = _feature_cols_inc(feat_slice)
        feat_slice = feat_slice.dropna(subset=fcols, thresh=max(1, len(fcols) - 5))
        feat_slice[fcols] = feat_slice[fcols].fillna(feat_slice[fcols].median())
        feat_slice = feat_slice.reset_index(drop=True)
    except Exception as _fe_err:
        _log(f"    [inc] {symbol}: feature computation failed: {_fe_err}")
        return cached

    if feat_slice.empty:
        return cached

    # Merge with existing cache: drop overlapping rows, append new ones
    if cached is not None and not cached.empty:
        new_rows = feat_slice[feat_slice["date"] > last_cached_date]
        if new_rows.empty:
            return cached  # nothing new
        merged = pd.concat([cached, new_rows], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date").reset_index(drop=True)
    else:
        merged = feat_slice

    # Save updated features cache
    try:
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = feat_path.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(feat_path)
    except Exception as _sv_err:
        _log(f"    [inc] {symbol}: could not save features cache: {_sv_err}")

    return merged


def _finetune_symbol(
    symbol: str,
    feat_df: pd.DataFrame,
    feat_cols: List[str],
    prod_path: Path,
) -> bool:
    """
    Warm-start LightGBM and XGBoost on the last _FINETUNE_ROWS rows.

    Returns True if at least one model was updated.
    """
    # Load production metadata (needed for feature column list + scaler)
    meta_path = prod_path / "metadata.json"
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return False

    prod_feat_cols: List[str] = meta.get("feature_names", feat_cols)

    # Load scaler + optional PCA + winsorisation bounds
    scaler_path = prod_path / "scaler.pkl"
    if not scaler_path.exists():
        return False
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except Exception:
        return False

    pca = None
    if (prod_path / "pca.pkl").exists():
        try:
            with open(prod_path / "pca.pkl", "rb") as f:
                pca = pickle.load(f)
        except Exception:
            pass

    wb = None
    if (prod_path / "winsor_bounds.pkl").exists():
        try:
            with open(prod_path / "winsor_bounds.pkl", "rb") as f:
                wb = pickle.load(f)
        except Exception:
            pass

    # Restrict feature DataFrame to production columns
    available = [c for c in prod_feat_cols if c in feat_df.columns]
    if len(available) < len(prod_feat_cols) * 0.8:
        _log(f"    [inc] {symbol}: only {len(available)}/{len(prod_feat_cols)} "
             f"prod features available — skipping fine-tune")
        return False

    # Use only the last _FINETUNE_ROWS rows for fine-tuning
    window = feat_df.tail(_FINETUNE_ROWS).copy()
    window = window.dropna(subset=["target"])
    if len(window) < 30:
        _log(f"    [inc] {symbol}: only {len(window)} labelled rows — skipping fine-tune")
        return False

    # Build X / y
    X_raw = window[available].values.astype(float)
    y     = window["target"].values.astype(float)
    X_raw = np.nan_to_num(X_raw, nan=0.0)

    # Winsorise then scale (using production-fitted transforms — no refitting)
    if wb is not None:
        X_raw = np.clip(X_raw, wb[0], wb[1])
    try:
        X_sc = np.clip(scaler.transform(X_raw), -5, 5)
    except Exception as _sc_err:
        _log(f"    [inc] {symbol}: scaler.transform failed: {_sc_err}")
        return False

    # PCA transform for tree models
    if pca is not None:
        try:
            X_tree = pca.transform(X_sc)
        except Exception:
            X_tree = X_sc
    else:
        X_tree = X_sc

    # Time-based train/val split (80/20)
    n_val   = max(10, int(len(X_tree) * 0.2))
    n_train = len(X_tree) - n_val
    if n_train < 20:
        _log(f"    [inc] {symbol}: not enough training rows ({n_train}) — skipping fine-tune")
        return False

    X_tr, y_tr = X_tree[:n_train], y[:n_train]
    X_va, y_va = X_tree[n_train:], y[n_train:]

    # Recency weighting: more recent rows get higher weight (linear ramp)
    ages_tr = np.arange(n_train, 0, -1, dtype=float)  # n_train…1
    weights  = 1.0 / (ages_tr + 1.0)                  # monotonically decreasing with age
    weights  = weights / weights.sum() * n_train        # normalise to sum=n_train

    updated = False

    # ── LightGBM warm-start ───────────────────────────────────────────────────
    lgb_pkl = prod_path / "lightgbm.pkl"
    if lgb_pkl.exists():
        try:
            import lightgbm as lgb
            with open(lgb_pkl, "rb") as f:
                lgb_clf = pickle.load(f)  # LightGBMClassifier wrapper

            existing_booster = lgb_clf.model.booster_

            # Build LightGBM Datasets
            dtrain = lgb.Dataset(X_tr, label=y_tr, weight=weights, free_raw_data=False)
            dval   = lgb.Dataset(X_va, label=y_va, free_raw_data=False, reference=dtrain)

            # Retrieve params from existing booster (stripped of non-serialisable keys)
            booster_params = existing_booster.params.copy()
            # Force these to avoid conflicts with the incremental run
            booster_params["verbosity"]        = -1
            booster_params["num_iterations"]   = _FINETUNE_ROUNDS_LGB
            # Remove keys that conflict with lgb.train API
            for _k in ("num_boost_round", "n_estimators", "early_stopping_rounds"):
                booster_params.pop(_k, None)

            new_booster = lgb.train(
                booster_params,
                dtrain,
                num_boost_round          = _FINETUNE_ROUNDS_LGB,
                valid_sets               = [dval],
                init_model               = existing_booster,
                callbacks                = [lgb.log_evaluation(period=-1)],
            )

            # Patch updated booster back into the wrapper and save
            lgb_clf.model.booster_ = new_booster
            tmp = lgb_pkl.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(lgb_clf, f, protocol=5)
            tmp.replace(lgb_pkl)
            updated = True
        except Exception as _lgb_err:
            _log(f"    [inc] {symbol}: LightGBM fine-tune error: {_lgb_err}")

    # ── XGBoost warm-start ────────────────────────────────────────────────────
    xgb_pkl = prod_path / "xgboost.pkl"
    if xgb_pkl.exists():
        try:
            import xgboost as xgb
            with open(xgb_pkl, "rb") as f:
                xgb_clf = pickle.load(f)  # XGBoostClassifier wrapper

            existing_booster = xgb_clf.model.get_booster()

            # Build DMatrix objects
            dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, weight=weights)
            dval_xgb   = xgb.DMatrix(X_va, label=y_va)

            # Extract params from existing XGBClassifier
            xgb_params = xgb_clf.model.get_params()
            # Convert to raw booster params (drop sklearn-specific keys)
            raw_params: Dict = {
                "max_depth":        xgb_params.get("max_depth",        5),
                "learning_rate":    xgb_params.get("learning_rate",    0.01),
                "subsample":        xgb_params.get("subsample",        0.8),
                "colsample_bytree": xgb_params.get("colsample_bytree", 0.8),
                "reg_alpha":        xgb_params.get("reg_alpha",        0.3),
                "reg_lambda":       xgb_params.get("reg_lambda",       1.5),
                "eval_metric":      "logloss",
                "objective":        "binary:logistic",
                "verbosity":        0,
                "seed":             42,
            }

            new_booster = xgb.train(
                raw_params,
                dtrain_xgb,
                num_boost_round  = _FINETUNE_ROUNDS_LGB,
                evals            = [(dval_xgb, "val")],
                xgb_model        = existing_booster,
                verbose_eval     = False,
            )

            # Patch the updated booster back into the XGBClassifier wrapper
            xgb_clf.model._Booster = new_booster
            tmp = xgb_pkl.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(xgb_clf, f, protocol=5)
            tmp.replace(xgb_pkl)
            updated = True
        except Exception as _xgb_err:
            _log(f"    [inc] {symbol}: XGBoost fine-tune error: {_xgb_err}")

    return updated


def _incremental_predict(
    symbol: str,
    raw_df: pd.DataFrame,
    global_cues_df: Optional[pd.DataFrame],
    usdinr_df: Optional[pd.DataFrame],
) -> Optional[Dict]:
    """
    Run next-day prediction via run_pipeline.predict_next_day using the (freshly
    fine-tuned) production models.
    """
    try:
        import importlib.util
        _pipeline_py = _PIPELINE_DIR / "run_pipeline.py"
        spec = importlib.util.spec_from_file_location("run_pipeline_pred", str(_pipeline_py))
        rp   = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(rp)  # type: ignore[union-attr]
        result = rp.predict_next_day(
            symbol,
            raw_df,
            global_cues_df=global_cues_df,
            usdinr_df=usdinr_df,
        )
        return result
    except Exception as _pe:
        _log(f"    [inc] {symbol}: prediction failed: {_pe}")
        return None


def run_incremental_retrain(capital: float = 500_000, paper: bool = False) -> None:
    """
    Fast daily incremental update for production LightGBM + XGBoost models.

    For each symbol:
    1. Loads raw parquet (already incremental from main pipeline).
    2. Loads / refreshes feature cache — only new rows are recomputed.
    3. Fine-tunes LightGBM + XGBoost on the last 252 rows (warm-start,
       20 additional rounds each).  Symbols without production models are
       skipped (they require a full pipeline run first).
    4. Runs prediction with the updated models.
    5. Writes next_day_predictions.csv to the latest run directory.

    Target: < 5 minutes for 100 symbols.
    """
    _log("=== INCREMENTAL RETRAIN STARTED ===")
    t0 = time.time()

    # ── Path setup ─────────────────────────────────────────────────────────────
    sys.path.insert(0, str(_V3_ROOT))
    sys.path.insert(0, str(_V3_ROOT / "02_models"))
    sys.path.insert(0, str(_PIPELINE_DIR))

    # ── Load shared auxiliary data once (global cues + USD/INR) ───────────────
    global_cues_df: Optional[pd.DataFrame] = None
    usdinr_df: Optional[pd.DataFrame]      = None
    gcues_path  = _RAW_DIR / "global_cues.parquet"
    usdinr_path = _RAW_DIR / "usdinr.parquet"
    if gcues_path.exists():
        global_cues_df = _load_parquet_safe(gcues_path)
    if usdinr_path.exists():
        usdinr_df = _load_parquet_safe(usdinr_path)

    # ── Determine symbol list from config ─────────────────────────────────────
    try:
        sys.path.insert(0, str(_V3_ROOT / "00_config"))
        from config import SYMBOLS as _SYMBOLS  # type: ignore
        symbols: List[str] = list(_SYMBOLS)
    except Exception:
        # Fallback: any symbol that has a raw parquet file
        symbols = [p.stem for p in _RAW_DIR.glob("*.parquet")
                   if p.stem not in {"global_cues", "usdinr", "market"}]

    predictions:   List[Dict] = []
    n_fine_tuned   = 0
    n_skipped      = 0
    n_pred_ok      = 0

    for i, symbol in enumerate(symbols, 1):
        sym_t0    = time.time()
        prod_path = _PROD_DIR / symbol
        raw_path  = _RAW_DIR / f"{symbol}.parquet"
        feat_path = _FEAT_RAW_DIR / f"{symbol}_features.parquet"

        # ── Skip if no production model exists ────────────────────────────────
        lgb_pkl = prod_path / "lightgbm.pkl"
        xgb_pkl = prod_path / "xgboost.pkl"
        if not prod_path.exists() or (not lgb_pkl.exists() and not xgb_pkl.exists()):
            _log(f"  [{i:3d}/{len(symbols)}] {symbol:<14} SKIP — no production model")
            n_skipped += 1
            continue

        # ── Load raw OHLCV ────────────────────────────────────────────────────
        if not raw_path.exists():
            _log(f"  [{i:3d}/{len(symbols)}] {symbol:<14} SKIP — raw parquet missing")
            n_skipped += 1
            continue
        raw_df = _load_parquet_safe(raw_path)
        if raw_df is None or raw_df.empty:
            _log(f"  [{i:3d}/{len(symbols)}] {symbol:<14} SKIP — raw parquet empty")
            n_skipped += 1
            continue

        # ── Incremental feature update ────────────────────────────────────────
        feat_df = _compute_incremental_features(
            symbol, raw_df, feat_path, global_cues_df, usdinr_df
        )
        if feat_df is None or feat_df.empty:
            _log(f"  [{i:3d}/{len(symbols)}] {symbol:<14} SKIP — feature computation failed")
            n_skipped += 1
            continue

        feat_cols = _feature_cols_inc(feat_df)

        # ── Fine-tune LightGBM + XGBoost ─────────────────────────────────────
        updated = _finetune_symbol(symbol, feat_df, feat_cols, prod_path)
        if updated:
            n_fine_tuned += 1

        # ── Next-day prediction using updated models ──────────────────────────
        pred = _incremental_predict(symbol, raw_df, global_cues_df, usdinr_df)
        if pred is not None:
            predictions.append(pred)
            n_pred_ok += 1

        elapsed_sym = time.time() - sym_t0
        _log(f"  [{i:3d}/{len(symbols)}] {symbol:<14} "
             f"{'fine-tuned' if updated else 'pred-only ':10} "
             f"| pred={'YES' if pred else 'no ':3} "
             f"| {elapsed_sym:.1f}s")

    # ── Write next_day_predictions.csv ────────────────────────────────────────
    run_id = _latest_run_id()
    if predictions and run_id:
        pred_path = _RESULTS_DIR / run_id / "next_day_predictions.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(predictions).to_csv(pred_path, index=False)
        _log(f"  Predictions written → {pred_path}")
    elif not run_id:
        _log("  WARNING: No run directory found — skipping predictions CSV write")

    # ── Build orders from predictions ─────────────────────────────────────────
    if run_id:
        _log("Step 2/2 — Building approved orders ...")
        rc = _run([
            _PYTHON, str(_LIVE_DIR / "signal_publisher.py"),
            "--run-id", run_id,
            "--capital", str(int(capital)),
            "--dry-run",
        ], f"signal_publisher.py  run_id={run_id}")
        if rc != 0:
            _log("WARNING: signal_publisher failed")

    elapsed = time.time() - t0
    _log(
        f"=== INCREMENTAL RETRAIN COMPLETE "
        f"| fine-tuned={n_fine_tuned} skipped={n_skipped} "
        f"predictions={n_pred_ok}/{len(symbols)} "
        f"| {elapsed:.1f}s ({elapsed/60:.1f} min) ==="
    )


# ══════════════════════════════════════════════════════════════════════════════
#  EVENING — Generate predictions & orders
# ══════════════════════════════════════════════════════════════════════════════

def run_evening(capital: float = 500_000, incremental: bool = False) -> None:
    """
    T-1 Evening sequence (run after 4 PM IST, before market close next day).

    incremental=False (default): full pipeline via orchestrator.py --fast (~30-60 min)
    incremental=True:            fast warm-start fine-tune of LightGBM + XGBoost (~5 min)
    """
    _log("=== EVENING RUN STARTED ===")

    # 1. Fetch news sentiment
    _log("Step 1/3 — Fetching news sentiment …")
    rc = _run([_PYTHON, str(_NEWS_DIR / "sentiment_history.py")], "sentiment_history.py")
    if rc != 0:
        _log("WARNING: Sentiment fetch failed — continuing with cached data")

    if incremental:
        # 2. Fast incremental fine-tune (warm-start, ~5 min for 100 symbols)
        _log("Step 2/3 — Running incremental fine-tune (warm-start mode) …")
        run_incremental_retrain(capital=capital, paper=False)
        # run_incremental_retrain already writes predictions + calls signal_publisher
        _log("=== EVENING RUN COMPLETE (incremental) ===")
        return

    # 2. Run V3 pipeline (fast mode = LightGBM + XGBoost only)
    _log("Step 2/3 — Running V3 pipeline (fast mode) …")
    rc = _run([
        _PYTHON, str(_PIPELINE_DIR / "orchestrator.py"),
        "--fast",
    ], "orchestrator.py --fast")
    if rc != 0:
        _log("ERROR: Pipeline failed — aborting evening run")
        sys.exit(1)

    # 3. Generate orders from predictions
    _log("Step 3/3 — Building approved orders …")
    run_id = _latest_run_id()
    if not run_id:
        _log("ERROR: No run directory found after pipeline")
        sys.exit(1)

    rc = _run([
        _PYTHON, str(_LIVE_DIR / "signal_publisher.py"),
        "--run-id", run_id,
        "--capital", str(int(capital)),
        "--dry-run",
    ], f"signal_publisher.py  run_id={run_id}")
    if rc != 0:
        _log("WARNING: signal_publisher failed")

    _log(f"=== EVENING RUN COMPLETE | run_id={run_id} ===")


# ══════════════════════════════════════════════════════════════════════════════
#  MORNING — Place orders
# ══════════════════════════════════════════════════════════════════════════════

def run_morning(capital: float = 500_000, paper: bool = False) -> None:
    """
    T Morning sequence (run at 9:00 AM IST, 15 min before market open).

    Step 0 (NEW): close any positions whose 10-trading-day hold has elapsed.
    Step 1+    : place today's BUY orders.
    """
    _log(f"=== MORNING RUN STARTED | {'PAPER MODE' if paper else 'LIVE MODE'} ===")

    # ── Step 0: time-based exits ──────────────────────────────────────────────
    # Backtest holds positions exactly 10 trading days; live must do the same.
    # Run exit_runner first so the SELLs go in before we add new BUYs.
    _log("Step 0/3 — Running exit_runner …")
    rc = _run([
        _PYTHON, str(_LIVE_DIR / "exit_runner.py"), "--execute",
    ], "exit_runner.py --execute")
    if rc != 0:
        _log("WARNING: exit_runner returned non-zero — continuing")

    # Load today's approved orders
    order_files = sorted(_ORDERS_DIR.glob("orders_*.json"), reverse=True)
    if not order_files:
        _log("ERROR: No order files found. Did evening run complete?")
        sys.exit(1)

    latest = order_files[0]
    _log(f"Loading orders: {latest.name}")
    with open(latest) as f:
        orders = json.load(f)

    if not orders:
        _log("No orders to place today.")
        return

    _log(f"  {len(orders)} orders to place | ₹{sum(o['order_value'] for o in orders):,.0f} total")

    if paper:
        # Paper mode — simulate fills only
        from order_manager import OrderManager
        mgr = OrderManager(client=None, paper_mode=True)
        mgr.execute_orders(orders)
        mgr.save_execution_log()
        _log("=== MORNING RUN COMPLETE (PAPER) ===")
        return

    # Live mode
    try:
        from angel_one_client import AngelOneClient
        from order_manager import OrderManager
        from risk_guard import RiskGuard
    except ImportError as e:
        _log(f"ERROR: Import failed: {e}")
        sys.exit(1)

    client = AngelOneClient()
    if not client.login():
        _log("ERROR: Angel One login failed")
        sys.exit(1)

    # Get current portfolio state for risk checks
    holdings     = client.get_holdings()
    funds        = client.get_funds()
    cash         = funds.get("available", capital)
    current_vals = {sym: pos.qty * pos.ltp for sym, pos in holdings.items()}
    portval      = cash + sum(current_vals.values())

    _log(f"  Portfolio: ₹{portval:,.0f}  Cash: ₹{cash:,.0f}  "
         f"Holdings: {len(holdings)} stocks")

    # Daily loss circuit breaker
    can_trade, reason = RiskGuard.check_daily_loss(portval, capital)
    if not can_trade:
        _log(f"CIRCUIT BREAKER: {reason} — no orders placed")
        sys.exit(0)

    # Market hours guard
    can_trade, reason = RiskGuard.check_market_hours()
    # Allow running slightly before open (will place orders at 9:15 open)

    # Full risk validation
    approved, rejected = RiskGuard.validate_batch(
        orders, portval, {s: p.qty for s, p in holdings.items()}, current_vals
    )
    _log(f"  RiskGuard: {len(approved)} approved, {len(rejected)} blocked")

    # Subscribe WebSocket for live prices
    syms = [o["symbol"] for o in approved]
    client.subscribe_ticks(syms)
    import time; time.sleep(3)  # let WS warm up

    # Execute
    mgr = OrderManager(client=client, paper_mode=False)
    fills = mgr.execute_orders(approved, max_slippage_pct=0.003)
    _log(f"  Orders placed: {len(fills)}")

    # Wait for fills (30 min timeout)
    mgr.wait_for_fills(timeout_min=30, poll_interval_sec=60)

    summary = mgr.summary()
    _log(f"  Fills: {summary['filled']} filled, {summary['rejected']} rejected, "
         f"{summary['timeout']} timeout | charges ₹{summary['total_charges']:,.0f}")

    log_path = mgr.save_execution_log(run_id=_latest_run_id())
    client.stop_websocket()
    _log(f"=== MORNING RUN COMPLETE | log={log_path.name} ===")


# ══════════════════════════════════════════════════════════════════════════════
#  RECONCILE — Evening portfolio reconciliation
# ══════════════════════════════════════════════════════════════════════════════

def run_reconcile() -> None:
    """
    T Evening (3:45 PM IST) — reconcile Angel One holdings with execution log.
    Appends to trade_history.parquet.
    """
    _log("=== RECONCILE STARTED ===")

    import pandas as pd

    try:
        from angel_one_client import AngelOneClient
        client = AngelOneClient()
        client.login()
    except Exception as e:
        _log(f"ERROR: {e}")
        sys.exit(1)

    holdings  = client.get_holdings()
    order_bk  = client.get_order_book()
    funds     = client.get_funds()

    today = datetime.now().strftime("%Y-%m-%d")
    rows  = []

    for sym, pos in holdings.items():
        rows.append({
            "date":       today,
            "symbol":     sym,
            "qty":        pos.qty,
            "avg_price":  pos.avg_price,
            "ltp":        pos.ltp,
            "pnl":        pos.pnl,
            "source":     "holdings",
        })

    for o in order_bk:
        status = o.get("orderstatus", "").lower()
        if status in ("complete", "filled"):
            rows.append({
                "date":       today,
                "symbol":     o.get("tradingsymbol", ""),
                "qty":        int(o.get("filledshares", 0)),
                "avg_price":  float(o.get("averageprice", 0)),
                "ltp":        float(o.get("ltp", 0)),
                "pnl":        float(o.get("realizedprofitloss", 0)),
                "source":     "orderbook",
            })

    if rows:
        df = pd.DataFrame(rows)
        if _HISTORY_PATH.exists():
            try:
                existing = pd.read_parquet(_HISTORY_PATH)
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=["date", "symbol", "source"], keep="last")
            except Exception:
                _log("  WARNING: trade_history.parquet corrupted — starting fresh")
        _tmp = _HISTORY_PATH.with_suffix(".parquet.tmp")
        df.to_parquet(_tmp, index=False, compression="snappy")
        _tmp.replace(_HISTORY_PATH)
        _log(f"  {len(rows)} records → {_HISTORY_PATH.name}")

    _log(f"  Funds: available=₹{funds.get('available', 0):,.0f}  "
         f"net=₹{funds.get('net', 0):,.0f}")
    client.stop_websocket()

    # Paper-trading P&L reconciliation: live-paper round-trips vs backtest predictions.
    # Always-on (paper or live) — runs on the local execution_log.parquet.
    _log("Running paper P&L reconciler …")
    _run([_PYTHON, str(_LIVE_DIR / "paper_pnl_reconciler.py")], "paper_pnl_reconciler.py")

    _log("=== RECONCILE COMPLETE ===")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily algo-trading runner")
    parser.add_argument("--mode",    required=True,
                        choices=["evening", "morning", "reconcile"],
                        help="Which phase to run")
    parser.add_argument("--capital", type=float, default=500_000,
                        help="Total portfolio capital in INR (default 500000)")
    parser.add_argument("--paper",   action="store_true",
                        help="Paper mode — simulate without placing real orders")
    parser.add_argument("--incremental", action="store_true",
                        help="Evening mode: skip full pipeline, do fast warm-start "
                             "fine-tune of LightGBM+XGBoost only (~5 min vs 30-60 min)")
    args = parser.parse_args()

    if args.mode == "evening":
        run_evening(capital=args.capital, incremental=args.incremental)
    elif args.mode == "morning":
        paper = _resolve_paper_mode(args.paper)
        _log(f"Trading mode resolved → {'PAPER' if paper else 'LIVE'} "
             f"(TRADING_MODE={os.getenv('TRADING_MODE', 'paper')})")
        run_morning(capital=args.capital, paper=paper)
    elif args.mode == "reconcile":
        run_reconcile()
