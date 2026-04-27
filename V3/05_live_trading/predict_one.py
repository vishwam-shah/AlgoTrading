"""
predict_one.py — Single-stock next-day prediction via the production pipeline
==============================================================================
Loads the production v2 models from V3/02_models/production/<SYMBOL>/, takes
raw OHLCV from V3/01_data/raw/<SYMBOL>.parquet (optionally truncated to a
specific cutoff date), recomputes features, and returns the same dict shape
as the orchestrator's Step 5 (`next_day_predictions.csv` per-row).

Usage:
    # Predict using all cached data (latest available bar):
    python V3/05_live_trading/predict_one.py FEDERALBNK

    # Predict using only data through a specific date (cutoff inclusive):
    python V3/05_live_trading/predict_one.py FEDERALBNK --as-of 2026-04-24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_V3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_V3_ROOT / "07_pipeline"))

from steps.predict import predict_next_day   # type: ignore  # noqa: E402

_RAW_DIR = _V3_ROOT / "01_data" / "raw"


def _load_raw(symbol: str, as_of: str | None) -> pd.DataFrame:
    p = _RAW_DIR / f"{symbol}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No raw cache at {p} — run downloader first.")
    df = pd.read_parquet(p).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    if as_of:
        cutoff = pd.to_datetime(as_of)
        df = df[df["date"] <= cutoff].reset_index(drop=True)
    return df


def _load_optional_cues(as_of: str | None = None) -> dict:
    """Best-effort load of global_cues / usdinr / market / peer_returns from cache."""
    cutoff = pd.to_datetime(as_of) if as_of else None
    out = {"global_cues_df": None, "usdinr_df": None, "market_df": None, "peer_returns": None}

    gc = _RAW_DIR / "global_cues.parquet"
    if gc.exists():
        try:
            df = pd.read_parquet(gc)
            if cutoff is not None and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["date"] <= cutoff]
            out["global_cues_df"] = df
        except Exception:
            pass

    fx = _RAW_DIR / "usdinr.parquet"
    if fx.exists():
        try:
            df = pd.read_parquet(fx)
            if cutoff is not None and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["date"] <= cutoff]
            out["usdinr_df"] = df
        except Exception:
            pass

    # Build peer_returns from the full Nifty-100 raw cache (sector features need it).
    peer = {}
    for p in _RAW_DIR.glob("*.parquet"):
        sym = p.stem
        if sym in ("global_cues", "usdinr"):
            continue
        try:
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            if cutoff is not None:
                df = df[df["date"] <= cutoff]
            if df.empty:
                continue
            peer[sym] = df.set_index("date")["close"].pct_change()
        except Exception:
            continue
    out["peer_returns"] = peer if peer else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", help="NSE symbol e.g. FEDERALBNK")
    ap.add_argument("--as-of", default=None,
                    help="Inclusive date cutoff (YYYY-MM-DD). Default: latest cached bar.")
    args = ap.parse_args()

    sym = args.symbol.upper()
    raw = _load_raw(sym, args.as_of)
    if raw.empty:
        print(f"  ERROR: no rows for {sym} at/before {args.as_of}"); sys.exit(1)

    last = raw.iloc[-1]
    print(f"\n  Symbol         : {sym}")
    print(f"  Bars used      : {len(raw)} (oldest {raw.date.iloc[0].date()} → newest {last['date'].date()})")
    print(f"  Last close     : ₹{float(last['close']):,.2f}")
    print(f"  Predicting for : next trading day after {last['date'].date()}")

    cues = _load_optional_cues(args.as_of)
    print(f"  Peer stocks    : {len(cues['peer_returns']) if cues['peer_returns'] else 0}")
    result = predict_next_day(sym, raw_df=raw, **cues)
    if result is None:
        print(f"\n  ✗ Prediction failed — no production model or feature mismatch for {sym}")
        sys.exit(2)

    print(f"\n  ── Prediction ───────────────────────────")
    print(f"  Direction      : {result['direction']}")
    print(f"  Action         : {result['action']}")
    print(f"  Confidence     : {result['confidence']:.4f}")
    print(f"  Primary  prob  : {result['avg_prob']:.4f}  (gate: ≥ 0.58)")
    print(f"  Meta-LdP prob  : {result['meta_prob']:.4f}  (gate: ≥ 0.60)")
    print(f"  Signal active  : {result['signal_active']}")
    print(f"  Regime         : {result['regime_label']}  ({result['regime']})")
    print(f"  Calibration T  : {result['temperature']}")
    print()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
