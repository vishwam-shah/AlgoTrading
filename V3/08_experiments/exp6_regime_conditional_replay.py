"""
exp6_regime_conditional_replay.py — Regime-conditional Sharpe / return.

Reads the production-pipeline portfolio curve and tags each day with the
NIFTY market regime (bull / sideways / bear) derived from the same SMA-50 /
SMA-200 + 20-day vol rule the pipeline uses internally. Reports per-regime
total return, annualised return, Sharpe, max-DD, and trade count.

Why: a +92% / Sharpe 1.72 result over a mostly-bullish 2.3-year window may
mask catastrophic bear-regime behaviour. Reviewers (and risk officers) need
the per-regime breakdown.

Inputs:
    V3/06_results/runs/<run_id>/backtest_portfolio.csv  (date, daily_return, equity)
    V3/01_data/raw/global_cues.parquet                  (us_vix, fallback)
    yfinance ^NSEI for NIFTY OHLC if local cache absent

Output:
    V3/08_experiments/results/exp6_regime_replay_<run_id>.csv
    Stdout summary table.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_RESULTS = Path(__file__).resolve().parents[1] / "06_results" / "runs"
_OUT_DIR = Path(__file__).resolve().parent / "results"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

ANNUAL = 252


def _load_nifty(start: str, end: str) -> pd.DataFrame:
    """NIFTY50 OHLC via yfinance — needed for regime classification."""
    import yfinance as yf
    df = yf.download("^NSEI", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "close"}).reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    return df.rename(columns={"Date": "date"})


def _classify_regime(nifty: pd.DataFrame) -> pd.DataFrame:
    """Same rule as run_pipeline.py:854 — SMA50 vs SMA200 + 20d realised vol terciles."""
    n = nifty.copy()
    n["sma50"]  = n["close"].rolling(50).mean()
    n["sma200"] = n["close"].rolling(200).mean()
    n["ret"]    = n["close"].pct_change()
    n["vol20"]  = n["ret"].rolling(20).std() * np.sqrt(ANNUAL)
    n["vol_q33"] = n["vol20"].rolling(252).quantile(0.33)
    n["vol_q66"] = n["vol20"].rolling(252).quantile(0.66)
    regime = pd.Series(1, index=n.index)  # default = sideways
    regime[(n["sma50"] > n["sma200"]) & (n["vol20"] < n["vol_q66"])] = 2  # bull
    regime[(n["sma50"] < n["sma200"]) & (n["vol20"] > n["vol_q33"])] = 0  # bear
    n["regime"] = regime
    return n[["date", "close", "regime"]]


def _per_regime_metrics(curve: pd.DataFrame, regime_col: str = "regime") -> pd.DataFrame:
    """Return one row per regime label."""
    rows = []
    name_map = {0: "bear", 1: "sideways", 2: "bull"}
    for r, sub in curve.groupby(regime_col):
        rets = sub["daily_return"].values
        n_days = len(sub)
        if n_days == 0:
            continue
        eq = (1 + rets).cumprod()
        total = float(eq[-1] - 1)
        ann = (1 + total) ** (ANNUAL / max(n_days, 1)) - 1
        sharpe = float(rets.mean() / rets.std() * np.sqrt(ANNUAL)) if rets.std() > 0 else 0.0
        peak = np.maximum.accumulate(eq)
        mdd = float(-((eq - peak) / peak).min()) if eq.size else 0.0
        active = int((rets != 0).sum())
        rows.append({
            "regime":      name_map.get(int(r), str(r)),
            "n_days":      n_days,
            "active_days": active,
            "total_ret":   round(total, 4),
            "ann_ret":     round(ann, 4),
            "sharpe":      round(sharpe, 3),
            "max_dd":      round(mdd, 4),
        })
    return pd.DataFrame(rows).sort_values("regime")


def replay(run_id: str) -> pd.DataFrame:
    run_dir = _RESULTS / run_id
    pcurve = run_dir / "backtest_portfolio.csv"
    if not pcurve.exists():
        print(f"  ERROR: {pcurve} not found"); sys.exit(1)

    curve = pd.read_csv(pcurve)
    curve["date"] = pd.to_datetime(curve["date"]).dt.strftime("%Y-%m-%d")
    print(f"  Portfolio curve: {len(curve)} rows, {curve.date.iloc[0]} → {curve.date.iloc[-1]}")

    nifty = _load_nifty(start=curve.date.iloc[0],
                        end=(pd.to_datetime(curve.date.iloc[-1]) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
    nifty = _classify_regime(nifty)
    merged = curve.merge(nifty[["date", "regime"]], on="date", how="left")
    # Forward-fill regime on weekends/holidays where curve has no rows but the merge logic should not need it.
    merged["regime"] = merged["regime"].ffill().bfill().astype(int)

    metrics = _per_regime_metrics(merged)
    overall_rets = curve["daily_return"].values
    overall_eq   = (1 + overall_rets).cumprod()
    overall_total = float(overall_eq[-1] - 1)
    overall_sh    = float(overall_rets.mean() / overall_rets.std() * np.sqrt(ANNUAL)) if overall_rets.std() > 0 else 0.0

    print(f"\n  Overall: total={overall_total:+.2%}  Sharpe={overall_sh:.2f}  n_days={len(curve)}")
    print(f"\n  Per-regime breakdown:")
    print(metrics.to_string(index=False))

    out = _OUT_DIR / f"exp6_regime_replay_{run_id}.csv"
    metrics.to_csv(out, index=False)
    print(f"\n  → {out}")
    return metrics


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="Run ID (default: latest)")
    args = ap.parse_args()
    runs = sorted(_RESULTS.glob("20*"), reverse=True)
    rid = args.run or (runs[0].name if runs else None)
    if not rid:
        print("No run found"); sys.exit(1)
    print(f"  Run: {rid}")
    replay(rid)


if __name__ == "__main__":
    main()
