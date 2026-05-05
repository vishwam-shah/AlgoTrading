"""
exp9_robustness_suite.py — Cross-cutting robustness checks for publication
==========================================================================
Runs five sensitivities against an existing run's predictions.csv corpus:

  A. Cost / slippage sensitivity   — how does Sharpe degrade as RT cost
     scales from 0% → 1%?
  B. Turnover sensitivity          — how does Sharpe change as min_confidence
     and meta_threshold are swept?
  C. Hold-horizon sensitivity      — Sharpe vs hold_days ∈ {3, 5, 7, 10, 15, 20}
  D. Regime-conditional metrics    — bull/bear/sideways NIFTY regimes
  E. Rolling calibration drift     — Brier score & ECE on a rolling 60-day window

Inputs (no retraining required):
  V3/06_results/runs/<run_id>/<symbol>/predictions.csv
  V3/01_data/raw/<symbol>.parquet  (close prices for hold-horizon)
  V3/01_data/raw/^NSEI.parquet     (optional — for regime classification)

Outputs (under V3/08_experiments/results/):
  exp9_cost_slippage.csv
  exp9_turnover.csv
  exp9_hold_horizon.csv
  exp9_regime.csv
  exp9_calibration_drift.csv
  exp9_summary.json

Usage:
  python V3/08_experiments/exp9_robustness_suite.py --run-id 20260430_131250
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_V3 = Path(__file__).resolve().parents[1]
_OUT = Path(__file__).resolve().parent / "results"
_OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(_V3 / "00_config"))
from risk_config import HOT as _RC, get as _rcget  # type: ignore  # noqa: E402

ANNUAL = 252


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_run_preds(run_id: str) -> Dict[str, pd.DataFrame]:
    base = _V3 / "06_results" / "runs" / run_id
    out: Dict[str, pd.DataFrame] = {}
    for p in sorted(base.glob("*/predictions.csv")):
        try:
            df = pd.read_csv(p)
            df["date"] = pd.to_datetime(df["date"])
            if "window_id" in df.columns:
                df = df.sort_values("window_id").drop_duplicates("date", keep="last")
            df = df.sort_values("date").reset_index(drop=True)
            out[p.parent.name] = df
        except Exception:
            continue
    return out


def _trade_returns(
    df: pd.DataFrame,
    *,
    hold_days: int,
    min_conf: float,
    meta_thr: float,
    cost_rt: float,
) -> np.ndarray:
    df = df.copy()
    df["exit_price"] = df["close_price"].shift(-hold_days)
    mask = (df["direction"] == "UP") & (df["prob_up"] >= min_conf)
    if "meta_prob" in df.columns and df["meta_prob"].notna().any() \
            and float(df["meta_prob"].std()) > 1e-6:
        mask &= (df["meta_prob"] >= meta_thr)
    mask &= df["exit_price"].notna()
    cand = df.index[mask].tolist()
    if len(cand) < 5:
        return np.array([])
    chosen, last_exit = [], -1
    for i in cand:
        if i < last_exit: continue
        chosen.append(i); last_exit = i + hold_days
    if len(chosen) < 5:
        return np.array([])
    sub = df.loc[chosen]
    raw = (sub["exit_price"].values - sub["close_price"].values) / sub["close_price"].values
    return raw - cost_rt


def _sharpe(rets: np.ndarray, hold_days: int) -> float:
    if len(rets) < 5 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * math.sqrt(ANNUAL / hold_days))


def _aggregate(corpus: Dict[str, pd.DataFrame], *,
               hold_days: int, min_conf: float, meta_thr: float,
               cost_rt: float) -> Dict[str, float]:
    sharpes, ret_totals, trade_counts = [], [], []
    for sym, df in corpus.items():
        rets = _trade_returns(df, hold_days=hold_days, min_conf=min_conf,
                              meta_thr=meta_thr, cost_rt=cost_rt)
        if len(rets) >= 5:
            sharpes.append(_sharpe(rets, hold_days))
            ret_totals.append(float(np.prod(1 + rets) - 1))
            trade_counts.append(len(rets))
    if not sharpes:
        return {"n_stocks": 0}
    return {
        "n_stocks":      int(len(sharpes)),
        "n_trades_mean": int(np.mean(trade_counts)),
        "sharpe_mean":   round(float(np.mean(sharpes)), 4),
        "sharpe_med":    round(float(np.median(sharpes)), 4),
        "ret_mean":      round(float(np.mean(ret_totals)), 4),
        "pct_pos":       round(float(np.mean(np.array(sharpes) > 0)), 4),
    }


# ── Experiments ───────────────────────────────────────────────────────────────

def exp_cost_slippage(corpus, base):
    rows = []
    for cost in [0.0, 0.0010, 0.0025, 0.0035, 0.0050, 0.0075, 0.0100]:
        agg = _aggregate(corpus, hold_days=base["hold"], min_conf=base["min_conf"],
                         meta_thr=base["meta"], cost_rt=cost)
        rows.append({"cost_round_trip": cost, **agg})
    return pd.DataFrame(rows)


def exp_turnover(corpus, base):
    rows = []
    for mc in [0.52, 0.55, 0.58, 0.60, 0.62, 0.65]:
        for mt in [0.50, 0.55, 0.58, 0.60, 0.62]:
            agg = _aggregate(corpus, hold_days=base["hold"], min_conf=mc,
                             meta_thr=mt, cost_rt=base["cost"])
            rows.append({"min_conf": mc, "meta_thr": mt, **agg})
    return pd.DataFrame(rows)


def exp_hold_horizon(corpus, base):
    rows = []
    for h in [3, 5, 7, 10, 15, 20]:
        agg = _aggregate(corpus, hold_days=h, min_conf=base["min_conf"],
                         meta_thr=base["meta"], cost_rt=base["cost"])
        rows.append({"hold_days": h, **agg})
    return pd.DataFrame(rows)


def exp_regime(corpus, base):
    """Classify each trade by NIFTY trend regime at entry — bull/bear/sideways."""
    nifty_path = _V3 / "01_data" / "raw" / "^NSEI.parquet"
    if not nifty_path.exists():
        # Try ^NSEI.NS naming
        nifty_path = _V3 / "01_data" / "raw" / "NIFTY.parquet"
    if not nifty_path.exists():
        return pd.DataFrame([{"regime": "unavailable", "n_trades": 0}])
    nifty = pd.read_parquet(nifty_path)
    nifty["date"] = pd.to_datetime(nifty["date"])
    nifty = nifty.sort_values("date").reset_index(drop=True)
    nifty["sma50"] = nifty["close"].rolling(50).mean()
    nifty["sma200"] = nifty["close"].rolling(200).mean()
    def _regime(row):
        if pd.isna(row.sma50) or pd.isna(row.sma200): return "unknown"
        if row.sma50 > row.sma200 * 1.02: return "bull"
        if row.sma50 < row.sma200 * 0.98: return "bear"
        return "sideways"
    nifty["regime"] = nifty.apply(_regime, axis=1)
    nifty_idx = nifty.set_index("date")["regime"]

    rows = []
    by_regime: Dict[str, List[float]] = {"bull": [], "bear": [], "sideways": []}
    for sym, df in corpus.items():
        rets = _trade_returns(df, hold_days=base["hold"], min_conf=base["min_conf"],
                              meta_thr=base["meta"], cost_rt=base["cost"])
        if len(rets) < 5:
            continue
        # Regime at entry: align by date in df where mask was True
        df = df.copy()
        df["exit_price"] = df["close_price"].shift(-base["hold"])
        mask = (df["direction"] == "UP") & (df["prob_up"] >= base["min_conf"])
        if "meta_prob" in df.columns and df["meta_prob"].notna().any() \
                and float(df["meta_prob"].std()) > 1e-6:
            mask &= (df["meta_prob"] >= base["meta"])
        mask &= df["exit_price"].notna()
        idx = df.index[mask].tolist()
        chosen, last_exit = [], -1
        for i in idx:
            if i < last_exit: continue
            chosen.append(i); last_exit = i + base["hold"]
        for i, r in zip(chosen, rets):
            d = df["date"].iloc[i].normalize()
            reg = nifty_idx.get(d, "unknown")
            if reg in by_regime:
                by_regime[reg].append(float(r))

    for reg, lst in by_regime.items():
        arr = np.array(lst)
        if len(arr) < 5:
            rows.append({"regime": reg, "n_trades": len(arr), "sharpe": 0.0})
            continue
        rows.append({
            "regime": reg, "n_trades": len(arr),
            "sharpe": round(float(arr.mean() / arr.std() * math.sqrt(ANNUAL / base["hold"])) if arr.std() > 0 else 0.0, 3),
            "win_rate": round(float((arr > 0).mean()), 3),
            "avg_ret": round(float(arr.mean()), 4),
        })
    return pd.DataFrame(rows)


def exp_calibration_drift(corpus, window: int = 60):
    """Rolling Brier score per symbol over 60-day windows."""
    rows = []
    for sym, df in corpus.items():
        if len(df) < window + 5 or "actual" not in df.columns:
            continue
        x = df.copy()
        if "prob_up" not in x.columns:
            continue
        x = x.dropna(subset=["prob_up", "actual"]).reset_index(drop=True)
        if len(x) < window:
            continue
        brier = []
        for i in range(window, len(x), max(1, window // 4)):
            seg = x.iloc[i - window: i]
            b = float(((seg["prob_up"] - seg["actual"]) ** 2).mean())
            brier.append({
                "symbol": sym, "window_end": str(seg["date"].iloc[-1].date()),
                "brier": round(b, 4), "n": len(seg),
            })
        rows.extend(brier)
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    base = {
        "hold":     int(_RC["HOLD_DAYS"]),
        "min_conf": float(_RC["MIN_CONFIDENCE"]),
        "meta":     float(_RC["META_THRESHOLD"]),
        "cost":     float(_RC["COST_RT"]),
    }
    print(f"  loading run {args.run_id} …")
    corpus = _load_run_preds(args.run_id)
    print(f"  loaded {len(corpus)} symbols")

    print("  A. cost/slippage …")
    df_a = exp_cost_slippage(corpus, base); df_a.to_csv(_OUT / "exp9_cost_slippage.csv", index=False)
    print(df_a.to_string(index=False))

    print("\n  B. turnover (min_conf × meta_thr) …")
    df_b = exp_turnover(corpus, base); df_b.to_csv(_OUT / "exp9_turnover.csv", index=False)
    # show the head ranked by sharpe_mean
    if "sharpe_mean" in df_b.columns:
        print(df_b.sort_values("sharpe_mean", ascending=False).head(10).to_string(index=False))

    print("\n  C. hold-horizon …")
    df_c = exp_hold_horizon(corpus, base); df_c.to_csv(_OUT / "exp9_hold_horizon.csv", index=False)
    print(df_c.to_string(index=False))

    print("\n  D. regime-conditional …")
    df_d = exp_regime(corpus, base); df_d.to_csv(_OUT / "exp9_regime.csv", index=False)
    print(df_d.to_string(index=False))

    print("\n  E. rolling Brier calibration …")
    df_e = exp_calibration_drift(corpus); df_e.to_csv(_OUT / "exp9_calibration_drift.csv", index=False)
    if not df_e.empty:
        print(df_e.groupby("symbol")["brier"].agg(["mean", "min", "max"]).head(10))

    summary = {
        "run_id": args.run_id,
        "n_symbols": len(corpus),
        "baseline": base,
        "cost_slippage_breakeven_cost":
            float(df_a[df_a["sharpe_mean"] > 0]["cost_round_trip"].max()) if not df_a.empty else None,
        "best_turnover": df_b.sort_values("sharpe_mean", ascending=False).head(1).to_dict(orient="records")[0]
                         if "sharpe_mean" in df_b.columns and not df_b.empty else None,
        "horizon_robust": (df_c["sharpe_mean"] > 0).mean() if not df_c.empty and "sharpe_mean" in df_c.columns else None,
        "regime_summary": df_d.to_dict(orient="records") if not df_d.empty else None,
    }
    with open(_OUT / "exp9_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  → {_OUT.name}/exp9_summary.json")


if __name__ == "__main__":
    main()
