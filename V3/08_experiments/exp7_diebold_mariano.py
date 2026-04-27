"""
exp7_diebold_mariano.py — Diebold-Mariano (1995) test of forecast-loss equality.

H0: model and baseline have equal expected forecast loss.
HA: model has lower expected forecast loss than baseline (one-sided DM ≤ 0).

Loss function: 0/1 loss (i.e. classification error) since target is binary
direction. Equivalent to McNemar in spirit but the DM stat correctly accounts
for serial correlation in the loss differential via Newey-West HAC.

Baselines (rebuilt from cached parquets):
  1. Always-UP        — predict 1 every day.
  2. Momentum-5       — predict UP if 5-day return > 0.
  3. AR(1) sign       — fit y_t = a + b*y_{t-1} + e on a rolling 252d window;
                         predict sign(b * y_{t-1}).

For each stock and each baseline, computes DM_stat and one-sided p-value
(model_better_than_baseline). Reports a stock-level table + an aggregated
test across all stocks (pooled).

Inputs:
    V3/06_results/runs/<run_id>/<symbol>/predictions.csv   (model probs)
    V3/01_data/raw/<symbol>.parquet                        (raw OHLCV)

Output:
    V3/08_experiments/results/exp7_dm_<run_id>.csv          (per-stock)
    Stdout summary.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

_V3      = Path(__file__).resolve().parents[1]
_RESULTS = _V3 / "06_results" / "runs"
_RAW     = _V3 / "01_data" / "raw"
_OUT     = Path(__file__).resolve().parent / "results"
_OUT.mkdir(exist_ok=True)

HORIZON = 5  # match features.HORIZON_DAYS — target is sign of 5-day fwd return


def _newey_west_var(x: np.ndarray, lag: int) -> float:
    """HAC long-run variance estimator."""
    x = x - x.mean()
    n = len(x)
    gamma0 = (x @ x) / n
    s = gamma0
    for k in range(1, lag + 1):
        gamma_k = (x[k:] @ x[:-k]) / n
        w = 1.0 - k / (lag + 1.0)   # Bartlett kernel
        s += 2.0 * w * gamma_k
    return float(max(s, 1e-12))


def _dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = HORIZON) -> Tuple[float, float]:
    """
    DM stat for one-sided test 'A has lower expected loss than B'.
    Returns (DM_statistic, one-sided p-value).
    Negative DM => A better than B (p small => significantly better).
    """
    d = loss_a - loss_b
    n = len(d)
    if n < 30:
        return float("nan"), float("nan")
    var_d = _newey_west_var(d, lag=max(h - 1, 1))
    dm = d.mean() / np.sqrt(var_d / n)
    # Harvey/Leybourne/Newbold small-sample correction
    k = ((n + 1 - 2 * h + h * (h - 1) / n) / n) ** 0.5
    dm_hln = dm * k
    # one-sided test: A better
    p = stats.norm.cdf(dm_hln)
    return float(dm_hln), float(p)


def _load_pred(run_id: str, symbol: str) -> pd.DataFrame:
    p = _RESULTS / run_id / symbol / "predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "window_id" in df.columns:
        df = df.sort_values("window_id").drop_duplicates("date", keep="last")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "close_price", "actual", "prob_up"]].dropna()


def _load_raw(symbol: str) -> pd.DataFrame:
    p = _RAW / f"{symbol}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"] if "date" in df.columns else df.index)
    return df.sort_values("date").reset_index(drop=True)


def _baseline_predictions(pred_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Attach baseline binary forecasts aligned to pred_df dates."""
    raw = raw_df.copy()
    raw["close"] = raw["close"].astype(float)
    raw["ret_5"] = raw["close"] / raw["close"].shift(5) - 1.0
    # AR(1) sign: rolling 252-day OLS on 1-day returns; predict sign(b * y_{t-1})
    raw["ret_1"] = raw["close"].pct_change()
    n = len(raw)
    ar_pred = np.full(n, np.nan)
    for i in range(252, n):
        y = raw["ret_1"].iloc[i - 252:i].dropna().values
        if len(y) < 100:
            continue
        x = y[:-1]; y_next = y[1:]
        if x.std() == 0:
            continue
        b = np.cov(x, y_next, bias=True)[0, 1] / x.var()
        ar_pred[i] = 1 if (b * raw["ret_1"].iloc[i - 1]) > 0 else 0
    raw["ar1_pred"] = ar_pred
    raw["mom5_pred"] = (raw["ret_5"] > 0).astype("Int64")
    raw["always_up"] = 1

    out = pred_df.merge(
        raw[["date", "always_up", "mom5_pred", "ar1_pred"]],
        on="date", how="left"
    )
    out["model_pred"] = (out["prob_up"] >= 0.5).astype(int)
    return out


def run_dm(run_id: str) -> pd.DataFrame:
    rows = []
    sym_dirs = sorted([d for d in (_RESULTS / run_id).glob("*") if d.is_dir() and (d / "predictions.csv").exists()])
    print(f"  Stocks with predictions: {len(sym_dirs)}")
    pooled = {"always_up": [], "mom5": [], "ar1": []}
    for sd in sym_dirs:
        sym = sd.name
        pred = _load_pred(run_id, sym)
        raw  = _load_raw(sym)
        if pred.empty or raw.empty:
            continue
        merged = _baseline_predictions(pred, raw).dropna(subset=["model_pred", "always_up"])
        if len(merged) < 60:
            continue
        y = merged["actual"].astype(int).values
        loss_model    = (merged["model_pred"].values  != y).astype(float)
        loss_alwaysup = (merged["always_up"].values   != y).astype(float)
        loss_mom5     = (merged["mom5_pred"].fillna(merged["always_up"]).astype(int).values != y).astype(float)
        loss_ar1      = (merged["ar1_pred"].fillna(merged["always_up"]).astype(int).values  != y).astype(float)

        dm_au, p_au = _dm_test(loss_model, loss_alwaysup)
        dm_mo, p_mo = _dm_test(loss_model, loss_mom5)
        dm_ar, p_ar = _dm_test(loss_model, loss_ar1)
        rows.append({
            "symbol": sym, "n": len(merged),
            "model_acc":    round(1 - loss_model.mean(),    4),
            "alwaysup_acc": round(1 - loss_alwaysup.mean(), 4),
            "mom5_acc":     round(1 - loss_mom5.mean(),     4),
            "ar1_acc":      round(1 - loss_ar1.mean(),      4),
            "dm_vs_alwaysup": round(dm_au, 3), "p_alwaysup": round(p_au, 4),
            "dm_vs_mom5":     round(dm_mo, 3), "p_mom5":     round(p_mo, 4),
            "dm_vs_ar1":      round(dm_ar, 3), "p_ar1":      round(p_ar, 4),
        })
        pooled["always_up"].append((loss_model, loss_alwaysup))
        pooled["mom5"].append((loss_model, loss_mom5))
        pooled["ar1"].append((loss_model, loss_ar1))

    df = pd.DataFrame(rows)
    out = _OUT / f"exp7_dm_{run_id}.csv"
    df.to_csv(out, index=False)

    # Pooled DM stats: stack all loss differentials
    print(f"\n  Per-stock test count: {len(df)}")
    print(f"\n  Stocks with model SIGNIFICANTLY better (p<0.05) vs each baseline:")
    print(f"    Always-UP : {(df.p_alwaysup < 0.05).sum()} / {len(df)}")
    print(f"    Momentum-5: {(df.p_mom5     < 0.05).sum()} / {len(df)}")
    print(f"    AR(1)     : {(df.p_ar1      < 0.05).sum()} / {len(df)}")
    print(f"\n  Pooled DM (stacked loss differentials, Newey-West HAC):")
    for name, pairs in pooled.items():
        la = np.concatenate([p[0] for p in pairs])
        lb = np.concatenate([p[1] for p in pairs])
        dm, p = _dm_test(la, lb)
        better = "✓ model better" if p < 0.05 else "✗ not significant"
        print(f"    vs {name:<10}  DM={dm:+.3f}  p={p:.4f}  {better}")

    print(f"\n  → {out}")
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    runs = sorted(_RESULTS.glob("20*"), reverse=True)
    rid = args.run or (runs[0].name if runs else None)
    print(f"  Run: {rid}")
    run_dm(rid)


if __name__ == "__main__":
    main()
