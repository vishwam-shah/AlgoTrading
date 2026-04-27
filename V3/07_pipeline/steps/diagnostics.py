"""
Step 7 — Diagnostics
====================
Post-backtest statistical analyses. Runs after `steps/backtest.py` finishes
and writes artefacts the dashboard / paper can consume directly.

Outputs (under <run_dir>/):
  diagnostics_regime.csv      per-regime (bull/sideways/bear) Sharpe & return
  diagnostics_dm.csv          per-stock Diebold-Mariano vs naive baselines
  diagnostics_summary.json    headline numbers from both

These are pure post-processing on the artefacts produced by the prior steps,
so they cost ~30 s on a 100-stock run and never gate downstream code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

ANNUAL = 252
HORIZON_DAYS = 5     # DM HAC bandwidth — matches features.HORIZON_DAYS

_V3_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _V3_ROOT / "01_data" / "raw"


# ──────────────────────────── Regime replay ─────────────────────────────────

def _load_nifty(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download("^NSEI", start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "close"}).reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    return df.rename(columns={"Date": "date"})


def _classify_regime(nifty: pd.DataFrame) -> pd.DataFrame:
    n = nifty.copy()
    n["sma50"]  = n["close"].rolling(50).mean()
    n["sma200"] = n["close"].rolling(200).mean()
    n["ret"]    = n["close"].pct_change()
    n["vol20"]  = n["ret"].rolling(20).std() * np.sqrt(ANNUAL)
    n["vol_q33"] = n["vol20"].rolling(252).quantile(0.33)
    n["vol_q66"] = n["vol20"].rolling(252).quantile(0.66)
    regime = pd.Series(1, index=n.index)
    regime[(n["sma50"] > n["sma200"]) & (n["vol20"] < n["vol_q66"])] = 2
    regime[(n["sma50"] < n["sma200"]) & (n["vol20"] > n["vol_q33"])] = 0
    n["regime"] = regime
    return n[["date", "regime"]]


def _per_regime_metrics(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    name_map = {0: "bear", 1: "sideways", 2: "bull"}
    for r, sub in curve.groupby("regime"):
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
        rows.append({
            "regime":      name_map.get(int(r), str(r)),
            "n_days":      n_days,
            "active_days": int((rets != 0).sum()),
            "total_ret":   round(total, 4),
            "ann_ret":     round(ann, 4),
            "sharpe":      round(sharpe, 3),
            "max_dd":      round(mdd, 4),
        })
    return pd.DataFrame(rows).sort_values("regime")


def run_regime_replay(run_dir: Path) -> Optional[pd.DataFrame]:
    pcurve = run_dir / "backtest_portfolio.csv"
    if not pcurve.exists():
        print("  [diag] no backtest_portfolio.csv — skipping regime replay")
        return None
    curve = pd.read_csv(pcurve)
    if curve.empty:
        return None
    curve["date"] = pd.to_datetime(curve["date"]).dt.strftime("%Y-%m-%d")
    nifty = _load_nifty(curve.date.iloc[0],
                        (pd.to_datetime(curve.date.iloc[-1]) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
    if nifty.empty:
        print("  [diag] NIFTY fetch failed — skipping regime replay")
        return None
    nifty = _classify_regime(nifty)
    merged = curve.merge(nifty, on="date", how="left")
    merged["regime"] = merged["regime"].ffill().bfill().astype(int)
    metrics = _per_regime_metrics(merged)
    out = run_dir / "diagnostics_regime.csv"
    metrics.to_csv(out, index=False)
    print(f"  [diag] regime replay → {out.name}  ({len(metrics)} regimes)")
    return metrics


# ──────────────────────────── DM tests ──────────────────────────────────────

def _newey_west_var(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    n = len(x); gamma0 = (x @ x) / n
    s = gamma0
    for k in range(1, lag + 1):
        gk = (x[k:] @ x[:-k]) / n
        w = 1.0 - k / (lag + 1.0)
        s += 2.0 * w * gk
    return float(max(s, 1e-12))


def _dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = HORIZON_DAYS):
    d = loss_a - loss_b
    n = len(d)
    if n < 30:
        return float("nan"), float("nan")
    var_d = _newey_west_var(d, lag=max(h - 1, 1))
    dm = d.mean() / np.sqrt(var_d / n)
    k = ((n + 1 - 2 * h + h * (h - 1) / n) / n) ** 0.5
    dm_hln = dm * k
    return float(dm_hln), float(stats.norm.cdf(dm_hln))


def _baselines(pred: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["close"] = raw["close"].astype(float)
    raw["ret_5"] = raw["close"] / raw["close"].shift(5) - 1.0
    raw["ret_1"] = raw["close"].pct_change()
    n = len(raw); ar_pred = np.full(n, np.nan)
    for i in range(252, n):
        y = raw["ret_1"].iloc[i - 252:i].dropna().values
        if len(y) < 100: continue
        x = y[:-1]; yn = y[1:]
        if x.std() == 0: continue
        b = np.cov(x, yn, bias=True)[0, 1] / x.var()
        ar_pred[i] = 1 if (b * raw["ret_1"].iloc[i - 1]) > 0 else 0
    raw["ar1_pred"]  = ar_pred
    raw["mom5_pred"] = (raw["ret_5"] > 0).astype("Int64")
    raw["always_up"] = 1
    out = pred.merge(raw[["date", "always_up", "mom5_pred", "ar1_pred"]], on="date", how="left")
    out["model_pred"] = (out["prob_up"] >= 0.5).astype(int)
    return out


def run_dm_tests(run_dir: Path) -> Optional[pd.DataFrame]:
    rows = []; pooled = {"always_up": [], "mom5": [], "ar1": []}
    sym_dirs = sorted([d for d in run_dir.glob("*") if d.is_dir() and (d / "predictions.csv").exists()])
    if not sym_dirs:
        print("  [diag] no per-stock predictions — skipping DM")
        return None
    for sd in sym_dirs:
        sym = sd.name
        pred_path = sd / "predictions.csv"
        raw_path  = _RAW_DIR / f"{sym}.parquet"
        if not raw_path.exists(): continue
        try:
            pred = pd.read_csv(pred_path)
            if "window_id" in pred.columns:
                pred = pred.sort_values("window_id").drop_duplicates("date", keep="last")
            pred["date"] = pd.to_datetime(pred["date"])
            pred = pred[["date", "actual", "prob_up"]].dropna().sort_values("date")
            raw  = pd.read_parquet(raw_path)
            raw["date"] = pd.to_datetime(raw["date"] if "date" in raw.columns else raw.index)
            merged = _baselines(pred, raw).dropna(subset=["model_pred", "always_up"])
            if len(merged) < 60: continue
            y = merged["actual"].astype(int).values
            lm  = (merged["model_pred"].values  != y).astype(float)
            lau = (merged["always_up"].values   != y).astype(float)
            lmo = (merged["mom5_pred"].fillna(merged["always_up"]).astype(int).values != y).astype(float)
            lar = (merged["ar1_pred"].fillna(merged["always_up"]).astype(int).values  != y).astype(float)
            dm_au, p_au = _dm_test(lm, lau); dm_mo, p_mo = _dm_test(lm, lmo); dm_ar, p_ar = _dm_test(lm, lar)
            rows.append({"symbol": sym, "n": len(merged),
                         "model_acc": round(1 - lm.mean(), 4),
                         "p_alwaysup": round(p_au, 4), "p_mom5": round(p_mo, 4), "p_ar1": round(p_ar, 4),
                         "dm_alwaysup": round(dm_au, 3), "dm_mom5": round(dm_mo, 3), "dm_ar1": round(dm_ar, 3)})
            pooled["always_up"].append((lm, lau)); pooled["mom5"].append((lm, lmo)); pooled["ar1"].append((lm, lar))
        except Exception as e:
            print(f"  [diag] {sym}: {e}")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out = run_dir / "diagnostics_dm.csv"
    df.to_csv(out, index=False)
    pooled_stats = {}
    for name, pairs in pooled.items():
        la = np.concatenate([p[0] for p in pairs]); lb = np.concatenate([p[1] for p in pairs])
        dm, p = _dm_test(la, lb)
        pooled_stats[name] = {"dm": round(dm, 3), "p": round(p, 4),
                              "model_better_than_baseline": bool(p < 0.05)}
    print(f"  [diag] DM tests → {out.name}  ({len(df)} stocks)")
    print(f"         pooled vs Always-UP : DM={pooled_stats['always_up']['dm']:+.3f} p={pooled_stats['always_up']['p']:.4f}")
    print(f"         pooled vs Momentum-5: DM={pooled_stats['mom5']['dm']:+.3f} p={pooled_stats['mom5']['p']:.4f}")
    print(f"         pooled vs AR(1)     : DM={pooled_stats['ar1']['dm']:+.3f} p={pooled_stats['ar1']['p']:.4f}")
    return df, pooled_stats


# ──────────────────────────── Driver ────────────────────────────────────────

def run_diagnostics(run_dir: Path) -> None:
    print(f"\n  ── Diagnostics ─────────────────────────────")
    regime_df = run_regime_replay(run_dir)
    dm_result = run_dm_tests(run_dir)
    summary = {}
    if regime_df is not None and not regime_df.empty:
        summary["regime"] = regime_df.to_dict(orient="records")
    if isinstance(dm_result, tuple):
        _, pooled_stats = dm_result
        summary["dm_pooled"] = pooled_stats
    if summary:
        with open(run_dir / "diagnostics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  [diag] summary → diagnostics_summary.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Path to 06_results/runs/<run_id>")
    args = p.parse_args()
    run_diagnostics(Path(args.run_dir))
