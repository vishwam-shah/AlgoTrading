"""
Experiment 1: Target Ablation
================================
Reuses cached features in V3/01_data/features/raw/{SYM}_features.parquet.
Rebuilds 5 target variants, runs LightGBM walk-forward, compares OOS metrics
AND realistic cost-aware backtest Sharpe on the same splits.

Target variants:
  [BASE]   binary-0.4%       : next_ret > 0.4% → 1, < -0.4% → 0, else drop   (current pipeline)
  [V1]     vol-0.5sigma      : next_ret > 0.5σ → 1, < -0.5σ → 0, else drop    (volatility-adaptive)
  [V2]     vol-1.0sigma      : stricter, less trades, higher conviction
  [V3]     TB-2sigma-5d      : triple-barrier (PT=+2σ, SL=-2σ, H=5 days) — hit first barrier
  [V4]     horizon-5d-sign   : 5-day cumulative return sign (longer horizon, less noise)

Baselines (measured on SAME splits):
  [B1]     Always UP
  [B2]     Momentum-5d (predict sign of prior 5-day return)
  [B3]     AR(1) sign

Per stock × variant, measure:
  - OOS accuracy on directional split
  - Precision on UP signals (conf ≥ 0.58)
  - Net Sharpe after 0.25% round-trip cost
  - Win rate, n_trades, total_return over OOS period

Output: V3/08_experiments/results/exp1_summary.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_EXP_DIR  = Path(__file__).resolve().parent
_V3_ROOT  = _EXP_DIR.parent
sys.path.insert(0, str(_V3_ROOT))
sys.path.insert(0, str(_V3_ROOT / "00_config"))

from config import (  # type: ignore
    FEAT_RAW_DIR, INITIAL_TRAIN_RATIO, EXPANSION_STEP, MAX_TRAIN_RATIO,
    MIN_TRAIN_SAMPLES, MIN_TEST_SAMPLES, CONFIDENCE_THRESHOLD, RANDOM_SEED,
)

OUT_DIR = _EXP_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

ROUND_TRIP_COST = 0.0025   # 0.25% — match steps/backtest.py
ANN_FACTOR      = 252


# ── Stock subset: 20 liquid names across sectors (faster iteration) ────────────
SUBSET = [
    "SBIN", "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK",  # Banking
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",                # IT
    "RELIANCE", "MARUTI", "LT", "BHARTIARTL", "ITC",           # Large-cap
    "ASIANPAINT", "TITAN", "SUNPHARMA", "NTPC", "TATASTEEL",   # Diversified
]


# ══════════════════════════════════════════════════════════════════════════════
#  TARGET BUILDERS  — each returns (target_series, effective_return_series)
# ══════════════════════════════════════════════════════════════════════════════

def _next_ret(df: pd.DataFrame) -> pd.Series:
    return (df["close"].shift(-1) - df["close"]) / (df["close"] + 1e-10)


def target_binary_fixed(df: pd.DataFrame, thr: float = 0.004) -> pd.DataFrame:
    """Base: 1 if next_ret > thr, 0 if < -thr, NaN else."""
    nr = _next_ret(df)
    y  = np.where(nr > thr, 1.0, np.where(nr < -thr, 0.0, np.nan))
    return pd.DataFrame({"y": y, "ret": nr.values})


def target_vol_normalised(df: pd.DataFrame, k: float = 0.5, win: int = 20) -> pd.DataFrame:
    """Volatility-adaptive: 1 if next_ret > k*σ_{past win}, 0 if < -k*σ. NaN else."""
    nr    = _next_ret(df)
    sigma = nr.rolling(win, min_periods=10).std().shift(1)  # no leakage
    thr   = k * sigma
    y     = np.where(nr >  thr, 1.0, np.where(nr < -thr, 0.0, np.nan))
    return pd.DataFrame({"y": y, "ret": nr.values})


def target_triple_barrier(df: pd.DataFrame, k_pt: float = 2.0, k_sl: float = 2.0,
                          horizon: int = 5, win: int = 20) -> pd.DataFrame:
    """
    Triple-barrier (López de Prado 2018 Advances in Financial ML, §3.4).
    For each t: set PT = +k_pt*σ, SL = -k_sl*σ (σ = trailing 20-day daily return std).
    Label 1 if PT hit first within `horizon` days, 0 if SL hit first, NaN if H hit first.
    Effective return = return when barrier was hit (or horizon return).
    """
    close = df["close"].values
    ret_1 = np.log(close[1:] / close[:-1])
    sigma = pd.Series(ret_1).rolling(win, min_periods=10).std().shift(1).bfill().values
    n     = len(df)
    y     = np.full(n, np.nan)
    eff_r = np.full(n, np.nan)

    for t in range(n - 1):
        s   = sigma[t] if t < len(sigma) else np.nan
        if not np.isfinite(s) or s <= 0:
            continue
        pt = k_pt * s
        sl = -k_sl * s
        end = min(t + horizon, n - 1)
        # walk forward from t+1 to end
        for u in range(t + 1, end + 1):
            r = np.log(close[u] / close[t])
            if r >= pt:
                y[t] = 1.0; eff_r[t] = r; break
            if r <= sl:
                y[t] = 0.0; eff_r[t] = r; break
        else:
            # horizon hit — drop (ambiguous, common practice)
            y[t] = np.nan
            eff_r[t] = np.log(close[end] / close[t])

    # Replace eff_r with simple next-day return for P&L simulation consistency
    nr = _next_ret(df).values
    return pd.DataFrame({"y": y, "ret": nr})


def target_horizon5_sign(df: pd.DataFrame) -> pd.DataFrame:
    """5-day cumulative return sign. Less noisy than 1-day."""
    close = df["close"].values
    future = pd.Series(close).shift(-5) / pd.Series(close) - 1.0
    y      = np.where(future > 0.01, 1.0, np.where(future < -0.01, 0.0, np.nan))
    # For trading P&L we still use next-day return (fair comparison)
    nr = _next_ret(df).values
    return pd.DataFrame({"y": y, "ret": nr})


TARGET_FNS = {
    "BASE_bin_0.4%":    lambda d: target_binary_fixed(d, thr=0.004),
    "V1_vol_0.5sigma":  lambda d: target_vol_normalised(d, k=0.5, win=20),
    "V2_vol_1.0sigma":  lambda d: target_vol_normalised(d, k=1.0, win=20),
    "V3_TB_2s_5d":      lambda d: target_triple_barrier(d, 2.0, 2.0, 5, 20),
    "V4_horizon5":      lambda d: target_horizon5_sign(d),
}


# ══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_windows(n: int) -> List[Dict]:
    w, r = [], INITIAL_TRAIN_RATIO
    while r <= MAX_TRAIN_RATIO:
        te    = int(n * r)
        vs    = te - max(int(te * 0.10), 20)
        nr    = r + EXPANSION_STEP
        t2e   = int(n * (nr + EXPANSION_STEP)) if nr <= MAX_TRAIN_RATIO else n
        t2e   = min(t2e, n)
        if t2e - te < MIN_TEST_SAMPLES:
            r = round(r + EXPANSION_STEP, 4); continue
        w.append(dict(train_start=0, train_end=vs, val_start=vs, val_end=te,
                      test_start=te, test_end=t2e, ratio=r))
        r = round(r + EXPANSION_STEP, 4)
    return w


def feature_columns(df: pd.DataFrame) -> List[str]:
    # Hard exclude: OHLCV, meta, targets, AND any forward-looking helpers we added
    exclude = {
        "date", "timestamp", "symbol",
        "open", "high", "low", "close", "volume",
        "target", "y", "ret", "next_ret",
    }
    # Anything with a forward-looking prefix/suffix we may have accidentally created
    def _is_leaky(name: str) -> bool:
        n = name.lower()
        return (n.startswith("next_") or n.startswith("future_") or n.startswith("tmrw_")
                or n.endswith("_target") or n.endswith("_tgt") or n == "y" or n == "ret")
    return [c for c in df.columns
            if c not in exclude and not _is_leaky(c)
            and df[c].dtype in ("float64", "float32", "int64", "int32", "int8", "uint8")]


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sharpe_annualised(rets: np.ndarray, rf_ann: float = 0.065) -> float:
    if len(rets) < 5 or rets.std() == 0:
        return 0.0
    rfd = (1 + rf_ann) ** (1 / ANN_FACTOR) - 1
    return float((rets - rfd).mean() / rets.std() * np.sqrt(ANN_FACTOR))


def max_drawdown(eq: np.ndarray) -> float:
    if len(eq) == 0: return 0.0
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak
    return float(-dd.min()) if dd.min() < 0 else 0.0


def backtest_metrics(prob_up: np.ndarray, y_dir: np.ndarray, ret: np.ndarray,
                     conf: float = 0.58) -> Dict:
    """P&L assuming trade = (prob_up>=conf) long next day, held 1 day, 0.25% cost."""
    mask = prob_up >= conf
    if mask.sum() < 5:
        return dict(n_trades=0, total_return=0.0, win_rate=0.0,
                    sharpe=0.0, max_dd=0.0, precision=0.0)
    trade_ret = ret[mask] - ROUND_TRIP_COST
    wins      = (trade_ret > 0).astype(int)
    eq        = np.cumprod(1 + trade_ret)
    precision = float(y_dir[mask].mean()) if len(y_dir[mask]) else 0.0
    return dict(
        n_trades     = int(mask.sum()),
        total_return = float(eq[-1] - 1),
        win_rate     = float(wins.mean()),
        sharpe       = sharpe_annualised(trade_ret),
        max_dd       = max_drawdown(eq),
        precision    = precision,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PRIMARY CLASSIFIER  (LightGBM — fast, dominant tree model)
# ══════════════════════════════════════════════════════════════════════════════

def train_lgbm(X_tr, y_tr, X_va, y_va) -> Tuple[Optional[object], float]:
    """Return (fitted model, val accuracy). Handles single-class train/val."""
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    import io, contextlib

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
        return None, 0.5

    mdl = LGBMClassifier(
        n_estimators=800, max_depth=5, learning_rate=0.02,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.3, reg_lambda=1.5, min_child_samples=20,
        is_unbalance=True, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mdl.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                callbacks=[early_stopping(50, verbose=False), log_evaluation(period=-1)])
    return mdl, float(mdl.score(X_va, y_va))


# ══════════════════════════════════════════════════════════════════════════════
#  BASELINES
# ══════════════════════════════════════════════════════════════════════════════

def baseline_probs(df: pd.DataFrame, test_idx: np.ndarray, name: str) -> np.ndarray:
    if name == "B1_alwaysUP":
        return np.full(len(test_idx), 1.0)
    if name == "B2_momentum5":
        return df["_bl_mom5"].values[test_idx]
    if name == "B3_AR1":
        return df["_bl_ar1"].values[test_idx]
    raise ValueError(name)


# ══════════════════════════════════════════════════════════════════════════════
#  PER-STOCK RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_one_stock(symbol: str, target_name: str, run_baselines: bool = True) -> List[Dict]:
    fpath = FEAT_RAW_DIR / f"{symbol}_features.parquet"
    if not fpath.exists():
        return []

    df = pd.read_parquet(fpath).copy()
    df = df.sort_values("date").reset_index(drop=True)
    if "close" not in df.columns or len(df) < 400:
        return []

    # Baseline probabilities computed on the FULL (unfiltered) close series so momentum
    # and AR(1) see contiguous lags. We compute them here, then align them to the
    # filtered df via date-join later.
    raw_close_full   = df["close"].copy()
    raw_date_full    = df["date"].copy()
    baseline_mom5_full = (raw_close_full.pct_change(5).shift(1) > 0).astype(float).clip(0.01, 0.99)
    baseline_ar1_full  = (raw_close_full.pct_change(1).shift(1) > 0).astype(float).clip(0.01, 0.99)

    # Build target
    tgt_df = TARGET_FNS[target_name](df)
    df["y"]   = tgt_df["y"].values
    df["ret"] = tgt_df["ret"].values

    # Next-day return for every row (for P&L); NOT used as a feature
    df["next_ret"] = _next_ret(df).values

    # Feature cols = everything not OHLCV/meta/leaky
    fcols = feature_columns(df)
    if len(fcols) < 20:
        return []

    # We need at least 20 rows with valid y to train. We drop NaN y BEFORE splitting
    # — matches the pipeline ('dropna(subset=["target"])' in features.py).
    keep_mask = df["y"].notna() & df["next_ret"].notna() & df[fcols].notna().all(axis=1)
    df_full   = df.copy()   # keep pre-filter copy to align baselines
    df        = df[keep_mask].reset_index(drop=True)
    if len(df) < MIN_TRAIN_SAMPLES + MIN_TEST_SAMPLES * 2:
        return []

    # Align baseline probs to the filtered df using date index
    full_prob_map_mom5 = dict(zip(raw_date_full.values, baseline_mom5_full.values))
    full_prob_map_ar1  = dict(zip(raw_date_full.values, baseline_ar1_full.values))
    df["_bl_mom5"] = df["date"].map(full_prob_map_mom5).fillna(0.5).values
    df["_bl_ar1"]  = df["date"].map(full_prob_map_ar1).fillna(0.5).values

    windows = build_windows(len(df))
    if not windows:
        return []

    X   = df[fcols].values.astype(np.float32)
    y   = df["y"].values.astype(int)
    nr  = df["next_ret"].values

    # Winsorise + scale globally (approx; ablation isn't training shape-sensitive)
    p1, p99 = np.nanpercentile(X, [1, 99], axis=0)
    X = np.clip(X, p1, p99)
    from sklearn.preprocessing import RobustScaler
    X = np.clip(np.nan_to_num(RobustScaler().fit_transform(X), nan=0.0), -5, 5)

    results: List[Dict] = []

    # ── Primary LGBM runs per window, concat OOS probs ─────────────────────────
    all_prob, all_y, all_ret = [], [], []
    for w in windows:
        X_tr, y_tr = X[w["train_start"]:w["train_end"]], y[w["train_start"]:w["train_end"]]
        X_va, y_va = X[w["val_start"]:w["val_end"]],     y[w["val_start"]:w["val_end"]]
        X_te, y_te = X[w["test_start"]:w["test_end"]],   y[w["test_start"]:w["test_end"]]
        nr_te      = nr[w["test_start"]:w["test_end"]]
        if len(y_tr) < MIN_TRAIN_SAMPLES or len(np.unique(y_tr)) < 2:
            continue
        mdl, _ = train_lgbm(X_tr, y_tr, X_va, y_va)
        if mdl is None:
            continue
        p = mdl.predict_proba(X_te)[:, 1]
        all_prob.extend(p); all_y.extend(y_te); all_ret.extend(nr_te)

    if len(all_prob) < 50:
        return []

    all_prob = np.array(all_prob); all_y = np.array(all_y); all_ret = np.array(all_ret)
    oos_acc  = float(((all_prob >= 0.5) == all_y).mean())
    bt       = backtest_metrics(all_prob, all_y, all_ret, conf=CONFIDENCE_THRESHOLD)
    results.append(dict(
        symbol=symbol, model=f"LGBM_{target_name}", oos_acc=oos_acc,
        n_oos=len(all_y), **bt,
    ))

    # ── Baselines (same OOS indices) ───────────────────────────────────────────
    if run_baselines:
        # Build a contiguous test index across windows
        test_idx = []
        for w in windows:
            test_idx.extend(range(w["test_start"], w["test_end"]))
        test_idx = np.array(test_idx)
        y_b   = y[test_idx]
        nr_b  = nr[test_idx]

        for bname in ["B1_alwaysUP", "B2_momentum5", "B3_AR1"]:
            p = baseline_probs(df, test_idx, bname)
            acc = float(((p >= 0.5) == y_b).mean())
            bt  = backtest_metrics(p, y_b, nr_b, conf=CONFIDENCE_THRESHOLD)
            results.append(dict(
                symbol=symbol, model=f"{bname}_{target_name}",
                oos_acc=acc, n_oos=len(y_b), **bt,
            ))

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    all_rows: List[Dict] = []
    for target_name in TARGET_FNS:
        # baselines share everything with the target split, so only run once
        run_bl = (target_name == "BASE_bin_0.4%")
        print(f"\n═══ target = {target_name}  run_baselines={run_bl} ═══")
        for sym in SUBSET:
            rows = run_one_stock(sym, target_name, run_baselines=run_bl)
            if rows:
                for r in rows:
                    tag = "BL " if r["model"].startswith(("B1","B2","B3")) else "   "
                    print(f"  {tag}{sym:<12} {r['model']:<30} acc={r['oos_acc']:.3f} "
                          f"n_trades={r['n_trades']:>4} sharpe={r['sharpe']:>+6.2f} "
                          f"ret={r['total_return']:+.2%}")
            all_rows.extend(rows)

    if not all_rows:
        print("No results.")
        return

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_DIR / "exp1_detail.csv", index=False)

    # ── Aggregate by model/target ──────────────────────────────────────────────
    agg = out.groupby("model").agg(
        n_stocks     = ("symbol",       "nunique"),
        avg_oos_acc  = ("oos_acc",      "mean"),
        median_acc   = ("oos_acc",      "median"),
        avg_trades   = ("n_trades",     "mean"),
        avg_precision= ("precision",    "mean"),
        avg_sharpe   = ("sharpe",       "mean"),
        med_sharpe   = ("sharpe",       "median"),
        avg_ret      = ("total_return", "mean"),
        avg_winrate  = ("win_rate",     "mean"),
        n_profitable = ("total_return", lambda s: (s > 0).sum()),
    ).round(4).sort_values("avg_sharpe", ascending=False)

    print("\n" + "="*100)
    print(" EXP1 — TARGET ABLATION SUMMARY  (20-stock subset, LightGBM primary, 0.25% cost, conf≥0.58)")
    print("="*100)
    print(agg.to_string())
    agg.to_csv(OUT_DIR / "exp1_summary.csv")

    print(f"\nDetail  → {OUT_DIR/'exp1_detail.csv'}")
    print(f"Summary → {OUT_DIR/'exp1_summary.csv'}")


if __name__ == "__main__":
    main()
