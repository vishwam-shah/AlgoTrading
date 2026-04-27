"""
Experiment 2: Meta-labeling on top of primary (López de Prado 2018)
====================================================================
Primary (M1)  — binary direction classifier (LightGBM), trained per window.
Secondary (M2)— binary "will this trade be profitable after cost" classifier,
                trained ONLY on samples where M1 predicted UP on the VAL set.
                Features: same X + M1 output probability.
Trade rule     : go long next day iff  M1(UP) AND M2(profitable) >= thresh.

Compared against:
  - Primary-only (same as Exp1 V4 horizon-5, our best) using plain conf >= 0.58

Target: V4 horizon-5 direction — our strongest primary from Exp1 (52.4% OOS acc).

Meta-model labels:
  M2_y = 1 if next_ret > ROUND_TRIP_COST else 0
  trained only on rows where M1 said UP (prob >= 0.5)
  this is the canonical "keep the trade" meta-label

Hypothesis: a well-trained M2 filters out false-positive trades; even if M1 has
only 52.4% directional accuracy, M2 can raise the precision of executed trades
to 55–60%+, which is enough to flip the cost-adjusted P&L positive.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

ROUND_TRIP_COST = 0.0025
ANN_FACTOR      = 252

SUBSET = [
    "SBIN", "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK",
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    "RELIANCE", "MARUTI", "LT", "BHARTIARTL", "ITC",
    "ASIANPAINT", "TITAN", "SUNPHARMA", "NTPC", "TATASTEEL",
]


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES  (copied from exp1 for independence)
# ══════════════════════════════════════════════════════════════════════════════

def _next_ret(df: pd.DataFrame) -> pd.Series:
    return (df["close"].shift(-1) - df["close"]) / (df["close"] + 1e-10)


def target_horizon5_sign(df: pd.DataFrame) -> pd.DataFrame:
    """Primary label: 5-day forward cumulative return sign."""
    close = df["close"].values
    future = pd.Series(close).shift(-5) / pd.Series(close) - 1.0
    y      = np.where(future > 0.01, 1.0, np.where(future < -0.01, 0.0, np.nan))
    nr     = _next_ret(df).values
    return pd.DataFrame({"y": y, "ret": nr})


def target_triple_barrier(df: pd.DataFrame, k_pt: float = 2.0, k_sl: float = 2.0,
                          horizon: int = 5, win: int = 20) -> pd.DataFrame:
    close = df["close"].values
    ret_1 = np.log(close[1:] / close[:-1])
    sigma = pd.Series(ret_1).rolling(win, min_periods=10).std().shift(1).bfill().values
    n     = len(df)
    y     = np.full(n, np.nan)
    for t in range(n - 1):
        s = sigma[t] if t < len(sigma) else np.nan
        if not np.isfinite(s) or s <= 0:
            continue
        pt, sl = k_pt * s, -k_sl * s
        end = min(t + horizon, n - 1)
        for u in range(t + 1, end + 1):
            r = np.log(close[u] / close[t])
            if r >= pt: y[t] = 1.0; break
            if r <= sl: y[t] = 0.0; break
    nr = _next_ret(df).values
    return pd.DataFrame({"y": y, "ret": nr})


def build_windows(n: int) -> List[Dict]:
    w, r = [], INITIAL_TRAIN_RATIO
    while r <= MAX_TRAIN_RATIO:
        te  = int(n * r)
        vs  = te - max(int(te * 0.10), 20)
        nr  = r + EXPANSION_STEP
        t2e = int(n * (nr + EXPANSION_STEP)) if nr <= MAX_TRAIN_RATIO else n
        t2e = min(t2e, n)
        if t2e - te < MIN_TEST_SAMPLES:
            r = round(r + EXPANSION_STEP, 4); continue
        w.append(dict(train_start=0, train_end=vs, val_start=vs, val_end=te,
                      test_start=te, test_end=t2e))
        r = round(r + EXPANSION_STEP, 4)
    return w


def feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"date", "timestamp", "symbol", "open", "high", "low", "close", "volume",
               "target", "y", "y_primary", "y_meta", "ret", "next_ret",
               "_bl_mom5", "_bl_ar1", "_m1_prob"}
    def _leaky(n):
        n = n.lower()
        return (n.startswith("next_") or n.startswith("future_") or n.startswith("tmrw_")
                or n.endswith("_target") or n.endswith("_tgt") or n.startswith("y_"))
    return [c for c in df.columns
            if c not in exclude and not _leaky(c)
            and df[c].dtype in ("float64", "float32", "int64", "int32", "int8", "uint8")]


def sharpe(rets: np.ndarray, rf_ann: float = 0.065) -> float:
    if len(rets) < 5 or rets.std() == 0: return 0.0
    rfd = (1 + rf_ann) ** (1 / ANN_FACTOR) - 1
    return float((rets - rfd).mean() / rets.std() * np.sqrt(ANN_FACTOR))


def max_dd(eq: np.ndarray) -> float:
    if len(eq) == 0: return 0.0
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak
    return float(-dd.min()) if dd.min() < 0 else 0.0


def backtest(mask: np.ndarray, ret: np.ndarray) -> Dict:
    if mask.sum() < 5:
        return dict(n_trades=0, total_return=0.0, win_rate=0.0, sharpe=0.0, max_dd=0.0, precision=0.0)
    trade_ret = ret[mask] - ROUND_TRIP_COST
    wins = trade_ret > 0
    eq   = np.cumprod(1 + trade_ret)
    return dict(
        n_trades=int(mask.sum()),
        total_return=float(eq[-1] - 1),
        win_rate=float(wins.mean()),
        sharpe=sharpe(trade_ret),
        max_dd=max_dd(eq),
        precision=float(wins.mean()),
    )


def train_lgbm(X_tr, y_tr, X_va, y_va, **kwargs):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    import io, contextlib
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
        return None
    defaults = dict(n_estimators=800, max_depth=5, learning_rate=0.02, num_leaves=31,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=1.5,
                    min_child_samples=20, is_unbalance=True, random_state=RANDOM_SEED,
                    n_jobs=-1, verbosity=-1)
    defaults.update(kwargs)
    mdl = LGBMClassifier(**defaults)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mdl.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                callbacks=[early_stopping(50, verbose=False), log_evaluation(period=-1)])
    return mdl


# ══════════════════════════════════════════════════════════════════════════════
#  PER-STOCK: Primary + Meta
# ══════════════════════════════════════════════════════════════════════════════

def run_one(symbol: str, primary_target: str = "V4_horizon5",
            meta_thresh: float = 0.55) -> List[Dict]:
    fpath = FEAT_RAW_DIR / f"{symbol}_features.parquet"
    if not fpath.exists():
        return []

    df = pd.read_parquet(fpath).sort_values("date").reset_index(drop=True)
    if "close" not in df.columns or len(df) < 400:
        return []

    # Build PRIMARY target
    if primary_target == "V4_horizon5":
        tgt = target_horizon5_sign(df)
    elif primary_target == "V3_TB_2s_5d":
        tgt = target_triple_barrier(df, 2.0, 2.0, 5, 20)
    else:
        raise ValueError(primary_target)

    df["y_primary"] = tgt["y"].values
    df["next_ret"]  = _next_ret(df).values

    fcols = feature_columns(df)
    if len(fcols) < 20:
        return []

    keep = df["y_primary"].notna() & df["next_ret"].notna() & df[fcols].notna().all(axis=1)
    df   = df[keep].reset_index(drop=True)
    if len(df) < MIN_TRAIN_SAMPLES + MIN_TEST_SAMPLES * 2:
        return []

    windows = build_windows(len(df))
    if not windows:
        return []

    X  = df[fcols].values.astype(np.float32)
    y1 = df["y_primary"].values.astype(int)
    nr = df["next_ret"].values

    # Winsorise + scale globally
    p1, p99 = np.nanpercentile(X, [1, 99], axis=0)
    X = np.clip(X, p1, p99)
    from sklearn.preprocessing import RobustScaler
    X = np.clip(np.nan_to_num(RobustScaler().fit_transform(X), nan=0.0), -5, 5)

    # Secondary label = 1 if trade was profitable after round-trip cost
    y2 = (nr > ROUND_TRIP_COST).astype(int)

    # Collect per-window results
    oos_primary_prob = []
    oos_meta_prob    = []
    oos_y1           = []
    oos_y2           = []
    oos_ret          = []

    for w in windows:
        X_tr, y1_tr = X[w["train_start"]:w["train_end"]], y1[w["train_start"]:w["train_end"]]
        X_va, y1_va = X[w["val_start"]:w["val_end"]],     y1[w["val_start"]:w["val_end"]]
        X_te, y1_te = X[w["test_start"]:w["test_end"]],   y1[w["test_start"]:w["test_end"]]
        y2_tr = y2[w["train_start"]:w["train_end"]]
        y2_va = y2[w["val_start"]:w["val_end"]]
        y2_te = y2[w["test_start"]:w["test_end"]]
        nr_te = nr[w["test_start"]:w["test_end"]]

        if len(y1_tr) < MIN_TRAIN_SAMPLES or len(np.unique(y1_tr)) < 2:
            continue

        # ── M1: primary direction ─────────────────────────────────────────────
        m1 = train_lgbm(X_tr, y1_tr, X_va, y1_va)
        if m1 is None:
            continue
        p_tr = m1.predict_proba(X_tr)[:, 1]
        p_va = m1.predict_proba(X_va)[:, 1]
        p_te = m1.predict_proba(X_te)[:, 1]

        # ── M2: meta-label "is this trade profitable after cost" ──────────────
        # Trained ONLY on rows where M1 said UP (p >= 0.5) in train+val,
        # features = X augmented with M1 prob.
        m1_pred_tr = p_tr >= 0.5
        m1_pred_va = p_va >= 0.5
        if m1_pred_tr.sum() < 100 or m1_pred_va.sum() < 30:
            # not enough positive primary signals → fall back to primary-only
            oos_primary_prob.extend(p_te); oos_meta_prob.extend(np.ones(len(p_te)))
            oos_y1.extend(y1_te); oos_y2.extend(y2_te); oos_ret.extend(nr_te)
            continue

        X_tr_m2 = np.column_stack([X_tr[m1_pred_tr], p_tr[m1_pred_tr]])
        y2_tr_m2 = y2_tr[m1_pred_tr]
        X_va_m2 = np.column_stack([X_va[m1_pred_va], p_va[m1_pred_va]])
        y2_va_m2 = y2_va[m1_pred_va]

        m2 = train_lgbm(X_tr_m2, y2_tr_m2, X_va_m2, y2_va_m2,
                        n_estimators=400, learning_rate=0.03)
        if m2 is None:
            oos_primary_prob.extend(p_te); oos_meta_prob.extend(np.ones(len(p_te)))
            oos_y1.extend(y1_te); oos_y2.extend(y2_te); oos_ret.extend(nr_te)
            continue

        X_te_m2 = np.column_stack([X_te, p_te])
        p_meta  = m2.predict_proba(X_te_m2)[:, 1]

        oos_primary_prob.extend(p_te); oos_meta_prob.extend(p_meta)
        oos_y1.extend(y1_te); oos_y2.extend(y2_te); oos_ret.extend(nr_te)

    if len(oos_primary_prob) < 50:
        return []

    p1 = np.array(oos_primary_prob); p2 = np.array(oos_meta_prob)
    y1 = np.array(oos_y1);            y2 = np.array(oos_y2)
    nr = np.array(oos_ret)

    rows: List[Dict] = []

    # Regime A: primary-only (conf >= 0.58)  — our Exp1 best
    mask_a = p1 >= CONFIDENCE_THRESHOLD
    bt_a   = backtest(mask_a, nr)
    rows.append(dict(symbol=symbol, strategy="primary_only_0.58",
                     oos_acc_dir=float(((p1 >= 0.5) == y1).mean()),
                     n_oos=len(y1), **bt_a))

    # Regime B: primary_up AND meta >= meta_thresh  — vanilla meta-labeling
    mask_b = (p1 >= 0.50) & (p2 >= meta_thresh)
    bt_b   = backtest(mask_b, nr)
    rows.append(dict(symbol=symbol, strategy=f"meta_label_thr{meta_thresh:.2f}",
                     oos_acc_dir=float(((p1 >= 0.5) == y1).mean()),
                     n_oos=len(y1), **bt_b))

    # Regime C: stricter meta (higher conviction)
    mask_c = (p1 >= 0.50) & (p2 >= 0.60)
    bt_c   = backtest(mask_c, nr)
    rows.append(dict(symbol=symbol, strategy="meta_label_thr0.60",
                     oos_acc_dir=float(((p1 >= 0.5) == y1).mean()),
                     n_oos=len(y1), **bt_c))

    # Regime D: stricter meta 0.65
    mask_d = (p1 >= 0.50) & (p2 >= 0.65)
    bt_d   = backtest(mask_d, nr)
    rows.append(dict(symbol=symbol, strategy="meta_label_thr0.65",
                     oos_acc_dir=float(((p1 >= 0.5) == y1).mean()),
                     n_oos=len(y1), **bt_d))

    # Regime E: stricter meta + primary conviction (belt-and-braces)
    mask_e = (p1 >= 0.58) & (p2 >= 0.60)
    bt_e   = backtest(mask_e, nr)
    rows.append(dict(symbol=symbol, strategy="primary_0.58_AND_meta_0.60",
                     oos_acc_dir=float(((p1 >= 0.5) == y1).mean()),
                     n_oos=len(y1), **bt_e))

    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="V4_horizon5",
                    choices=["V4_horizon5", "V3_TB_2s_5d"])
    args = ap.parse_args()

    all_rows: List[Dict] = []
    print(f"\n═══ Meta-labeling experiment  primary={args.primary}  cost=0.25% ═══")
    for sym in SUBSET:
        rows = run_one(sym, primary_target=args.primary)
        if rows:
            for r in rows:
                print(f"  {sym:<12} {r['strategy']:<28} "
                      f"n={r['n_trades']:>4} win={r['win_rate']:.2%} "
                      f"sharpe={r['sharpe']:>+6.2f} ret={r['total_return']:+.2%}")
        all_rows.extend(rows)

    if not all_rows:
        print("No results."); return

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_DIR / f"exp2_detail_{args.primary}.csv", index=False)

    agg = out.groupby("strategy").agg(
        n_stocks      = ("symbol",       "nunique"),
        avg_trades    = ("n_trades",     "mean"),
        avg_precision = ("precision",    "mean"),
        avg_sharpe    = ("sharpe",       "mean"),
        med_sharpe    = ("sharpe",       "median"),
        avg_ret       = ("total_return", "mean"),
        avg_winrate   = ("win_rate",     "mean"),
        n_profitable  = ("total_return", lambda s: (s > 0).sum()),
        n_pos_sharpe  = ("sharpe",       lambda s: (s > 0).sum()),
    ).round(4).sort_values("avg_sharpe", ascending=False)

    print("\n" + "="*105)
    print(f" EXP2 — META-LABELING SUMMARY  (primary={args.primary}, 20 stocks, 0.25% cost)")
    print("="*105)
    print(agg.to_string())
    agg.to_csv(OUT_DIR / f"exp2_summary_{args.primary}.csv")
    print(f"\nDetail  → {OUT_DIR}/exp2_detail_{args.primary}.csv")
    print(f"Summary → {OUT_DIR}/exp2_summary_{args.primary}.csv")


if __name__ == "__main__":
    main()
