"""
Experiment 3: Winning config — pushing toward actual profitability
===================================================================
Combines the components that individually helped in Exp1/Exp2:

  PRIMARY (M1) : Horizon-5 target  (best directional edge — Exp1 52.4%)
               : Ensemble of LightGBM + XGBoost + CatBoost with calibrated
                 val-logloss weighted soft vote  (mirrors the pipeline ensemble)
               : Temperature-scaling calibration on val

  META (M2)    : LightGBM trained on {primary says UP}
               : Label = next_ret > 0.25% (profitable after round-trip cost)
               : Features = X augmented with M1 prob

  REGIME GATE : skip trades when VIX z-score > 1.5 (stress regime)
              : also skip when abs(nifty50_ret_20d) > 6% sharp sell-offs

  PORTFOLIO   : per day, rank stocks by primary*meta score; trade TOP_K=5
              : equal-weighted, half-Kelly sized at portfolio level

Compared to:
  - Baseline V4 primary-only (our Exp2 starting point)
  - Current pipeline approximation (V0 binary-0.4%, primary-only, 0.58)

Reports per-stock AND portfolio-level metrics (Sharpe, DD, total return,
n_stocks_profitable, NIFTY comparison).
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
    FEAT_RAW_DIR, RAW_DATA_DIR, INITIAL_TRAIN_RATIO, EXPANSION_STEP, MAX_TRAIN_RATIO,
    MIN_TRAIN_SAMPLES, MIN_TEST_SAMPLES, CONFIDENCE_THRESHOLD, RANDOM_SEED,
    SYMBOLS_100,
)

OUT_DIR = _EXP_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

ROUND_TRIP_COST = 0.0025
ANN_FACTOR      = 252

# 20-stock subset for iteration; full for final
SUBSET = [
    "SBIN", "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK",
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    "RELIANCE", "MARUTI", "LT", "BHARTIARTL", "ITC",
    "ASIANPAINT", "TITAN", "SUNPHARMA", "NTPC", "TATASTEEL",
]


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _next_ret(df: pd.DataFrame) -> pd.Series:
    return (df["close"].shift(-1) - df["close"]) / (df["close"] + 1e-10)


def target_horizon5_sign(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].values
    future = pd.Series(close).shift(-5) / pd.Series(close) - 1.0
    y = np.where(future > 0.01, 1.0, np.where(future < -0.01, 0.0, np.nan))
    nr = _next_ret(df).values
    return pd.DataFrame({"y": y, "ret": nr})


def build_windows(n: int) -> List[Dict]:
    w, r = [], INITIAL_TRAIN_RATIO
    while r <= MAX_TRAIN_RATIO:
        te  = int(n * r)
        vs  = te - max(int(te * 0.10), 20)
        nr_ = r + EXPANSION_STEP
        t2e = int(n * (nr_ + EXPANSION_STEP)) if nr_ <= MAX_TRAIN_RATIO else n
        t2e = min(t2e, n)
        if t2e - te < MIN_TEST_SAMPLES:
            r = round(r + EXPANSION_STEP, 4); continue
        w.append(dict(train_start=0, train_end=vs, val_start=vs, val_end=te,
                      test_start=te, test_end=t2e))
        r = round(r + EXPANSION_STEP, 4)
    return w


def feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"date", "timestamp", "symbol", "open", "high", "low", "close", "volume",
               "target", "y", "y_primary", "y_meta", "ret", "next_ret"}
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
    peak = np.maximum.accumulate(eq); dd = (eq - peak) / peak
    return float(-dd.min()) if dd.min() < 0 else 0.0


def temperature_scale(probs: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    from scipy.special import expit
    p  = np.clip(probs, 1e-7, 1 - 1e-7)
    lg = np.log(p / (1 - p))
    def nll(T):
        T = max(T, 0.05)
        cal = np.clip(expit(lg / T), 1e-7, 1 - 1e-7)
        return -float(np.mean(y * np.log(cal) + (1 - y) * np.log(1 - cal)))
    try:
        r = minimize_scalar(nll, bounds=(0.1, 3.0), method="bounded")
        T = float(r.x)
        if T >= 2.9 or nll(T) > nll(1.0) - 1e-4: return 1.0
        return T
    except Exception:
        return 1.0


def apply_T(p: np.ndarray, T: float) -> np.ndarray:
    from scipy.special import expit
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.array(expit(np.log(p / (1 - p)) / max(T, 0.05)), dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINERS
# ══════════════════════════════════════════════════════════════════════════════

def _silent(fn, *a, **kw):
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **kw)


def train_lgbm(X_tr, y_tr, X_va, y_va, **kw):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2: return None
    p = dict(n_estimators=800, max_depth=5, learning_rate=0.02, num_leaves=31,
             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=1.5,
             min_child_samples=20, is_unbalance=True,
             random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1)
    p.update(kw)
    m = LGBMClassifier(**p)
    def _fit():
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[early_stopping(50, verbose=False), log_evaluation(period=-1)])
    _silent(_fit)
    return m


def train_xgb(X_tr, y_tr, X_va, y_va):
    from xgboost import XGBClassifier
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2: return None
    spw = max(1, (y_tr == 0).sum()) / max(1, (y_tr == 1).sum())
    m = XGBClassifier(
        n_estimators=800, max_depth=5, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=1.5,
        scale_pos_weight=spw, early_stopping_rounds=50, eval_metric="logloss",
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
    )
    _silent(lambda: m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False))
    return m


def train_cat(X_tr, y_tr, X_va, y_va):
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2: return None
    m = CatBoostClassifier(
        iterations=800, depth=6, learning_rate=0.02,
        l2_leaf_reg=3.0, border_count=128,
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED, thread_count=-1, verbose=False,
        early_stopping_rounds=50,
    )
    _silent(lambda: m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False))
    return m


def ensemble_predict(models: Dict, X: np.ndarray) -> np.ndarray:
    """val-logloss weighted vote (weights passed as attribute)."""
    probs = []; wts = []
    for name, (m, w) in models.items():
        p = m.predict_proba(X)[:, 1]
        probs.append(p); wts.append(w)
    if not probs:
        return np.full(len(X), 0.5)
    probs = np.array(probs); wts = np.array(wts); wts = wts / wts.sum()
    return (probs * wts[:, None]).sum(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  PER-STOCK RUN — Primary ensemble + Meta + Regime gate
# ══════════════════════════════════════════════════════════════════════════════

def _vix_proxy(df: pd.DataFrame) -> np.ndarray:
    """Build a stress gate from the global cues we already have in features."""
    col_vix_z    = "us_vix_zscore" if "us_vix_zscore" in df.columns else None
    col_n50_20d  = "nifty50_ret_20d" if "nifty50_ret_20d" in df.columns else None
    n = len(df)
    stress = np.zeros(n, dtype=bool)
    if col_vix_z is not None:
        stress |= (df[col_vix_z].fillna(0).values > 1.5)
    if col_n50_20d is not None:
        stress |= (np.abs(df[col_n50_20d].fillna(0).values) > 0.06)
    return stress


def run_one(symbol: str, use_regime_gate: bool = True) -> Optional[Dict]:
    fpath = FEAT_RAW_DIR / f"{symbol}_features.parquet"
    if not fpath.exists():
        return None

    df = pd.read_parquet(fpath).sort_values("date").reset_index(drop=True)
    if "close" not in df.columns or len(df) < 400:
        return None

    tgt = target_horizon5_sign(df)
    df["y_primary"] = tgt["y"].values
    df["next_ret"]  = _next_ret(df).values

    fcols = feature_columns(df)
    if len(fcols) < 20:
        return None

    stress_full = _vix_proxy(df)

    keep = df["y_primary"].notna() & df["next_ret"].notna() & df[fcols].notna().all(axis=1)
    df_full = df.copy()
    df = df[keep].reset_index(drop=True)
    if len(df) < MIN_TRAIN_SAMPLES + MIN_TEST_SAMPLES * 2:
        return None

    # Align stress mask to filtered df by date
    s_map = dict(zip(df_full["date"].values, stress_full))
    df["_stress"] = df["date"].map(s_map).fillna(False).astype(bool).values

    X  = df[fcols].values.astype(np.float32)
    y1 = df["y_primary"].values.astype(int)
    nr = df["next_ret"].values

    p01, p99 = np.nanpercentile(X, [1, 99], axis=0)
    X = np.clip(X, p01, p99)
    from sklearn.preprocessing import RobustScaler
    X = np.clip(np.nan_to_num(RobustScaler().fit_transform(X), nan=0.0), -5, 5)

    y2 = (nr > ROUND_TRIP_COST).astype(int)

    windows = build_windows(len(df))
    out_dates, out_p1, out_p2, out_y1, out_ret, out_stress = [], [], [], [], [], []

    from sklearn.metrics import log_loss

    for w in windows:
        X_tr, y_tr = X[w["train_start"]:w["train_end"]], y1[w["train_start"]:w["train_end"]]
        X_va, y_va = X[w["val_start"]:w["val_end"]],     y1[w["val_start"]:w["val_end"]]
        X_te, y_te = X[w["test_start"]:w["test_end"]],   y1[w["test_start"]:w["test_end"]]
        nr_te   = nr[w["test_start"]:w["test_end"]]
        y2_tr   = y2[w["train_start"]:w["train_end"]]
        y2_va   = y2[w["val_start"]:w["val_end"]]
        y2_te   = y2[w["test_start"]:w["test_end"]]
        date_te = df["date"].values[w["test_start"]:w["test_end"]]
        stress_te = df["_stress"].values[w["test_start"]:w["test_end"]]

        if len(y_tr) < MIN_TRAIN_SAMPLES or len(np.unique(y_tr)) < 2: continue

        # ── Primary ensemble ──────────────────────────────────────────────────
        prim_models: Dict = {}
        for name, tr in [("lgbm", train_lgbm), ("xgb", train_xgb), ("cat", train_cat)]:
            m = tr(X_tr, y_tr, X_va, y_va)
            if m is not None:
                p_va = m.predict_proba(X_va)[:, 1]
                ll   = max(log_loss(y_va, p_va), 1e-6)
                prim_models[name] = (m, 1.0 / ll)
        if not prim_models:
            continue

        p_tr_primary = ensemble_predict({k: v for k, v in prim_models.items()}, X_tr)
        p_va_primary = ensemble_predict(prim_models, X_va)
        p_te_primary = ensemble_predict(prim_models, X_te)

        # Temperature calibration
        T = temperature_scale(p_va_primary, y_va)
        p_te_primary = apply_T(p_te_primary, T)
        p_tr_primary = apply_T(p_tr_primary, T)
        p_va_primary = apply_T(p_va_primary, T)

        # ── Meta model on primary-says-UP subset ──────────────────────────────
        m1_up_tr = p_tr_primary >= 0.5
        m1_up_va = p_va_primary >= 0.5
        p_meta_te = np.ones(len(p_te_primary))  # default: let primary decide
        if m1_up_tr.sum() >= 100 and m1_up_va.sum() >= 30:
            Xm_tr = np.column_stack([X_tr[m1_up_tr], p_tr_primary[m1_up_tr]])
            Xm_va = np.column_stack([X_va[m1_up_va], p_va_primary[m1_up_va]])
            Xm_te = np.column_stack([X_te, p_te_primary])
            m2 = train_lgbm(Xm_tr, y2_tr[m1_up_tr], Xm_va, y2_va[m1_up_va],
                            n_estimators=400, learning_rate=0.03)
            if m2 is not None:
                p_meta_te = m2.predict_proba(Xm_te)[:, 1]

        out_dates.extend(date_te); out_p1.extend(p_te_primary); out_p2.extend(p_meta_te)
        out_y1.extend(y_te); out_ret.extend(nr_te); out_stress.extend(stress_te)

    if len(out_p1) < 50:
        return None

    dates = np.array(out_dates)
    p1    = np.array(out_p1); p2 = np.array(out_p2)
    y1    = np.array(out_y1); ret = np.array(out_ret)
    stress = np.array(out_stress, dtype=bool)

    # Configurations to evaluate
    cfgs = {
        "primary_only_0.58":         (p1 >= 0.58),
        "primary_meta_0.58_0.60":    ((p1 >= 0.58) & (p2 >= 0.60)),
        "primary_meta_0.55_0.60":    ((p1 >= 0.55) & (p2 >= 0.60)),
        "primary_meta_0.60_0.65":    ((p1 >= 0.60) & (p2 >= 0.65)),
    }
    if use_regime_gate:
        cfgs["PRIM+META+REGIME_gate"] = ((p1 >= 0.58) & (p2 >= 0.60) & ~stress)
        cfgs["PRIM+META+REGIME_0.55"] = ((p1 >= 0.55) & (p2 >= 0.60) & ~stress)

    metrics_per_cfg: Dict[str, Dict] = {}
    for name, mask in cfgs.items():
        if mask.sum() < 5:
            metrics_per_cfg[name] = dict(n_trades=0, total_return=0.0, win_rate=0.0,
                                         sharpe=0.0, max_dd=0.0)
            continue
        trade_ret = ret[mask] - ROUND_TRIP_COST
        eq = np.cumprod(1 + trade_ret)
        metrics_per_cfg[name] = dict(
            n_trades=int(mask.sum()),
            total_return=float(eq[-1] - 1),
            win_rate=float((trade_ret > 0).mean()),
            sharpe=sharpe(trade_ret),
            max_dd=max_dd(eq),
        )

    return dict(symbol=symbol, dates=dates, p1=p1, p2=p2, y1=y1, ret=ret,
                stress=stress, per_cfg=metrics_per_cfg, n_oos=len(p1))


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO-LEVEL: Top-K each day, equal weight
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_backtest(stock_results: List[Dict], top_k: int = 5,
                       score_fn=lambda p1, p2: p1 * p2) -> Dict:
    """
    Per day, rank stocks where (p1>=0.58 AND p2>=0.60 AND not stress) by score.
    Go long TOP_K equal-weighted for next day's return minus cost.
    Portfolio daily return = mean of selected stocks' (ret - cost).
    """
    per_day: Dict[str, List[Tuple[str, float, float]]] = {}

    for r in stock_results:
        sym  = r["symbol"]; dates = r["dates"]; p1 = r["p1"]; p2 = r["p2"]
        ret = r["ret"]; stress = r["stress"]
        mask = (p1 >= 0.58) & (p2 >= 0.60) & ~stress
        for i in np.where(mask)[0]:
            d = pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
            per_day.setdefault(d, []).append(
                (sym, float(score_fn(p1[i], p2[i])), float(ret[i]))
            )

    rows = []
    for d, picks in sorted(per_day.items()):
        picks.sort(key=lambda t: -t[1])
        chosen = picks[:top_k]
        daily_ret = np.mean([c[2] for c in chosen]) - ROUND_TRIP_COST
        rows.append({"date": d, "n_picks": len(chosen), "daily_return": daily_ret})

    if not rows:
        return dict(n_trading_days=0, total_return=0.0, ann_return=0.0, sharpe=0.0,
                    max_dd=0.0, win_rate=0.0)

    port_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    rets    = port_df["daily_return"].values
    eq      = np.cumprod(1 + rets)
    n_days  = len(rets)
    ann_ret = (eq[-1]) ** (ANN_FACTOR / max(n_days, 1)) - 1
    return dict(
        n_trading_days=n_days,
        total_return=float(eq[-1] - 1),
        ann_return=float(ann_ret),
        sharpe=sharpe(rets),
        max_dd=max_dd(eq),
        win_rate=float((rets > 0).mean()),
        daily_df=port_df,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run on full 100-stock universe")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    symbols = SYMBOLS_100 if args.full else SUBSET
    print(f"\n═══ Exp3 — Winning Config  n_stocks={len(symbols)}  top_k={args.top_k} ═══")

    stock_results: List[Dict] = []
    for i, sym in enumerate(symbols):
        r = run_one(sym, use_regime_gate=True)
        if r is None:
            print(f"  [{i+1:>3}/{len(symbols)}] {sym:<12} SKIP")
            continue
        stock_results.append(r)
        # One-line per stock: show winning config Sharpe
        best_cfg_name, best_cfg_metrics = max(
            r["per_cfg"].items(), key=lambda kv: kv[1].get("sharpe", -999))
        print(f"  [{i+1:>3}/{len(symbols)}] {sym:<12}  "
              f"n_oos={r['n_oos']:>4}  "
              f"best={best_cfg_name:<28} "
              f"sharpe={best_cfg_metrics['sharpe']:>+6.2f} "
              f"n={best_cfg_metrics['n_trades']:>4} "
              f"ret={best_cfg_metrics['total_return']:+.2%}")

    # Aggregate per-config per-stock
    rows = []
    for r in stock_results:
        for cfg_name, m in r["per_cfg"].items():
            rows.append({"symbol": r["symbol"], "strategy": cfg_name, **m})
    out = pd.DataFrame(rows)
    suffix = "full" if args.full else "subset"
    out.to_csv(OUT_DIR / f"exp3_detail_{suffix}.csv", index=False)

    agg = out.groupby("strategy").agg(
        n_stocks=("symbol", "nunique"),
        avg_trades=("n_trades", "mean"),
        avg_sharpe=("sharpe", "mean"),
        med_sharpe=("sharpe", "median"),
        avg_ret=("total_return", "mean"),
        med_ret=("total_return", "median"),
        avg_winrate=("win_rate", "mean"),
        avg_mdd=("max_dd", "mean"),
        n_profitable=("total_return", lambda s: (s > 0).sum()),
        n_pos_sharpe=("sharpe", lambda s: (s > 0).sum()),
    ).round(4).sort_values("avg_sharpe", ascending=False)

    print("\n" + "="*115)
    print(f" EXP3 — PER-STOCK SUMMARY  (n_stocks={len(stock_results)})")
    print("="*115)
    print(agg.to_string())
    agg.to_csv(OUT_DIR / f"exp3_summary_{suffix}.csv")

    # Portfolio-level backtest with multiple top_k values
    print("\n" + "="*115)
    print(" PORTFOLIO-LEVEL BACKTEST  (equal-weight Top-K, primary>=0.58 & meta>=0.60 & not stress)")
    print("="*115)
    port_rows = []
    for k in [3, 5, 8, 10]:
        pm = portfolio_backtest(stock_results, top_k=k)
        port_rows.append({"top_k": k, **{kk: vv for kk, vv in pm.items() if kk != "daily_df"}})
        print(f"  top_k={k:>2}  n_days={pm['n_trading_days']:>4}  "
              f"total_ret={pm['total_return']:>+7.2%}  ann={pm['ann_return']:>+7.2%}  "
              f"Sharpe={pm['sharpe']:>+5.2f}  MaxDD={pm['max_dd']:>6.2%}  "
              f"WinDays={pm['win_rate']:>6.2%}")
    pd.DataFrame(port_rows).to_csv(OUT_DIR / f"exp3_portfolio_{suffix}.csv", index=False)

    # NIFTY comparison for the same window
    try:
        gc_path = RAW_DATA_DIR / "global_cues.parquet"
        gc = pd.read_parquet(gc_path)
        gc["date"] = pd.to_datetime(gc["date"])
        best = portfolio_backtest(stock_results, top_k=args.top_k)
        if best.get("daily_df") is not None and len(best["daily_df"]) > 0:
            d0, d1 = best["daily_df"]["date"].iloc[0], best["daily_df"]["date"].iloc[-1]
            sub = gc[(gc["date"] >= d0) & (gc["date"] <= d1)]
            if "nifty50_close" in sub.columns and len(sub) >= 2:
                nr = float(sub["nifty50_close"].iloc[-1] / sub["nifty50_close"].iloc[0] - 1)
                print(f"\n  NIFTY buy-and-hold {d0} → {d1}: {nr:+.2%}")
                print(f"  Top-{args.top_k} portfolio total : {best['total_return']:+.2%}  "
                      f"(ann {best['ann_return']:+.2%})")
    except Exception as e:
        print(f"  [nifty] comparison skipped: {e}")


if __name__ == "__main__":
    main()
