"""
exp8_purged_cv.py — Purged Combinatorial K-fold CV with embargo
================================================================
López de Prado (2018) §7. The expanding walk-forward gives one Sharpe point
estimate. Purged combinatorial K-fold gives N choose k = many non-overlapping
test paths, so we can build a real CI on Sharpe instead of a single bootstrap
on the OOS window.

Procedure:
  1. Split the OOS-eligible time index into K equal contiguous folds.
  2. For each combination of k test folds (we use k=2 → C(K,2) paths):
        - test_idx   = union of those k folds
        - train_idx  = everything else, MINUS:
            * a 'purge' band of HORIZON days around each test fold
              (label leakage: target uses t+5 close, so any train sample
               whose label spans into a test fold is dropped)
            * an 'embargo' of HORIZON days AFTER each test fold
              (serial-correlation leakage)
        - Fit a LightGBM on purged train, score on test.
  3. Stitch the held-out probability for each test fold into a single time
     series, simulate the v2 strategy (hold=10, t1=0.58), record Sharpe.
  4. Report mean ± Sharpe, 95% bootstrap CI of Sharpe across the C(K,2) paths.

Why this matters
  Walk-forward = 1 contiguous test path. Purged CCV = many non-contiguous
  test paths drawn from the same data. If Sharpe holds up across paths,
  the result is not a fluke of one window's market regime.

Universe: 100-stock Nifty100 cache (uses cached scaled features under
  V3/01_data/features/scaled/<symbol>_scaled.parquet). Falls back to
  rebuilding from raw if cache absent.

Outputs:
  V3/08_experiments/results/exp8_purged_cv_summary.csv  per-symbol path stats
  V3/08_experiments/results/exp8_purged_cv_pooled.csv   pooled-path stats
  Stdout: headline Sharpe distribution + CI.

Run:
  python V3/08_experiments/exp8_purged_cv.py                       # all stocks
  python V3/08_experiments/exp8_purged_cv.py --k-folds 6 --k-test 2
  python V3/08_experiments/exp8_purged_cv.py --symbols SBIN HDFCBANK
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_V3 = Path(__file__).resolve().parents[1]
_OUT = Path(__file__).resolve().parent / "results"
_OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(_V3 / "07_pipeline"))

from steps.features import HORIZON_DAYS, feature_cols  # type: ignore  # noqa

ANNUAL = 252
HOLD_DAYS = 10
ENTRY_THR = 0.58
COST = 0.0025


def _load_features(symbol: str) -> Optional[pd.DataFrame]:
    cache = _V3 / "01_data" / "features" / "raw" / f"{symbol}_features.parquet"
    if not cache.exists():
        return None
    df = pd.read_parquet(cache)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _purged_train_idx(n: int, test_idx: np.ndarray, purge: int, embargo: int) -> np.ndarray:
    """All indices outside test_idx, minus purge band on both sides + embargo after."""
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    test_set = set(test_idx.tolist())
    starts = sorted(test_set)
    # Identify contiguous runs of test
    runs = []
    if starts:
        cur_start = cur_end = starts[0]
        for i in starts[1:]:
            if i == cur_end + 1:
                cur_end = i
            else:
                runs.append((cur_start, cur_end))
                cur_start = cur_end = i
        runs.append((cur_start, cur_end))
    for s, e in runs:
        # purge: drop train samples whose target overlaps test range [s, e]
        # target uses [t, t+H]; sample t leaks if t+H >= s, i.e. t >= s - H
        purge_lo = max(0, s - purge)
        mask[purge_lo:s] = False
        # embargo: drop train samples right after the test fold
        emb_hi = min(n, e + 1 + embargo)
        mask[e + 1:emb_hi] = False
    return np.flatnonzero(mask)


def _fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    """Quick LightGBM primary — the focus is on path-level Sharpe, not model tuning."""
    from lightgbm import LGBMClassifier, early_stopping as _es, log_evaluation as _le
    # Use last 15% of train as internal val
    n = len(X_tr); cut = int(n * 0.85)
    X_t, X_v = X_tr[:cut], X_tr[cut:]
    y_t, y_v = y_tr[:cut], y_tr[cut:]
    if len(np.unique(y_v)) < 2:
        cut = int(n * 0.90); X_t, X_v, y_t, y_v = X_tr[:cut], X_tr[cut:], y_tr[:cut], y_tr[cut:]
    if len(np.unique(y_v)) < 2 or len(np.unique(y_t)) < 2:
        return np.full(len(X_te), 0.5)
    m = LGBMClassifier(n_estimators=400, max_depth=5, learning_rate=0.03, num_leaves=31,
                       subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=1.5,
                       min_child_samples=20, is_unbalance=True, random_state=42, n_jobs=-1, verbosity=-1)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        m.fit(X_t, y_t, eval_set=[(X_v, y_v)], callbacks=[_es(30, verbose=False), _le(period=-1)])
    return m.predict_proba(X_te)[:, 1]


def _path_sharpe(df: pd.DataFrame, prob: np.ndarray) -> Dict:
    """v2 strategy on a single test path: hold=10, t1=0.58, no overlapping per-stock."""
    if len(df) < HOLD_DAYS + 5:
        return {}
    closes = df["close"].values if "close" in df.columns else df.get("close_price", pd.Series(np.nan)).values
    n = len(prob)
    chosen = []
    last_exit = -1
    for i in range(n - HOLD_DAYS):
        if prob[i] < ENTRY_THR or i < last_exit:
            continue
        chosen.append(i); last_exit = i + HOLD_DAYS
    if len(chosen) < 5:
        return {}
    returns = np.array([closes[i + HOLD_DAYS] / closes[i] - 1 - COST for i in chosen])
    sharpe = float(returns.mean() / returns.std() * np.sqrt(ANNUAL / HOLD_DAYS)) if returns.std() > 0 else 0.0
    return {"n_trades": len(chosen),
            "total_ret": float(np.prod(1 + returns) - 1),
            "win_rate": float((returns > 0).mean()),
            "sharpe": round(sharpe, 3)}


def run_for_symbol(symbol: str, K: int, k_test: int) -> List[Dict]:
    feat = _load_features(symbol)
    if feat is None or "target" not in feat.columns or "next_ret" not in feat.columns:
        return []
    feat = feat.dropna(subset=["target"]).reset_index(drop=True)
    if len(feat) < 600:
        return []
    fcols = feature_cols(feat)
    if not fcols:
        return []

    X = feat[fcols].values.astype(float)
    y = feat["target"].values.astype(int)
    n = len(feat)

    # K equal-sized contiguous folds
    fold_size = n // K
    folds = [np.arange(i * fold_size, (i + 1) * fold_size if i < K - 1 else n) for i in range(K)]

    results = []
    purge = HORIZON_DAYS
    embargo = HORIZON_DAYS
    for combo in combinations(range(K), k_test):
        test_idx = np.concatenate([folds[i] for i in combo])
        train_idx = _purged_train_idx(n, test_idx, purge, embargo)
        if len(train_idx) < 200 or len(test_idx) < 30:
            continue
        if len(np.unique(y[train_idx])) < 2:
            continue
        try:
            p_te = _fit_predict(X[train_idx], y[train_idx], X[test_idx])
            df_te = feat.iloc[test_idx].reset_index(drop=True)
            stats = _path_sharpe(df_te, p_te)
            if stats:
                results.append({"symbol": symbol, "fold_combo": "+".join(str(c) for c in combo),
                                "n_train": len(train_idx), "n_test": len(test_idx), **stats})
        except Exception as e:
            print(f"    [{symbol} {combo}] {e}")
    return results


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95):
    if len(values) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    a = (1 - ci) / 2
    return float(np.percentile(boots, a * 100)), float(np.percentile(boots, (1 - a) * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--k-folds", type=int, default=6, help="K (number of contiguous folds)")
    ap.add_argument("--k-test",  type=int, default=2, help="k (test folds per combo); paths = C(K,k)")
    ap.add_argument("--max-symbols", type=int, default=None)
    args = ap.parse_args()

    if args.symbols:
        syms = args.symbols
    else:
        cache = (_V3 / "01_data" / "features" / "raw")
        syms = sorted([p.stem.replace("_features", "") for p in cache.glob("*_features.parquet")])
    if args.max_symbols:
        syms = syms[: args.max_symbols]

    print(f"  Purged Combinatorial K-fold CV  K={args.k_folds}  k_test={args.k_test}  paths/symbol={len(list(combinations(range(args.k_folds), args.k_test)))}")
    print(f"  Symbols: {len(syms)}\n")

    all_rows: List[Dict] = []
    for i, sym in enumerate(syms):
        rows = run_for_symbol(sym, args.k_folds, args.k_test)
        all_rows.extend(rows)
        if rows:
            sharpes = [r["sharpe"] for r in rows]
            print(f"  [{i+1:3d}/{len(syms)}] {sym:<13} paths={len(rows):>2}  "
                  f"sharpe mean={np.mean(sharpes):+.2f} std={np.std(sharpes):.2f} "
                  f"min={min(sharpes):+.2f} max={max(sharpes):+.2f}")

    if not all_rows:
        print("  No results — check feature cache."); return

    df = pd.DataFrame(all_rows)
    out1 = _OUT / "exp8_purged_cv_summary.csv"
    df.to_csv(out1, index=False)

    # Pooled per-symbol
    pooled = df.groupby("symbol").agg(
        n_paths=("sharpe", "size"),
        sharpe_mean=("sharpe", "mean"),
        sharpe_std=("sharpe", "std"),
        sharpe_min=("sharpe", "min"),
        sharpe_max=("sharpe", "max"),
        ret_mean=("total_ret", "mean"),
        win_rate_mean=("win_rate", "mean"),
        n_trades_mean=("n_trades", "mean"),
    ).reset_index().round(3)
    out2 = _OUT / "exp8_purged_cv_pooled.csv"
    pooled.to_csv(out2, index=False)

    # Universe-level: Sharpe distribution across all paths × stocks
    all_sharpes = df["sharpe"].values
    ci_lo, ci_hi = _bootstrap_ci(all_sharpes)
    print(f"\n  ── Universe-level Sharpe distribution ({len(all_sharpes)} paths × stocks) ──")
    print(f"    Mean Sharpe: {np.mean(all_sharpes):+.3f}")
    print(f"    Median     : {np.median(all_sharpes):+.3f}")
    print(f"    Std        : {np.std(all_sharpes):.3f}")
    print(f"    Bootstrap 95% CI of mean: [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"    % paths with Sharpe > 0: {(all_sharpes > 0).mean():.1%}")
    print(f"    % paths with Sharpe > 1: {(all_sharpes > 1).mean():.1%}")
    print(f"\n  → {out1.name}")
    print(f"  → {out2.name}")


if __name__ == "__main__":
    main()
