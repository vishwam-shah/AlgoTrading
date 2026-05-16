"""
extract_feature_importance.py
Loads the production LightGBM models for all symbols and extracts
feature importances. Aggregates across symbols and saves:
  - feature_importance_by_symbol.csv
  - feature_importance_aggregate.csv   (mean gain across all stocks)
  - feature_importance_groups.csv      (grouped by feature category)

Run after a full pipeline run:
  python V3/07_pipeline/extract_feature_importance.py
"""
from __future__ import annotations
import sys
from pathlib import Path

_V3   = Path(__file__).resolve().parent.parent
_ROOT = _V3.parent
for p in [str(_V3), str(_V3 / "02_models"), str(_V3 / "00_config")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import joblib
import numpy as np
import pandas as pd

_PROD   = _V3 / "02_models" / "production"
_OUTDIR = _V3 / "06_results"

# Feature group tags — any feature name containing these substrings belongs to the group.
GROUPS = {
    "Indian Calendar": ["days_to_expiry", "is_expiry", "days_to_rbi", "is_rbi",
                        "days_to_budget", "is_budget", "is_result_season"],
    "Global Macro":    ["sp500", "nasdaq", "vix", "dxy", "crude", "nikkei",
                        "usdinr", "global_", "nifty_ret"],
    "Technical":       ["rsi", "macd", "bb_", "atr", "ema", "sma", "adx",
                        "stoch", "cci", "williams", "obv", "vwap"],
    "Price / Return":  ["ret_", "return", "gap", "close_", "open_", "high_", "low_"],
    "Volume":          ["vol_", "volume", "mfi", "obv"],
    "Sentiment":       ["sentiment", "news", "bullish", "bearish"],
    "Statistical":     ["skew", "kurt", "zscore", "autocorr", "hurst"],
    "Regime":          ["regime", "trend_", "bull", "bear", "sideways"],
}


def _classify(feat: str) -> str:
    fl = feat.lower()
    for group, keywords in GROUPS.items():
        if any(kw in fl for kw in keywords):
            return group
    return "Other"


def run() -> None:
    rows = []
    missing = []

    for sym_dir in sorted(_PROD.iterdir()):
        if not sym_dir.is_dir():
            continue
        sym = sym_dir.name
        lgbm_path = sym_dir / "lightgbm.pkl"
        feat_path  = sym_dir / "feature_names.txt"

        if not lgbm_path.exists():
            missing.append(sym)
            continue

        try:
            model    = joblib.load(lgbm_path)
            pca_path = sym_dir / "pca.pkl"
            meta_path = sym_dir / "metadata.json"

            pca_imp = model.feature_importances_.astype(float)  # shape: (n_components,)

            # Back-project PCA importance to original feature space:
            # importance of original feature j = sum_k |loading[k,j]| * pca_imp[k]
            if pca_path.exists() and meta_path.exists():
                pca = joblib.load(pca_path)
                with open(meta_path) as _mf:
                    meta = __import__("json").load(_mf)
                feat_names = meta.get("feature_names", [])
                n_orig = len(feat_names)
                if n_orig > 0 and hasattr(pca, "components_") and pca.components_.shape[0] == len(pca_imp):
                    # loadings: shape (n_components, n_orig_features)
                    loadings = np.abs(pca.components_[:, :n_orig])
                    orig_imp = (pca_imp[:, None] * loadings).sum(axis=0)
                else:
                    orig_imp  = pca_imp
                    feat_names = [f"pca_{i}" for i in range(len(pca_imp))]
            else:
                orig_imp  = pca_imp
                feat_names = [f"pca_{i}" for i in range(len(pca_imp))]

            for fname, val in zip(feat_names, orig_imp):
                rows.append({
                    "symbol":     sym,
                    "feature":    fname,
                    "importance": float(val),
                    "group":      _classify(fname),
                })
        except Exception as e:
            print(f"  [skip] {sym}: {e}")
            missing.append(sym)

    if not rows:
        print("No production models found. Run full pipeline first.")
        return

    df = pd.DataFrame(rows)
    n_syms = df["symbol"].nunique()
    print(f"Loaded importances for {n_syms} symbols, {df['feature'].nunique()} unique features")

    # Per-symbol CSV
    by_sym = (df.groupby(["symbol", "feature", "group"])["importance"]
                .sum().reset_index()
                .sort_values(["symbol", "importance"], ascending=[True, False]))
    by_sym.to_csv(_OUTDIR / "feature_importance_by_symbol.csv", index=False)
    print(f"  -> feature_importance_by_symbol.csv  ({len(by_sym)} rows)")

    # Aggregate: normalise within symbol then average across symbols
    df["imp_norm"] = df.groupby("symbol")["importance"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else x
    )
    agg = (df.groupby("feature")
             .agg(mean_norm_importance=("imp_norm", "mean"),
                  std_norm_importance=("imp_norm", "std"),
                  n_symbols=("symbol", "nunique"),
                  group=("group", "first"))
             .reset_index()
             .sort_values("mean_norm_importance", ascending=False))
    agg.to_csv(_OUTDIR / "feature_importance_aggregate.csv", index=False)
    print(f"  -> feature_importance_aggregate.csv  (top feature: {agg.iloc[0]['feature']})")
    print()
    print("Top 20 features:")
    for _, r in agg.head(20).iterrows():
        bar = "#" * int(r["mean_norm_importance"] * 500)
        print(f"  {r['feature']:<35} {r['mean_norm_importance']:.4f}  [{r['group']}]  {bar}")

    # Group-level rollup
    grp = (df.groupby("group")["imp_norm"]
             .mean().reset_index()
             .rename(columns={"imp_norm": "mean_norm_importance"})
             .sort_values("mean_norm_importance", ascending=False))
    grp.to_csv(_OUTDIR / "feature_importance_groups.csv", index=False)
    print()
    print("By feature group:")
    for _, r in grp.iterrows():
        bar = "#" * int(r["mean_norm_importance"] * 300)
        print(f"  {r['group']:<20} {r['mean_norm_importance']:.4f}  {bar}")

    if missing:
        print(f"\n  Skipped {len(missing)} symbols (no model): {missing[:10]}")


if __name__ == "__main__":
    run()
