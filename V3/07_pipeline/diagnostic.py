"""
Quick diagnostic: test if ANY features actually predict 5-day returns on NSE stocks.
Checks:
1. Class balance (what % of 5-day periods are UP?)
2. Best individual feature correlations with target
3. Max achievable accuracy with all features (train=test, perfect overfit check)
4. Signal in regime-filtered subset (only trending days)
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

V3_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(V3_PATH))
sys.path.insert(0, str(V3_PATH / '02_models'))

# ── 1. Load existing features (from latest run) ───────────────────────────────
feat_dir = sorted((V3_PATH / 'data' / 'features').glob('*'))[-1]
print(f"Using features from: {feat_dir.name}\n")

results = []

for csv in sorted(feat_dir.glob('*_features.csv')):
    sym = csv.stem.replace('_features', '')
    df = pd.read_csv(csv)

    if 'target' not in df.columns:
        continue

    df = df.dropna(subset=['target'])
    n = len(df)
    up_rate = df['target'].mean()

    feat_cols = [c for c in df.columns if c not in
                 ['date', 'open', 'high', 'low', 'close', 'volume', 'target']]
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], 0)

    X = df[feat_cols].values.astype(float)
    y = df['target'].values.astype(int)

    # ── 2. Best feature correlations ──────────────────────────────────────────
    corrs = {}
    for i, col in enumerate(feat_cols):
        try:
            c = np.corrcoef(X[:, i], y)[0, 1]
            if not np.isnan(c):
                corrs[col] = abs(c)
        except:
            pass
    top5 = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:5]

    # ── 3. Overfit check: RF on full data (train=test) ────────────────────────
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_sc = np.nan_to_num(X_sc, nan=0.0)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 min_samples_leaf=10, random_state=42)
    rf.fit(X_sc, y)
    train_acc = (rf.predict(X_sc) == y).mean()

    # ── 4. Holdout split accuracy ─────────────────────────────────────────────
    split = int(0.8 * n)
    rf2 = RandomForestClassifier(n_estimators=100, max_depth=5,
                                  min_samples_leaf=10, random_state=42)
    scaler2 = StandardScaler()
    rf2.fit(scaler2.fit_transform(X[:split]), y[:split])
    oos_acc = (rf2.predict(scaler2.transform(X[split:])) == y[split:]).mean()

    # ── 5. Regime-filtered OOS (only trending days) ──────────────────────────
    if 'regime_trending' in df.columns:
        trend_mask = df['regime_trending'].values.astype(bool)
        trend_oos  = trend_mask[split:]
        if trend_oos.sum() > 20:
            X_te_sc = scaler2.transform(X[split:])
            preds_te = rf2.predict(X_te_sc)
            regime_acc = (preds_te[trend_oos] == y[split:][trend_oos]).mean()
            n_trend = trend_oos.sum()
        else:
            regime_acc = 0; n_trend = 0
    else:
        regime_acc = 0; n_trend = 0

    print(f"{'─'*65}")
    print(f"{sym}: n={n}  UP={up_rate:.1%}  tracc={train_acc:.1%}  OOS={oos_acc:.1%}  RegimeOOS={regime_acc:.1%}(n={n_trend})")
    print(f"  Top features: {', '.join(f'{k}:{v:.3f}' for k, v in top5[:3])}")

    results.append({
        'symbol': sym, 'n': n, 'up_rate': up_rate,
        'train_acc': train_acc, 'oos_acc': oos_acc,
        'regime_oos_acc': regime_acc, 'n_trending': n_trend,
    })

print(f"\n{'='*65}")
df_r = pd.DataFrame(results)
print(f"Mean OOS all      : {df_r['oos_acc'].mean():.1%}")
print(f"Mean OOS trending : {df_r[df_r['n_trending']>20]['regime_oos_acc'].mean():.1%}")
print(f"Mean UP-rate      : {df_r['up_rate'].mean():.1%}  ← target class imbalance")
print(f"Mean train-acc    : {df_r['train_acc'].mean():.1%}  ← if >90% = overfitting")
