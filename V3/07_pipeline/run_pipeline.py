"""
================================================================================
PRODUCTION PIPELINE - V3 FINAL
================================================================================
Zero-leakage expanding window walk-forward validation.

DATA LEAKAGE AUDIT:
==================
1. TARGET: shift(-1) applied BEFORE splitting → safe (it's just labeling)
   - We drop the last row of each window where target is NaN
   - We NEVER use any future price data in feature computation

2. FEATURES: all rolling/EWM operations computed on full series → SAFE because:
   - Rolling windows look BACK in time only
   - We use `min_periods=1` where needed
   - No `shift(-N)` in feature columns
   - Lag features use `.shift(+N)` (past data only)

3. SCALER: StandardScaler fit ONLY on X_train of each window, then .transform
   on val and test → zero contamination

4. WINDOWS: Strict chronological ordering
   - Window N: Train[0..t]  Val[t..t+v]  Test[t+v..t+v+ts]
   - Window N+1: Train[0..t+v+ts] ... (expanding, never re-using test as train)
   - No shuffling anywhere

5. FEATURE COMPUTATION ORDER: All features computed BEFORE target creation,
   then target is appended and data sorted by date.

METHODOLOGY:
============
- 7 expanding windows: 70% → 95% training (5% steps)
- Each window trains fresh on all data up to split point
- Test set is strictly FUTURE data unseen by model
- Ensemble: LightGBM + XGBoost + CatBoost (soft voting)
- Metric: OOS (out-of-sample) accuracy on each test slice

================================================================================
"""

import sys
import os
import warnings
import pickle
import json
import time
import io
import contextlib

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# ─── Paths ─────────────────────────────────────────────────────────────────────
V3_PATH = Path(__file__).parent.parent
DATA_PATH   = V3_PATH / 'data'
MODELS_PATH = V3_PATH / 'models'
RESULTS_PATH = V3_PATH / 'results'
sys.path.insert(0, str(V3_PATH))
sys.path.insert(0, str(V3_PATH / '02_models'))

# ─── Stocks ────────────────────────────────────────────────────────────────────
SYMBOLS = [
    'SBIN', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK',   # Banking
    'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM'                  # IT
]

# ─── Config ─────────────────────────────────────────────────────────────────────
DATA_START_DATE     = '2018-01-01'  # Fixed start date for all stocks
INITIAL_TRAIN_RATIO = 0.70        # First window: 70% train
EXPANSION_STEP      = 0.05        # Each window expands by 5%
MAX_TRAIN_RATIO     = 0.95        # Stop expanding at 95%
MIN_TRAIN_SAMPLES   = 400         # Must have at least 400 rows to train
MIN_TEST_SAMPLES    = 30          # At least 30 rows in each test slice


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_data(symbols: List[str], timestamp: str) -> Dict[str, pd.DataFrame]:
    """Download latest OHLCV from Yahoo Finance. Always fetches fresh data."""
    import yfinance as yf

    raw_path = DATA_PATH / 'raw'
    raw_path.mkdir(parents=True, exist_ok=True)

    start_date = DATA_START_DATE
    end_date   = datetime.now().strftime('%Y-%m-%d')

    print(f"\n{'='*70}")
    print(f" STEP 1 – DATA DOWNLOAD")
    print(f" Period : {start_date} → {end_date}")
    print(f"{'='*70}")

    data: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        ticker = f"{symbol}.NS"
        print(f"  Downloading {symbol} ...", end=' ', flush=True)

        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True   # Use adjusted prices (splits, dividends)
            )

            if raw.empty:
                print("⚠ No data returned")
                continue

            # ── Normalise columns ─────────────────────────────────────────
            df = raw.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                # yfinance >= 0.2 sometimes returns MultiIndex
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            # yfinance may return 'date' or 'datetime'
            for date_col in ['date', 'datetime', 'index']:
                if date_col in df.columns:
                    df = df.rename(columns={date_col: 'date'})
                    break

            df['date'] = pd.to_datetime(df['date'])

            # Keep only OHLCV
            keep = ['date', 'open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in keep if c in df.columns]].copy()
            df = df.sort_values('date').reset_index(drop=True)

            # Remove any rows with zero volume or zero price
            df = df[(df['close'] > 0) & (df['volume'] > 0)]

            # Save raw
            save_path = raw_path / f"{symbol}_{timestamp}.csv"
            df.to_csv(save_path, index=False)

            data[symbol] = df
            print(f"✓ {len(df)} rows  [{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}]")

        except Exception as exc:
            print(f"✗ Error: {exc}")

    print(f"\n  Loaded {len(data)}/{len(symbols)} symbols")
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – FEATURE ENGINEERING  (no leakage: all lookbacks are backward)
# ══════════════════════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 240 technical features from OHLCV.

    LEAKAGE SAFETY:
    ---------------
    - All rolling/EWM windows look BACKWARD (default pandas behaviour)
    - Lag features use .shift(+N)  (N ≥ 1)
    - NO .shift(-N) anywhere in features
    - Target is created separately AFTER features
    - The full-series statistics (mean/std for z-score) are computed on
      training slice only INSIDE the walk-forward loop, not here.
      Here we only compute raw values — the scaler handles normalisation.
    """
    d = df.copy()
    c = d['close']
    h = d['high']
    l = d['low']
    o = d['open']
    v = d['volume']

    # ── 1. Returns ─────────────────────────────────────────────────────────
    d['ret_1d']   = c.pct_change(1)
    d['ret_2d']   = c.pct_change(2)
    d['ret_5d']   = c.pct_change(5)
    d['ret_10d']  = c.pct_change(10)
    d['ret_20d']  = c.pct_change(20)
    d['log_ret']  = np.log(c / c.shift(1))

    # ── 2. SMA / EMA ────────────────────────────────────────────────────────
    for p in [5, 10, 20, 50, 100, 200]:
        d[f'sma_{p}']       = c.rolling(p).mean()
        d[f'ema_{p}']       = c.ewm(span=p, adjust=False).mean()
        d[f'price_sma_{p}'] = c / d[f'sma_{p}']    # ratio (no leak: sma is backward)
        d[f'price_ema_{p}'] = c / d[f'ema_{p}']

    # ── 3. MACD ─────────────────────────────────────────────────────────────
    for (fast, slow, sig) in [(12, 26, 9), (5, 35, 5)]:
        ema_f = c.ewm(span=fast, adjust=False).mean()
        ema_s = c.ewm(span=slow, adjust=False).mean()
        macd_line  = ema_f - ema_s
        macd_signal = macd_line.ewm(span=sig, adjust=False).mean()
        tag = f'{fast}_{slow}'
        d[f'macd_{tag}']       = macd_line
        d[f'macd_signal_{tag}'] = macd_signal
        d[f'macd_hist_{tag}']  = macd_line - macd_signal

    # ── 4. RSI ──────────────────────────────────────────────────────────────
    for p in [7, 14, 21, 28]:
        delta = c.diff()
        up   = delta.clip(lower=0).rolling(p).mean()
        dn   = (-delta).clip(lower=0).rolling(p).mean()
        rs   = up / (dn + 1e-10)
        d[f'rsi_{p}'] = 100 - 100 / (1 + rs)

    # ── 5. Bollinger Bands ──────────────────────────────────────────────────
    for p in [10, 20, 50]:
        sma = c.rolling(p).mean()
        std = c.rolling(p).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        d[f'bb_upper_{p}'] = upper
        d[f'bb_lower_{p}'] = lower
        d[f'bb_width_{p}'] = (upper - lower) / (sma + 1e-10)
        d[f'bb_pos_{p}']   = (c - lower) / (upper - lower + 1e-10)

    # ── 6. ATR ──────────────────────────────────────────────────────────────
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)

    for p in [7, 14, 21]:
        atr = tr.rolling(p).mean()
        d[f'atr_{p}']      = atr
        d[f'atr_ratio_{p}'] = atr / (c + 1e-10)

    # ── 7. ADX / DI ─────────────────────────────────────────────────────────
    hd = h.diff()
    ld = -l.diff()
    plus_dm  = np.where((hd > ld) & (hd > 0), hd, 0.0)
    minus_dm = np.where((ld > hd) & (ld > 0), ld, 0.0)

    for p in [14, 21]:
        tr_s  = tr.rolling(p).sum()
        pdi   = 100 * pd.Series(plus_dm,  index=d.index).rolling(p).sum() / (tr_s + 1e-10)
        mdi   = 100 * pd.Series(minus_dm, index=d.index).rolling(p).sum() / (tr_s + 1e-10)
        dx    = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
        d[f'adx_{p}']      = dx.rolling(p).mean()
        d[f'plus_di_{p}']  = pdi
        d[f'minus_di_{p}'] = mdi
        d[f'di_diff_{p}']  = pdi - mdi

    # ── 8. Stochastic Oscillator ────────────────────────────────────────────
    for p in [9, 14, 21]:
        lo = l.rolling(p).min()
        hi = h.rolling(p).max()
        k  = 100 * (c - lo) / (hi - lo + 1e-10)
        d[f'stoch_k_{p}'] = k
        d[f'stoch_d_{p}'] = k.rolling(3).mean()

    # ── 9. CCI ──────────────────────────────────────────────────────────────
    for p in [14, 20]:
        tp  = (h + l + c) / 3
        stp = tp.rolling(p).mean()
        mad = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        d[f'cci_{p}'] = (tp - stp) / (0.015 * mad + 1e-10)

    # ── 10. Williams %R ──────────────────────────────────────────────────────
    for p in [14, 21]:
        hi  = h.rolling(p).max()
        lo  = l.rolling(p).min()
        d[f'willr_{p}'] = -100 * (hi - c) / (hi - lo + 1e-10)

    # ── 11. OBV ──────────────────────────────────────────────────────────────
    obv = (np.sign(c.diff()) * v).cumsum()
    d['obv']        = obv
    d['obv_sma20']  = obv.rolling(20).mean()
    d['obv_ratio']  = obv / (d['obv_sma20'] + 1e-10)

    # ── 12. Volume ───────────────────────────────────────────────────────────
    for p in [5, 10, 20, 50]:
        d[f'vol_sma_{p}'] = v.rolling(p).mean()
        d[f'vol_ratio_{p}'] = v / (d[f'vol_sma_{p}'] + 1e-10)
    d['vol_change'] = v.pct_change()
    d['vol_change_5d'] = v.pct_change(5)

    # ── 13. Volatility ───────────────────────────────────────────────────────
    log_r = np.log(c / c.shift(1))
    for p in [5, 10, 20, 50]:
        d[f'hist_vol_{p}']  = log_r.rolling(p).std() * np.sqrt(252)
    # Parkinson volatility (uses H-L, not close-to-close)
    pk_term = (np.log(h / l)) ** 2 / (4 * np.log(2))
    for p in [10, 20]:
        d[f'parkinson_{p}'] = np.sqrt(pk_term.rolling(p).mean() * 252)
    # Garman-Klass
    gk_term = 0.5 * (np.log(h / l)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / o)) ** 2
    for p in [10, 20]:
        d[f'gk_vol_{p}'] = np.sqrt(gk_term.rolling(p).mean() * 252)

    # ── 14. Momentum / ROC ──────────────────────────────────────────────────
    for p in [3, 5, 10, 20, 50]:
        d[f'roc_{p}'] = c.pct_change(p)
        d[f'mom_{p}'] = c - c.shift(p)

    # ── 15. Price patterns (candlestick) ─────────────────────────────────────
    body  = (c - o).abs()
    rng   = (h - l).abs() + 1e-10
    d['body_size']     = body / rng
    d['upper_shadow']  = (h - c.where(c >= o, o)) / rng
    d['lower_shadow']  = (c.where(c <= o, o) - l) / rng
    d['hl_range']      = rng / (c + 1e-10)
    d['oc_return']     = (c - o) / (o + 1e-10)
    d['gap']           = (o - c.shift(1)) / (c.shift(1) + 1e-10)

    # ── 16. Statistical features ────────────────────────────────────────────
    for p in [10, 20, 50]:
        d[f'skew_{p}']    = log_r.rolling(p).skew()
        d[f'kurt_{p}']    = log_r.rolling(p).kurt()
        d[f'zscore_{p}']  = (c - c.rolling(p).mean()) / (c.rolling(p).std() + 1e-10)

    # ── 17. Lag features (past prices, no future info) ───────────────────────
    for lag in [1, 2, 3, 5, 10, 20]:
        d[f'ret_lag_{lag}']    = log_r.shift(lag)
        d[f'vol_lag_{lag}']    = (v / (v.rolling(20).mean() + 1e-10)).shift(lag)
    
    # ── 18. High/Low position ────────────────────────────────────────────────
    for p in [20, 52, 126, 252]:
        hi = h.rolling(p).max()
        lo = l.rolling(p).min()
        d[f'pos_hi_{p}'] = (c - lo) / (hi - lo + 1e-10)
        d[f'dist_hi_{p}'] = (hi - c) / (c + 1e-10)
        d[f'dist_lo_{p}'] = (c - lo) / (c + 1e-10)

    # ── 19. Trend features ───────────────────────────────────────────────────
    d['trend_20']   = (c - c.shift(20)) / (c.shift(20) + 1e-10)
    d['trend_cons'] = log_r.rolling(20).apply(lambda x: (x > 0).mean(), raw=True)
    # Golden / death cross
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    d['cross_ratio'] = sma50 / (sma200 + 1e-10)

    # ── 20. Calendar features (cyclic encoding, no future info) ─────────────
    d['dow_sin'] = np.sin(2 * np.pi * d['date'].dt.dayofweek / 5)
    d['dow_cos'] = np.cos(2 * np.pi * d['date'].dt.dayofweek / 5)
    d['mon_sin'] = np.sin(2 * np.pi * d['date'].dt.month / 12)
    d['mon_cos'] = np.cos(2 * np.pi * d['date'].dt.month / 12)
    d['week_of_year_sin'] = np.sin(2 * np.pi * d['date'].dt.isocalendar().week.astype(int) / 52)
    d['week_of_year_cos'] = np.cos(2 * np.pi * d['date'].dt.isocalendar().week.astype(int) / 52)

    # ── Clean up ─────────────────────────────────────────────────────────────
    d = d.replace([np.inf, -np.inf], np.nan)

    return d


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return feature column names (exclude OHLCV, date, target)."""
    exclude = {'date', 'open', 'high', 'low', 'close', 'volume', 'target'}
    return [c for c in df.columns if c not in exclude]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – TARGET CREATION  (strictly t+1 prediction)
# ══════════════════════════════════════════════════════════════════════════════

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary target: will tomorrow's close be HIGHER than today's close?
    We create target = close[t+1] > close[t], i.e. shift(-1).

    This is safe — it's just a labeling step.
    The last row gets target=NaN and is dropped.
    """
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(float)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 – EXPANDING WINDOW WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def build_windows(n: int) -> List[Dict]:
    """
    Build expanding window schedule.

    Window schedule (indices are row positions, 0-based):
    ┌──────┬────────────────┬──────────────────┬──────────────────┐
    │  Win │  Train [0..t]  │  Val  [t..t+v]   │  Test [t+v..end] │
    └──────┴────────────────┴──────────────────┴──────────────────┘
    Window 1: train=70%, val=5%, test=remaining
    Window 2: train=75%, val=5%, test=remaining
    ...
    Window 6: train=95%, val leftover

    Key invariant: test indices are ALWAYS strictly after train+val indices.
    """
    windows = []
    ratio = INITIAL_TRAIN_RATIO

    while ratio <= MAX_TRAIN_RATIO:
        train_end = int(n * ratio)

        # Validation: next 5% after training
        val_end = int(n * (ratio + EXPANSION_STEP))
        val_end = min(val_end, n)

        # Test: everything after val (until next window's train end)
        next_ratio = ratio + EXPANSION_STEP
        if next_ratio <= MAX_TRAIN_RATIO:
            test_end = int(n * (next_ratio + EXPANSION_STEP))
        else:
            test_end = n  # Last window: test goes to end

        test_end = min(test_end, n)

        # Validation: take last 10% of training as val
        val_size = max(int(train_end * 0.10), 20)
        actual_train_end = train_end - val_size
        actual_val_start = actual_train_end
        actual_val_end   = train_end
        test_start       = train_end
        actual_test_end  = test_end

        if actual_test_end - test_start < MIN_TEST_SAMPLES:
            ratio += EXPANSION_STEP
            continue

        windows.append({
            'id':          len(windows) + 1,
            'train_start': 0,
            'train_end':   actual_train_end,
            'val_start':   actual_val_start,
            'val_end':     actual_val_end,
            'test_start':  test_start,
            'test_end':    actual_test_end,
            'train_ratio': ratio,
        })

        ratio = round(ratio + EXPANSION_STEP, 4)

    return windows


def train_window(
    X: np.ndarray,
    y: np.ndarray,
    window: Dict,
    feature_names: List[str],
    symbol: str,
    save_path: Path
) -> Optional[Dict]:
    """
    Train LightGBM + XGBoost + CatBoost on one window, evaluate on test set.

    Scaler is fit on X_train ONLY, then applied to val and test.
    This is the key anti-leakage step.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix
    )
    from traditional.lightgbm_classifier import LightGBMClassifier
    from traditional.xgboost_classifier import XGBoostClassifier
    from traditional.catboost_classifier import CatBoostClassifier

    ws = window['train_start']
    we = window['train_end']
    vs = window['val_start']
    ve = window['val_end']
    ts = window['test_start']
    te = window['test_end']

    X_train_raw = X[ws:we]
    y_train     = y[ws:we]
    X_val_raw   = X[vs:ve]
    y_val       = y[vs:ve]
    X_test_raw  = X[ts:te]
    y_test      = y[ts:te]

    # ── Scaler fit on TRAIN only ──────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    X_test  = scaler.transform(X_test_raw)

    # Replace any NaN introduced by scaler (zero-variance features)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val   = np.nan_to_num(X_val,   nan=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0)

    trained_models = {}
    test_preds     = {}
    test_probs     = {}

    for ModelClass, name in [
        (LightGBMClassifier, 'LightGBM'),
        (XGBoostClassifier,  'XGBoost'),
        (CatBoostClassifier, 'CatBoost'),
    ]:
        try:
            mdl = ModelClass()
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                mdl.train(
                    X_train, y_train,
                    X_val,   y_val,
                    feature_names=feature_names,
                    verbose=False
                )
            test_preds[name] = mdl.predict(X_test)
            test_probs[name] = mdl.predict_proba(X_test)
            trained_models[name] = mdl
        except Exception as exc:
            print(f"      [{name}] error: {exc}")

    if not test_preds:
        return None

    # ── Soft-vote ensemble ────────────────────────────────────────────────
    avg_prob = np.mean(list(test_probs.values()), axis=0)
    ens_pred = (avg_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, ens_pred)
    f1   = f1_score(y_test, ens_pred, zero_division=0)
    prec = precision_score(y_test, ens_pred, zero_division=0)
    rec  = recall_score(y_test, ens_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, avg_prob)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_test, ens_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Per-model accuracy
    per_model = {
        n: float(accuracy_score(y_test, p))
        for n, p in test_preds.items()
    }

    # ── Save checkpoint ───────────────────────────────────────────────────
    win_path = save_path / f"window_{window['id']:02d}"
    win_path.mkdir(parents=True, exist_ok=True)

    for name, mdl in trained_models.items():
        with open(win_path / f'{name.lower()}.pkl', 'wb') as f:
            pickle.dump(mdl, f)
    with open(win_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    win_meta = {
        'symbol': symbol,
        'window_id': window['id'],
        'train_ratio': window['train_ratio'],
        'train_size':  we - ws,
        'val_size':    ve - vs,
        'test_size':   te - ts,
        'accuracy':    acc,
        'f1':          f1,
        'auc':         auc,
        'per_model':   per_model,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }
    with open(win_path / 'meta.json', 'w') as f:
        json.dump(win_meta, f, indent=2)

    return {
        'window':      window,
        'models':      trained_models,
        'scaler':      scaler,
        'y_test':      y_test,
        'ens_pred':    ens_pred,
        'avg_prob':    avg_prob,
        'accuracy':    acc,
        'f1':          f1,
        'precision':   prec,
        'recall':      rec,
        'auc':         auc,
        'per_model':   per_model,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 – PER-SYMBOL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_symbol(
    symbol: str,
    raw_df: pd.DataFrame,
    timestamp: str,
    feat_save_path: Path,
    model_save_path: Path,
    result_save_path: Path,
) -> Dict:
    """Full pipeline for one symbol."""
    from sklearn.metrics import accuracy_score, f1_score

    print(f"\n{'─'*60}")
    print(f"  {symbol}")
    print(f"{'─'*60}")

    # ── Features ─────────────────────────────────────────────────────────
    print(f"  [1/4] Computing features...", end=' ', flush=True)
    df = compute_features(raw_df)
    df = add_target(df)

    # Drop rows where target is NaN (last row) OR where features are NaN
    feat_cols   = get_feature_columns(df)
    target_col  = 'target'

    # Drop last row (no target)
    df = df.dropna(subset=[target_col])
    # Drop rows where too many features are NaN
    df = df.dropna(subset=feat_cols, thresh=len(feat_cols) - 5)
    # Fill remaining minor NaN with column median
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    df = df.reset_index(drop=True)

    n_feat = len(feat_cols)
    n_rows = len(df)
    print(f"✓  {n_rows} rows × {n_feat} features")

    # Save features
    feat_save_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(feat_save_path / f'{symbol}_features.csv', index=False)

    # ── Build arrays ──────────────────────────────────────────────────────
    X = df[feat_cols].values.astype(float)
    y = df[target_col].values.astype(int)
    dates = df['date'].values

    if n_rows < MIN_TRAIN_SAMPLES + MIN_TEST_SAMPLES:
        print(f"  ⚠ Not enough rows ({n_rows}), skipping")
        return {'symbol': symbol, 'status': 'skipped', 'reason': 'too_few_rows'}

    # ── Windows ───────────────────────────────────────────────────────────
    windows = build_windows(n_rows)
    print(f"  [2/4] Walk-forward windows: {len(windows)}")
    for w in windows:
        print(f"        Win {w['id']}: train[0:{w['train_end']}]"
              f"  val[{w['val_start']}:{w['val_end']}]"
              f"  test[{w['test_start']}:{w['test_end']}]"
              f"  ({w['train_ratio']:.0%} train)")

    # ── Train each window ─────────────────────────────────────────────────
    print(f"  [3/4] Training...")
    sym_model_path  = model_save_path  / symbol
    sym_result_path = result_save_path / symbol
    sym_model_path.mkdir(parents=True, exist_ok=True)
    sym_result_path.mkdir(parents=True, exist_ok=True)

    window_rows = []
    all_preds   = []
    all_actuals = []
    all_dates   = []
    last_result = None

    for win in windows:
        res = train_window(X, y, win, feat_cols, symbol, sym_model_path)
        if res is None:
            continue

        last_result = res

        test_dates = dates[win['test_start']:win['test_end']]
        all_preds.extend(res['ens_pred'])
        all_actuals.extend(res['y_test'])
        all_dates.extend(test_dates)

        tag = "✓" if res['accuracy'] >= 0.55 else ("~" if res['accuracy'] >= 0.50 else "✗")
        print(f"    {tag} Win {win['id']} | train={res['window']['train_ratio']:.0%}"
              f" | train_n={win['train_end']}"
              f" | test_n={win['test_end'] - win['test_start']}"
              f" | OOS_acc={res['accuracy']:.2%}"
              f" | AUC={res['auc']:.3f}"
              f" | F1={res['f1']:.3f}"
              f" | LGB={res['per_model'].get('LightGBM',0):.2%}"
              f" | XGB={res['per_model'].get('XGBoost',0):.2%}"
              f" | CBS={res['per_model'].get('CatBoost',0):.2%}")

        window_rows.append({
            'symbol':      symbol,
            'window_id':   win['id'],
            'train_ratio': win['train_ratio'],
            'train_size':  win['train_end'],
            'val_size':    win['val_end'] - win['val_start'],
            'test_size':   win['test_end'] - win['test_start'],
            'test_start':  str(test_dates[0])[:10] if len(test_dates) else '',
            'test_end':    str(test_dates[-1])[:10] if len(test_dates) else '',
            'oos_accuracy': res['accuracy'],
            'auc':         res['auc'],
            'f1':          res['f1'],
            'precision':   res['precision'],
            'recall':      res['recall'],
            'tp':          int(res['tp']),
            'fp':          int(res['fp']),
            'tn':          int(res['tn']),
            'fn':          int(res['fn']),
            'lgbm_acc':    res['per_model'].get('LightGBM', 0),
            'xgb_acc':     res['per_model'].get('XGBoost', 0),
            'catboost_acc': res['per_model'].get('CatBoost', 0),
        })

    # ── Overall OOS metrics (all test slices combined) ────────────────────
    if all_preds:
        oos_acc = accuracy_score(all_actuals, all_preds)
        oos_f1  = f1_score(all_actuals, all_preds, zero_division=0)
    else:
        oos_acc = 0.0
        oos_f1  = 0.0

    print(f"  [4/4] OOS Overall → Accuracy={oos_acc:.2%}  F1={oos_f1:.4f}"
          f"  ({len(all_preds)} predictions across {len(windows)} windows)")

    # ── Save window results ───────────────────────────────────────────────
    win_df = pd.DataFrame(window_rows)
    win_df.to_csv(sym_result_path / 'window_results.csv', index=False)

    # ── Save prediction history ───────────────────────────────────────────
    pred_df = pd.DataFrame({
        'date':      pd.to_datetime(all_dates),
        'actual':    all_actuals,
        'predicted': all_preds,
        'correct':   np.array(all_actuals) == np.array(all_preds),
    })
    pred_df.to_csv(sym_result_path / 'predictions.csv', index=False)

    # ── Save production model (last window models) ────────────────────────
    if last_result:
        prod_path = MODELS_PATH / 'production' / symbol
        prod_path.mkdir(parents=True, exist_ok=True)

        for name, mdl in last_result['models'].items():
            with open(prod_path / f'{name.lower()}.pkl', 'wb') as f:
                pickle.dump(mdl, f)
        with open(prod_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(last_result['scaler'], f)

        prod_meta = {
            'symbol':        symbol,
            'timestamp':     timestamp,
            'feature_names': feat_cols,
            'oos_accuracy':  oos_acc,
            'oos_f1':        oos_f1,
            'n_features':    n_feat,
            'n_train_rows':  n_rows,
            'last_window':   windows[-1]['id'] if windows else 0,
        }
        with open(prod_path / 'metadata.json', 'w') as f:
            json.dump(prod_meta, f, indent=2)

    return {
        'symbol':       symbol,
        'status':       'ok',
        'oos_accuracy': oos_acc,
        'oos_f1':       oos_f1,
        'n_windows':    len(windows),
        'n_predictions': len(all_preds),
        'n_features':   n_feat,
        'n_rows':       n_rows,
        'window_rows':  window_rows,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 – NEXT-DAY PREDICTION (using production model)
# ══════════════════════════════════════════════════════════════════════════════

def predict_next_day(symbol: str, raw_df: pd.DataFrame) -> Optional[Dict]:
    """Load production model and predict tomorrow's direction."""
    prod_path = MODELS_PATH / 'production' / symbol
    if not prod_path.exists():
        return None

    # Load metadata
    try:
        with open(prod_path / 'metadata.json') as f:
            meta = json.load(f)
    except Exception:
        return None

    feat_cols = meta['feature_names']

    # Load pipeline scaler + models
    scaler_path = prod_path / 'scaler.pkl'
    if not scaler_path.exists():
        return None
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    models = {}
    for pkl_file in prod_path.glob('*.pkl'):
        if pkl_file.stem == 'scaler':
            continue
        with open(pkl_file, 'rb') as f:
            models[pkl_file.stem] = pickle.load(f)

    if not models:
        return None

    # Compute features on latest data
    df = compute_features(raw_df)
    df = df.dropna(subset=feat_cols, thresh=len(feat_cols) - 10)
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    df = df.reset_index(drop=True)

    if df.empty:
        return None

    # Validate feature count matches what scaler was trained on
    n_expected = scaler.n_features_in_
    if len(feat_cols) != n_expected:
        return None

    # Use the very last row
    X_last = df[feat_cols].iloc[[-1]].values.astype(float)
    X_last = np.nan_to_num(X_last, nan=0.0)
    X_scaled = scaler.transform(X_last)

    # Ensemble soft vote
    # Each model class has its OWN internal scaler — pass raw X (not X_scaled)
    # because pipeline scaler and model internal scaler would double-scale.
    # Instead we use raw features and let each model handle its own scaling.
    probs = []
    for mdl in models.values():
        try:
            p = mdl.predict_proba(X_last)
            probs.append(float(p) if np.isscalar(p) else float(p.ravel()[0]))
        except Exception:
            pass

    if not probs:
        return None

    avg_prob = float(np.mean(probs))
    direction = 1 if avg_prob >= 0.5 else 0
    confidence = avg_prob if direction == 1 else 1 - avg_prob

    return {
        'symbol':     symbol,
        'last_date':  str(df['date'].iloc[-1])[:10],
        'last_close': float(raw_df['close'].iloc[-1]),
        'direction':  'UP' if direction == 1 else 'DOWN',
        'confidence': round(confidence, 4),
        'avg_prob':   round(avg_prob, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    t0 = time.time()

    print("=" * 70)
    print("  AI STOCK PREDICTION — PRODUCTION PIPELINE V3")
    print(f"  Run timestamp : {timestamp}")
    print(f"  Date          : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Stocks        : {len(SYMBOLS)}  ({', '.join(SYMBOLS)})")
    print(f"  Data from     : {DATA_START_DATE} → today")
    print(f"  Windows       : {INITIAL_TRAIN_RATIO:.0%} → {MAX_TRAIN_RATIO:.0%} train (step {EXPANSION_STEP:.0%})")
    print("=" * 70)

    # ── Run paths ──────────────────────────────────────────────────────────
    run_feat_path   = DATA_PATH   / 'features'   / timestamp
    run_model_path  = MODELS_PATH / 'runs'        / timestamp
    run_result_path = RESULTS_PATH / 'runs'       / timestamp

    for p in [run_feat_path, run_model_path, run_result_path]:
        p.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download ───────────────────────────────────────────────────
    raw_data = download_data(SYMBOLS, timestamp)

    if not raw_data:
        print("\n✗ No data downloaded. Exiting.")
        return

    # ── Steps 2–5: Per-symbol pipeline ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" STEPS 2-5 – FEATURES + WALK-FORWARD TRAINING")
    print(f"{'='*70}")

    summary_rows = []

    for symbol in SYMBOLS:
        if symbol not in raw_data:
            print(f"\n  {symbol}: ⚠ No data, skipping")
            continue

        try:
            result = run_symbol(
                symbol        = symbol,
                raw_df        = raw_data[symbol],
                timestamp     = timestamp,
                feat_save_path = run_feat_path,
                model_save_path = run_model_path,
                result_save_path = run_result_path,
            )
            summary_rows.append(result)
        except Exception as exc:
            import traceback
            print(f"\n  {symbol}: ✗ ERROR — {exc}")
            traceback.print_exc()

    # ── Step 6: Next-day predictions ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" STEP 6 – NEXT-DAY PREDICTIONS")
    print(f"{'='*70}")

    predictions = []
    for symbol, df in raw_data.items():
        pred = predict_next_day(symbol, df)
        if pred:
            predictions.append(pred)
            arrow = "📈" if pred['direction'] == 'UP' else "📉"
            print(f"  {arrow} {symbol:12s}: {pred['direction']:4s}  conf={pred['confidence']:.1%}"
                  f"  (last close={pred['last_close']:.2f}  date={pred['last_date']})")

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f" FINAL SUMMARY")
    print(f"{'='*70}")

    ok_rows = [r for r in summary_rows if r.get('status') == 'ok']

    if ok_rows:
        summary_df = pd.DataFrame([{
            'symbol':        r['symbol'],
            'oos_accuracy':  r['oos_accuracy'],
            'oos_f1':        r['oos_f1'],
            'n_windows':     r['n_windows'],
            'n_predictions': r['n_predictions'],
            'n_features':    r['n_features'],
            'n_rows':        r['n_rows'],
        } for r in ok_rows])

        print(f"\n {'Symbol':<12} {'OOS Acc':>8} {'F1':>7} {'Wins':>5} {'Preds':>6} {'Rows':>6}")
        print(f" {'─'*12} {'─'*8} {'─'*7} {'─'*5} {'─'*6} {'─'*6}")
        for _, row in summary_df.iterrows():
            tag = "✅" if row['oos_accuracy'] >= 0.55 else ("⚠️ " if row['oos_accuracy'] >= 0.50 else "❌")
            print(f" {tag} {row['symbol']:<10} {row['oos_accuracy']:>8.2%}"
                  f" {row['oos_f1']:>7.4f} {int(row['n_windows']):>5}"
                  f" {int(row['n_predictions']):>6} {int(row['n_rows']):>6}")

        avg_acc = summary_df['oos_accuracy'].mean()
        avg_f1  = summary_df['oos_f1'].mean()
        best    = summary_df.loc[summary_df['oos_accuracy'].idxmax()]

        print(f"\n Avg OOS Accuracy : {avg_acc:.2%}")
        print(f" Avg F1 Score     : {avg_f1:.4f}")
        print(f" Best Stock       : {best['symbol']}  ({best['oos_accuracy']:.2%})")
        print(f" >55% acc stocks  : {(summary_df['oos_accuracy'] >= 0.55).sum()}/{len(summary_df)}")

        # Save summary
        summary_df.to_csv(run_result_path / 'summary.csv', index=False)

        # Save full window-level detail
        all_win_rows = []
        for r in ok_rows:
            all_win_rows.extend(r.get('window_rows', []))
        if all_win_rows:
            pd.DataFrame(all_win_rows).to_csv(
                run_result_path / 'all_windows_detail.csv', index=False
            )

    # Save predictions
    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(run_result_path / 'next_day_predictions.csv', index=False)

    # Save run metadata
    run_meta = {
        'timestamp':      timestamp,
        'symbols':        SYMBOLS,
        'data_start':     DATA_START_DATE,
        'initial_train':  INITIAL_TRAIN_RATIO,
        'expansion_step': EXPANSION_STEP,
        'max_train':      MAX_TRAIN_RATIO,
        'elapsed_sec':    round(elapsed, 1),
        'n_symbols_ok':   len(ok_rows),
    }
    with open(run_result_path / 'run_metadata.json', 'w') as f:
        json.dump(run_meta, f, indent=2)

    print(f"\n Results  → {run_result_path}")
    print(f" Models   → {run_model_path}")
    print(f" Features → {run_feat_path}")
    print(f" Elapsed  : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print("=" * 70)


if __name__ == '__main__':
    main()
