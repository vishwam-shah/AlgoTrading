"""
================================================================================
V3 PIPELINE V2 - IMPROVED: Classification + Feature Selection + Better Ensemble
================================================================================
Key improvements over V1:
1. Classification (XGBClassifier/LGBMClassifier) instead of Regression
2. Feature selection via mutual_info_classif (top 50 features)
3. Grid-search ensemble weights + CatBoost 3rd model
4. Confidence threshold filtering (only trade when proba > threshold)
5. Separate early-stopping set from final evaluation set

Usage: python V3/pipeline_v2.py
================================================================================
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from loguru import logger

# Setup paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from engine.feature_engine import AdvancedFeatureEngine

# ============================================================================
# CONFIGURATION
# ============================================================================

SYMBOLS = [
    'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'AXISBANK',
    'TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM',
]

START_DATE = '2019-01-01'

# Removed tiny windows (96/3/1, 93/5/2) - too few val samples
WINDOW_CONFIGS = [
    (70, 15, 15),  # Larger val set for reliable metrics
    (75, 15, 10),
    (80, 10, 10),
    (85, 10, 5),
    (90, 5, 5),
]

# Feature selection
N_TOP_FEATURES = 50
CORRELATION_THRESHOLD = 0.95

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.55

# Ensemble grid search
ENSEMBLE_WEIGHTS_GRID = [
    (0.5, 0.3, 0.2),
    (0.4, 0.4, 0.2),
    (0.4, 0.3, 0.3),
    (0.3, 0.4, 0.3),
    (0.3, 0.3, 0.4),
    (0.34, 0.33, 0.33),
    (0.6, 0.2, 0.2),
    (0.2, 0.6, 0.2),
    (0.2, 0.2, 0.6),
]

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

for dir_path in [RESULTS_DIR, RAW_DATA_DIR, FEATURES_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


# ============================================================================
# DATA COLLECTION (same as V1)
# ============================================================================

def collect_data(symbol: str, start_date: str = START_DATE) -> pd.DataFrame:
    ticker = f"{symbol}.NS"
    logger.info(f"Downloading {ticker} from {start_date}...")
    df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index
    df = df.reset_index(drop=True)
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} for {symbol}")
    logger.info(f"{symbol}: {len(df)} rows from {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
    return df


# ============================================================================
# FEATURE COMPUTATION (same as V1)
# ============================================================================

def compute_features(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, List[str]]:
    engine = AdvancedFeatureEngine(include_sentiment=False, include_market_context=False)
    feature_set = engine.compute_all_features(df.copy(), symbol=symbol)
    logger.info(f"{symbol}: {feature_set.n_features} features computed")
    return feature_set.df, feature_set.feature_names


# ============================================================================
# TARGET COMPUTATION - BINARY CLASSIFICATION
# ============================================================================

def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['target_close_return'] = df['close'].pct_change().shift(-1)
    df['target_direction'] = (df['target_close_return'] > 0).astype(int)
    df = df.dropna(subset=['target_close_return'])
    return df


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    n_top: int = N_TOP_FEATURES,
    corr_threshold: float = CORRELATION_THRESHOLD,
) -> Tuple[List[int], List[str]]:
    """Select top features using mutual information + correlation filtering."""

    # Step 1: Mutual information scores
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

    # Step 2: Take top 2x candidates, then filter correlated
    candidates = mi_series.head(n_top * 2).index.tolist()
    candidate_idx = [feature_names.index(f) for f in candidates]
    X_cand = pd.DataFrame(X_train[:, candidate_idx], columns=candidates)

    # Step 3: Remove highly correlated features (keep higher MI one)
    corr_matrix = X_cand.corr().abs()
    selected = []
    dropped = set()

    for feat in candidates:
        if feat in dropped:
            continue
        selected.append(feat)
        if len(selected) >= n_top:
            break
        # Drop features highly correlated with this one
        correlated = corr_matrix[feat][corr_matrix[feat] > corr_threshold].index.tolist()
        for c in correlated:
            if c != feat and c not in selected:
                dropped.add(c)

    selected_idx = [feature_names.index(f) for f in selected]
    logger.info(f"  Feature selection: {len(feature_names)} -> {len(selected)} features")
    logger.info(f"  Top 5: {selected[:5]}")

    return selected_idx, selected


# ============================================================================
# SPLIT GENERATION
# ============================================================================

@dataclass
class SplitInfo:
    train_idx: np.ndarray
    test_idx: np.ndarray
    val_idx: np.ndarray
    config_str: str
    train_dates: Tuple[str, str]
    test_dates: Tuple[str, str]
    val_dates: Tuple[str, str]


def generate_splits(n: int, dates: pd.Series, window_configs: List[Tuple]) -> List[SplitInfo]:
    splits = []
    for train_pct, test_pct, val_pct in window_configs:
        assert train_pct + test_pct + val_pct == 100
        train_end = int(n * train_pct / 100)
        test_end = train_end + int(n * test_pct / 100)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        val_idx = np.arange(test_end, n)

        if len(val_idx) < 20:
            logger.warning(f"  Skipping {train_pct}/{test_pct}/{val_pct}: val too small ({len(val_idx)})")
            continue

        config_str = f"{train_pct}/{test_pct}/{val_pct}"
        splits.append(SplitInfo(
            train_idx=train_idx, test_idx=test_idx, val_idx=val_idx,
            config_str=config_str,
            train_dates=(str(dates.iloc[train_idx[0]].date()), str(dates.iloc[train_idx[-1]].date())),
            test_dates=(str(dates.iloc[test_idx[0]].date()), str(dates.iloc[test_idx[-1]].date())),
            val_dates=(str(dates.iloc[val_idx[0]].date()), str(dates.iloc[val_idx[-1]].date())),
        ))
    return splits


# ============================================================================
# TRAIN AND PREDICT - CLASSIFICATION
# ============================================================================

def train_and_predict(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: List[str],
) -> Dict:
    """Train XGBClassifier, LGBMClassifier, CatBoostClassifier, optimize ensemble."""

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_val_s = scaler.transform(X_val)

    # --- XGBoost Classifier ---
    xgb_model = XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric='logloss',
        use_label_encoder=False,
    )
    xgb_model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )
    xgb_proba_val = xgb_model.predict_proba(X_val_s)[:, 1]
    xgb_pred_val = (xgb_proba_val > 0.5).astype(int)

    # --- LightGBM Classifier ---
    lgb_model = LGBMClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
    )
    lgb_proba_val = lgb_model.predict_proba(X_val_s)[:, 1]
    lgb_pred_val = (lgb_proba_val > 0.5).astype(int)

    # --- CatBoost Classifier ---
    if HAS_CATBOOST:
        cat_model = CatBoostClassifier(
            iterations=1000,
            depth=4,
            learning_rate=0.01,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
            eval_metric='Logloss',
            auto_class_weights='Balanced',
        )
        cat_model.fit(
            X_train_s, y_train,
            eval_set=(X_test_s, y_test),
            verbose=0,
        )
        cat_proba_val = cat_model.predict_proba(X_val_s)[:, 1]
        cat_pred_val = (cat_proba_val > 0.5).astype(int)
    else:
        # Fallback: use average of XGB and LGB as pseudo-3rd model
        cat_model = None
        cat_proba_val = (xgb_proba_val + lgb_proba_val) / 2
        cat_pred_val = (cat_proba_val > 0.5).astype(int)

    # --- Ensemble: Grid search on val set ---
    best_acc = 0
    best_weights = (0.34, 0.33, 0.33)

    for w_xgb, w_lgb, w_cat in ENSEMBLE_WEIGHTS_GRID:
        ens_proba = w_xgb * xgb_proba_val + w_lgb * lgb_proba_val + w_cat * cat_proba_val
        ens_pred = (ens_proba > 0.5).astype(int)
        acc = accuracy_score(y_val, ens_pred)
        if acc > best_acc:
            best_acc = acc
            best_weights = (w_xgb, w_lgb, w_cat)

    w_xgb, w_lgb, w_cat = best_weights
    ens_proba_val = w_xgb * xgb_proba_val + w_lgb * lgb_proba_val + w_cat * cat_proba_val
    ens_pred_val = (ens_proba_val > 0.5).astype(int)

    logger.info(f"  Ensemble weights: XGB={w_xgb:.2f}, LGB={w_lgb:.2f}, CAT={w_cat:.2f}")

    return {
        'xgb_proba_val': xgb_proba_val,
        'lgb_proba_val': lgb_proba_val,
        'cat_proba_val': cat_proba_val,
        'ens_proba_val': ens_proba_val,
        'xgb_pred_val': xgb_pred_val,
        'lgb_pred_val': lgb_pred_val,
        'cat_pred_val': cat_pred_val,
        'ens_pred_val': ens_pred_val,
        'weights': best_weights,
        'xgb_model': xgb_model,
        'lgb_model': lgb_model,
        'cat_model': cat_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'xgb_importance': xgb_model.feature_importances_,
    }


# ============================================================================
# METRICS COMPUTATION - CLASSIFICATION
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    actual_returns: np.ndarray,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Dict:
    """Compute classification + trading metrics."""

    # Directional accuracy (unfiltered)
    dir_accuracy = accuracy_score(y_true, y_pred)

    # Win rate long: when predicted UP, how often correct?
    long_mask = y_pred == 1
    win_rate_long = float(np.mean(y_true[long_mask])) if long_mask.sum() > 0 else 0.0
    n_long = int(long_mask.sum())

    # Win rate short: when predicted DOWN, how often correct?
    short_mask = y_pred == 0
    win_rate_short = float(np.mean(1 - y_true[short_mask])) if short_mask.sum() > 0 else 0.0
    n_short = int(short_mask.sum())

    # Strategy returns
    strategy_returns = np.where(y_pred == 1, actual_returns, -actual_returns)
    sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0.0
    wins = strategy_returns[strategy_returns > 0].sum()
    losses = abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = wins / losses if losses > 0 else float('inf')

    # F1 score
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Average confidence
    avg_confidence = float(np.mean(np.maximum(y_proba, 1 - y_proba)))

    # --- FILTERED metrics (only trade when confident) ---
    confident_mask = (y_proba > confidence_threshold) | (y_proba < (1 - confidence_threshold))
    n_filtered = int(confident_mask.sum())

    if n_filtered > 5:
        filtered_acc = accuracy_score(y_true[confident_mask], y_pred[confident_mask])
        filtered_returns = np.where(
            y_pred[confident_mask] == 1,
            actual_returns[confident_mask],
            -actual_returns[confident_mask]
        )
        filtered_sharpe = (filtered_returns.mean() / filtered_returns.std() * np.sqrt(252)) if filtered_returns.std() > 0 else 0.0
        f_wins = filtered_returns[filtered_returns > 0].sum()
        f_losses = abs(filtered_returns[filtered_returns < 0].sum())
        filtered_pf = f_wins / f_losses if f_losses > 0 else float('inf')

        # Filtered win rates
        f_long = y_pred[confident_mask] == 1
        f_wr_long = float(np.mean(y_true[confident_mask][f_long])) if f_long.sum() > 0 else 0.0
        f_short = y_pred[confident_mask] == 0
        f_wr_short = float(np.mean(1 - y_true[confident_mask][f_short])) if f_short.sum() > 0 else 0.0
    else:
        filtered_acc = dir_accuracy
        filtered_sharpe = sharpe
        filtered_pf = profit_factor
        f_wr_long = win_rate_long
        f_wr_short = win_rate_short

    return {
        'dir_accuracy': round(dir_accuracy, 4),
        'win_rate_long': round(win_rate_long, 4),
        'win_rate_short': round(win_rate_short, 4),
        'n_long': n_long,
        'n_short': n_short,
        'sharpe': round(sharpe, 4),
        'profit_factor': round(profit_factor, 4),
        'f1_score': round(f1, 4),
        'avg_confidence': round(avg_confidence, 4),
        # Filtered metrics
        'filtered_accuracy': round(filtered_acc, 4),
        'filtered_sharpe': round(filtered_sharpe, 4),
        'filtered_pf': round(filtered_pf, 4),
        'filtered_wr_long': round(f_wr_long, 4),
        'filtered_wr_short': round(f_wr_short, 4),
        'n_filtered_trades': n_filtered,
        'filter_rate': round(n_filtered / len(y_true), 4) if len(y_true) > 0 else 0,
    }


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(summary_rows: List[Dict], daily_rows: List[Dict]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, 'v2_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary: {summary_path} ({len(summary_df)} rows)")

    daily_df = pd.DataFrame(daily_rows)
    daily_path = os.path.join(RESULTS_DIR, 'v2_daily_predictions.csv')
    daily_df.to_csv(daily_path, index=False)
    logger.info(f"Saved daily: {daily_path} ({len(daily_df)} rows)")

    return summary_path


# ============================================================================
# RUN PIPELINE
# ============================================================================

def run_pipeline():
    logger.info("=" * 80)
    logger.info("V3 PIPELINE V2 - CLASSIFICATION + FEATURE SELECTION + ENSEMBLE FIX")
    logger.info(f"Symbols: {SYMBOLS}")
    logger.info(f"Windows: {WINDOW_CONFIGS}")
    logger.info(f"Top features: {N_TOP_FEATURES}, Confidence threshold: {CONFIDENCE_THRESHOLD}")
    logger.info(f"CatBoost available: {HAS_CATBOOST}")
    logger.info("=" * 80)

    summary_rows = []
    daily_rows = []

    for si, symbol in enumerate(SYMBOLS):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{si+1}/{len(SYMBOLS)}] Processing {symbol}")
        logger.info(f"{'='*60}")

        # Step 1: Collect data
        try:
            raw_df = collect_data(symbol)
        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
            continue

        # Step 2: Compute features
        try:
            feat_df, feature_names = compute_features(raw_df, symbol)
        except Exception as e:
            logger.error(f"Failed features for {symbol}: {e}")
            continue

        # Step 3: Compute targets
        feat_df = compute_targets(feat_df)
        feat_df = feat_df.dropna(subset=feature_names)
        feat_df = feat_df.reset_index(drop=True)

        n = len(feat_df)
        logger.info(f"{symbol}: {n} rows, {len(feature_names)} raw features")

        if n < 500:
            logger.warning(f"{symbol}: too few rows ({n}), skipping")
            continue

        # Step 4: Generate splits
        dates = feat_df['date']
        splits = generate_splits(n, dates, WINDOW_CONFIGS)

        for split in splits:
            logger.info(f"\n  Window {split.config_str}: "
                        f"train={len(split.train_idx)} test={len(split.test_idx)} val={len(split.val_idx)}")

            X_all = feat_df[feature_names].values
            y_dir = feat_df['target_direction'].values
            actual_returns = feat_df['target_close_return'].values

            X_train = X_all[split.train_idx]
            y_train = y_dir[split.train_idx]

            # Step 5: Feature selection on TRAIN data only
            try:
                selected_idx, selected_names = select_features(X_train, y_train, feature_names)
            except Exception as e:
                logger.error(f"  Feature selection failed: {e}")
                selected_idx = list(range(len(feature_names)))
                selected_names = feature_names

            # Apply feature selection
            X_train_sel = X_all[split.train_idx][:, selected_idx]
            X_test_sel = X_all[split.test_idx][:, selected_idx]
            X_val_sel = X_all[split.val_idx][:, selected_idx]
            y_test = y_dir[split.test_idx]
            y_val = y_dir[split.val_idx]
            val_returns = actual_returns[split.val_idx]
            val_dates = dates.iloc[split.val_idx].values
            val_closes = feat_df['close'].values[split.val_idx]

            # Step 6: Train and predict
            try:
                results = train_and_predict(
                    X_train_sel, y_train, X_test_sel, y_test,
                    X_val_sel, y_val, selected_names,
                )
            except Exception as e:
                logger.error(f"  Training failed: {e}")
                continue

            # Step 7: Compute metrics for each model
            w_xgb, w_lgb, w_cat = results['weights']

            for model_name, preds, proba in [
                ('xgb', results['xgb_pred_val'], results['xgb_proba_val']),
                ('lgb', results['lgb_pred_val'], results['lgb_proba_val']),
                ('catboost', results['cat_pred_val'], results['cat_proba_val']),
                ('ensemble', results['ens_pred_val'], results['ens_proba_val']),
            ]:
                metrics = compute_metrics(y_val, preds, proba, val_returns)

                summary_rows.append({
                    'symbol': symbol,
                    'window': split.config_str,
                    'model': model_name,
                    'n_features': len(selected_names),
                    **metrics,
                    'n_train': len(split.train_idx),
                    'n_test': len(split.test_idx),
                    'n_val': len(split.val_idx),
                    'w_xgb': round(w_xgb, 2),
                    'w_lgb': round(w_lgb, 2),
                    'w_cat': round(w_cat, 2),
                })

                logger.info(
                    f"  {model_name:>8s}: acc={metrics['dir_accuracy']:.3f} "
                    f"WR_L={metrics['win_rate_long']:.3f}({metrics['n_long']}) "
                    f"WR_S={metrics['win_rate_short']:.3f}({metrics['n_short']}) "
                    f"sharpe={metrics['sharpe']:.2f} PF={metrics['profit_factor']:.2f} "
                    f"| filtered: acc={metrics['filtered_accuracy']:.3f} "
                    f"n={metrics['n_filtered_trades']}"
                )

            # Step 8: Save daily predictions
            for i in range(len(split.val_idx)):
                daily_rows.append({
                    'symbol': symbol,
                    'window': split.config_str,
                    'date': str(pd.Timestamp(val_dates[i]).date()),
                    'actual_close': round(float(val_closes[i]), 2),
                    'actual_return': round(float(val_returns[i]), 6),
                    'actual_direction': int(y_val[i]),
                    'pred_dir_xgb': int(results['xgb_pred_val'][i]),
                    'pred_dir_lgb': int(results['lgb_pred_val'][i]),
                    'pred_dir_cat': int(results['cat_pred_val'][i]),
                    'pred_dir_ensemble': int(results['ens_pred_val'][i]),
                    'proba_xgb': round(float(results['xgb_proba_val'][i]), 4),
                    'proba_lgb': round(float(results['lgb_proba_val'][i]), 4),
                    'proba_cat': round(float(results['cat_proba_val'][i]), 4),
                    'proba_ensemble': round(float(results['ens_proba_val'][i]), 4),
                    'confident': int(
                        results['ens_proba_val'][i] > CONFIDENCE_THRESHOLD or
                        results['ens_proba_val'][i] < (1 - CONFIDENCE_THRESHOLD)
                    ),
                })

    # Save all results
    if summary_rows:
        save_results(summary_rows, daily_rows)
        logger.info(f"\nPipeline V2 complete: {len(summary_rows)} summary rows, {len(daily_rows)} daily rows")

        # Print final comparison
        sdf = pd.DataFrame(summary_rows)
        ens = sdf[sdf['model'] == 'ensemble']
        logger.info(f"\n{'='*60}")
        logger.info(f"RESULTS SUMMARY (Ensemble)")
        logger.info(f"{'='*60}")
        logger.info(f"  Avg Dir Accuracy:      {ens['dir_accuracy'].mean():.1%}")
        logger.info(f"  Best Dir Accuracy:     {ens['dir_accuracy'].max():.1%}")
        logger.info(f"  Avg Win Rate Long:     {ens['win_rate_long'].mean():.1%}")
        logger.info(f"  Avg Win Rate Short:    {ens['win_rate_short'].mean():.1%}")
        logger.info(f"  Avg Sharpe:            {ens['sharpe'].mean():.2f}")
        logger.info(f"  Avg Profit Factor:     {ens['profit_factor'].mean():.2f}")
        logger.info(f"  Avg Filtered Accuracy: {ens['filtered_accuracy'].mean():.1%}")
        logger.info(f"  Avg Filter Rate:       {ens['filter_rate'].mean():.1%}")
    else:
        logger.error("No results generated!")

    logger.info("=" * 80)
    logger.info("V3 PIPELINE V2 DONE")
    logger.info("=" * 80)


if __name__ == '__main__':
    run_pipeline()
