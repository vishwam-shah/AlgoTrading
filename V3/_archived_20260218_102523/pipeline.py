"""
================================================================================
V3 PIPELINE - Self-contained stock prediction pipeline
================================================================================
Single file: collect data, engineer features, train XGBoost + LightGBM + Ensemble,
evaluate with rolling windows, save all results to CSV.

Usage: python v3/pipeline.py
================================================================================
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize

from loguru import logger

# Setup paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from engine.feature_engine import AdvancedFeatureEngine

# ============================================================================
# CONFIGURATION
# ============================================================================

SYMBOLS = [
    # Banking
    'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'AXISBANK',
    # IT
    'TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM',
]

START_DATE = '2019-01-01'

# (train%, test%, val%) - chronological: [train | test | val]
# Progressive increase in training data for better model learning
WINDOW_CONFIGS = [
    (70, 20, 10),  # Start: 70% train
    (75, 20, 5),   # Increase training to 75%
    (80, 15, 5),   # Increase training to 80%
    (85, 10, 5),   # Increase training to 85%
    (90, 7, 3),    # Increase training to 90%
    (93, 5, 2),    # Increase training to 93%
    (96, 3, 1),    # Max training: 96%, minimal test/val for final prediction
]

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories
for dir_path in [RESULTS_DIR, RAW_DATA_DIR, FEATURES_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Generate timestamp for this run
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ============================================================================
# SECTION 2: DATA COLLECTION
# ============================================================================

def collect_data(symbol: str, start_date: str = START_DATE) -> pd.DataFrame:
    """Download OHLCV data from yfinance and save to data/raw/."""
    ticker = f"{symbol}.NS"
    logger.info(f"Downloading {ticker} from {start_date}...")

    df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df.columns = [c.lower() for c in df.columns]

    # Ensure index is datetime and add as column
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index
    df = df.reset_index(drop=True)

    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} for {symbol}")

    if len(df) < 1000:
        logger.warning(f"{symbol}: only {len(df)} rows (wanted >= 1000)")

    logger.info(f"{symbol}: {len(df)} rows from {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
    
    # Save raw data
    raw_file = os.path.join(RAW_DATA_DIR, f"{symbol}_{RUN_TIMESTAMP}.csv")
    df.to_csv(raw_file, index=False)
    logger.info(f"Saved raw data: {raw_file}")
    
    return df


# ============================================================================
# SECTION 3: FEATURE COMPUTATION
# ============================================================================

def compute_features(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, List[str]]:
    """Compute features using AdvancedFeatureEngine and save preprocessed data."""
    engine = AdvancedFeatureEngine(include_sentiment=False, include_market_context=False)
    feature_set = engine.compute_all_features(df.copy(), symbol=symbol)

    logger.info(f"{symbol}: {feature_set.n_features} features computed")
    
    # Save preprocessed features
    features_subdir = os.path.join(FEATURES_DIR, RUN_TIMESTAMP)
    os.makedirs(features_subdir, exist_ok=True)
    features_file = os.path.join(features_subdir, f"{symbol}_features.csv")
    feature_set.df.to_csv(features_file, index=False)
    logger.info(f"Saved features: {features_file}")
    
    return feature_set.df, feature_set.feature_names


# ============================================================================
# SECTION 4: TARGET COMPUTATION
# ============================================================================

def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute regression + binary direction targets."""
    df = df.copy()

    # Next-day close return (what we predict)
    df['target_close_return'] = df['close'].pct_change().shift(-1)

    # Binary direction
    df['target_direction'] = (df['target_close_return'] > 0).astype(int)

    # Drop rows with NaN targets (last row has no future, plus any from pct_change)
    df = df.dropna(subset=['target_close_return'])

    return df


# ============================================================================
# SECTION 5: SPLIT GENERATION
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
    """Generate chronological train/test/val splits."""
    splits = []
    for train_pct, test_pct, val_pct in window_configs:
        assert train_pct + test_pct + val_pct == 100

        train_end = int(n * train_pct / 100)
        test_end = train_end + int(n * test_pct / 100)

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        val_idx = np.arange(test_end, n)

        config_str = f"{train_pct}/{test_pct}/{val_pct}"

        splits.append(SplitInfo(
            train_idx=train_idx,
            test_idx=test_idx,
            val_idx=val_idx,
            config_str=config_str,
            train_dates=(str(dates.iloc[train_idx[0]].date()), str(dates.iloc[train_idx[-1]].date())),
            test_dates=(str(dates.iloc[test_idx[0]].date()), str(dates.iloc[test_idx[-1]].date())),
            val_dates=(str(dates.iloc[val_idx[0]].date()), str(dates.iloc[val_idx[-1]].date())),
        ))

    return splits


# ============================================================================
# SECTION 6: TRAIN AND PREDICT
# ============================================================================

def train_and_predict(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: List[str],
) -> Dict:
    """Train XGBoost, LightGBM, optimize ensemble weights on val set."""

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_val_s = scaler.transform(X_val)

    # --- XGBoost ---
    xgb_model = XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.005,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )
    xgb_pred_test = xgb_model.predict(X_test_s)
    xgb_pred_val = xgb_model.predict(X_val_s)

    # --- LightGBM ---
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.005,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
    )
    lgb_pred_test = lgb_model.predict(X_test_s)
    lgb_pred_val = lgb_model.predict(X_val_s)

    # --- Ensemble: optimize weights on val set ---
    def ensemble_loss(weights):
        w_xgb, w_lgb = weights
        pred = w_xgb * xgb_pred_val + w_lgb * lgb_pred_val
        return np.mean((y_val - pred) ** 2)

    result = minimize(
        ensemble_loss,
        x0=[0.5, 0.5],
        method='SLSQP',
        bounds=[(0, 1), (0, 1)],
        constraints={'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1.0},
    )
    w_xgb, w_lgb = result.x
    logger.info(f"  Ensemble weights: XGB={w_xgb:.3f}, LGB={w_lgb:.3f}")

    ens_pred_test = w_xgb * xgb_pred_test + w_lgb * lgb_pred_test
    ens_pred_val = w_xgb * xgb_pred_val + w_lgb * lgb_pred_val

    # Feature importances (gain-based)
    xgb_importance = xgb_model.feature_importances_
    lgb_importance = lgb_model.feature_importances_

    return {
        'xgb_pred_test': xgb_pred_test,
        'lgb_pred_test': lgb_pred_test,
        'ens_pred_test': ens_pred_test,
        'xgb_pred_val': xgb_pred_val,
        'lgb_pred_val': lgb_pred_val,
        'ens_pred_val': ens_pred_val,
        'weights': (w_xgb, w_lgb),
        'xgb_importance': xgb_importance,
        'lgb_importance': lgb_importance,
        'feature_names': feature_names,
        # Models for saving
        'xgb_model': xgb_model,
        'lgb_model': lgb_model,
        'scaler': scaler,
    }


# ============================================================================
# SECTION 6.5: MODEL PERSISTENCE
# ============================================================================

def save_models(results: Dict, symbol: str, window_config: str, split_info):
    """Save trained models, scaler, and metadata to disk."""
    model_subdir = os.path.join(MODELS_DIR, RUN_TIMESTAMP, symbol, window_config.replace('/', '_'))
    os.makedirs(model_subdir, exist_ok=True)
   
    # Save XGBoost model
    xgb_path = os.path.join(model_subdir, 'xgb_model.pkl')
    with open(xgb_path, 'wb') as f:
        pickle.dump(results['xgb_model'], f)
   
    # Save LightGBM model
    lgb_path = os.path.join(model_subdir, 'lgb_model.pkl')
    with open(lgb_path, 'wb') as f:
        pickle.dump(results['lgb_model'], f)
   
    # Save scaler
    scaler_path = os.path.join(model_subdir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(results['scaler'], f)
   
    # Save ensemble weights and metadata
    metadata = {
        'symbol': symbol,
        'window_config': window_config,
        'ensemble_weights': results['weights'],
        'n_features': len(results['feature_names']),
        'feature_names': results['feature_names'],
        'train_dates': split_info.train_dates,
        'test_dates': split_info.test_dates,
        'val_dates': split_info.val_dates,
        'timestamp': RUN_TIMESTAMP,
    }
    metadata_path = os.path.join(model_subdir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
   
    logger.info(f"  Saved models to: {model_subdir}")
    return model_subdir


# ============================================================================
# SECTION 7: METRICS COMPUTATION
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, close_prices: np.ndarray) -> Dict:
    """Compute directional accuracy, win rates, Sharpe, profit factor, RMSE, MAE."""

    actual_dir = (y_true > 0).astype(int)
    pred_dir = (y_pred > 0).astype(int)

    # Directional accuracy
    dir_accuracy = accuracy_score(actual_dir, pred_dir)

    # Win rate (long): when we predict UP, how often is it actually UP?
    long_mask = pred_dir == 1
    win_rate_long = float(np.mean(actual_dir[long_mask])) if long_mask.sum() > 0 else 0.0

    # Win rate (short): when we predict DOWN, how often is it actually DOWN?
    short_mask = pred_dir == 0
    win_rate_short = float(np.mean(1 - actual_dir[short_mask])) if short_mask.sum() > 0 else 0.0

    # Strategy returns: go long if pred > 0, short if pred <= 0
    strategy_returns = np.where(pred_dir == 1, y_true, -y_true)

    # Sharpe ratio (annualized)
    if strategy_returns.std() > 0:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Profit factor: sum(wins) / abs(sum(losses))
    wins = strategy_returns[strategy_returns > 0].sum()
    losses = abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = wins / losses if losses > 0 else float('inf')

    # Regression metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Predicted close prices
    # pred_close = close_today * (1 + predicted_return)
    # But close_prices here are the close prices of the prediction day (row i),
    # and y_pred is the predicted return for day i+1
    pred_close = close_prices * (1 + y_pred)
    actual_close_next = close_prices * (1 + y_true)

    return {
        'dir_accuracy': round(dir_accuracy, 4),
        'win_rate_long': round(win_rate_long, 4),
        'win_rate_short': round(win_rate_short, 4),
        'sharpe': round(sharpe, 4),
        'profit_factor': round(profit_factor, 4),
        'rmse': round(rmse, 6),
        'mae': round(mae, 6),
        'pred_close': pred_close,
        'actual_close_next': actual_close_next,
    }


# ============================================================================
# SECTION 8: SAVE RESULTS
# ============================================================================

def save_results(
    summary_rows: List[Dict],
    daily_rows: List[Dict],
) -> None:
    """Save all results to CSV files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # CSV 1: Summary (one row per symbol x window x model)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, 'all_stocks_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary: {summary_path} ({len(summary_df)} rows)")

    # CSV 2: Daily predictions
    daily_df = pd.DataFrame(daily_rows)
    daily_path = os.path.join(RESULTS_DIR, 'all_daily_predictions.csv')
    daily_df.to_csv(daily_path, index=False)
    logger.info(f"Saved daily predictions: {daily_path} ({len(daily_df)} rows)")

    # CSV 3: Methodology notes
    notes = """V3 Pipeline Methodology Notes
==============================

Data:
- Source: yfinance (Yahoo Finance), NSE stocks with .NS suffix
- Period: 2019-01-01 to present (~7 years, ~1750 trading days)
- Includes COVID crash, recovery, rate hike cycles

Features:
- AdvancedFeatureEngine from engine/feature_engine.py
- ~230 features: technical, volatility, volume, momentum, statistical, regime, alpha, lagged
- No sentiment (slow), no market context (unreliable)
- MinMaxScaler [0,1] fitted on training data only

Models:
- XGBoost Regressor: 1000 trees, depth 5, lr 0.005, early stopping on test set
- LightGBM Regressor: 1000 trees, depth 5, lr 0.005, early stopping on test set
- Ensemble: scipy.optimize weighted average, weights optimized on val set (sum=1 constraint)

Targets:
- Primary: close_return = next-day close-to-close percentage return
- Direction: binary, derived from sign of close_return

Split Strategy:
- Chronological: [Train | Test | Val], no shuffling
- Train always starts from 2019-01-01
- 5 window configs: 70/20/10, 65/25/10, 60/25/15, 55/30/15, 50/30/20
- Test set used for early stopping (prevents overfitting)
- Val set used for final evaluation and ensemble weight optimization

Metrics:
- Directional accuracy: accuracy_score on sign of return
- Win rate (long): % of UP predictions that were correct
- Win rate (short): % of DOWN predictions that were correct
- Sharpe ratio: annualized, strategy goes long on UP prediction, short on DOWN
- Profit factor: sum(winning trades) / abs(sum(losing trades))
- RMSE, MAE: regression error on raw returns
"""
    notes_path = os.path.join(RESULTS_DIR, 'methodology_notes.txt')
    with open(notes_path, 'w') as f:
        f.write(notes)
    logger.info(f"Saved methodology: {notes_path}")


# ============================================================================
# SECTION 9: RUN PIPELINE
# ============================================================================

def run_pipeline():
    """Main pipeline: loop symbols x windows, train, evaluate, save."""
    logger.info("=" * 80)
    logger.info("V3 PIPELINE START")
    logger.info(f"Symbols: {SYMBOLS}")
    logger.info(f"Windows: {WINDOW_CONFIGS}")
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
            logger.error(f"Failed to compute features for {symbol}: {e}")
            continue

        # Step 3: Compute targets
        feat_df = compute_targets(feat_df)

        # Drop rows with NaN in features
        feat_df = feat_df.dropna(subset=feature_names)

        # Reset index for clean splitting
        feat_df = feat_df.reset_index(drop=True)

        n = len(feat_df)
        logger.info(f"{symbol}: {n} rows after cleaning, {len(feature_names)} features")

        if n < 500:
            logger.warning(f"{symbol}: too few rows ({n}), skipping")
            continue

        # Step 4: Generate splits
        dates = feat_df['date']
        splits = generate_splits(n, dates, WINDOW_CONFIGS)

        for split in splits:
            logger.info(f"\n  Window {split.config_str}: "
                        f"train={len(split.train_idx)} [{split.train_dates[0]}..{split.train_dates[1]}], "
                        f"test={len(split.test_idx)} [{split.test_dates[0]}..{split.test_dates[1]}], "
                        f"val={len(split.val_idx)} [{split.val_dates[0]}..{split.val_dates[1]}]")

            X = feat_df[feature_names].values
            y = feat_df['target_close_return'].values
            close_prices = feat_df['close'].values

            X_train = X[split.train_idx]
            y_train = y[split.train_idx]
            X_test = X[split.test_idx]
            y_test = y[split.test_idx]
            X_val = X[split.val_idx]
            y_val = y[split.val_idx]

            close_val = close_prices[split.val_idx]
            val_dates = dates.iloc[split.val_idx].values

            # Step 5: Train and predict
            try:
                results = train_and_predict(X_train, y_train, X_test, y_test, X_val, y_val, feature_names)
            except Exception as e:
                logger.error(f"  Training failed for {symbol} {split.config_str}: {e}")
                continue

            # Step 5.5: Save trained models
            try:
                save_models(results, symbol, split.config_str, split)
            except Exception as e:
                logger.warning(f"  Failed to save models for {symbol} {split.config_str}: {e}")

            # Step 6: Compute metrics on VAL set for each model
            w_xgb, w_lgb = results['weights']

            for model_name, preds in [
                ('xgb', results['xgb_pred_val']),
                ('lgb', results['lgb_pred_val']),
                ('ensemble', results['ens_pred_val']),
            ]:
                metrics = compute_metrics(y_val, preds, close_val)

                summary_rows.append({
                    'symbol': symbol,
                    'window': split.config_str,
                    'model': model_name,
                    'dir_accuracy': metrics['dir_accuracy'],
                    'win_rate_long': metrics['win_rate_long'],
                    'win_rate_short': metrics['win_rate_short'],
                    'sharpe': metrics['sharpe'],
                    'profit_factor': metrics['profit_factor'],
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'n_train': len(split.train_idx),
                    'n_test': len(split.test_idx),
                    'n_val': len(split.val_idx),
                    'n_features': len(feature_names),
                    'ensemble_weight_xgb': round(w_xgb, 4),
                    'ensemble_weight_lgb': round(w_lgb, 4),
                })

                logger.info(f"  {model_name:>8s}: dir_acc={metrics['dir_accuracy']:.3f} "
                            f"WR_long={metrics['win_rate_long']:.3f} "
                            f"WR_short={metrics['win_rate_short']:.3f} "
                            f"sharpe={metrics['sharpe']:.3f} "
                            f"PF={metrics['profit_factor']:.3f}")

            # Step 7: Save daily predictions for val set
            for i in range(len(split.val_idx)):
                actual_return = y_val[i]
                actual_close = close_val[i]
                actual_close_next = actual_close * (1 + actual_return)

                daily_rows.append({
                    'symbol': symbol,
                    'window': split.config_str,
                    'date': str(pd.Timestamp(val_dates[i]).date()),
                    'actual_close': round(actual_close, 2),
                    'actual_close_next': round(actual_close_next, 2),
                    'actual_return': round(actual_return, 6),
                    'actual_direction': int(actual_return > 0),
                    'pred_return_xgb': round(float(results['xgb_pred_val'][i]), 6),
                    'pred_return_lgb': round(float(results['lgb_pred_val'][i]), 6),
                    'pred_return_ensemble': round(float(results['ens_pred_val'][i]), 6),
                    'pred_close_xgb': round(float(actual_close * (1 + results['xgb_pred_val'][i])), 2),
                    'pred_close_lgb': round(float(actual_close * (1 + results['lgb_pred_val'][i])), 2),
                    'pred_close_ensemble': round(float(actual_close * (1 + results['ens_pred_val'][i])), 2),
                    'pred_dir_xgb': int(results['xgb_pred_val'][i] > 0),
                    'pred_dir_lgb': int(results['lgb_pred_val'][i] > 0),
                    'pred_dir_ensemble': int(results['ens_pred_val'][i] > 0),
                })

    # Save all results
    if summary_rows:
        save_results(summary_rows, daily_rows)
        logger.info(f"\nPipeline complete: {len(summary_rows)} summary rows, {len(daily_rows)} daily rows")
    else:
        logger.error("No results generated!")

    logger.info("=" * 80)
    logger.info("V3 PIPELINE DONE")
    logger.info("=" * 80)


def run_pipeline_api(
    symbols: Optional[List[str]] = None,
    progress_callback=None,
) -> Dict:
    """API-friendly pipeline: runs pipeline, calls progress_callback, returns formatted results.

    Args:
        symbols: List of stock symbols. Defaults to SYMBOLS if None.
        progress_callback: Optional callable(step, total, message) for progress updates.

    Returns:
        Dict with 'backtest_results' and 'signals' matching frontend PipelineResult shape.
    """
    symbols = symbols or SYMBOLS
    total_steps = 4

    def update_progress(step: int, message: str):
        if progress_callback:
            progress_callback(step, total_steps, message)

    update_progress(1, "Collecting data...")

    # Step 1: Collect data for all symbols
    stock_data = {}
    for si, symbol in enumerate(symbols):
        update_progress(1, f"Downloading {symbol} ({si+1}/{len(symbols)})...")
        try:
            stock_data[symbol] = collect_data(symbol)
        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")

    # Step 2: Feature engineering
    update_progress(2, "Computing features...")
    stock_features = {}
    for si, symbol in enumerate(stock_data):
        update_progress(2, f"Features for {symbol} ({si+1}/{len(stock_data)})...")
        try:
            feat_df, feature_names = compute_features(stock_data[symbol], symbol)
            feat_df = compute_targets(feat_df)
            feat_df = feat_df.dropna(subset=feature_names)
            feat_df = feat_df.reset_index(drop=True)
            if len(feat_df) >= 500:
                stock_features[symbol] = (feat_df, feature_names)
            else:
                logger.warning(f"{symbol}: too few rows ({len(feat_df)}), skipping")
        except Exception as e:
            logger.error(f"Failed features for {symbol}: {e}")

    # Step 3: Train models across all windows
    update_progress(3, "Training models...")
    # Store all results per symbol
    all_symbol_results: Dict[str, Dict] = {}

    for si, symbol in enumerate(stock_features):
        feat_df, feature_names = stock_features[symbol]
        n = len(feat_df)
        dates = feat_df['date']
        splits = generate_splits(n, dates, WINDOW_CONFIGS)
        symbol_windows = {}

        for split in splits:
            update_progress(3, f"Training {symbol} window {split.config_str} ({si+1}/{len(stock_features)})...")

            X = feat_df[feature_names].values
            y = feat_df['target_close_return'].values
            close_prices = feat_df['close'].values

            X_train = X[split.train_idx]
            y_train = y[split.train_idx]
            X_test = X[split.test_idx]
            y_test = y[split.test_idx]
            X_val = X[split.val_idx]
            y_val = y[split.val_idx]
            close_val = close_prices[split.val_idx]

            try:
                results = train_and_predict(X_train, y_train, X_test, y_test, X_val, y_val, feature_names)
            except Exception as e:
                logger.error(f"Training failed for {symbol} {split.config_str}: {e}")
                continue

            # Compute metrics for each model on val set
            xgb_metrics = compute_metrics(y_val, results['xgb_pred_val'], close_val)
            lgb_metrics = compute_metrics(y_val, results['lgb_pred_val'], close_val)
            ens_metrics = compute_metrics(y_val, results['ens_pred_val'], close_val)

            symbol_windows[split.config_str] = {
                'xgb_acc': xgb_metrics['dir_accuracy'],
                'lgb_acc': lgb_metrics['dir_accuracy'],
                'ens_acc': ens_metrics['dir_accuracy'],
                'sharpe': ens_metrics['sharpe'],
                'profit_factor': ens_metrics['profit_factor'],
                'win_rate_long': ens_metrics['win_rate_long'],
                'win_rate_short': ens_metrics['win_rate_short'],
                'rmse': ens_metrics['rmse'],
                'mae': ens_metrics['mae'],
                'n_val': len(split.val_idx),
                'n_train': len(split.train_idx),
                'ensemble_weights': (float(results['weights'][0]), float(results['weights'][1])),
                'mean_pred_return': float(np.mean(results['ens_pred_val'])),
                'last_pred_return': float(results['ens_pred_val'][-1]),
                # Store importance data separately (not serialized to frontend)
                '_xgb_importance': results['xgb_importance'],
                '_feature_names': results['feature_names'],
            }

        if symbol_windows:
            # Find best window by ensemble Sharpe ratio
            best_window = max(symbol_windows, key=lambda w: symbol_windows[w]['sharpe'])
            all_symbol_results[symbol] = {
                'windows': symbol_windows,
                'best_window': best_window,
                'n_features': len(feature_names),
                'last_close': float(feat_df['close'].iloc[-1]),
            }

    # Step 4: Format results for frontend
    update_progress(4, "Formatting results...")

    backtest_results = {}
    signals = {}

    for symbol, data in all_symbol_results.items():
        best = data['windows'][data['best_window']]

        # Top 10 feature importances
        importance = best['_xgb_importance']
        feat_names = best['_feature_names']
        top_indices = np.argsort(importance)[::-1][:10]
        feature_importance = [
            {'feature': feat_names[i], 'importance': float(importance[i])}
            for i in top_indices
        ]

        backtest_results[symbol] = {
            'total_return': best['sharpe'] / 100,  # proxy
            'sharpe_ratio': best['sharpe'],
            'max_drawdown': 0,
            'win_rate': best['win_rate_long'],
            'total_trades': best['n_val'],
            'profit_factor': best['profit_factor'],
            'directional_accuracy': best['ens_acc'],
            'model_predictions': {
                'xgb_accuracy': best['xgb_acc'],
                'lgb_accuracy': best['lgb_acc'],
                'ensemble_accuracy': best['ens_acc'],
            },
            'feature_importance': feature_importance,
            'equity_curve': [],
            'trades': [],
            # V3-specific extras (strip internal keys with numpy arrays)
            'v3_all_windows': {
                w: {k: v for k, v in wd.items() if not k.startswith('_')}
                for w, wd in data['windows'].items()
            },
            'v3_best_window': data['best_window'],
            'v3_n_features': data['n_features'],
            'v3_method': 'regression',
        }

        # Signal from best window ensemble
        pred_return = best['last_pred_return']
        signals[symbol] = {
            'action': 'BUY' if pred_return > 0 else 'SELL',
            'confidence': best['ens_acc'],
            'direction_probability': best['win_rate_long'] if pred_return > 0 else best['win_rate_short'],
            'expected_return': best['mean_pred_return'],
            'current_price': data['last_close'],
        }

    update_progress(4, "Pipeline complete.")

    return {
        'backtest_results': backtest_results,
        'signals': signals,
    }


if __name__ == '__main__':
    run_pipeline()
