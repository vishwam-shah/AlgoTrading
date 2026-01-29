"""
Configuration file for AI Stock Prediction System
=================================================
Target: Next Day Closing Price Prediction
"""
import os
from datetime import datetime

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')           # Raw OHLCV data
MARKET_DATA_DIR = os.path.join(DATA_DIR, 'market')     # Nifty, VIX, USD/INR, Sector indices
FEATURES_DIR = os.path.join(DATA_DIR, 'features')      # Engineered features
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')    # Train/Val/Test splits

# Model directories
MODEL_DIR = os.path.join(BASE_DIR, 'models')
XGBOOST_DIR = os.path.join(MODEL_DIR, 'xgboost')
LSTM_DIR = os.path.join(MODEL_DIR, 'lstm')
GRU_DIR = os.path.join(MODEL_DIR, 'gru')
TRANSFORMER_DIR = os.path.join(MODEL_DIR, 'transformer')
ENSEMBLE_DIR = os.path.join(MODEL_DIR, 'ensemble')
SCALERS_DIR = os.path.join(MODEL_DIR, 'scalers')

# Results directories
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, 'predictions')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

# Logs directory
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# ============================================================================
# DATA COLLECTION PARAMETERS
# ============================================================================
# 10 years of data (2015-2025) for 1-2 full market cycles
START_DATE = '2015-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Expected rows: 252 trading days x 10 years = ~2,520 rows per stock
MIN_REQUIRED_ROWS = 2000  # Minimum acceptable

# Data source
DATA_SOURCE = 'yfinance'

# Sentiment analysis
ENABLE_SENTIMENT = True  # Enable sentiment features - can significantly improve accuracy  # Set to True to enable sentiment (slower but more accurate)
SENTIMENT_MAX_FETCH = 100  # Max number of recent dates to fetch sentiment
SENTIMENT_LOOKBACK_DAYS = 365  # Only fetch sentiment for last N days (None = all history)

# ============================================================================
# MARKET CONTEXT DATA (Additional features)
# ============================================================================
MARKET_SYMBOLS = {
    'NIFTY50': '^NSEI',           # Nifty 50 index
    'BANKNIFTY': '^NSEBANK',      # Bank Nifty index
    'INDIA_VIX': '^INDIAVIX',     # India VIX (volatility)
    'USD_INR': 'INR=X',           # USD/INR exchange rate
}

# Sector indices for relative strength
SECTOR_INDICES = {
    'NIFTY_BANK': '^NSEBANK',
    'NIFTY_IT': '^CNXIT',
    'NIFTY_PHARMA': '^CNXPHARMA',
    'NIFTY_AUTO': '^CNXAUTO',
    'NIFTY_METAL': '^CNXMETAL',
    'NIFTY_ENERGY': '^CNXENERGY',
    'NIFTY_FMCG': '^CNXFMCG',
    'NIFTY_REALTY': '^CNXREALTY',
    'NIFTY_INFRA': '^CNXINFRA',
    'NIFTY_FIN_SERVICE': '^CNXFIN',
}

# Stock to Sector mapping
STOCK_SECTOR_MAP = {
    # Banking
    'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'KOTAKBANK': 'Banking',
    'SBIN': 'Banking', 'AXISBANK': 'Banking', 'INDUSINDBK': 'Banking',
    'BAJFINANCE': 'Banking', 'BAJAJFINSV': 'Banking', 'PNB': 'Banking', 'BANKBARODA': 'Banking',
    # IT
    'TCS': 'IT', 'INFY': 'IT', 'HCLTECH': 'IT', 'WIPRO': 'IT', 'TECHM': 'IT',
    # Auto
    'MARUTI': 'Auto', 'TATAMOTORS': 'Auto', 'M&M': 'Auto', 'BAJAJ-AUTO': 'Auto', 'HEROMOTOCO': 'Auto',
    # Pharma
    'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma', 'CIPLA': 'Pharma', 'DIVISLAB': 'Pharma', 'APOLLOHOSP': 'Pharma',
    # Energy
    'RELIANCE': 'Energy', 'ONGC': 'Energy', 'NTPC': 'Energy', 'POWERGRID': 'Energy', 'ADANIGREEN': 'Energy',
    # Metals
    'TATASTEEL': 'Metal', 'JSWSTEEL': 'Metal', 'HINDALCO': 'Metal', 'VEDL': 'Metal', 'COALINDIA': 'Metal',
    # FMCG
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG', 'BRITANNIA': 'FMCG', 'TATACONSUM': 'FMCG',
    # Infra
    'LT': 'Infra', 'ADANIENT': 'Infra', 'ADANIPORTS': 'Infra', 'ULTRACEMCO': 'Infra', 'GRASIM': 'Infra',
    # Others
    'BHARTIARTL': 'Telecom', 'ASIANPAINT': 'Consumer', 'TITAN': 'Consumer', 'HDFCLIFE': 'Insurance', 'SBILIFE': 'Insurance',
}

# ============================================================================
# STOCK UNIVERSE (50 Stocks for Production)
# ============================================================================
# Phase 1: 10 stocks for proof of concept
PHASE1_STOCKS = {
    'Banking': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'AXISBANK'],
    'IT': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM']
}

# Phase 2: 50 stocks across sectors
ALL_STOCKS = [
    # Banking (10)
    'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'AXISBANK',
    'INDUSINDBK', 'BAJFINANCE', 'BAJAJFINSV', 'PNB', 'BANKBARODA',
    # IT (5)
    'TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM',
    # Auto (5)
    'MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO',
    # Pharma (5)
    'SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP',
    # Energy (5)
    'RELIANCE', 'ONGC', 'NTPC', 'POWERGRID', 'ADANIGREEN',
    # Metals (5)
    'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'COALINDIA',
    # FMCG (5)
    'HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'TATACONSUM',
    # Infra (5)
    'LT', 'ADANIENT', 'ADANIPORTS', 'ULTRACEMCO', 'GRASIM',
    # Others (5)
    'BHARTIARTL', 'ASIANPAINT', 'TITAN', 'HDFCLIFE', 'SBILIFE'
]

# ============================================================================
# PREDICTION TARGET
# ============================================================================
# Single Target: Next Day Closing Price
# Formula: close_price_tomorrow (actual price, not return)
# For training: predict close_return = (close_tomorrow - close_today) / close_today

TARGET_TYPE = 'close_price'  # What we ultimately want
TRAINING_TARGET = 'close_return'  # What model learns (normalized)

# Direction classification thresholds (for confusion matrix)
DIRECTION_THRESHOLDS = {
    'strong_bear': -0.015,   # < -1.5%
    'weak_bear': -0.005,     # -1.5% to -0.5%
    'neutral': 0.005,        # -0.5% to +0.5%
    'weak_bull': 0.015,      # +0.5% to +1.5%
    'strong_bull': float('inf')  # > +1.5%
}

# ============================================================================
# ROBUST MODE CONFIGURATION (Anti-Overfitting)
# ============================================================================
# Enable this when you have limited data (< 500 samples)
# This mode prevents overfitting by:
# 1. Reducing features from 118 to ~20 curated ones
# 2. Using heavy regularization on models
# 3. Requiring statistical significance in backtests
# 4. Using walk-forward cross-validation

ROBUST_MODE = True  # Set to True for statistically sound results

# Feature Selection Mode
# 'strict':     20 features max - safest, for < 600 samples
# 'moderate':   40 features max - balanced, for 600-1200 samples  
# 'aggressive': 60 features max - risky, needs 1800+ samples
# 'auto':       Automatically choose based on sample count
FEATURE_MODE = 'moderate'  # Using moderate for more features with ~700 samples

# Feature Selection (Robust Mode)
ROBUST_MAX_FEATURES = 20        # Maximum features to use (rule of thumb: samples/30)
ROBUST_MIN_SAMPLES_PER_FEATURE = 30  # Minimum samples needed per feature

# Auto-mode thresholds (samples needed for each mode)
AUTO_MODE_THRESHOLDS = {
    'strict': 0,        # Always available
    'moderate': 1200,   # Need 1200+ samples (40 × 30)
    'aggressive': 1800, # Need 1800+ samples (60 × 30)
}

# Statistical Validation
MIN_BACKTEST_TRADES = 30        # Minimum trades for statistical significance
RECOMMENDED_BACKTEST_TRADES = 100  # Recommended for reliable results
MIN_WIN_RATE_SIGNIFICANCE = 0.55   # Win rate must be significantly > 50%
SIGNIFICANCE_LEVEL = 0.05       # p-value threshold (95% confidence)

# Walk-Forward Cross-Validation
WALK_FORWARD_FOLDS = 5          # Number of CV folds
WALK_FORWARD_PURGE_DAYS = 5     # Days gap between train/test (prevent leakage)
WALK_FORWARD_EMBARGO_DAYS = 5   # Days to skip after test fold

# Model Complexity Limits (Robust Mode)
ROBUST_MODEL_PARAMS = {
    'max_depth': 3,             # Shallow trees (prevents overfitting)
    'n_estimators': 100,        # Fewer trees
    'min_samples_leaf': 50,     # Higher minimum samples per leaf
    'reg_alpha': 1.0,           # L1 regularization (10x higher)
    'reg_lambda': 5.0,          # L2 regularization (5x higher)
}

# Data Requirements
MIN_SAMPLES_FOR_TRAINING = 250  # At least 1 year of data
RECOMMENDED_SAMPLES = 750       # 3 years for reliable results

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================
# Sequence length for LSTM
SEQUENCE_LENGTH = 60  # 60 trading days (~3 months)

# Feature selection
MAX_FEATURES = 75  # Increased to accommodate fundamentals
MIN_FEATURE_CORRELATION = 0.02  # Minimum correlation with target

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
# Models to train (set to True to enable)
ENABLE_MODELS = {
    'xgboost': True,
    'lstm': True,
    'gru': True,
    'transformer': False,  # More complex, optional
    'ensemble': True
}

# Ensemble configuration
ENSEMBLE_BASE_MODELS = ['xgboost', 'lstm', 'gru']  # Models to combine
ENSEMBLE_META_LEARNER = 'ridge'  # ridge, linear, or xgboost

# XGBoost (Regression for close prediction)
XGBOOST_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 42,
    'n_jobs': -1
}

# LSTM
LSTM_PARAMS = {
    'units': [128, 64, 32],
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 8
}

# GRU (Gated Recurrent Unit - faster than LSTM, similar performance)
GRU_PARAMS = {
    'units': [128, 64, 32],
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 8
}

# Transformer (Optional - more complex)
TRANSFORMER_PARAMS = {
    'num_heads': 4,
    'd_model': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 20
}

# ============================================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================================
# Rolling window mode (recommended for production)
USE_ROLLING_WINDOW = True  # If True, uses last N years; if False, uses fixed dates
ROLLING_TRAIN_YEARS = 5    # Last 5 years for training
ROLLING_VAL_YEARS = 1      # 1 year for validation

# Fixed date mode (for backtesting)
# Training:   2015-2021 (7 years) -> 70%
# Validation: 2022-2023 (2 years) -> 20%
# Testing:    2024-2025 (1 year)  -> 10%
TRAIN_START_DATE = '2015-01-01'
TRAIN_END_DATE = '2021-12-31'
VAL_START_DATE = '2022-01-01'
VAL_END_DATE = '2023-12-31'
TEST_START_DATE = '2024-01-01'
# Test: everything after VAL_END_DATE

# Alternative: ratio-based split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# ============================================================================
# EVALUATION METRICS
# ============================================================================
# For close price prediction
TARGET_METRICS = {
    'rmse': 0.02,          # < 2% RMSE
    'mae': 0.015,          # < 1.5% MAE
    'mape': 2.0,           # < 2% MAPE
    'direction_accuracy': 0.55,  # > 55% direction correct
    'r2_score': 0.10       # > 0.10 R2 (hard for stock prediction)
}

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}'

# Pipeline run tracking
PIPELINE_LOG_FILE = os.path.join(LOGS_DIR, 'pipeline_runs.csv')
DATA_COLLECTION_LOG = os.path.join(LOGS_DIR, 'data_collection_log.csv')
TRAINING_LOG = os.path.join(LOGS_DIR, 'training_log.csv')
PREDICTION_LOG = os.path.join(LOGS_DIR, 'prediction_log.csv')

# ============================================================================
# RANDOM SEED
# ============================================================================
RANDOM_SEED = 42
import numpy as np
import random
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================================
# CREATE DIRECTORIES
# ============================================================================
ALL_DIRS = [
    RAW_DATA_DIR, MARKET_DATA_DIR, FEATURES_DIR, PROCESSED_DIR,
    XGBOOST_DIR, LSTM_DIR, GRU_DIR, TRANSFORMER_DIR, ENSEMBLE_DIR, SCALERS_DIR,
    PREDICTIONS_DIR, PLOTS_DIR, METRICS_DIR, LOGS_DIR
]

for dir_path in ALL_DIRS:
    os.makedirs(dir_path, exist_ok=True)
