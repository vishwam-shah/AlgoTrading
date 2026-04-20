"""
================================================================================
V3 PIPELINE — CENTRALIZED CONFIGURATION
================================================================================
Single source of truth for all V3 pipeline settings.
Phase 1 upgrades (Feb 2026):
  - MIN_MOVE raised to 0.004 (above round-trip transaction costs)
  - CONFIDENCE_THRESHOLD raised to 0.58 (trade less, win more)
  - Global market cues (S&P500, DXY, VIX, Crude, Nikkei, Nasdaq)
  - NSE calendar features (F&O expiry, RBI MPC, Budget, result season)
  - Temperature-scaling probability calibration
================================================================================
"""

from pathlib import Path
from datetime import datetime

# ── Root paths ────────────────────────────────────────────────────────────────
V3_ROOT = Path(__file__).parent.parent        # .../AI_IN_STOCK_V2/V3/

# ── Data paths (01_data) ──────────────────────────────────────────────────────
DATA_ROOT       = V3_ROOT / "01_data"
RAW_DATA_DIR    = DATA_ROOT / "raw"
FEATURES_DIR    = DATA_ROOT / "features"
FEAT_RAW_DIR    = FEATURES_DIR / "raw"
FEAT_SCALED_DIR = FEATURES_DIR / "scaled"

# ── Model paths (02_models) ───────────────────────────────────────────────────
MODELS_ROOT      = V3_ROOT / "02_models"
MODELS_RUNS_DIR  = MODELS_ROOT / "runs"
MODELS_PROD_DIR  = MODELS_ROOT / "production"

# ── Results paths (06_results) ────────────────────────────────────────────────
RESULTS_ROOT     = V3_ROOT / "06_results"
RESULTS_RUNS_DIR = RESULTS_ROOT / "runs"

# ── Logging paths (07_pipeline/logs) ──────────────────────────────────────────
LOG_DIR          = V3_ROOT / "07_pipeline" / "logs"

# ── All directories to auto-create on import ─────────────────────────────────
ALL_DIRS = [
    RAW_DATA_DIR, FEAT_RAW_DIR, FEAT_SCALED_DIR,
    MODELS_RUNS_DIR, MODELS_PROD_DIR, RESULTS_RUNS_DIR, LOG_DIR,
]
for _d in ALL_DIRS:
    _d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STOCK UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════

# Full Nifty-100 universe (Nifty 50 + Nifty Next 50 constituents, Feb 2026)
SYMBOLS_100: list = [
    # ── Banking & Finance (18) ───────────────────────────────────────────────
    "SBIN",       "HDFCBANK",   "ICICIBANK",  "AXISBANK",   "KOTAKBANK",
    "INDUSINDBK", "BANDHANBNK", "IDFCFIRSTB", "FEDERALBNK", "AUBANK",
    "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE",   "SBILIFE",    "ICICIGI",
    "MUTHOOTFIN", "CHOLAFIN",   "SHRIRAMFIN",
    # ── IT & Technology (10) ─────────────────────────────────────────────────
    "TCS",        "INFY",       "WIPRO",      "HCLTECH",    "TECHM",
    "LTIM",       "MPHASIS",    "PERSISTENT", "COFORGE",    "TATAELXSI",
    # ── Automobiles (7) ──────────────────────────────────────────────────────
    "MARUTI",     "M&M",        "TVSMOTOR",   "BAJAJ-AUTO", "HEROMOTOCO",
    "EICHERMOT",  "MOTHERSON",
    # ── FMCG (8) ─────────────────────────────────────────────────────────────
    "HINDUNILVR", "ITC",        "NESTLEIND",  "BRITANNIA",  "TATACONSUM",
    "MARICO",     "COLPAL",     "GODREJCP",
    # ── Pharma & Healthcare (8) ──────────────────────────────────────────────
    "SUNPHARMA",  "DRREDDY",    "CIPLA",      "DIVISLAB",   "LUPIN",
    "TORNTPHARM", "AUROPHARMA", "ALKEM",
    # ── Energy & Utilities (8) ───────────────────────────────────────────────
    "RELIANCE",   "ONGC",       "BPCL",       "NTPC",       "POWERGRID",
    "COALINDIA",  "GAIL",       "TATAPOWER",
    # ── Metals & Mining (5) ──────────────────────────────────────────────────
    "TATASTEEL",  "HINDALCO",   "JSWSTEEL",   "VEDL",       "SAIL",
    # ── Telecom (2) ──────────────────────────────────────────────────────────
    "BHARTIARTL", "INDUSTOWER",
    # ── Capital Goods & Engineering (7) ──────────────────────────────────────
    "LT",         "BHEL",       "SIEMENS",    "ABB",        "HAVELLS",
    "POLYCAB",    "CUMMINSIND",
    # ── Cement (4) ───────────────────────────────────────────────────────────
    "ULTRACEMCO", "GRASIM",     "AMBUJACEM",  "SHREECEM",
    # ── Consumer Durables & Retail (7) ───────────────────────────────────────
    "TITAN",      "ASIANPAINT", "PIDILITIND", "BERGEPAINT", "VOLTAS",
    "PAGEIND",    "DMART",
    # ── Real Estate (2) ──────────────────────────────────────────────────────
    "DLF",        "GODREJPROP",
    # ── Conglomerate, Infra & Others (14) ────────────────────────────────────
    "ADANIENT",   "ADANIPORTS", "BEL",        "HAL",        "IRFC",
    "OFSS",       "ETERNAL",    "NAUKRI",     "NMDC",       "BOSCHLTD",
    "BAJAJHFL",   "MANAPPURAM", "RBLBANK",    "EXIDEIND",
]

IT_SYMBOLS: set = {
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM",
    "LTIM", "MPHASIS", "PERSISTENT", "COFORGE", "TATAELXSI",
    "OFSS", "NAUKRI",                          # IT-adjacent (high USDINR/Nasdaq exposure)
}

BANKING_SYMBOLS: set = {
    "SBIN", "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK",
    "INDUSINDBK", "BANDHANBNK", "IDFCFIRSTB", "FEDERALBNK", "AUBANK",
    "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "ICICIGI",
    "MUTHOOTFIN", "CHOLAFIN", "SHRIRAMFIN", "MANAPPURAM", "RBLBANK",
    "BAJAJHFL",
}

SECTOR_MAP: dict = {
    # Banking & Finance
    "SBIN":       "banking",   "HDFCBANK":   "banking",   "ICICIBANK":  "banking",
    "AXISBANK":   "banking",   "KOTAKBANK":  "banking",   "INDUSINDBK": "banking",
    "BANDHANBNK": "banking",   "IDFCFIRSTB": "banking",   "FEDERALBNK": "banking",
    "AUBANK":     "banking",   "BAJFINANCE": "banking",   "BAJAJFINSV": "banking",
    "HDFCLIFE":   "banking",   "SBILIFE":    "banking",   "ICICIGI":    "banking",
    "MUTHOOTFIN": "banking",   "CHOLAFIN":   "banking",   "SHRIRAMFIN": "banking",
    "MANAPPURAM": "banking",   "RBLBANK":    "banking",   "BAJAJHFL":   "banking",
    # IT & Technology
    "TCS":        "IT",        "INFY":       "IT",        "WIPRO":      "IT",
    "HCLTECH":    "IT",        "TECHM":      "IT",        "LTIM":       "IT",
    "MPHASIS":    "IT",        "PERSISTENT": "IT",        "COFORGE":    "IT",
    "TATAELXSI":  "IT",        "OFSS":       "IT",        "NAUKRI":     "IT",
    # Automobiles
    "MARUTI":     "auto",      "M&M":        "auto",      "TVSMOTOR":   "auto",
    "BAJAJ-AUTO": "auto",      "HEROMOTOCO": "auto",      "EICHERMOT":  "auto",
    "MOTHERSON":  "auto",      "EXIDEIND":   "auto",
    # FMCG
    "HINDUNILVR": "fmcg",      "ITC":        "fmcg",      "NESTLEIND":  "fmcg",
    "BRITANNIA":  "fmcg",      "TATACONSUM": "fmcg",      "MARICO":     "fmcg",
    "COLPAL":     "fmcg",      "GODREJCP":   "fmcg",
    # Pharma
    "SUNPHARMA":  "pharma",    "DRREDDY":    "pharma",    "CIPLA":      "pharma",
    "DIVISLAB":   "pharma",    "LUPIN":      "pharma",    "TORNTPHARM": "pharma",
    "AUROPHARMA": "pharma",    "ALKEM":      "pharma",
    # Energy
    "RELIANCE":   "energy",    "ONGC":       "energy",    "BPCL":       "energy",
    "NTPC":       "energy",    "POWERGRID":  "energy",    "COALINDIA":  "energy",
    "GAIL":       "energy",    "TATAPOWER":  "energy",
    # Metals
    "TATASTEEL":  "metals",    "HINDALCO":   "metals",    "JSWSTEEL":   "metals",
    "VEDL":       "metals",    "SAIL":       "metals",    "NMDC":       "metals",
    # Telecom
    "BHARTIARTL": "telecom",   "INDUSTOWER": "telecom",
    # Capital Goods
    "LT":         "capgoods",  "BHEL":       "capgoods",  "SIEMENS":    "capgoods",
    "ABB":        "capgoods",  "HAVELLS":    "capgoods",  "POLYCAB":    "capgoods",
    "CUMMINSIND": "capgoods",  "BOSCHLTD":   "capgoods",
    # Cement
    "ULTRACEMCO": "cement",    "GRASIM":     "cement",    "AMBUJACEM":  "cement",
    "SHREECEM":   "cement",
    # Consumer Durables
    "TITAN":      "consumer",  "ASIANPAINT": "consumer",  "PIDILITIND": "consumer",
    "BERGEPAINT": "consumer",  "VOLTAS":     "consumer",  "PAGEIND":    "consumer",
    "DMART":      "consumer",
    # Real Estate
    "DLF":        "realty",    "GODREJPROP": "realty",
    # Conglomerate / Infra / Other
    "ADANIENT":   "infra",     "ADANIPORTS": "infra",     "BEL":        "defense",
    "HAL":        "defense",   "IRFC":       "infra",     "ETERNAL":    "consumer",
}

SYMBOLS = SYMBOLS_100  # Run full 100-stock universe


# ══════════════════════════════════════════════════════════════════════════════
#  DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

DATA_START_DATE = "2018-01-01"
YFINANCE_DELAY  = 0.3           # seconds between downloads (rate-limit safety)

# Global market cues — downloaded once per run (incremental)
# Saved to 01_data/raw/global_cues.parquet
GLOBAL_CUES_TICKERS: dict = {
    "sp500":      "^GSPC",      # S&P 500 (US market sentiment)
    "nasdaq":     "^IXIC",      # Nasdaq (US tech, critical for IT stocks)
    "us_vix":     "^VIX",       # CBOE VIX (fear gauge)
    "dxy":        "DX-Y.NYB",   # US Dollar Index (FII flow proxy)
    "crude":      "CL=F",       # Crude Oil (India is major importer)
    "nikkei":     "^N225",      # Nikkei 225 (Asian session cue)
    "nifty50":    "^NSEI",      # Nifty 50 index (broad Indian market)
    "niftybank":  "^NSEBANK",   # Nifty Bank index (banking sector)
}


# ══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

INITIAL_TRAIN_RATIO = 0.70
EXPANSION_STEP      = 0.05
MAX_TRAIN_RATIO     = 0.95
MIN_TRAIN_SAMPLES   = 400
MIN_TEST_SAMPLES    = 30


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL GATING  — Phase 1 upgrades
# ══════════════════════════════════════════════════════════════════════════════

# RAISED from 0.001 → 0.004 (0.4%).
# Rationale: NSE delivery round-trip cost is ~0.1-0.15%. Predicting sub-0.1%
# moves is predicting pure noise; 0.4% threshold ensures every signal covers
# transaction costs and still has meaningful edge.
MIN_MOVE = 0.004

# RAISED from 0.55 → 0.58.
# Rationale: With better features (global cues + calibration), we can afford
# to be more selective. 58% confident predictions should be right ~62-64% of
# the time after calibration. Trade less, win more.
CONFIDENCE_THRESHOLD = 0.58


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

N_TOP_FEATURES     = 50
MIN_REGIME_SAMPLES = 150

# Global cues features force-included in feature selection for ALL stocks.
# These capture 70% of next-day move drivers not in OHLCV alone.
GLOBAL_CUES_FEATURES: list = [
    "sp500_ret_prev",     # S&P 500 previous day return
    "sp500_ret_5d",       # S&P 500 5-day return (trend)
    "us_vix_level",       # US VIX level (fear gauge)
    "us_vix_zscore",      # US VIX z-score vs 20-day mean
    "us_vix_spike",       # Binary: VIX spike > 1.5 std
    "nifty50_ret_prev",   # Nifty 50 previous day return (broad market)
    "nifty50_ret_5d",     # Nifty 50 5-day trend
    "days_to_earnings",   # Days until next results announcement
    "pre_results_drift",  # Binary: 1-5 days before results (buy-the-rumour)
    "post_results_day",   # Binary: 0-2 days after results (sell-the-news)
    "earnings_proximity", # Smooth 1/(1+days) proximity score
    "dxy_ret_prev",     # Dollar Index previous day return
    "dxy_ret_5d",       # Dollar Index 5-day trend
    "crude_ret_prev",   # Crude oil previous day return
    "nikkei_ret_prev",  # Nikkei previous day return (Asian cue)
]

# Additional force-included for IT stocks (on top of GLOBAL_CUES_FEATURES)
USDINR_FEATURES: list = [
    "usdinr_ret_1d", "usdinr_ret_5d", "usdinr_ret_20d",
    "usdinr_rsi_14", "usdinr_ma20_ratio",
    "alpha_vs_usdinr_1d", "alpha_vs_usdinr_5d",
    "nasdaq_ret_prev",    # Nasdaq cue (force-include for IT)
    "nasdaq_ret_5d",
]

# Additional force-included for Banking stocks
BANKING_CUES_FEATURES: list = [
    "crude_ret_prev",      "crude_ret_5d",
    "dxy_ret_prev",        "dxy_ret_5d",
    "days_to_rbi",         "is_rbi_week",
    "niftybank_ret_prev",  "niftybank_ret_5d",   # Nifty Bank sector index
    "niftybank_ret_20d",                          # Nifty Bank 20-day trend
]


# ══════════════════════════════════════════════════════════════════════════════
#  NSE CALENDAR — RBI MPC MEETING DATES
# ══════════════════════════════════════════════════════════════════════════════
# RBI Monetary Policy Committee meets 6 times/year.
# Bank stocks move significantly on MPC days ± 2 days.
# These are announced months in advance by RBI.
RBI_MPC_DATES: list = [
    # 2021
    "2021-04-07", "2021-06-04", "2021-08-06", "2021-10-08", "2021-12-08",
    # 2022
    "2022-02-09", "2022-04-08", "2022-05-04", "2022-06-08",
    "2022-08-05", "2022-09-30", "2022-12-07",
    # 2023
    "2023-02-08", "2023-04-06", "2023-06-08",
    "2023-08-10", "2023-10-06", "2023-12-08",
    # 2024
    "2024-02-08", "2024-04-05", "2024-06-07",
    "2024-08-08", "2024-10-09", "2024-12-06",
    # 2025
    "2025-02-07", "2025-04-09", "2025-06-06",
    "2025-08-07", "2025-10-08", "2025-12-05",
    # 2026
    "2026-02-07", "2026-04-09", "2026-06-05",
    "2026-08-06", "2026-10-08", "2026-12-04",
]

# ══════════════════════════════════════════════════════════════════════════════
#  NSE CALENDAR — UNION BUDGET DATES
# ══════════════════════════════════════════════════════════════════════════════
# Budget day ± 3 days sees anomalous volatility across all sectors.
BUDGET_DATES: list = [
    "2019-02-01", "2019-07-05",   # Interim + full 2019
    "2020-02-01",
    "2021-02-01",
    "2022-02-01",
    "2023-02-01",
    "2024-02-01", "2024-07-23",   # Interim + full 2024
    "2025-02-01",
    "2026-02-01",
]

# ══════════════════════════════════════════════════════════════════════════════
#  NSE CALENDAR — RESULT SEASON
# ══════════════════════════════════════════════════════════════════════════════
# Q1 results: Jul-Aug | Q2: Oct-Nov | Q3: Jan-Feb | Q4: Apr-May
# During result season, fundamental surprises dominate over technicals.
RESULT_SEASON_MONTHS: set = {1, 2, 4, 5, 7, 8, 10, 11}


# ══════════════════════════════════════════════════════════════════════════════
#  LIGHTGBM HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

LGBM_PARAMS: dict = {
    "n_estimators":      1000,
    "max_depth":         5,
    "learning_rate":     0.01,
    "num_leaves":        31,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.3,
    "reg_lambda":        1.5,
    "min_child_samples": 20,
    "early_stopping_rounds": 50,
    "objective":         "binary",
    "metric":            "binary_logloss",
    "random_state":      42,
    "n_jobs":            -1,
    "verbosity":         -1,
}

LGBM_FS_PARAMS: dict = {
    "n_estimators":      600,
    "max_depth":         5,
    "learning_rate":     0.03,
    "num_leaves":        31,
    "subsample":         0.8,
    "colsample_bytree":  0.7,
    "reg_alpha":         0.5,
    "reg_lambda":        2.0,
    "min_child_samples": 30,
    "objective":         "binary",
    "random_state":      42,
    "n_jobs":            -1,
    "verbosity":         -1,
}


# ══════════════════════════════════════════════════════════════════════════════
#  XGBOOST HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

XGB_PARAMS: dict = {
    "n_estimators":       1000,
    "max_depth":          5,
    "learning_rate":      0.01,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.3,
    "reg_lambda":         1.5,
    "early_stopping_rounds": 50,
    "eval_metric":        "logloss",
    "objective":          "binary:logistic",
    "random_state":       42,
    "n_jobs":             -1,
    "verbosity":          0,
}


# ══════════════════════════════════════════════════════════════════════════════
#  DEEP LEARNING — SEQUENCE MODELS
# ══════════════════════════════════════════════════════════════════════════════
# All 7 models (LGBM + XGB + LSTM + BiLSTM + GRU + CNN-LSTM + CNN-GRU) always train.
# DL models use RobustScaler output (no PCA) as 3D sequences: (batch, SEQ_LEN, features)
# Tree models use PCA output as 2D arrays — preprocessing fork after scaling.

DL_SEQ_LEN     = 20    # 20 trading days = 1 calendar month (4 F&O weeks)
DL_BATCH_SIZE  = 32    # Smaller batch = noisier gradients, better generalisation
DL_MAX_EPOCHS  = 100   # Capped; EarlyStopping fires well before this

# EarlyStopping — monitor val_loss, restore best weights
DL_ES_PATIENCE  = 8    # Tight patience — small datasets converge fast
DL_ES_MIN_DELTA = 5e-5 # Finer threshold — counts smaller real improvements

# ReduceLROnPlateau — halve LR when stuck
DL_RLROP_FACTOR   = 0.5   # Multiply LR by this on plateau
DL_RLROP_PATIENCE = 8     # Raised from 5
DL_RLROP_MIN_LR   = 1e-5  # Floor on learning rate

# ── LSTM ─────────────────────────────────────────────────────────────────────
# 2-layer stacked LSTM. Units 64→32 halving prevents memorization.
# recurrent_dropout=0.2 injects noise at each time step (inside BPTT).
LSTM_PARAMS: dict = {
    "units_1":           64,
    "units_2":           32,
    "dropout":           0.3,
    "recurrent_dropout": 0.2,
    "l2":                1e-4,
    "dense_units":       32,
    "learning_rate":     1e-3,
}

# ── BiLSTM ────────────────────────────────────────────────────────────────────
# Units halved (32/16 per direction) so merged output (64/32) matches LSTM size.
# Bidirectional is safe: all features are lagged (no future target in feature values).
BILSTM_PARAMS: dict = {
    "units_1":           32,   # per direction → 64 merged after concat
    "units_2":           16,   # per direction → 32 merged
    "dropout":           0.3,
    "recurrent_dropout": 0.2,
    "l2":                1e-4,
    "dense_units":       32,
    "learning_rate":     1e-3,
}

# ── GRU ───────────────────────────────────────────────────────────────────────
# Same topology as LSTM. GRU trains faster (fewer params), similar accuracy.
# Acts as diversity partner — when LSTM overfits, GRU often does not.
GRU_PARAMS: dict = {
    "units_1":           64,
    "units_2":           32,
    "dropout":           0.3,
    "recurrent_dropout": 0.2,
    "l2":                1e-4,
    "dense_units":       32,
    "learning_rate":     1e-3,
}

# ── 1D-CNN + LSTM ─────────────────────────────────────────────────────────────
# Causal Conv1D extracts local patterns (3-day candlestick shapes, week formations).
# MaxPool(2) compresses 20→10 steps before LSTM; reduces LSTM compute.
# padding='causal' guarantees no future information leak through convolution.
CNN_LSTM_PARAMS: dict = {
    "cnn_filters":       64,
    "cnn_kernel_size":   3,
    "pool_size":         2,
    "lstm_units":        32,
    "dropout":           0.3,
    "recurrent_dropout": 0.2,
    "l2":                1e-4,
    "dense_units":       32,
    "learning_rate":     1e-3,
}

# ── 1D-CNN + GRU ──────────────────────────────────────────────────────────────
# Identical CNN backbone to CNN-LSTM; GRU at the recurrent stage.
# Provides distinct inductive bias with same CNN feature extractor structure.
CNN_GRU_PARAMS: dict = {
    "cnn_filters":       64,
    "cnn_kernel_size":   3,
    "pool_size":         2,
    "gru_units":         32,
    "dropout":           0.3,
    "recurrent_dropout": 0.2,
    "l2":                1e-4,
    "dense_units":       32,
    "learning_rate":     1e-3,
}

# ── Temporal CNN + GRU ────────────────────────────────────────────────────────
# Dilated causal convolutions (TCN) + GRU temporal compression.
# Dilations [1,2,4,8]: receptive field = 1 + (kernel-1)*2*sum(dilations) = 61 steps.
# Captures multi-scale local patterns before GRU sequential aggregation.
# Ref: Bai et al. 2018 "An Empirical Evaluation of Generic Conv/Recurrent Networks"
TCN_GRU_PARAMS: dict = {
    "filters":           64,
    "kernel_size":       3,
    "dilations":         [1, 2, 4, 8],
    "gru_units":         32,
    "dropout":           0.3,
    "recurrent_dropout": 0.2,
    "l2":                1e-4,
    "dense_units":       32,
    "learning_rate":     1e-3,
}

# ── Temporal CNN + Transformer ─────────────────────────────────────────────────
# TCN local feature extraction + Multi-Head Self-Attention for global dependencies.
# Inspired by TFT (Lim et al. 2021): combine local recurrent/conv with attention.
# d_model=64, 4 heads × key_dim=16. No causal mask (features already lagged).
TCN_TRANSFORMER_PARAMS: dict = {
    "d_model":       64,
    "kernel_size":   3,
    "dilations":     [1, 2, 4],
    "num_heads":     4,
    "dropout":       0.2,
    "l2":            1e-4,
    "dense_units":   32,
    "learning_rate": 1e-3,
}

# ── N-BEATS Generic (adapted for binary classification) ───────────────────────
# Residual backcast blocks: each block reconstructs (explains away) its input,
# passes only the unexplained residual to the next block.
# Ref: Oreshkin et al. 2019 ICLR — 11% over statistical benchmarks on M4.
# Adaptation: flattened multivariate input, sigmoid classification head.
NBEATS_PARAMS: dict = {
    "n_blocks":      3,      # Reduced from 4 — fewer but functional blocks
    "n_layers":      4,      # FC layers per block (matches paper generic config)
    "fc_dim":        512,    # Raised from 256 — capacity to model projected 256-dim space
    "proj_dim":      256,    # Input projection: (seq*feat) → 256 before blocks
    "forecast_dim":  64,     # classification representation size per block
    "dense_units":   64,     # classification head hidden size
    "dropout":       0.3,
    "l2":            1e-4,
    "learning_rate": 5e-4,   # lower LR: N-BEATS has more parameters
}


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

PLOT_DPI           = 120
PLOT_STYLE         = "seaborn-v0_8-darkgrid"
PLOT_FIGSIZE_WIDE  = (14, 5)
PLOT_FIGSIZE_TALL  = (10, 8)
PLOT_FIGSIZE_SQUARE = (8, 6)


# ══════════════════════════════════════════════════════════════════════════════
#  MISC
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42
