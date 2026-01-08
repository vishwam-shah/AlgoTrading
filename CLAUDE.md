# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Stock Prediction System - Multi-Task Deep Learning for NSE Stock Market Prediction. Research-grade implementation of a 4-target prediction system for 100 NSE stocks using XGBoost, LSTM, and Transformer models.

## Common Commands

### Running the Pipeline

```bash
# Phase 1: Data collection (10 proof-of-concept stocks)
python main.py --phase 1 --step data

# Phase 1: Feature engineering
python main.py --phase 1 --step features

# Phase 1: Train models
python main.py --phase 1 --step train

# Phase 1: Complete pipeline (all steps)
python main.py --phase 1 --step all

# Phase 2: Scale to 100 stocks
python main.py --phase 2 --step all
```

### Testing Individual Modules

```bash
# Test data collection
python src/data_collection.py

# Test target computation
python src/target_computation.py

# Test feature engineering
python src/feature_engineering.py

# Test model architectures
python src/models.py
```

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Architecture Overview

### 4-Target Multi-Task Learning System

This system predicts 4 targets simultaneously:

1. **Direction** (5-class classification): Strong Bear / Weak Bear / Neutral / Weak Bull / Strong Bull
2. **Intraday High** (regression): log(high_tomorrow / open_tomorrow) - for profit targets
3. **Intraday Low** (regression): log(low_tomorrow / open_tomorrow) - for stop loss
4. **Closing Price** (regression): log(close_tomorrow / close_today) - for swing decisions

All regression targets use log returns for normalization and stationarity.

### Core Architecture Patterns

**Data Pipeline Flow:**
1. `DataCollector` (data_collection.py) → Downloads OHLCV data from yfinance
2. `FeatureEngineer` (feature_engineering.py) → Computes 202 features across 10 categories
3. `TargetComputer` (target_computation.py) → Computes 4 targets with consistency validation
4. Models (models.py) → XGBoost (4 separate), LSTM multi-task, Transformer multi-task
5. Ensemble (ensemble.py) → Stacking meta-learners combine base models

**Model Architecture:**
- **XGBoostMultiTask**: 4 separate XGBoost models (1 classifier for direction, 3 regressors)
- **LSTMMultiTask**: Single LSTM backbone with attention, 4 output heads
- **TransformerMultiTask**: Transformer encoder with positional encoding, 4 output heads

All deep learning models:
- Take sequences of length 60 (configurable via `config.SEQUENCE_LENGTH`)
- Use StandardScaler for feature normalization
- Share backbone architecture with task-specific heads
- Trained with weighted multi-task loss (weights in `config.LOSS_WEIGHTS`)

### Configuration System

All parameters centralized in `config.py`:
- **Stock universe**: `SECTORS` dict organizes 100 stocks across 13 sectors
- **Phase control**: `PHASE1_STOCKS` (10 stocks) vs `ALL_STOCKS` (100 stocks)
- **Model hyperparameters**: `XGBOOST_PARAMS`, `LSTM_PARAMS`, `TRANSFORMER_PARAMS`
- **Target thresholds**: `DIRECTION_THRESHOLDS` for 5-class classification
- **Trading strategy**: `MIN_CONFIDENCE`, `RISK_PER_TRADE`, `MAX_POSITIONS`
- **Paths**: All data/model/results directories defined at top

Never hardcode parameters - always reference `config.py`.

### Phased Development Approach

**Phase 1** (Current): Proof of concept with 10 stocks (5 Banking + 5 IT)
- Goal: Validate approach works (Direction >50%, Close >58%)
- Models: XGBoost + LSTM only
- Set via `CURRENT_PHASE = 1` in config.py

**Phase 2**: Scale to all 100 stocks across 13 sectors
- Sector-based model training
- Add Transformer models
- Implement ensemble stacking

**Phase 3**: Backtesting & optimization
- Walk-forward validation (2020-2024)
- Test 3 strategies (intraday, swing, hybrid)
- Portfolio-level backtesting

**Phase 4**: Research paper & live trading

### Data Structure Conventions

**Raw OHLCV DataFrames** must have columns:
- `timestamp` (datetime): Trading date
- `open`, `high`, `low`, `close` (float): Price data
- `volume` (int): Trading volume
- `symbol` (str): Stock symbol

**Feature DataFrames** add 200+ computed feature columns while preserving OHLCV.

**Target DataFrames** add these columns:
- `direction_target` (int 0-4): 5-class direction
- `high_target` (float): Log return from open to high
- `low_target` (float): Log return from open to low
- `close_target` (float): Log return close-to-close
- `next_day_log_return` (float): Intermediate computation

### Target Computation Details

The `TargetComputer` class computes forward-looking targets:
- Uses `.shift(-1)` to get tomorrow's values
- Computes log returns: `np.log(price_t+1 / price_t)`
- Direction classes based on `DIRECTION_THRESHOLDS` from config
- **Always validate consistency**: Use `check_target_consistency()` to ensure:
  - High > Low (100% of time)
  - High ≥ 0 (relative to open)
  - Low ≤ 0 (relative to open)
  - Direction-Close agreement for extreme moves

### Feature Engineering Patterns

The `FeatureEngineer` computes 202 features in 10 categories:
1. Price-based (20): Returns, momentum, gaps, MA ratios
2. Technical indicators (50): MACD, RSI, Bollinger Bands, ADX, Stochastic
3. Volume indicators (15): OBV, MFI, VWAP, volume momentum
4. Intraday range (15): Candlestick patterns, shadows, range metrics
5. Temporal (15): Day/week/month features with cyclic encoding
6. Volatility (20): Historical vol, Parkinson, Garman-Klass estimators
7. Momentum (25): Multi-period momentum, acceleration, divergence
8. Statistical (20): Skew, kurtosis, z-scores, autocorrelation
9. Regime (10): Trend/volatility/volume regimes, regime persistence
10. Interaction (12): Non-linear feature combinations

**Technical Analysis Library:**
- Primary: `ta` library (pure Python, no compilation needed)
- Fallback: Manual pandas/numpy implementations
- Never use `ta-lib` (requires C++ compiler, commented out in requirements)

Feature computation creates NaN values due to rolling windows - always call `.dropna()` after computing features.

### Model Training Conventions

**XGBoost Models:**
- Direction: `objective='multi:softmax'`, `num_class=5`, `eval_metric='mlogloss'`
- Regression: `objective='reg:squarederror'`, `eval_metric='rmse'`
- All use early stopping with 50 rounds patience
- Feature importance tracked via `get_score(importance_type='gain')`

**Deep Learning Models (LSTM/Transformer):**
- Create sequences using `create_sequences()` method (drops first `sequence_length-1` samples)
- Standardize features with `StandardScaler` (fit on train, transform on val/test)
- Multi-task loss: Sparse categorical crossentropy (direction) + MSE (high/low/close)
- Callbacks: EarlyStopping (patience=20), ReduceLROnPlateau (patience=10)
- Save scaler alongside model for inference

**Training/Validation Split:**
- Walk-forward validation preferred: `TRAIN_PERIOD=504`, `VAL_PERIOD=63`, `TEST_PERIOD=63`
- Alternative: `TRAIN_RATIO=0.7`, `VAL_RATIO=0.15`, `TEST_RATIO=0.15`

### Logging System

Uses `loguru` for structured logging:
- Each module has dedicated log file in `logs/` directory (see `config.LOG_FILES`)
- Format: `{time} | {level} | {name}:{function}:{line} | {message}`
- Data downloads logged to `logs/data_downloads.csv` for tracking
- Always log at INFO level for pipeline steps, DEBUG for detailed computations

### Data Source Strategy

**Primary**: yfinance (Yahoo Finance)
- Symbols: Add `.NS` suffix for NSE stocks (e.g., `TCS.NS`)
- Handles multi-index columns automatically
- More reliable than nsepy

**Alternative**: nsepy (optional, often has issues)
- Use only if yfinance fails
- Check `NSEPY_AVAILABLE` flag before using

**Date Range**: `START_DATE='2019-01-01'` to `END_DATE=today` (configurable in config.py)

## Important Implementation Notes

### When Working with Stock Data

1. **Always check for missing values** after operations that might introduce NaN (rolling windows, shifts)
2. **Validate data quality** using `DataCollector.check_data_quality()` before using data
3. **Handle symbol formatting**: Remove `.NS` suffix when saving to ensure clean filenames
4. **Log all downloads** via `_log_download()` to track data provenance

### When Adding Features

1. **Update feature count** if adding new categories (currently 202 total)
2. **Test on sample data** before running on full dataset (use `main()` function in module)
3. **Store feature names** in `FeatureEngineer.feature_names` for downstream selection
4. **Exclude OHLCV and target columns** from feature list

### When Training Models

1. **Always use validation set** to detect overfitting early
2. **Track all metrics** in training history for analysis
3. **Save models** immediately after training to avoid data loss
4. **Check predictions** for sanity (e.g., high > low, reasonable ranges)

### Target Consistency Validation

Always run `TargetComputer.check_target_consistency()` after computing targets. Expected:
- `high_positive_rate ≈ 1.0` (high should always be ≥ open)
- `low_negative_rate ≈ 1.0` (low should always be ≤ open)
- `high_gt_low_rate = 1.0` (high must always > low)
- `strong_bull_close_agreement > 0.9` (strong bullish moves should have positive close)

If any consistency check fails, investigate data quality issues.

### Model Output Interpretation

**Direction predictions** (0-4 integer):
- 0 = Strong Bear (<-1.5%)
- 1 = Weak Bear (-1.5% to -0.5%)
- 2 = Neutral (-0.5% to +0.5%)
- 3 = Weak Bull (+0.5% to +1.5%)
- 4 = Strong Bull (>+1.5%)

**Regression predictions** (float, log returns):
- Convert to percentage: `(exp(pred) - 1) * 100`
- High/Low are relative to tomorrow's open
- Close is relative to today's close

### Performance Expectations

Realistic targets (defined in `config.TARGET_METRICS`):
- Direction accuracy: 50%+ (vs 20% random for 5-class)
- Close directional accuracy: 58%+ (vs 50% random for 2-class)
- High/Low/Close MAE: 1.2-1.5% (reasonable prediction error)
- Sharpe ratio: 1.5+
- Annual return: 22%+
- Max drawdown: <20%

Do not expect or claim unrealistic performance (e.g., 80% accuracy, 100% returns).

## Module-Specific Guidelines

### data_collection.py
- Use `download_phase1_stocks()` for Phase 1, `download_all_stocks()` for Phase 2
- Always add delay between downloads to avoid rate limiting (default 1 second)
- Save raw data to `config.RAW_DATA_DIR` as `{symbol}.csv`

### feature_engineering.py
- Compute all features via `compute_all_features()` which calls all private `_compute_*` methods
- Feature selection via `select_features()` - supports correlation, variance, and mutual_info methods
- Always call `.dropna()` to remove NaN rows from rolling windows

### target_computation.py
- Compute all 4 targets together via `compute_all_targets()`
- Use `check_target_consistency()` to validate logical relationships
- Use `get_target_statistics()` to understand target distributions

### models.py
- XGBoost: Train 4 models independently, combine predictions at inference
- LSTM/Transformer: Single model with 4 output heads, joint training with weighted loss
- Sequence models need `create_sequences()` preprocessing step
- Always save scaler with deep learning models for deployment

### main.py
- Entry point for all pipeline operations
- Use `--phase` to select development phase (1-4)
- Use `--step` to run specific steps or 'all' for complete pipeline
- Orchestrates all modules in correct order

## Current Development Status

**Completed:**
- Project infrastructure and configuration
- Data collection module (tested)
- Target computation module (tested, 100% consistency)
- Feature engineering module (implemented, 202 features)
- Model architectures (XGBoost, LSTM, Transformer implemented)
- Main pipeline orchestrator

**In Progress:**
- Training pipeline integration
- End-to-end testing on Phase 1 stocks

**Pending:**
- Ensemble stacking
- Backtesting engine
- Walk-forward validation
- Performance evaluation
- Research paper documentation

See `PROGRESS.md` for detailed status tracking.

## Research Context

This is a research project aimed at publication. Code should be:
- **Reproducible**: Fixed random seeds (`RANDOM_SEED=42`)
- **Well-documented**: Comprehensive logging and comments
- **Statistically sound**: Proper validation, significance testing
- **Realistic**: Conservative performance expectations

Reference documents in repository:
- `RESEARCH_PLAN.md`: Research methodology and validation approach
- `ENSEMBLE_TRUTH.md`: Ensemble stacking strategy with literature review
- `100_STOCKS_STRATEGY.md`: Stock universe and sector modeling approach
- `REVISED_STRATEGY_DIRECTION_INTRADAY.md`: Direction classification and target strategy
