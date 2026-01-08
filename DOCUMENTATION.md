# AI Stock Prediction System - Complete Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Quick Start](#2-quick-start)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Data Collection](#4-data-collection)
5. [Feature Engineering](#5-feature-engineering)
6. [Target Computation](#6-target-computation)
7. [Model Architecture](#7-model-architecture)
8. [Training Process](#8-training-process)
9. [Prediction System](#9-prediction-system)
10. [Configuration Reference](#10-configuration-reference)
11. [Results Interpretation](#11-results-interpretation)
12. [Performance Metrics](#12-performance-metrics)

---

## 1. Project Overview

### What This System Does
Multi-task deep learning system for NSE (National Stock Exchange of India) stock prediction. The system predicts 4 targets simultaneously for each stock:

1. **Direction** - 5-class classification (Strong Bear to Strong Bull)
2. **Intraday High** - Maximum potential gain from opening price
3. **Intraday Low** - Maximum potential loss from opening price
4. **Closing Price** - End-of-day price change

### Stock Universe
- **Phase 1 (Proof of Concept)**: 10 stocks (5 Banking + 5 IT)
- **Phase 2 (Full Scale)**: 100 stocks across 13 sectors

### Phase 1 Stocks
| Sector | Stocks |
|--------|--------|
| Banking/Finance | HDFCBANK, ICICIBANK, KOTAKBANK, SBIN, BAJFINANCE |
| IT Services | TCS, INFY, HCLTECH, WIPRO, TECHM |

---

## 2. Quick Start

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
# Single stock (e.g., TCS)
python pipeline/run_pipeline.py --symbol TCS

# Phase 1 stocks (10 stocks)
python pipeline/run_pipeline.py

# All 100 stocks
python pipeline/run_pipeline.py --all
```

### Run Individual Steps
```bash
# Step 1: Download data
python pipeline/01_data_collection.py --symbol TCS

# Step 2: Compute features
python pipeline/02_feature_engineering.py --symbol TCS

# Step 3: Train models
python pipeline/03_train_models.py --symbol TCS

# Step 4: Generate predictions
python pipeline/04_predict.py --symbol TCS
```

---

## 3. Pipeline Architecture

### Serial Execution Flow
```
01_data_collection.py  →  02_feature_engineering.py  →  03_train_models.py  →  04_predict.py
        ↓                         ↓                          ↓                      ↓
   data/raw/*.csv         data/features/*.csv         models/*/*.json       predictions/*.csv
```

### Directory Structure
```
AI_IN_STOCK_V2/
├── pipeline/                    # Main pipeline scripts
│   ├── run_pipeline.py         # Master pipeline runner
│   ├── 01_data_collection.py   # Download OHLCV data
│   ├── 02_feature_engineering.py # Compute features + targets
│   ├── 03_train_models.py      # Train XGBoost + LSTM
│   └── 04_predict.py           # Generate predictions
│
├── config.py                   # Centralized configuration
│
├── data/
│   ├── raw/                    # Downloaded OHLCV data
│   └── features/               # Processed features + targets
│
├── models/
│   ├── xgboost/{symbol}/       # XGBoost models (4 per stock)
│   └── lstm/{symbol}/          # LSTM models + scalers
│
├── results/
│   ├── predictions/            # Daily predictions
│   └── training_results.csv    # Model performance metrics
│
└── logs/                       # Pipeline execution logs
```

---

## 4. Data Collection

### Source
- **Primary**: Yahoo Finance (via `yfinance` library)
- **Symbol Format**: `{SYMBOL}.NS` (e.g., `TCS.NS`)

### Date Range
- **Start Date**: 2019-01-01
- **End Date**: Current date

### Data Columns
| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Trading date |
| open | float | Opening price |
| high | float | Highest price of day |
| low | float | Lowest price of day |
| close | float | Closing price |
| volume | int | Trading volume |
| symbol | str | Stock symbol |

### Data Quality Checks
- Missing values detection
- Duplicate date detection
- Zero volume days detection
- Negative price detection

---

## 5. Feature Engineering

### Feature Selection Strategy

**Why not use all features?**
- 202 features with ~1400 samples = overfitting risk
- Many features are highly correlated (RSI_7 vs RSI_14)
- NaN values from different lookback periods cause data loss
- LSTM struggles with high-dimensional input

**Our Approach:**
1. Compute ~80 candidate features
2. Remove features with >5% NaN values
3. Remove redundant features (>85% correlation)
4. Select top 50 by correlation with close_target
5. Final: 50 clean features per stock

### Feature Categories (80 Candidates → 50 Selected)

#### 1. Returns (6 features)
- return_1d, return_5d, return_10d, return_20d
- log_return_1d, log_return_5d

#### 2. Price Position (5 features)
- price_position (where in high-low range)
- gap, intraday_return
- high_low_range, close_to_high

#### 3. Moving Average Ratios (10 features)
- price_to_sma10, price_to_sma20, price_to_sma50
- sma10_to_sma20, sma20_to_sma50

#### 4. Momentum Indicators (12 features)
- rsi_14, rsi_oversold, rsi_overbought
- macd, macd_signal, macd_histogram, macd_cross_above
- roc_10, roc_20

#### 5. Volatility (10 features)
- volatility_10, volatility_20
- atr_pct, bb_width, bb_position
- vol_ratio_10_20

#### 6. Volume Indicators (10 features)
- volume_ratio, volume_change
- obv_slope, price_volume_trend
- volume_price_confirm

#### 7. Trend Indicators (8 features)
- trend_up, trend_strength
- higher_high_5d, lower_low_5d
- positive_days_10, positive_days_20

#### 8. Candlestick Patterns (8 features)
- body_size, upper_shadow, lower_shadow
- is_bullish, is_doji
- bullish_streak, bearish_streak, body_to_range

#### 9. Statistical Features (8 features)
- return_skew_20, return_kurt_20
- zscore_20, mean_reversion_20
- dist_from_high_20, dist_from_low_20
- percentile_20, percentile_50

#### 10. Temporal Features (6 features)
- day_of_week, day_of_month, month
- is_month_end, is_month_start, day_sin

---

## 6. Target Computation

### Target 1: Direction (5-class Classification)
Based on next-day log return: `log(close_tomorrow / close_today)`

| Class | Label | Threshold |
|-------|-------|-----------|
| 0 | STRONG BEAR | < -1.5% |
| 1 | WEAK BEAR | -1.5% to -0.5% |
| 2 | NEUTRAL | -0.5% to +0.5% |
| 3 | WEAK BULL | +0.5% to +1.5% |
| 4 | STRONG BULL | > +1.5% |

### Target 2: Intraday High (Regression)
```
high_target = log(high_tomorrow / open_tomorrow)
```
- Always positive (high >= open)
- Represents maximum potential profit from buy at open

### Target 3: Intraday Low (Regression)
```
low_target = log(low_tomorrow / open_tomorrow)
```
- Always negative (low <= open)
- Represents maximum potential loss (stop loss level)

### Target 4: Close (Regression)
```
close_target = log(close_tomorrow / close_today)
```
- Can be positive or negative
- Represents end-of-day return

### Target Consistency Validation
- High target should be >= 0 (100% of time)
- Low target should be <= 0 (100% of time)
- High target > Low target (always)
- Strong Bull direction should correlate with positive close

---

## 7. Model Architecture

### Model 1: XGBoost Multi-Task (4 Separate Models)

#### Direction Model (Classifier)
```
Objective: multi:softmax
Classes: 5
Evaluation: mlogloss
```

#### High/Low/Close Models (Regressors)
```
Objective: reg:squarederror
Evaluation: rmse
```

#### XGBoost Hyperparameters
| Parameter | Value |
|-----------|-------|
| n_estimators | 1000 |
| max_depth | 5 |
| learning_rate | 0.01 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| min_child_weight | 5 |
| gamma | 0.1 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |
| early_stopping | 50 rounds |

### Model 2: LSTM Multi-Task (Single Model, 4 Heads)

#### Architecture
```
Input: (batch, 60, 202)  # 60 days sequence, 202 features
    ↓
LSTM Layer 1: 128 units, return_sequences=True
Dropout: 0.3
    ↓
LSTM Layer 2: 64 units, return_sequences=True
Dropout: 0.3
    ↓
Attention Mechanism
    ↓
Global Average Pooling
    ↓
Shared Dense: 64 units, ReLU
Dropout: 0.2
    ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Direction   │ High        │ Low         │ Close       │
│ Dense(32)   │ Dense(32)   │ Dense(32)   │ Dense(32)   │
│ Softmax(5)  │ Linear(1)   │ Linear(1)   │ Linear(1)   │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

#### LSTM Hyperparameters
| Parameter | Value |
|-----------|-------|
| Sequence Length | 60 days |
| LSTM Units | 128 → 64 |
| Dropout | 0.3 |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Max Epochs | 100 |
| Early Stopping | 20 epochs |

### Multi-Task Loss
```python
loss = 1.0 * direction_loss + 0.4 * high_loss + 0.4 * low_loss + 0.6 * close_loss
```

| Target | Loss Function | Weight |
|--------|---------------|--------|
| Direction | Sparse Categorical Crossentropy | 1.0 |
| High | MSE | 0.4 |
| Low | MSE | 0.4 |
| Close | MSE | 0.6 |

---

## 8. Training Process

### Data Split
- **Training**: 70%
- **Validation**: 30%
- **Method**: Chronological split (no shuffle to preserve time order)

### XGBoost Training
1. Train direction classifier with early stopping
2. Train high regressor with early stopping
3. Train low regressor with early stopping
4. Train close regressor with early stopping
5. Save 4 separate models per stock

### LSTM Training
1. Scale features with StandardScaler
2. Create sequences of 60 timesteps
3. Train single model with 4 output heads
4. Use early stopping and learning rate reduction
5. Save model + scaler per stock

### Model Saving
```
models/
├── xgboost/{symbol}/
│   ├── xgboost_direction.json
│   ├── xgboost_high.json
│   ├── xgboost_low.json
│   └── xgboost_close.json
│
└── lstm/{symbol}/
    ├── lstm_model.keras
    └── scaler.pkl
```

---

## 9. Prediction System

### Prediction Modes
1. **XGBoost Only**: Use XGBoost predictions
2. **LSTM Only**: Use LSTM predictions
3. **Ensemble**: Average XGBoost and LSTM (default)

### Ensemble Logic
- **Direction**: Voting (mode of predictions)
- **Regression**: Average of predictions

### Converting Log Returns to Prices
```python
# Log return to percentage
pct_change = (exp(log_return) - 1) * 100

# Predicted prices
predicted_high = current_close * exp(high_log_return)
predicted_low = current_close * exp(low_log_return)
predicted_close = current_close * exp(close_log_return)
```

### Output Format
```
Symbol: TCS
Direction: STRONG BULL [++]
High Target: +1.52% -> Rs.4125.50
Low Target: -0.73% -> Rs.4034.20
Close: +0.89% -> Rs.4099.75
```

---

## 10. Configuration Reference

### config.py Parameters

#### Data Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| START_DATE | 2019-01-01 | Data start date |
| END_DATE | Today | Data end date |
| DATA_SOURCE | yfinance | Data provider |

#### Feature Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| SEQUENCE_LENGTH | 60 | Days for LSTM sequence |
| N_FEATURES | 100 | Features after selection |

#### Direction Thresholds
| Threshold | Value | Description |
|-----------|-------|-------------|
| strong_bear | -0.015 | < -1.5% |
| weak_bear | -0.005 | -1.5% to -0.5% |
| neutral | 0.005 | -0.5% to +0.5% |
| weak_bull | 0.015 | +0.5% to +1.5% |
| strong_bull | +inf | > +1.5% |

#### Training Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| TRAIN_RATIO | 0.7 | Training data percentage |
| VAL_RATIO | 0.15 | Validation data percentage |
| TEST_RATIO | 0.15 | Test data percentage |
| RANDOM_SEED | 42 | Random seed for reproducibility |

---

## 11. Results Interpretation

### Direction Predictions
| Direction | Trading Action |
|-----------|----------------|
| STRONG BULL | Aggressive buy, full position |
| WEAK BULL | Moderate buy, partial position |
| NEUTRAL | No trade, wait for clarity |
| WEAK BEAR | Reduce positions, caution |
| STRONG BEAR | Exit longs, consider shorts |

### Using High/Low Predictions
```
For BULLISH trades:
  Entry: At market open
  Target: Predicted High price
  Stop Loss: Predicted Low price

For BEARISH trades:
  Entry: At market open (short)
  Target: Predicted Low price
  Stop Loss: Predicted High price
```

### Risk-Reward Calculation
```python
risk = abs(low_pct)      # Potential loss
reward = abs(high_pct)   # Potential gain
rr_ratio = reward / risk # Should be > 1.5 for trade
```

---

## 12. Performance Metrics

### Target Metrics
| Metric | Target | Description |
|--------|--------|-------------|
| Direction Accuracy | > 50% | vs 20% random baseline |
| Close Directional Acc | > 58% | vs 50% random baseline |
| High MAE | < 1.5% | Mean Absolute Error |
| Low MAE | < 1.3% | Mean Absolute Error |
| Close MAE | < 1.4% | Mean Absolute Error |

### Evaluation Metrics
```python
# Direction
accuracy = correct_predictions / total_predictions

# Regression
mae = mean(abs(predicted - actual))
mse = mean((predicted - actual)^2)
rmse = sqrt(mse)
```

### Training Results Output
```
XGBoost Results:
Symbol      Dir Acc    High MAE   Low MAE    Close MAE
TCS          32.44%      0.0088     0.0087      0.0131
INFY         31.77%      0.0094     0.0092      0.0139
...
AVERAGE      31.50%      0.0091     0.0089      0.0135

LSTM Results:
Symbol      Dir Acc    High MAE   Low MAE    Close MAE
TCS          28.12%      0.0092     0.0091      0.0142
...
```

---

## Appendix A: Dependencies

```
# Core
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.10.0

# Technical Analysis
ta>=0.10.0

# Data
yfinance>=0.1.70

# Utilities
loguru>=0.6.0
tqdm>=4.62.0
joblib>=1.1.0
```

---

## Appendix B: Troubleshooting

### Common Issues

**1. No data downloaded**
```
Solution: Check internet connection, verify symbol exists on NSE
```

**2. Feature file not found**
```
Solution: Run data collection before feature engineering
```

**3. Model not found**
```
Solution: Run training before prediction
```

**4. LSTM memory error**
```
Solution: Reduce batch size or sequence length
```

**5. yfinance rate limiting**
```
Solution: Increase delay between downloads (default: 1 second)
```

---

## Appendix C: Command Reference

```bash
# Full pipeline
python pipeline/run_pipeline.py --symbol TCS
python pipeline/run_pipeline.py                    # Phase 1 (10 stocks)
python pipeline/run_pipeline.py --all              # All 100 stocks

# Individual steps
python pipeline/01_data_collection.py --symbol TCS
python pipeline/02_feature_engineering.py --symbol TCS
python pipeline/03_train_models.py --symbol TCS --model xgboost
python pipeline/03_train_models.py --symbol TCS --model lstm
python pipeline/04_predict.py --symbol TCS --model ensemble

# Batch processing
python pipeline/run_pipeline.py --symbols TCS,INFY,HDFCBANK
```

---

*Documentation Version: 1.0*
*Last Updated: December 2024*
*AI Stock Prediction System*
