# V3 AlgoTrading Pipeline: Technical Analysis

## 1. Executive Summary

This project is a full end-to-end quantitative equity trading pipeline for Indian large-cap stocks, built around a zero-lookahead walk-forward validation framework. It downloads daily NSE OHLCV data plus macro/global context, engineers a large technical and cross-market feature set, trains a per-stock ensemble model, calibrates probabilities, generates next-day signals, and simulates a live trading workflow with order generation, risk checks, and execution helpers.

At a high level:

- Universe: Nifty-100 style large-cap Indian stocks
- Prediction objective: binary directional classification
- Primary target: whether the 5-day forward return is above +1% or below -1%
- Core philosophy: trade only high-confidence directional predictions and then hold for a fixed 10-trading-day horizon
- Validation style: expanding-window walk-forward, per stock
- Signal output: next-day `UP` / `DOWN` direction, confidence, action, and range estimate

Important clarification:

- The pipeline uses data available through trading day `T`.
- The primary model output is a prediction for trading day `T+1` direction.
- The displayed predicted price, estimated close, and range are secondary ATR-based estimates layered on top of the directional probability.
- The active production pipeline is not a dedicated next-day closing-price regression model.

Current verdict:

- Research quality: promising
- Backtest profitability: yes, on the filtered tradeable subset
- Live-trading readiness: not yet
- Publication readiness: not yet

Why this is the current verdict:

- The current research results are strong enough to justify continued development.
- The current backtest shows strong positive performance on a filtered subset of stocks.
- However, there are still important methodology and execution gaps between backtest and actual live trading.
- Those gaps are large enough that the system should not yet be presented as production-profitable or publication-grade without additional controls and validation.

## 2. What the Pipeline Is Trying to Do

The pipeline is not trying to predict exact prices directly. Its main job is:

1. Learn whether a stock is likely to move meaningfully up or down over the next multi-day horizon.
2. Convert that into a next-day trade decision.
3. Filter those decisions down to only the strongest candidates.
4. Rank and size the positions.
5. Measure whether the resulting strategy is profitable after costs.

In other words, the system is a directional signal engine plus a trade-selection and risk-filtering framework.

The clean mental model is:

- `T close` is part of the input data
- `T+1 direction` is the true model prediction
- `T+1 price range / estimated close` is a derived estimate for interpretation

## 3. Full Architecture

The active code path is centered on:

- `v3/07_pipeline/orchestrator.py`
- `v3/07_pipeline/steps/download.py`
- `v3/07_pipeline/steps/features.py`
- `v3/07_pipeline/steps/train.py`
- `v3/07_pipeline/steps/evaluate.py`
- `v3/07_pipeline/steps/predict.py`
- `v3/07_pipeline/steps/backtest.py`
- `v3/05_live_trading/*`
- `backend/main.py`
- `frontend/src/*`

End-to-end flow:

1. Download raw market data.
2. Build cached engineered features.
3. Add the training target.
4. Select the top features for each stock.
5. Train multiple models in walk-forward windows.
6. Build an ensemble from the model probabilities.
7. Calibrate the final probabilities.
8. Save per-window and per-stock outputs.
9. Generate next-day prediction signals.
10. Run a rule-based backtest on those signals.
11. Publish signals to the dashboard and CSV outputs.
12. Optionally convert signals into orders for paper/live execution.

## 4. Pipeline Steps in Detail

### Step 1: Data Download

Source code:

- `v3/07_pipeline/steps/download.py`
- `v3/00_config/config.py`

Data sources:

- NSE stock OHLCV via Yahoo Finance (`{symbol}.NS`)
- USD/INR exchange rate (`USDINR=X`)
- Global market cues:
  - S&P 500
  - Nasdaq
  - US VIX
  - Dollar Index
  - Crude Oil
  - Nikkei
  - Nifty 50
  - Nifty Bank

Storage:

- Raw stock data: `v3/01_data/raw/{symbol}.parquet`
- USD/INR: `v3/01_data/raw/usdinr.parquet`
- Global cues: `v3/01_data/raw/global_cues.parquet`

Download design:

- Incremental updates only
- Existing files are reused
- Recent dates are re-fetched with a small overlap to avoid gaps
- `yfinance` calls are serialized with a lock because the code explicitly notes that concurrent `yfinance` downloads are not thread-safe

Important design strength:

- The code tries to be leakage-safe when merging global cues by shifting US market data forward one India trading day, so Indian date `T` only sees foreign close data that was actually known before Indian trading on `T`.

### Step 2: Feature Engineering

Source code:

- `v3/07_pipeline/steps/features.py`

Raw inputs:

- `open`
- `high`
- `low`
- `close`
- `volume`
- `date`

Raw market variables count:

- 5 market fields plus date per stock row

Engineered feature count:

- The pipeline documentation and code describe this as `260+` engineered features

Post-selection feature count used in training:

- `N_TOP_FEATURES = 50`

Feature groups:

1. Returns
   - 1d, 2d, 5d, 10d, 20d
   - log returns

2. Trend and moving average structure
   - SMA ratios
   - EMA ratios
   - 50/200 cross structure
   - trend strength

3. Momentum
   - RSI across multiple windows
   - MACD variants
   - ROC across multiple windows
   - stochastic oscillator
   - Williams %R
   - CCI

4. Volatility
   - ATR ratios
   - rolling realized volatility
   - Parkinson volatility
   - Garman-Klass volatility
   - optional GARCH conditional volatility

5. Volume
   - OBV ratio
   - volume ratios
   - volume change / volume momentum

6. Candlestick structure
   - body size
   - upper/lower shadow
   - high-low range
   - open-close return
   - overnight gap

7. Statistical descriptors
   - skew
   - kurtosis
   - rolling z-scores

8. Lagged features
   - lagged returns
   - lagged volume ratios

9. Relative positioning
   - distance to rolling highs/lows
   - high/low position over different windows

10. Regime features
   - rule-based market regime
   - bull/bear indicators

11. Cross-sectional and sector features
   - relative strength vs peers
   - sector average return
   - sector momentum
   - sector volatility
   - sector correlation

12. Global cues
   - previous-day S&P return
   - Nasdaq return
   - VIX level / z-score / spike regime
   - Dollar index returns
   - crude returns
   - Nikkei returns
   - Nifty 50 / Nifty Bank context

13. Macro / currency features
   - USD/INR return and RSI for IT-sensitive names

14. Calendar features
   - day-of-week cyclic encoding
   - month cyclic encoding
   - week-of-year cyclic encoding

15. NSE-specific event features
   - days to monthly expiry
   - expiry week / expiry day
   - days to RBI MPC
   - RBI week
   - budget week
   - result season

16. News sentiment features
   - daily FinBERT raw score
   - positive / negative ratios
   - article counts
   - rolling sentiment mean, trend, z-score, volatility

17. Earnings proximity features
   - days to earnings
   - pre-results drift
   - post-results day
   - earnings proximity score

### Feature Engineering Design Principles

The code clearly tries to avoid lookahead:

- Rolling features are backward-looking
- Global cues are lagged correctly
- Sentiment is shifted by one day
- Earnings are used as proximity/event features, not post-event future labels

### Feature Selection

Source:

- `select_top_features()` in `v3/07_pipeline/steps/features.py`

Method:

- LightGBM-based feature importance
- Trained only on the train+validation portion of the first walk-forward window
- Uses `LGBM_FS_PARAMS`
- Returns the top `50` features

Forced-included features:

- global cue features for all stocks
- USD/INR-related features for IT stocks
- banking cue features for banking stocks

This is a sensible design because it avoids losing structurally important macro features simply because they are not the top split-count features in one window.

## 5. Target Definition

Source:

- `add_target()` in `v3/07_pipeline/steps/features.py`

Primary target:

- `target = 1` if 5-day forward return > `+1%`
- `target = 0` if 5-day forward return < `-1%`
- otherwise `NaN`

Important implications:

- This is not a pure next-day target.
- It is a 5-day directional move target with a deadband around zero.
- The deadband removes noise and forces the model to learn only meaningful moves.

Why this is good:

- Daily equity moves are noisy.
- A 5-day horizon is more stable than a 1-day noise-level target.
- The code comments explicitly say the old short-horizon target was too close to the transaction-cost noise floor.

Extra stored field:

- `next_ret`

This is kept for trade meta-labeling and P&L logic, but excluded from model features to avoid leakage.

## 6. Preprocessing and Scaling

Training preprocessing in `train_window()`:

1. Winsorization
   - train-set 1st and 99th percentile clipping

2. Scaling
   - `RobustScaler`
   - fit on train only
   - then clip scaled values to `[-5, 5]`

3. PCA
   - `PCA(n_components=0.90)`
   - keeps 90% explained variance
   - fit on train only

Preprocessing fork:

- Tree models use the PCA-transformed 2D inputs
- Deep learning models use sequence tensors formed from the scaled/PCA pipeline prepared for contiguous temporal blocks

Why RobustScaler:

- Stock features have outliers, fat tails, and volatility spikes
- RobustScaler is less sensitive to extreme values than StandardScaler

## 7. Training Methodology

Source:

- `v3/07_pipeline/steps/train.py`

Validation style:

- Expanding-window walk-forward validation

Window schedule:

- Initial train ratio: `0.70`
- Expansion step: `0.05`
- Maximum train ratio: `0.95`
- Minimum training samples: `400`
- Minimum test samples: `30`

Typical per-window split:

- train
- validation
- test

How the windows work:

- Window 1 trains on the earliest 70% of history and tests on the next segment
- Each later window expands the training set forward
- This mimics how a real live system would be retrained over time

This is one of the strongest aspects of the entire project.

## 8. Models Used

### 8.1 Tree Models

Always available:

1. LightGBM
2. XGBoost
3. CatBoost

#### LightGBM

Config:

- `n_estimators = 1000`
- `max_depth = 5`
- `learning_rate = 0.01`
- `num_leaves = 31`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `reg_alpha = 0.3`
- `reg_lambda = 1.5`
- `min_child_samples = 20`
- `early_stopping_rounds = 50`
- `is_unbalance = True` when used in train pipeline

Regularization:

- shallow trees
- low learning rate
- subsampling
- column subsampling
- L1 (`reg_alpha`)
- L2 (`reg_lambda`)
- early stopping

Special detail:

- The wrapper uses a custom focal-loss objective rather than plain binary logloss
- This is intended to emphasize hard-to-classify examples

#### XGBoost

Config:

- `n_estimators = 1000`
- `max_depth = 5`
- `learning_rate = 0.01`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `reg_alpha = 0.3`
- `reg_lambda = 1.5`
- `early_stopping_rounds = 50`
- `scale_pos_weight = n_down / n_up`

Regularization:

- shallow trees
- low learning rate
- subsampling
- column subsampling
- L1 and L2 regularization
- early stopping
- class imbalance weighting

#### CatBoost

Config:

- `iterations = 1000`
- `depth = 6`
- `learning_rate = 0.01`
- `subsample = 0.8`
- `colsample_bylevel = 0.8`
- `l2_leaf_reg = 3.0`
- `min_data_in_leaf = 20`
- `early_stopping_rounds = 50`

Regularization:

- controlled depth
- subsampling
- per-level column subsampling
- L2 leaf regularization
- early stopping

### 8.2 Deep Learning Models

Supported by the codebase:

- LSTM
- BiLSTM
- GRU
- CNN-LSTM
- CNN-GRU
- TCN-GRU
- TCN-Transformer
- N-BEATS

However, the current `train.py` fast-path loads only:

- BiLSTM
- TCN-Transformer
- N-BEATS

and if fast mode is enabled, deep learning is skipped entirely.

Global deep-learning training settings:

- sequence length = `20`
- batch size = `32`
- max epochs = `100`
- optimizer = `Adam`
- early stopping patience = `8`
- min delta = `5e-5`
- reduce LR on plateau factor = `0.5`
- reduce LR patience = `8`
- min LR = `1e-5`

Loss:

- custom directional loss (`dir_loss`) with asymmetric penalty
- code comment says direction errors cost `2x`

#### BiLSTM

Architecture:

- Bidirectional LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.2, L2=1e-4)
- Bidirectional LSTM(16, return_sequences=False, dropout=0.3, recurrent_dropout=0.2, L2=1e-4)
- Dense(32, relu, L2=1e-4)
- Dropout(0.3)
- Dense(1, sigmoid)

#### TCN-Transformer

Architecture:

- Dense projection to `d_model = 64`
- 3 causal TCN residual blocks with dilations `[1, 2, 4]`
- MultiHeadAttention with `4` heads
- feed-forward network
- global average pooling
- Dense(32)
- Dropout(0.2)
- Dense(1, sigmoid)

Regularization:

- dropout
- L2 = `1e-4`
- residual connections
- layer normalization

#### N-BEATS

Architecture:

- flatten sequence
- dense projection to `256`
- `3` residual N-BEATS blocks
- each block has `4` FC layers of width `512`
- forecast dimension `64`
- final Dense(64) + Dropout(0.3) + sigmoid

Regularization:

- dropout `0.3`
- L2 = `1e-4`
- lower learning rate `5e-4`

## 9. How the Ensemble Is Formed

This is one of the most important parts of the system.

The ensemble is not a single averaging rule; it is built in layers.

### Layer 1: Base model probabilities

Each trained model outputs `P(UP)` on the validation and test sets.

### Layer 2: Validation-logloss-weighted averaging

The initial ensemble probability is computed from inverse validation logloss weights:

- better-calibrated models get higher weight
- worse models get lower weight

Fallback:

- if this fails, the code falls back to simple averaging

### Layer 3: Meta-learner stacking

If at least two models are available:

- a `LogisticRegression` meta-model is trained on validation-set model probabilities
- penalty: elastic net
- `C = 2.0`
- `l1_ratio = 0.3`
- solver = `saga`

The meta-learner is only accepted if:

- it beats the base average on validation accuracy
- it produces non-trivial coefficients
- it does not collapse to a degenerate constant-style prediction

### Layer 4: Regime-specific LightGBM routing

The code also trains regime-specific LightGBM models:

- bull
- sideways
- bear

If enough regime samples exist:

- a regime-specific LightGBM probability is routed for samples in that regime
- then adaptively blended with the global ensemble

Blend weight:

- based on regime-vs-global validation logloss
- clipped into `[0.40, 0.75]`

### Layer 5: Temperature scaling

Finally, the ensemble probability is calibrated using temperature scaling:

- fit on validation probabilities only
- single-parameter probability calibration
- stored per window in `calibration.json`

This produces the final calibrated `avg_prob`.

### Layer 6: Secondary meta-labeling filter

There is an additional trade-selection model inspired by López de Prado:

- trained only on rows where the primary model said `UP`
- target = whether the next-day trade was profitable after round-trip cost
- model = LightGBM secondary classifier
- output = `meta_prob`

This secondary layer is not the main direction predictor.
It is a trade filter on top of the primary directional signal.

## 10. Signal Generation

Source:

- `v3/07_pipeline/steps/predict.py`

The production prediction pipeline:

1. Load the production artifacts for a stock.
2. Recompute the latest features using information available through the most recent completed trading day `T`.
3. Apply winsorization, scaler, and PCA.
4. Run each available model.
5. Build the ensemble probability.
6. Apply regime logic and temperature scaling.
7. Compute a prediction for trading day `T+1` with `direction = UP if prob >= 0.5 else DOWN`.
8. Compute `confidence`.
9. Apply the trade gate:
   - `confidence >= 0.58`
   - and secondary `meta_prob >= 0.60` if the secondary model exists

Generated prediction fields include:

- `symbol`
- `last_date`
- `prediction_date`
- `prediction_for`
- `last_close`
- `direction`
- `action`
- `confidence`
- `avg_prob`
- `meta_prob`
- `predicted_price`
- `range_low`
- `range_high`
- `predicted_move_pct`
- `range_down_pct`
- `range_up_pct`
- `atr_14`
- `signal_active`
- `regime`
- `regime_label`
- `temperature`

Important note:

The predicted price range is not a model-trained price forecast. It is an ATR-based estimate layered on top of a directional probability signal for `T+1`. It is useful for interpretation and dashboarding, but it should not be confused with a true next-day close regression model or a direct return-regression model.

## 11. Backtest Method

Source:

- `v3/07_pipeline/steps/backtest.py`

Main assumptions:

- long-only
- enter only on `UP` signals
- confidence threshold around `0.58`
- optional `meta_prob >= 0.60`
- hold for exactly `10` trading days
- no overlapping positions per stock
- round-trip cost = `0.25%`

Backtest outputs:

- per-stock results
- cross-sectional top-15 ranking
- tradeable universe
- portfolio-level equity curve
- bootstrap confidence interval for trade accuracy
- NIFTY buy-and-hold comparison

Important caveat:

The current backtest uses the same day `close_price` as the entry reference for a signal that is created from that day’s completed data. In real life, that signal can only be acted on the next session. This is a major realism gap and must be corrected before claiming live profitability.

## 12. CSV and Output Files

The main output directory is:

- `v3/06_results/runs/{run_id}/`

### 12.1 `summary.csv`

Purpose:

- one row per stock plus an `AVERAGE` row

Main columns:

- `symbol`
- `oos_accuracy`
- `oos_f1`
- `n_windows`
- `n_predictions`
- `n_features`
- `n_rows`

Meaning:

- the primary per-stock research summary

### 12.2 `all_windows_detail.csv`

Purpose:

- one row per stock per walk-forward window

Main columns:

- `symbol`
- `window_id`
- `train_ratio`
- `train_size`
- `val_size`
- `test_size`
- `test_start`
- `test_end`
- `oos_accuracy`
- `auc`
- `f1`
- `precision`
- `recall`
- `tp`, `fp`, `tn`, `fn`
- per-model accuracies:
  - `lgbm_acc`
  - `xgb_acc`
  - `catboost_acc` when present
  - DL accuracies when present
- `dir_acc_up`
- `dir_acc_down`
- `pct_neutral`
- `pct_up`
- `pct_down`
- `temperature`
- `dl_models_trained`

Meaning:

- the fine-grained validation record for every walk-forward segment

### 12.3 `model_comparison.csv`

Purpose:

- aggregate per-model statistics across all windows/stocks

Columns:

- `model`
- `avg_accuracy`
- `median_accuracy`
- `max_accuracy`
- `min_accuracy`
- `std_accuracy`
- `n_datapoints`

### 12.4 `next_day_predictions.csv`

Purpose:

- latest inference output for all stocks in the run

Observed current columns:

- `symbol`
- `last_date`
- `last_close`
- `direction`
- `action`
- `confidence`
- `avg_prob`
- `meta_prob`
- `signal_active`
- `regime`
- `regime_label`
- `temperature`
- `oos_accuracy`
- `tradeable`
- `cross_sectional_top15`
- `sharpe_rank`

After recent extension, the API/dashboard path also supports:

- `prediction_date`
- `prediction_for`
- `predicted_price`
- `range_low`
- `range_high`
- `predicted_move_pct`
- `range_down_pct`
- `range_up_pct`
- `atr_14`
- `ensemble_accuracy`
- `up_signal_accuracy`
- `down_signal_accuracy`
- `directional_accuracy_for_signal`
- `best_model`
- `best_model_accuracy`

### 12.5 `backtest_results.csv`

Purpose:

- per-stock trading metrics from the signal-based simulation

Main columns:

- `symbol`
- `oos_accuracy`
- `meta_val_auc`
- `tradeable`
- `cross_sectional_top15`
- `sharpe_rank`
- `n_trades`
- `total_return`
- `ann_return`
- `win_rate`
- `avg_win_pct`
- `avg_loss_pct`
- `profit_factor`
- `sharpe`
- `max_drawdown`
- `calmar`
- `binary_dir_acc`
- `up_signal_acc`
- `date_range`

### 12.6 `backtest_portfolio.csv`

Purpose:

- portfolio equity curve for the selected tradeable/top-ranked set

Main columns:

- `date`
- `daily_return`
- `equity`

### 12.7 `backtest_summary.json`

Purpose:

- compressed portfolio-level summary

Current latest-run fields:

- `bootstrap_acc_mean`
- `bootstrap_ci_lower`
- `bootstrap_ci_upper`
- `bootstrap_significant`
- `bootstrap_n_signals`
- `nifty_return`
- `nifty_start_date`
- `nifty_end_date`
- `portfolio_total_return`
- `portfolio_sharpe`
- `portfolio_max_dd`
- `avg_per_stock_return`
- `portfolio_return`

### 12.8 Per-stock outputs

Each stock directory contains:

- `window_results.csv`
- `predictions.csv`
- `summary_row.json`
- `plots/*`

`predictions.csv` includes the full OOS prediction history per stock, including:

- `date`
- `window_id`
- `close_price`
- `next_close_price`
- `actual`
- `prob_up`
- `meta_prob`
- model-specific predictions
- `ensemble_pred`
- `direction`
- `correct`

## 13. Dashboard / Backend / Frontend Integration

Backend:

- `backend/main.py`

Frontend:

- `frontend/src/app/page.tsx`
- `frontend/src/components/SignalsTable.tsx`

What the dashboard currently shows:

- pipeline status
- run selector
- prediction table
- per-stock research summary
- chart view
- signal counts
- OOS accuracy context
- CSV export of signals

The backend serves:

- latest runs
- per-run summary
- per-run predictions
- per-stock prediction history
- backtest outputs

## 14. Latest Run Results Interpreted

Reference run:

- `20260427_122004`

### 14.1 Research metrics

From `summary.csv`:

- Stocks trained: `97`
- Average OOS accuracy: `51.43%`
- Stocks >= 50% accuracy: `68`
- Stocks >= 52% accuracy: `48`
- Stocks >= 54% accuracy: `22`
- Stocks >= 58% accuracy: `5`
- Best stock by OOS accuracy: `FEDERALBNK` at `60.22%`

Interpretation:

- This is clearly above a random 50/50 binary baseline.
- The system does show predictive edge.
- But the average edge is still modest at the full-universe level.
- The performance is concentrated in a subset of stocks rather than uniformly strong everywhere.

### 14.2 Model comparison

From `model_comparison.csv` in the latest run:

- XGBoost average accuracy: `51.21%`
- LightGBM average accuracy: `51.10%`

Important interpretation:

- This latest run appears to be a fast-mode tree-only run.
- DL models were not active in this run, as seen from zero DL metrics in `all_windows_detail.csv`.
- Therefore, the latest production-style result reflects the practical live path more than the full theoretical model stack.

### 14.3 Trading/backtest metrics

From `backtest_summary.json`:

- Bootstrap trade accuracy mean: `61.53%`
- 95% bootstrap CI: `[58.98%, 64.02%]`
- Significant above 50%: `true`
- Signals in bootstrap sample: `1448`
- Portfolio total return: `92.30%`
- Portfolio Sharpe: `1.723`
- Portfolio max drawdown: `13.2%`
- NIFTY buy-and-hold over the same broad period: `14.0%`

From `backtest_results.csv`:

- Tradeable stocks: `17`
- Average tradeable-stock Sharpe: `5.6598`
- Average tradeable-stock total return: `21.54%`

Interpretation:

- The filtered subset backtest is strong.
- The strategy is not “all stocks are profitable”; it is “a selected subset of stocks appears profitable under the current backtest rules”.
- This distinction is very important for honest reporting.

## 15. Are We Profitable?

### Short answer

Backtest profitable: yes.

Live profitability proven: no.

### Honest assessment

What can be said:

- The research pipeline shows predictive signal.
- The filtered strategy backtest is profitable.
- The bootstrap directional-trade accuracy is statistically above 50%.
- The portfolio backtest materially outperforms the benchmark in-sample/out-of-sample simulation.

What cannot be said yet:

- that the live deployed system is already proven profitable
- that the execution path is production-safe
- that the current backtest is realistic enough to publish a definitive profitability claim

The main reason is that there are still research-to-live mismatches, especially around entry timing and execution realism.

## 16. Are We Research Publication Ready?

### Short answer

Not yet.

### Why not yet

The project has many things a mentor or reviewer would like:

- zero-leakage intent
- walk-forward validation
- multi-model ensemble
- calibration
- sector/global context
- backtest metrics
- bootstrap significance

However, it is still not publication-ready because:

1. The backtest entry timing is not fully realistic.
2. Live execution assumptions are not tightly matched to research assumptions.
3. The latest production-style run is tree-only, while the architecture claims a richer ensemble.
4. The instrument/token mapping in live trading code is currently unsafe.
5. More statistical robustness work is needed:
   - ablation studies
   - regime stability analysis
   - transaction cost sensitivity
   - retraining frequency sensitivity
   - turnover/slippage impact
   - benchmark comparisons beyond one strategy framing

### Publication readiness verdict

- Interesting and strong thesis/research prototype: yes
- Submission-ready empirical paper: not yet

## 17. Pros of the Current Pipeline

1. Strong architecture discipline.
   The pipeline is modular, readable, and clearly separated into download, feature engineering, training, evaluation, prediction, backtest, and live trading.

2. Good anti-leakage intent.
   The code explicitly tries to prevent lookahead in features, scaling, PCA, and calibration.

3. Walk-forward validation is correctly prioritized.
   This is much better than naive train/test splitting for time-series trading.

4. Feature stack is broad and market-aware.
   The pipeline does not rely only on OHLCV; it includes macro, sector, event, and sentiment information.

5. Ensemble design is thoughtful.
   It uses averaging, stacking, regime routing, and calibration instead of a simplistic vote.

6. Probability calibration is a major strength.
   Temperature scaling improves interpretation and trade gating.

7. There is a clear trade-selection layer.
   The system does not blindly trade every signal.

8. Rich output artifacts.
   The CSVs and plots are already suitable for analysis and mentor discussion.

## 18. Cons / Weaknesses of the Current Pipeline

1. Backtest/live mismatch.
   This is the most important weakness.

2. Live execution infrastructure still has correctness risk.
   In particular, symbol-token integrity and some execution-path bugs need tightening.

3. Sizing logic and risk logic are not fully harmonized.

4. Deep learning support exists, but the latest live-style pipeline run appears to rely only on trees.

5. Price-range output is heuristic, not a trained price forecast.

6. Profitability is concentrated in a filtered subset, not universal across the universe.

7. Some live trading guards are present in code but not always enforced cleanly in the execution path.

## 19. Ten High-Impact Improvements for the Next Version

These are the ten most important improvements to make the system more credible for real trading and stronger for research publication.

### 1. Fix the backtest/live timing mismatch

Rebuild the backtest so signals generated from day `T` data are entered at day `T+1` open, VWAP, or a defined execution window. This is the single most important correction.

### 2. Replace the static Angel instrument-token map

Generate symbol-token mapping daily from the official instrument master instead of maintaining a fragile hard-coded map.

### 3. Persist a true portfolio state ledger

Maintain a canonical ledger of:

- previous-day NAV
- open lots
- realized P&L
- pending orders
- current holdings

This should drive risk checks, exits, and reconciliation.

### 4. Unify strategy sizing and risk limits

The order generator and risk guard should use one shared configuration for:

- max stock exposure
- max sector exposure
- max holdings
- slippage limits
- hold duration

### 5. Move from direction-only ranking to expectancy ranking

Use expected return after cost rather than only `prob_up` for ordering and capital sizing.

### 6. Add execution-realistic slippage and fill modeling

Use next-open or intraday VWAP assumptions, partial fills, cancel/replace logic, and liquidity-aware slippage curves.

### 7. Strengthen statistical research validation

Add:

- proper ablation studies
- turnover sensitivity
- slippage sensitivity
- regime-specific validation
- rolling calibration diagnostics
- significance tests beyond one bootstrap summary

### 8. Improve exit logic

Test:

- volatility stop + time stop
- trailing stop
- signal decay exit
- partial profit-taking

### 9. Add portfolio construction optimization

Replace simple top-3 score selection with allocation logic based on:

- correlation
- marginal contribution to risk
- sector diversification
- volatility budgeting

### 10. Build a paper-to-live promotion framework

Require a minimum sample of live-paper trades and monitor:

- fill quality
- slippage drift
- rolling Sharpe
- rolling win rate
- calibration drift
- drawdown

before promoting to real capital.

## 20. Final Verdict

### What is already good enough to show confidently

- The system architecture
- The step-by-step training and validation methodology
- The feature engineering depth
- The ensemble design
- The current CSV outputs and dashboard integration
- The fact that the research subset backtest is profitable

### What should be presented carefully

- Profitability should be described as backtest profitability, not proven live profitability.
- Publication claims should be described as “promising but not publication-ready yet”.
- The mentor should be told clearly that the system has a real signal, but the live-trading bridge still needs tightening.

### Final answer to the key question

Is the system profitable?

- In research backtest form on filtered tradeable stocks: yes, promisingly so.
- In actual real-time proven deployed trading: not proven yet.

Is the system publication ready?

- Not yet.
- It is a strong thesis-quality prototype, but not yet a defensible publication-quality final result.

## 21. Recommended One-Line Summary

This is a well-structured walk-forward ensemble trading research pipeline with promising predictive and backtest results, but it still needs a more realistic execution-aligned backtest, stronger live-trading controls, and tighter validation before it can be called production-profitable or publication-ready.
