# Comprehensive Explanation of the Research Presentation
## "Development of Positional Trading Strategy Using Deep Learning and Its Training, Testing, and Implementation on a Real-Time Platform Using API"

**Author:** Vishwam Shah  
**Guide:** Dr. Jigarkumar Shah  
**Institution:** Pandit Deendayal Energy University, School of Technology  
**Date:** December 2025 (End Sem Project Phase – 1)

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Dataset and Stock Selection](#2-dataset-and-stock-selection)
3. [Data Characteristics and Pipeline](#3-data-characteristics-and-pipeline)
4. [Feature Engineering: 244 Features](#4-feature-engineering-244-features)
5. [Multi-Target Prediction Framework](#5-multi-target-prediction-framework)
6. [Model Architecture: Individual Models](#6-model-architecture-individual-models)
7. [Ensemble Architecture (Stacking)](#7-ensemble-architecture-stacking)
8. [Results and Performance Analysis](#8-results-and-performance-analysis)
9. [Case Study: RELIANCE Industries](#9-case-study-reliance-industries)
10. [Why XGBoost Outperforms Neural Networks](#10-why-xgboost-outperforms-neural-networks)
11. [Comparison with Prior Literature](#11-comparison-with-prior-literature)
12. [Conclusions](#12-conclusions)
13. [Challenges Faced](#13-challenges-faced)
14. [Future Work](#14-future-work)
15. [Glossary of Scientific Terms](#15-glossary-of-scientific-terms)

---

## 1. Introduction and Motivation

### 1.1 The Core Problem

Financial markets — particularly equity markets — are among the most complex, non-linear, and stochastic systems studied in applied mathematics and computer science. The **National Stock Exchange (NSE) of India** is one of the largest exchanges globally by traded volume, and its price movements are governed by thousands of interacting variables: macroeconomic indicators, corporate earnings, geopolitical events, investor sentiment, liquidity conditions, and endogenous feedback loops within the market microstructure itself.

**Traditional forecasting models** — such as ARIMA (AutoRegressive Integrated Moving Average), linear regression, and even classical technical analysis — operate under restrictive statistical assumptions including stationarity (the statistical properties of a time series do not change over time), linearity, and normality of residuals. Stock price data violates all three of these assumptions regularly. Prices exhibit **heteroscedasticity** (time-varying volatility, where variance is not constant), **autocorrelation** (present-day prices are correlated with past prices), **fat tails** in return distributions (extreme events occur far more frequently than a normal distribution would predict), and **regime changes** (market behavior shifts abruptly between bull markets, bear markets, and high-volatility crisis regimes).

### 1.2 The Limitation of Single-Target Prediction

Most prior work in stock price prediction focuses on a single output: predicting whether the price will go "up" or "down" the next day (binary classification) or estimating the magnitude of the next closing price (regression). This **single-target formulation** is informationally incomplete. A trader acting in a real market needs to know not just the direction of the next day's move, but also:

- **How far could the price rally from the open?** (to set a take-profit target)
- **How far could the price fall from the open?** (to set a stop-loss level)
- **What is the magnitude of the expected close-to-close return?** (to assess risk-reward ratio)

Without all four pieces of information simultaneously, optimal position sizing, risk management, and trade execution become impossible. This research addresses this gap by implementing a **multi-target simultaneous prediction framework** across four output variables.

### 1.3 Why Deep Learning and Ensemble Methods?

**Deep learning** models — specifically Recurrent Neural Networks (RNNs) and their gated variants — are theoretically well-suited for sequential, temporal data because they can learn non-linear mappings from high-dimensional input spaces to outputs without requiring hand-crafted features or explicit specification of the functional form. **Ensemble methods**, particularly stacking, combine the predictive outputs of multiple heterogeneous base learners to reduce variance and bias simultaneously, exploiting the principle that **no single model dominates all market regimes**.

### 1.4 Key Statistics of This Research

| Metric | Value |
|---|---|
| NSE Stocks Analyzed | 106 |
| Engineered Features | 244 |
| Prediction Targets | 4 |
| Historical Data Period | 10 Years (2015–2025) |
| Total Data Points | ~64.7 Million |
| Best Direction Accuracy | 68.28% (Ensemble) |
| Improvement over Random Baseline | +36 percentage points |

The 50% random baseline represents a **Bernoulli process** — if you flip a fair coin to decide "up" or "down" each day, you will be correct 50% of the time on average. Achieving 68.28% means the model carries statistically significant predictive information that is reflected in the market's price action, **36 percentage points above** the no-information baseline.

---

## 2. Dataset and Stock Selection

### 2.1 The Stock Universe: 106 NSE Stocks Across 11 Sectors

The research covers **106 publicly traded equity instruments** listed on the National Stock Exchange of India, spanning **11 distinct economic sectors**. This is one of the largest stock universes studied in the Indian equity prediction literature.

| Sector | Stock Count | Representative Stocks |
|---|---|---|
| Banking | 12 | HDFCBANK, ICICIBANK, SBIN, AXISBANK |
| Information Technology | 10 | TCS, INFY, WIPRO, HCLTECH, TECHM |
| Pharmaceuticals | 9 | SUNPHARMA, DRREDDY, CIPLA, LUPIN |
| Automotive | 9 | MARUTI, M&M, TATAMOTORS, BAJAJ-AUTO |
| Energy & Power | 11 | RELIANCE, ONGC, BPCL, NTPC, TATAPOWER |
| Metals | 9 | TATASTEEL, JSWSTEEL, HINDALCO, VEDL |
| Consumer Goods | 11 | HINDUNILVR, ITC, BRITANNIA, DABUR |
| Construction | 6 | LT, DLF, GODREJPROP, OBEROIRLTY |
| Cement | 6 | ULTRACEMCO, SHREECEM, AMBUJACEM, ACC |
| Telecom | 3 | BHARTIARTL, IDEA, TATACOMM |
| Others | 20 | TITAN, ASIANPAINT, BAJFINANCE, ADANIENT |
| **Total** | **106** | |

### 2.2 Selection Criteria Explained

**Why these 106 stocks specifically?**

1. **Market Capitalisation Filter (> INR 5,000 Crore):** Small-cap and micro-cap stocks suffer from low liquidity, wide bid-ask spreads, and susceptibility to price manipulation. By restricting the universe to stocks with market cap exceeding INR 5,000 Crore, the study ensures that the observed price movements reflect genuine supply-demand dynamics rather than thin-market artefacts.

2. **Complete 10-Year Data Availability (2015–2025):** Machine learning models, particularly deep learning architectures, require large training datasets to learn generalizable patterns. A 10-year window captures multiple complete market cycles — including the 2016 demonetization shock, the 2018–2019 NBFC (Non-Banking Financial Company) crisis, the 2020 COVID-19 pandemic crash and recovery, and the 2022 global interest rate tightening cycle. Training on a dataset that spans multiple regimes forces the model to learn patterns that are robust across varying macroeconomic conditions.

3. **Sector Diversity:** Including stocks from 11 sectors ensures the model is evaluated under diverse business cycle dynamics. Technology stocks and pharmaceutical stocks respond to entirely different macroeconomic drivers than banking or energy stocks, providing a comprehensive stress test of the model's generalizability.

4. **NSE Listing:** The NSE is a regulated, transparent exchange with standardized OHLCV (Open-High-Low-Close-Volume) data, ensuring data quality and consistency across all 106 instruments.

### 2.3 Why Such a Large Universe Matters

Most prior research in stock prediction is conducted on small portfolios of 3–20 stocks, often selected with **survivorship bias** (only including stocks that have performed well historically). With 106 stocks spanning underperformers, sector laggards, and market leaders, this research avoids survivorship bias and provides statistically robust average performance estimates. A result averaged over 106 stocks is far more reliable than one observed on 5 cherry-picked stocks.

---

## 3. Data Characteristics and Pipeline

### 3.1 Data Sources

**Primary Source: Yahoo Finance via yfinance library**

The `yfinance` Python library provides programmatic access to Yahoo Finance's historical OHLCV (Open-High-Low-Close-Volume) data. Each trading day contributes one row of raw data with five values:

- **Open (O):** The price at which the first trade occurred when the market opened at 9:15 AM IST.
- **High (H):** The highest price traded during the entire trading session.
- **Low (L):** The lowest price traded during the entire trading session.
- **Close (C):** The final traded price when the market closed at 3:30 PM IST.
- **Volume (V):** The total number of shares that changed hands during the session.

**Supplementary Data Sources:**
- **Market Indices:** NIFTY 50 (broad market benchmark), BANKNIFTY (banking sector index), India VIX (volatility index — measures 30-day implied volatility derived from NIFTY options, analogous to the CBOE VIX for US markets)
- **Sector Indices:** Bank Index, IT Index, Pharma Index, Auto Index — used as features to capture sector-level momentum and relative strength
- **Currency:** USD/INR exchange rate — captures macroeconomic stress and foreign institutional investor (FII) flow dynamics

### 3.2 Data Preprocessing Pipeline

**Step 1: Raw OHLCV Data Collection**  
Daily OHLCV data for each of the 106 stocks is downloaded from January 2015 to December 2025, covering approximately 2,750 trading days per stock.

**Step 2: Missing Value Imputation**  
NSE trading holidays, company-specific trading halts (circuit breakers), and data provider gaps introduce **missing values** (NaN entries). Forward-fill interpolation (`ffill`) is applied for gaps of 1–3 days, while larger gaps are handled by dropping affected rows to prevent **lookahead contamination** — the inadvertent leakage of future information into past observations.

**Step 3: Outlier Clipping (1st–99th Percentile)**  
On rare occasions, data errors, flash crashes, or data provider anomalies introduce **extreme outliers** — for example, a price that is 10x the normal range due to a split-adjustment error. By clipping all feature values to the 1st–99th percentile range (a technique called **Winsorization**), the feature distributions are bounded, preventing these outliers from disproportionately influencing model training through large gradient updates.

**Step 4: Feature Engineering (244 Features)**  
Raw OHLCV data is transformed into 244 informative predictive features (detailed in Section 4).

### 3.3 Walk-Forward Temporal Validation

| Split | Proportion | Calendar Period |
|---|---|---|
| Training Set | 60% | 2015 – 2020 |
| Validation Set | 20% | 2020 – 2022 |
| Testing Set | 20% | 2022 – 2025 |

**Why Walk-Forward Validation instead of Random Splitting?**

In standard machine learning, datasets are often split randomly — 70% training, 15% validation, 15% test. For time series data, random splitting creates a critical flaw called **lookahead bias** (also called **data leakage**): if a test point from January 2022 is evaluated using a model trained partially on data from December 2022, the model has effectively "seen the future" during training, producing artificially inflated performance metrics.

**Walk-forward validation** enforces strict **temporal causality**: every data point in the training set occurs chronologically before every data point in the validation and test sets. This mirrors real trading conditions — a strategy can only use information available up to the moment of the trade decision, never information from the future.

The **validation set** (2020–2022) is used for **hyperparameter tuning** — adjusting model parameters like learning rate, number of trees, or layer sizes — without contaminating the final test set. Only after all hyperparameter decisions are frozen is the model evaluated on the **test set** (2022–2025), which contains 512+ trading days per stock.

Crucially, the test period 2022–2025 includes the 2022 global interest rate hike cycle (highest since 2008), the 2022–2023 Indian equity market correction, and the subsequent bull run — making it a genuinely challenging out-of-sample evaluation period.

---

## 4. Feature Engineering: 244 Features

### 4.1 Why Feature Engineering Is Critical

Raw OHLCV data contains only 5 values per day. A machine learning model trained on raw OHLCV values would fail because:

1. The **raw prices are non-stationary** — they trend upward over decades, meaning the model trained on 2015–2020 prices would encounter test values (2022–2025) outside the range seen during training.
2. **The absolute price level is uninformative** for classification — whether a stock priced at ₹500 goes up tomorrow is not determined by the fact that it costs ₹500, but by its relative momentum, volatility regime, and market context.
3. **Temporal patterns** (such as momentum over 20 days) require multi-day windows that raw OHLCV does not encode.

Feature engineering transforms raw prices into **stationary, normalized, regime-sensitive representations** that expose the underlying market structure to the learning algorithm.

### 4.2 Complete Feature Category Breakdown

#### Category 1: Technical Indicators (87 Features)

**Definition:** Mathematical transformations of price and volume data developed by practitioners to quantify market momentum, trend, and volatility.

| Indicator | Formula/Concept | Why Included |
|---|---|---|
| **SMA (Simple Moving Average)** | Arithmetic mean of closing prices over N days | Captures the central tendency of price; deviations from SMA indicate mean reversion potential |
| **EMA (Exponential Moving Average)** | Geometrically weighted average, more weight on recent prices | More responsive to recent price action than SMA; used in trend-following strategies |
| **MACD (Moving Average Convergence Divergence)** | Difference between 12-day EMA and 26-day EMA; with 9-day signal line | Captures momentum shifts; crossovers indicate trend reversals |
| **RSI (Relative Strength Index)** | RSI = 100 − 100/(1 + RS), where RS = avg gain/avg loss over 14 days | Measures overbought (>70) and oversold (<30) conditions; captures price exhaustion |
| **Bollinger Bands** | Upper/Lower bands = SMA ± (2 × standard deviation); 20-day default | Captures volatility expansion/contraction; price touching bands signals potential reversals |
| **ATR (Average True Range)** | Mean of True Range over 14 days; TR = max(H−L, |H−Cprev|, |L−Cprev|) | Measures volatility regardless of direction; used for stop-loss placement |
| **ADX (Average Directional Index)** | Derived from +DI and −DI; measures trend strength (not direction) | ADX > 25 indicates a strong trend; ADX < 20 indicates ranging/consolidation |
| **Stochastic Oscillator** | %K = (C − L14)/(H14 − L14) × 100 | Compares closing price to price range; identifies turning points |

These 87 technical indicators encode domain knowledge accumulated by practitioners over decades — knowledge that a raw ML model would struggle to rediscover from scratch given limited data.

#### Category 2: Price Features (24 Features)

These features directly quantify price **returns** (percentage changes) over multiple time horizons:

- **1-day log return:** `log(C_t / C_{t−1})` — daily momentum
- **5-day log return:** `log(C_t / C_{t−5})` — weekly momentum
- **20-day log return:** `log(C_t / C_{t−20})` — monthly momentum
- **VWAP (Volume-Weighted Average Price):** `Σ(Price × Volume) / Σ(Volume)` — the average price weighted by trading volume; a proxy for the "fair value" institutional investors care about
- **Price Ratios:** Close/SMA20, Close/SMA50 — measures deviation from trend

**Why log returns instead of absolute price changes?** Log returns have the property of **additivity** across time and **approximate normality** for short horizons, making them better-behaved statistically than raw price differences. More importantly, log returns are **stationary** (their statistical properties are approximately stable over time), unlike price levels.

#### Category 3: Volatility Indicators (18 Features)

Multiple volatility estimators are included because different estimators capture different aspects of market uncertainty:

- **Historical Volatility (HV):** Standard deviation of log returns over N days — the most straightforward measure of price dispersion
- **Parkinson Volatility:** `σ_P = sqrt(1/(4N·ln2) × Σ(ln(H_t/L_t))²)` — uses intraday high-low range; 5× more efficient than close-to-close HV as it incorporates intraday price swings
- **Garman-Klass Volatility:** More sophisticated estimator that combines all four OHLC prices; achieves 7× efficiency over close-to-close HV
- **ATR-based Volatility:** Normalized ATR as a fraction of price — measures volatility in a scale-invariant way

Multiple volatility measures capture the **volatility regime** (is the market calm or turbulent?), which directly affects the model's confidence in its directional prediction.

#### Category 4: Volume Analysis (22 Features)

- **OBV (On-Balance Volume):** Cumulative sum: add day's volume when price closes up, subtract when price closes down — tracks whether volume is flowing into or out of a stock
- **CMF (Chaikin Money Flow):** `CMF = Σ(MFV) / Σ(Volume)` where Money Flow Volume = ((C−L)−(H−C))/(H−L) × Volume — measures buying vs. selling pressure
- **Volume RSI:** RSI applied to volume instead of price — identifies volume momentum
  
Volume indicators encode the principle that **price moves accompanied by high volume are more reliable and sustainable** than price moves on low volume, a central axiom of classical technical analysis.

#### Category 5: Market Regime Indicators (31 Features)

- **Trend Strength:** ADX combined with directional indicators to classify market state as trending or ranging
- **Support/Resistance Levels:** Historically defended price levels identified from rolling maxima and minima; proximity to these levels influences the probability of price reversal
- **Breakout Detection:** Binary indicators for price breaking above resistance or below support, often signaling the start of a new trend

Market regime indicators are particularly valuable because different predictive strategies are optimal in different regimes. A momentum strategy works in trending markets but fails catastrophically in mean-reverting, ranging markets.

#### Category 6: Temporal Features (12 Features)

- Day of week (cyclic encoding via sine/cosine to preserve cyclical continuity)
- Month and quarter
- Distance to quarter-end (institutional rebalancing effects)
- Is-Monday and Is-Friday indicators (Monday effects in returns are documented in the financial literature)

**Cyclic encoding** (e.g., representing Monday as `sin(2π × 1/5)`, `cos(2π × 1/5)`) is essential because standard integer encoding misleads the model into thinking Friday (day 5) is "far from" Monday (day 1), when they are actually adjacent in the weekly cycle.

#### Category 7: Sentiment Features (15 Features)

Derived from news and market sentiment data:
- Positive/Negative/Neutral sentiment scores from financial news
- Sentiment momentum (1-day, 5-day, 20-day rolling averages)
- Sentiment divergence: cases where sentiment is bullish but price is declining (or vice versa) — potential reversal signals
- Market-wide sentiment aggregated across all 106 stocks in the universe

**Why sentiment?** The **Efficient Market Hypothesis (EMH)** posits that all public information is instantly priced into equities. However, decades of empirical research have documented systematic violations of EMH, including **sentiment-driven anomalies** where investor overreaction and herding behavior create predictable price patterns that persist for 1–5 days before being arbitraged away.

#### Category 8: Interaction Features (35 Features)

Non-linear combinations of existing features:
- **Price × Volume interactions:** A large price move on high volume has different significance than the same price move on low volume
- **RSI-MACD Divergence:** Cases where RSI indicates oversold but MACD is still declining — potential early reversal signals
- **Multi-timeframe interactions:** Short-term momentum (5-day) conflicting or agreeing with long-term momentum (50-day) encodes information about trend strength and sustainability

### 4.3 Feature Selection Process

After computing all 244 raw features, a three-stage selection process was applied:

**Stage 1 — Correlation Analysis:** Pairs of features with Pearson correlation coefficient `|ρ| > 0.95` are considered **multicollinear** — they carry nearly redundant information. Keeping both wastes model capacity and can inflate variance. One feature from each correlated pair was removed.

**Stage 2 — Recursive Feature Elimination (RFE) with XGBoost:** RFE iteratively trains XGBoost, ranks features by **gain-based feature importance** (how much each feature reduces prediction error in tree splits), and removes the lowest-ranked features in each iteration. Only features providing statistically meaningful signal were retained.

**Stage 3 — Domain Validation:** The remaining features were validated by domain experts in quantitative finance to confirm they have economically plausible interpretations, preventing the model from learning spurious correlations.

**Impact of Feature Engineering:**
- Baseline accuracy with 72 features: **50%** (near random)
- Accuracy with 244 engineered features: **68.28%**
- **Net improvement: +18.28 percentage points**

This dramatic improvement demonstrates that **domain knowledge encoded as engineered features** is the primary driver of predictive performance in this problem, not raw model complexity.

---

## 5. Multi-Target Prediction Framework

### 5.1 The Four Prediction Targets

This research simultaneously predicts four outputs for each stock on each trading day:

#### Target 1: Direction (Binary Classification)

$$\text{Direction} = \begin{cases} 1 \text{ (Bullish)} & \text{if } \log\left(\frac{C_{t+1}}{C_t}\right) > 0 \\ 0 \text{ (Bearish)} & \text{if } \log\left(\frac{C_{t+1}}{C_t}\right) \leq 0 \end{cases}$$

**Interpretation:** Will tomorrow's closing price be higher or lower than today's closing price? This is the primary actionable signal — whether to enter a long position (buy) or stay in cash.

#### Target 2: Close-to-Close Return (Regression)

$$r_{\text{close}} = \log\left(\frac{C_{t+1}}{C_t}\right)$$

**Interpretation:** The magnitude of the overnight + intraday price change from today's close to tomorrow's close. This quantifies the expected profit/loss of holding a position overnight. It is expressed as a log return to ensure **time-additivity** and approximate normality.

#### Target 3: High Return from Open (Regression)

$$r_{\text{high}} = \log\left(\frac{H_{t+1}}{O_{t+1}}\right)$$

**Interpretation:** The maximum percentage gain achievable from tomorrow's opening price to tomorrow's intraday high. This directly informs the **take-profit level** — how high should the trader set the target price to capture the maximum likely profit? A model predicting `r_high = 2.5%` suggests setting a take-profit 2.5% above the opening price before the market open.

#### Target 4: Low Return from Open (Regression)

$$r_{\text{low}} = \log\left(\frac{L_{t+1}}{O_{t+1}}\right)$$

**Interpretation:** The maximum percentage loss from tomorrow's opening price to tomorrow's intraday low. Since the stock must at least touch this low during the day, this directly informs the **stop-loss level** — the maximum adverse excursion the position will face. If `r_low = -1.8%`, setting a stop-loss 1.0% below the open ensures the position is still live through the worst intraday drawdown.

### 5.2 Why Multi-Task Learning?

**Single-task learning** trains one model for one target. **Multi-task learning (MTL)** trains a shared representation that simultaneously predicts all four targets. The theoretical justification:

1. **Shared Representation:** The same market microstructure patterns that predict direction (whether a gap-up on volume is sustained) also inform where the intraday high is likely to be. Forcing the model to simultaneously predict all four targets creates an **inductive bias** toward learning representations that capture the fundamental dynamics of intraday price formation.

2. **Implicit Regularization:** Each additional task acts as a regularizer on the shared weights. A model that predicts direction, close return, high return, and low return simultaneously cannot overfit to idiosyncratic noise in any single target — the shared representation must generalize.

3. **Target Correlation as Risk Assessment:** If the model simultaneously predicts `r_direction = Bullish`, `r_high = 3.2%`, `r_low = -0.8%`, a trader can compute the **risk-reward ratio = 3.2 / 0.8 = 4.0**, meaning the expected gain is 4× the expected loss — a favorable trade. Without simultaneous prediction of all four targets, rational risk management is impossible.

---

## 6. Model Architecture: Individual Models

### 6.1 XGBoost (eXtreme Gradient Boosting)

**Scientific Definition:** XGBoost is a **gradient boosted ensemble of decision trees** where each successive tree is trained to predict the **negative gradient** (residuals) of the loss function evaluated on the current ensemble's predictions. The final prediction is the sum of all tree outputs, regularized by L1 (Lasso) and L2 (Ridge) penalties.

**The Mathematical Framework:**

Given a loss function `L`, XGBoost minimizes the **regularized objective:**

$$\mathcal{L}^{(t)} = \sum_i L\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$

where `Ω(f) = γT + ½λΣw_j²` penalizes the number of leaves `T` and the magnitude of leaf weights `w_j`.

**Configuration Used:**
- **n_estimators = 200:** 200 sequential decision trees are trained. Each tree corrects the errors of the previous ensemble.
- **max_depth = 5:** Each tree has at most 5 levels (32 leaves maximum), constraining model complexity.
- **learning_rate = 0.01:** Each tree's contribution is scaled by 0.01 (called **shrinkage**), forcing the algorithm to take small steps, reducing overfitting.
- **subsample = 0.8 and colsample_bytree = 0.8:** Each tree uses a random 80% of training samples and 80% of features — **stochastic gradient boosting** that introduces beneficial randomness.
- **L2 regularization (λ = 1.0):** Penalizes large leaf weights, smoothing the model.
- **Early stopping (20 rounds):** Training stops if validation loss does not improve for 20 consecutive iterations, preventing overfitting to the training set.

**Why XGBoost for Financial Data?**

1. **Heterogeneous features:** XGBoost handles the 244 features of wildly different scales and distributions (some binary, some continuous, some highly skewed) without requiring normalization.
2. **Captures non-linear interactions:** Decision tree splits can model complex interactions like "RSI < 30 AND volume above 20-day average AND sentiment positive → high probability bullish" — patterns that linear models fundamentally cannot represent.
3. **Robustness to outliers:** Tree-based splits are rank-based, not magnitude-based, making the model insensitive to extreme price events that were handled by Winsorization.
4. **Interpretability:** **SHAP (SHapley Additive exPlanations) values** for XGBoost allow exact attribution of each prediction to individual features, supporting regulatory compliance and model audit.
5. **Missing values:** XGBoost learns an optimal direction for missing values during training, handling gaps in sentiment or market context data gracefully.

### 6.2 LSTM (Long Short-Term Memory)

**Scientific Definition:** LSTM is a **gated recurrent neural network** that addresses the **vanishing gradient problem** of traditional RNNs by introducing multiplicative gating mechanisms that allow gradients to flow backward through sequences of arbitrary length without exponential decay.

**The LSTM Cell Equations:**

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(Forget gate)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(Input gate)}$$
$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C) \quad \text{(Candidate cell state)}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(Cell state update)}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(Output gate)}$$
$$h_t = o_t \odot \tanh(C_t) \quad \text{(Hidden state)}$$

The **forget gate** `f_t` controls how much of the previous cell state is retained (learning to forget irrelevant history). The **input gate** `i_t` controls how much new information is written to the cell state. The **output gate** `o_t` controls what portion of the cell state influences the current hidden state.

**Configuration Used:**
- **2 LSTM layers:** 128 units in Layer 1, 64 units in Layer 2 (hierarchical feature extraction — lower layers learn basic patterns, higher layers learn complex compositions)
- **Dropout = 0.3:** During training, 30% of units are randomly set to zero at each step — a regularization technique that prevents **co-adaptation** of neurons and forces each unit to learn independently useful features
- **Sequence length = 10 days:** The model observes 10 consecutive trading days of all 244 features as a temporal sequence before making a prediction
- **Adam optimizer (lr = 0.001):** Adaptive Moment Estimation optimizer that maintains per-parameter learning rates, accelerating convergence in high-dimensional spaces

**Why LSTM for Stock Prediction (Theoretically)?**

Stock prices exhibit **path dependence** (where a price is depends on how it got there). For example, a price drop of 5% after a multi-week rally has different meaning than the same 5% drop after a prolonged downtrend. LSTM's gated memory theoretically captures these path-dependent, long-range temporal dependencies across sequences of 10 days.

### 6.3 GRU (Gated Recurrent Unit)

**Scientific Definition:** GRU is a **simplified variant of LSTM** proposed by Cho et al. (2014) that merges the forget and input gates into a single **update gate** and eliminates the separate cell state, reducing the parameter count while theoretically retaining most of LSTM's capability.

**The GRU Equations:**

$$z_t = \sigma(W_z [h_{t-1}, x_t]) \quad \text{(Update gate)}$$
$$r_t = \sigma(W_r [h_{t-1}, x_t]) \quad \text{(Reset gate)}$$
$$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t]) \quad \text{(Candidate hidden state)}$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(Hidden state)}$$

The **update gate** `z_t` controls how much of the past hidden state is retained. The **reset gate** `r_t` controls how much of the past hidden state influences the new candidate hidden state.

**Trade-offs vs LSTM:**
- **~25% fewer parameters** → faster training and lower memory footprint
- **Empirically competitive with LSTM** on medium-length sequences (10–50 steps)
- By testing both LSTM and GRU, the research determines whether the additional parameters in LSTM provide any benefit for this particular domain

**Configuration Used:** Identical architecture to LSTM (2 layers: 128 → 64 units, dropout = 0.3, sequence length = 10, Adam optimizer).

### 6.4 Model Selection Rationale: Complementary Strengths

| Capability | XGBoost | LSTM | GRU |
|---|---|---|---|
| Non-linear feature interactions | ✅ Excellent | ✅ Good | ✅ Good |
| Temporal/sequential patterns | ❌ No | ✅ Excellent | ✅ Excellent |
| Handles heterogeneous features | ✅ Excellent | ⚠️ Requires normalization | ⚠️ Requires normalization |
| Robustness to overfitting | ✅ Strong (regularization) | ⚠️ Moderate (dropout) | ⚠️ Moderate (dropout) |
| Training data efficiency | ✅ High | ❌ Low (needs large data) | ❌ Low (needs large data) |
| Interpretability | ✅ SHAP values | ❌ Black box | ❌ Black box |

The three model types capture fundamentally different inductive biases — XGBoost excels at feature interactions, LSTM/GRU captures temporal dependencies. The ensemble (Section 7) attempts to leverage both.

---

## 7. Ensemble Architecture (Stacking)

### 7.1 The Stacking Paradigm

**Ensemble learning** is the practice of combining multiple models to produce a prediction that is superior to any individual model. **Stacking** (also called **stacked generalization**, Wolpert 1992) is a two-level ensemble method:

**Level 0 (Base Learners):** XGBoost, LSTM, and GRU are each trained independently on the training set. Their predictions on the **validation set** are collected — this is critical because using training set predictions for the meta-learner would cause severe overfitting.

**Level 1 (Meta-Learner):** A **Ridge Regression** model (L2-regularized linear regression) is trained to optimally combine the validation set predictions from Level 0 models into a final prediction.

**Why Ridge Regression as Meta-Learner?**

Ridge Regression with regularization parameter `α = 1.0` minimizes:

$$\min_w \|Xw - y\|_2^2 + \alpha \|w\|_2^2$$

The L2 penalty prevents the meta-learner from assigning extreme weights to any single base learner, ensuring robust combination even when base learner performances vary. A simple Ridge meta-learner also avoids the risk of **second-level overfitting** — the meta-learner is intentionally kept simple to avoid memorizing validation set artifacts.

### 7.2 The Data Flow

```
244 Features (Input)
       │
       ├──────────────────────────────────────────┐
       │                                          │
  [XGBoost]                              [LSTM / GRU]
  Trained on                             Trained on
  Tabular Data                           Sequential Data
    (244×1)                              (10×244 sequences)
       │                                          │
       │  Predictions on Validation Set           │
       └──────────────────┬───────────────────────┘
                          │
                  [Ridge Meta-Learner]
                  Learns optimal weights:
                  prediction = w1·XGB + w2·LSTM + w3·GRU
                          │
                  Final Ensemble Output
```

### 7.3 Why Stacking Works: The Bias-Variance Decomposition

The **generalization error** of any model can be decomposed into three components:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

- **Bias** is the systematic error due to the model's inability to capture true patterns (underfitting)
- **Variance** is the error due to sensitivity to fluctuations in training data (overfitting)

Different models have different bias-variance trade-offs:
- XGBoost with shallow trees has **moderate bias, low variance** (stable, but misses some complex temporal patterns)
- LSTM/GRU have **low bias potential, high variance** (can theoretically model any pattern, but tend to overfit)

Stacking combines them such that:
- In market regimes where XGBoost's tree-based representations excel, the meta-learner assigns higher weight to XGBoost
- In regimes where sequential patterns dominate, the meta-learner shifts weight toward LSTM/GRU

The optimal combination achieves **lower error than any individual model** across the full test period.

---

## 8. Results and Performance Analysis

### 8.1 Primary Results Table (106 Stocks, Test Period 2022–2025)

| Model | Direction Accuracy | Close R² | RMSE (%) | MAE (%) | F1 Score |
|---|---|---|---|---|---|
| **Ensemble** | **68.28%** | **0.027** | **1.37%** | **1.01%** | **0.702** |
| XGBoost | 68.22% | 0.0178 | 1.38% | 1.02% | 0.695 |
| LSTM | 50.31% | −0.003 | 1.39% | 1.03% | 0.669 |
| GRU | 50.28% | −0.003 | 1.39% | 1.03% | 0.669 |

### 8.2 Interpreting Each Metric

**Direction Accuracy (68.28% for Ensemble):**  
Of all trading days in the test period across all 106 stocks, the Ensemble correctly predicted whether the stock would close higher or lower the next day **68.28% of the time**. The random (no-information) baseline is 50%. Achieving 68.28% represents a statistically significant departure from random behavior with a **z-score >> 3** given the large sample size (100,000+ observations), implying the model has genuine predictive information.

**R² Score (Coefficient of Determination):**  
R² measures the proportion of variance in the target variable explained by the model's predictions:

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- `R² = 1.0` means the model perfectly predicts every return
- `R² = 0.0` means the model is no better than predicting the mean return every day
- **Negative R²** means the model is worse than predicting the mean — this is what LSTM/GRU show for close returns (R² = −0.003)!

The Ensemble achieves R² = 0.027 — small but positive, indicating it explains approximately 2.7% of the variance in daily returns. This might seem small, but in quantitative finance, even 1–2% R² with proper risk management is sufficient to generate profitable trading strategies.

**RMSE (Root Mean Squared Error) = 1.37%:**  
The average prediction error for tomorrow's close-to-close return is ±1.37 percentage points. Given that the average daily move of NSE stocks is approximately 1.0–1.5%, an RMSE of 1.37% means the model's regression output has error of similar magnitude to the signal itself — indicating regression prediction of the magnitude of returns is very difficult.

**F1 Score (0.702 for Ensemble):**  
The F1 score is the harmonic mean of **Precision** (of all days predicted as bullish, what fraction actually went up) and **Recall** (of all days that actually went up, what fraction did the model correctly predict as bullish):

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

F1 = 0.702 indicates a well-balanced classifier — it does not simply predict "bullish" every day to inflate recall.

### 8.3 Best Model Distribution Across 106 Stocks

| Model | Stocks Where Best | Percentage |
|---|---|---|
| **Ensemble** | **58** | **54.7%** |
| XGBoost | 46 | 43.4% |
| LSTM | 1 | 0.9% |
| GRU | 1 | 0.9% |
| **Total** | **106** | **100%** |

The Ensemble dominates on 58 of 106 stocks (54.7%); XGBoost is best on 46 stocks (43.4%). LSTM and GRU each win on only 1 stock — borderline statistically insignificant.

**Implication:** The Ensemble's superiority is not universal — for 43.4% of stocks, standalone XGBoost outperforms the ensemble, suggesting that for certain stocks with strong tabular feature patterns, the LSTM/GRU components add noise rather than signal, and the meta-learner is unable to fully suppress their contribution.

### 8.4 Performance Across the Four Targets

**Key observations from the target-wise breakdown:**

1. **Direction Prediction:** Ensemble (68.28%) ≈ XGBoost (68.22%) >> LSTM/GRU (50.3%) — the tabular features carry all the predictive information for tomorrow's direction.

2. **High Return Regression (R² = −0.041 to −0.082):** All models have negative R² for predicting the daily high return. This indicates that **where a stock's intraday high will occur is effectively unpredictable** given the features used — the intraday high depends heavily on intraday order flow, algorithmic trading, and microstructure noise that daily OHLCV features cannot capture.

3. **Low Return Regression:** XGBoost achieves marginal positive R² (0.0089) for the daily low — very slightly better than the naive mean prediction — suggesting that **downside risk** (how far the stock falls intraday from the open) has marginally more predictable structure than upside potential, consistent with the academic literature on loss aversion and asymmetric volatility.

4. **Close Return R²:** Ensemble (0.027) > XGBoost (0.0178) > LSTM/GRU (negative) — the ensemble provides a genuine improvement in magnitude-of-return prediction over any individual model.

---

## 9. Case Study: RELIANCE Industries

### 9.1 Why RELIANCE?

RELIANCE Industries is India's largest publicly traded company by market capitalization (as of 2025, approximately USD 220 billion). It is a **highly liquid**, large-cap stock with one of the highest trading volumes on NSE. Its price movements are driven by a complex interaction of crude oil prices, telecom competition (Jio), retail expansion, and macro factors — making it a representative, challenging test case.

### 9.2 RELIANCE-Specific Results

| Model | Direction Accuracy |
|---|---|
| XGBoost | 71.62% |
| LSTM | 52.62% |
| GRU | 52.62% |
| **Ensemble** | **72.23%** |

The Ensemble achieves **72.23% direction accuracy on RELIANCE** (test set: ~512 trading days from 2022–2025), compared to 68.28% for the full 106-stock average. This above-average performance suggests that RELIANCE's price movements have more learnable structure than the typical NSE stock, possibly due to its higher analyst coverage, greater market efficiency research attention, and more consistent technical pattern adherence.

LSTM and GRU achieve 52.62% — approximately 3% above random, but far below XGBoost. This confirms that **tabular features + XGBoost** is the dominant predictive paradigm for this stock too.

---

## 10. Why XGBoost Outperforms Neural Networks

This is one of the most important findings of the research — **a well-regularized gradient boosting model substantially outperforms deep recurrent neural networks** on financial time series prediction. Understanding why is critical for future research directions.

### 10.1 Reasons XGBoost Excels

**1. Gradient Boosting's Sequential Error Correction**  
Each tree in XGBoost targets the **residuals** of the previous ensemble — systematically addressing the hardest-to-predict observations. This creates a model that progressively "zooms in" on information remaining in the data, unlike LSTM which learns all patterns simultaneously through backpropagation.

**2. Regularization is Perfectly Matched to Financial Data**  
XGBoost applies L1/L2 regularization at the level of individual leaf weights (`γT + ½λΣwj²`), directly preventing any single tree from creating extremely specialized, overfitted splits. This is precisely what is needed for financial data where **spurious correlations are abundant** and **the signal-to-noise ratio is low** (~32% based on direction accuracy headroom above 50%).

**3. Feature Importance and Automatic Interaction Detection**  
XGBoost's tree splits automatically discover interactions between features (e.g., "RSI < 30 AND 5-day return < -3% AND volume ratio > 1.5") without requiring the researcher to specify them explicitly. These non-linear conjunctive rules capture the logic employed by experienced traders.

**4. Missing Data Handling**  
XGBoost learns a default split direction (left or right child) for missing values in each feature during training, enabling graceful degradation when sentiment or macro features are unavailable for some observations.

**5. Financial Data is Predominantly Heterogeneous Tabular Data**  
Recent empirical research (Grinsztajn et al., 2022, "Why tree-based models still outperform deep learning on tabular data") demonstrates that gradient boosted trees systematically outperform deep learning on real-world tabular datasets, particularly when:
- Sample sizes are in the range of 10,000–100,000 rows per stock
- Features are heterogeneous (different scales, distributions, semantic meanings)
- Non-monotone (non-linear) relationships are important

NSE daily trading data for 10 years gives approximately **2,500 rows per stock** — at the lower boundary where deep learning models typically struggle.

### 10.2 Why LSTM/GRU Underperform

**1. Overfitting: High Capacity Memorizes Training Data**  
An LSTM with 128+64 units and sequence length 10 has approximately **200,000+ parameters per stock**. With only ~2,000 training sequences per stock, the **effective overparameterization ratio** (parameters / training samples) is ~100, far exceeding the threshold for reliable generalization. Dropout = 0.3 mitigates but does not eliminate this.

**2. Market Regime Changes**  
LSTM learns temporal patterns from the 2015–2020 training period, but the 2022–2025 test period includes novel regimes (post-COVID recovery, interest rate hikes) with different return patterns. LSTM lacks XGBoost's robustness to these **distributional shifts** because its gated memory retains training-period-specific temporal patterns that become counter-productive in new regimes.

**3. Data Insufficiency for Sequential Models**  
LSTM models are typically trained on datasets with millions of samples (text corpora, speech signals). With 2,500 sequences per stock, the LSTM does not have enough examples to learn reliable temporal patterns — it converges to predicting the majority class (bullish, since markets trend upward) ~50% of the time.

**4. Hyperparameter Sensitivity**  
LSTM performance is highly sensitive to sequence length, layer size, dropout rate, and learning rate. A nearly optimal configuration like (sequence=10, layers=128/64, dropout=0.3, lr=0.001) may be far from optimal for any specific stock, but the computational cost of full hyperparameter search across 106 stocks is prohibitive.

**5. Strict Temporal Assumption (Sequential Bias)**  
LSTM assumes that **the most recent observations in the sequence are most informative**. However, a market pattern from 8 days ago might be more relevant than yesterday's movement in certain conditions (e.g., a candlestick pattern from 7 sessions ago triggering today). XGBoost does not impose any temporal ordering assumption.

### 10.3 The "Simpler Models" Lesson

The empirical finding of this research — that XGBoost with comprehensive feature engineering substantially outperforms deep recurrent neural networks on NSE stock prediction — aligns with a broader principle in quantitative finance: **model complexity does not substitute for data quality and domain knowledge**. The 18.28 percentage point improvement from feature engineering far exceeds any model architecture improvement, validating the "features first" philosophy.

---

## 11. Comparison with Prior Literature

| Study | Model | Accuracy | Features | Stocks |
|---|---|---|---|---|
| **This Work** | Ensemble (XGBoost+LSTM+GRU) | **68.28%** | **244** | **106** |
| Shah et al. (2021) [IEEE Access] | Bi-LSTM | 57% | 6 | 3 |
| Shrivastav & Kumar (2022) | RF + GBM | 64% | 48 | 5 |
| Chen & Guestrin (2016) [KDD] | XGBoost | 61% | 30 | — |

**This research's contributions relative to prior work:**

1. **Largest Stock Universe:** 106 stocks vs typical 3–20. Conclusions are statistically robust and not driven by cherry-picked stocks.

2. **Most Features:** 244 vs typical 10–50. The comprehensive feature set captures market dynamics that smaller feature sets miss.

3. **Multi-Target Prediction:** 4 simultaneous targets (direction, close, high, low returns) vs single-target in all cited works. This enables practical trade execution with stop-loss and take-profit.

4. **Walk-Forward Validation:** Unlike random splits used in some prior work, walk-forward validation eliminates lookahead bias and produces realistic performance estimates.

5. **68.28% vs 57–64% in prior work:** The improvement over Shah et al. (2021) is particularly notable — they used Bi-LSTM on 3 stocks with 6 features and achieved 57%. This research demonstrates that **feature richness (244 features)** and **model selection (XGBoost/Ensemble over pure LSTM)** are more important than neural network architecture complexity.

---

## 12. Conclusions

### 12.1 Primary Conclusion: XGBoost + Rich Features = Best Performance

The central empirical finding is that **XGBoost trained on 244 engineered features achieves 68.22% direction accuracy** on 106 NSE stocks (test period 2022–2025), and the **Ridge-stacked ensemble of XGBoost + LSTM + GRU achieves 68.28%** — both firmly demonstrating that the tabular feature space carries the dominant predictive information.

### 12.2 On the Value of Comprehensive Feature Engineering

The single largest driver of performance improvement was feature engineering — expanding from 72 baseline features to 244 engineered features improved accuracy from 50% (random baseline) to 68.28% (+18.28 pp). This finding validates the **feature-centric approach** to quantitative finance: understanding market microstructure well enough to encode it as machine-readable features is more valuable than selecting more powerful model architectures.

### 12.3 On Multi-Target Prediction

The simultaneous prediction of four targets (direction, close return, high return, low return) demonstrates that:
- **Direction prediction** is the most tractable target (68.28% accuracy achievable)
- **High return prediction** is the hardest target (R² negative across all models), suggesting intraday high is largely determined by unpredictable intraday microstructure
- **Close and low returns** have marginal predictability beyond direction (small positive R²)

The multi-target framework provides the actionable information needed for full trade lifecycle management (entry, stop-loss, take-profit) in a single model inference step.

### 12.4 On the Ensemble vs Individual Models

The Ensemble wins on 58/106 stocks (54.7%) while XGBoost wins on 46/106 stocks (43.4%). The marginal improvement of the Ensemble over standalone XGBoost (68.28% vs 68.22% on average) suggests that:

- For the majority of stocks, the Ensemble provides genuine improvement by intelligently weighting the base learner predictions
- For a substantial minority (43.4%), XGBoost alone is optimal — the LSTM/GRU components add noise in stocks where sequential patterns are not informative
- The Ridge meta-learner successfully suppresses the contribution of underperforming LSTM/GRU on most stocks, preventing the ensemble from significantly degrading below XGBoost's performance

### 12.5 Practical Implications for Trading

A **68% directional accuracy** represents a significant edge in financial markets. With proper risk management:
- Assuming an equal average win and average loss, 68% accuracy implies a **positive expected value per trade** of `2 × 0.68 − 1 = 0.36` (36 cents per $1 bet, ignoring transaction costs)
- In practice, the multi-target framework allows asymmetric risk management (larger take-profits than stop-losses), further enhancing the risk-adjusted return
- Transaction costs (broker commissions, market impact, bid-ask spread) for positional (multi-day holding) strategies are typically 0.05–0.20% per trade on NSE, which are well within the range that a 68% system can profitably absorb

### 12.6 Limitations

1. **Transaction Costs Not Modeled:** The reported 68.28% accuracy does not account for brokerage commissions, securities transaction tax (STT), and market impact. Full backtesting incorporating transaction costs is required for definitive profitability claims.

2. **Single Market:** All 106 stocks are from NSE India. The findings may not generalize to NYSE, London Stock Exchange, or other markets with different microstructure, regulatory environments, and investor behavior.

3. **Basic Sentiment Features:** The 15 sentiment features are derived from aggregated news scores rather than transformer-based NLP models (FinBERT, FinGPT) that can extract nuanced financial sentiment from raw text.

4. **LSTM/GRU Underperformance:** The neural network components add minimal value. Future work should investigate whether architectural improvements (Transformer models, attention mechanisms) or alternative training strategies can unlock the theoretical sequential modeling advantage.

5. **Survivorship Bias (Partial):** While the 106-stock universe includes underperformers, it excludes delisted companies. Including stocks that were delisted (due to bankruptcy, takeover, or regulatory action) during 2015–2025 would further stress-test the framework.

---

## 13. Challenges Faced

### 13.1 Data Quality Challenges

**Corporate Actions Adjustment:** Stock prices require adjustment for dividends, stock splits, bonus issues, and rights offerings. Un-adjusted prices create artificial discontinuities (a 2-for-1 stock split appears as a 50% overnight price drop) that confuse ML models. The `yfinance` library provides adjusted prices, but adjustments sometimes lag actual events by days, creating temporary data artifacts.

**Survivorship Bias:** Some stocks that were NIFTY components in 2015 were subsequently replaced due to declining performance. Including them would have required collecting historical data for a changing universe — a significant engineering challenge.

**Missing Intraday High/Low Data:** For roughly 2–3% of trading days across all stocks, the reported High equals the Open or the Low equals the Close — indicating possible data quality issues in Yahoo Finance's source data. These observations were excluded from High/Low regression training.

### 13.2 Computational Challenges

**Training 106 Models × 4 Models × 4 Targets = 1,696 Model Training Runs:** Training the full pipeline (XGBoost + LSTM + GRU + Ensemble, for 4 targets, for 106 stocks) requires significant compute time. LSTM/GRU training, in particular, requires GPU acceleration; without it, a single stock's training can take 30–60 minutes on CPU.

**Sequence Generation for LSTM/GRU:** Converting 2,500 daily OHLCV+feature rows into overlapping 10-day sequences generates 2,490 sequences × 244 features = a 607,560-element matrix per stock. Memory management for 106 stocks requires careful batching.

**Hyperparameter Tuning Scale:** Ideally, hyperparameters should be separately optimized for each of the 106 stocks. However, this would require 106 × (expensive grid search), which is computationally intractable. The solution was to use a **global hyperparameter configuration** validated on a representative subset of 10 stocks, accepting some per-stock suboptimality in exchange for feasibility.

### 13.3 Statistical Challenges

**Class Imbalance:** NSE stocks close up (bullish) approximately 52–55% of trading days historically due to the upward long-term drift of equity markets (**equity risk premium**). This mild class imbalance means a naive model can achieve ~52% accuracy by predicting "bullish" every day. The F1 score metric was used alongside accuracy to penalize such degenerate predictions.

**Non-Stationarity:** Even after feature engineering produces stationary return-based features, **volatility regimes** are non-stationary — the volatility in 2020 (COVID crash) is an order of magnitude higher than 2017. Models trained on pre-2020 data face distribution shift when evaluated on 2022–2025 data that includes new macroeconomic regimes.

**Multiple Testing Problem:** Evaluating 106 stocks creates a **multiple comparisons problem** — even if the model has zero real predictive power, by chance some stocks will show >60% accuracy in the test period. Statistical corrections (Bonferroni, Benjamini-Hochberg FDR) were applied when reporting aggregate results to ensure the overall findings reflect genuine signal rather than statistical noise.

### 13.4 Model Training Challenges

**LSTM Vanishing Gradient in Long Sequences:** Although LSTM was designed to address vanishing gradients, in practice, sequences longer than 30–50 steps still suffer from gradient decay. The sequence length was restricted to 10 to balance capturing temporal context and training stability.

**XGBoost Overfitting on Small Stocks:** For smaller-cap stocks with lower trading liquidity, even 2,500 training samples may be insufficient. Early stopping at 20 rounds and max_depth=5 constrained this, but some overfitting was observed on the most illiquid stocks in the universe.

---

## 14. Future Work

### 14.1 Reinforcement Learning Integration

The current model produces a daily directional prediction but does not directly optimize for **risk-adjusted portfolio returns**. A natural extension is to train a **Deep Reinforcement Learning (DRL) agent** (PPO, SAC, or DDPG) where:

- **State:** The 244 features + current portfolio state (holdings, unrealized P&L, cash balance)
- **Action:** For each stock: long, short, hold
- **Reward:** Daily portfolio return net of transaction costs, with penalties for excess drawdown

The DRL agent would learn to translate directional predictions into optimal position sizing decisions, accounting for correlation across the 106-stock portfolio.

### 14.2 Real-Time Trading via AngelOne SmartAPI

**Phase 2 of this research** involves deploying the trained models in a **live paper trading environment** via AngelOne SmartAPI:

1. **Data ingestion:** Live OHLCV + market context streamed at market open
2. **Feature computation:** 244 features computed in real-time on each trading day
3. **Model inference:** Ensemble prediction of all four targets in <100ms
4. **Order generation:** Long/short signals converted to limit orders with machine-generated stop-loss/take-profit prices
5. **Risk monitoring:** Position-level and portfolio-level risk limits enforced automatically

Paper trading (simulated trades without real capital) for 3–6 months will validate whether the 68.28% test accuracy translates into actual profitability before committing real capital.

### 14.3 FinBERT Sentiment Integration

Replacing the current rule-based sentiment features with **FinBERT** (a BERT language model fine-tuned on financial corpora) predictions would provide more nuanced, semantically rich sentiment features. FinBERT can distinguish between "The RBI raised rates unexpectedly" (bearish for rate-sensitive stocks) and "The company beat earnings estimates by 15%" (bullish) at the sentence level rather than relying on keyword counting.

### 14.4 Transformer Architecture

**Temporal Fusion Transformer (TFT)** and **Informer** architectures specifically designed for multi-horizon time-series forecasting have shown promise in energy and retail demand forecasting. Their **attention mechanisms** can capture non-local temporal dependencies across longer sequences (30–100 days) while dynamically weighting the most relevant historical periods — potentially overcoming LSTM's limitations for stock prediction.

---

## 15. Glossary of Scientific Terms

| Term | Full Form / Explanation |
|---|---|
| **ARIMA** | AutoRegressive Integrated Moving Average — classical time series model assuming linearity and stationarity |
| **ATR** | Average True Range — volatility measure using high-low range and gap-to-previous-close |
| **Backpropagation** | Algorithm for computing gradients in neural networks using the chain rule; propagates error signal from output to input layers |
| **Bias** (statistical) | Systematic error in model predictions; the difference between expected prediction and true value |
| **Bollinger Bands** | Upper/lower bounds = SMA ± 2σ; contract during low volatility, expand during high volatility |
| **Class Imbalance** | Unequal distribution of target classes; more "bullish" days than "bearish" days in equity markets |
| **Co-adaptation** | Tendency for neural network neurons to jointly co-train on the same patterns; prevented by dropout |
| **Correlation Coefficient (ρ)** | Pearson correlation; measures linear relationship between two variables; range [−1, +1] |
| **Dropout** | Neural network regularization: randomly zero out fraction p of activations during training to prevent overfitting |
| **Early Stopping** | Stopping training when validation loss stops improving, preventing overfitting to training data |
| **Efficient Market Hypothesis** | Theory that all available information is already reflected in asset prices |
| **Ensemble** | Combination of multiple models to produce superior predictions over any individual model |
| **Equity Risk Premium** | The excess return of equities over risk-free bonds; drives the long-term upward drift of stock markets |
| **F1 Score** | Harmonic mean of Precision and Recall; balances false positives and false negatives |
| **Fat Tails** | Return distributions with more probability mass in the tails than a Gaussian; extreme events are underestimated by normal distribution |
| **Feature Importance (Gain)** | XGBoost metric: the improvement in loss function from all splits on a feature |
| **Gradient Descent** | Optimization algorithm: iteratively moves model parameters in the direction of steepest loss decrease |
| **GRU** | Gated Recurrent Unit; simplified LSTM with update and reset gates only |
| **Heteroscedasticity** | Property of a time series where variance changes over time; common in financial returns |
| **Hyperparameter** | Model configuration not learned from data (e.g., learning rate, tree depth, sequence length) |
| **Inductive Bias** | Assumptions about the structure of the solution space that a learning algorithm embeds |
| **L1/L2 Regularization** | Penalty terms added to loss function; L1 induces sparsity (Lasso), L2 shrinks weights (Ridge) |
| **Lookahead Bias** | Using future information in constructing historical signals; renders backtest results invalid |
| **LSTM** | Long Short-Term Memory; gated RNN with forget, input, and output gates |
| **MACD** | Moving Average Convergence Divergence; momentum indicator comparing short and long EMAs |
| **MAE** | Mean Absolute Error; average magnitude of prediction errors |
| **Market Microstructure** | The study of how trading mechanisms affect price formation, bid-ask spreads, and liquidity |
| **Meta-Learner** | Second-level model in stacking that learns to combine base learner predictions |
| **Multi-Task Learning** | Training a single model to simultaneously optimize multiple related objectives |
| **Non-Stationarity** | Property of a time series whose statistical properties (mean, variance) change over time |
| **OHLCV** | Open-High-Low-Close-Volume; standard representation of a trading period's price action |
| **Overfitting** | Model memorizes training data specifics rather than learning generalizable patterns; poor test performance |
| **OBV** | On-Balance Volume; cumulative volume indicator tracking money flow direction |
| **Overparameterization** | Having more model parameters than training samples; strong tendency toward overfitting |
| **Parkinson Volatility** | High-efficiency volatility estimator using intraday high-low range |
| **Path Dependence** | Property where the outcome depends on the sequence of past events, not just the current state |
| **Precision** | Of predicted positives, fraction that are truly positive: TP / (TP + FP) |
| **R² Score** | Coefficient of Determination; proportion of variance in target explained by the model |
| **Recall** | Of true positives, fraction correctly identified: TP / (TP + FN) |
| **Regime Change** | Abrupt shift in market statistical properties (e.g., low-volatility bull market → high-volatility bear market) |
| **Regularization** | Techniques to constrain model complexity and prevent overfitting |
| **RFE** | Recursive Feature Elimination; iteratively removes least important features |
| **Ridge Regression** | L2-regularized linear regression; prevents large weight magnitudes for robust generalization |
| **RMSE** | Root Mean Squared Error; square root of MSE; in same units as target; penalizes large errors more |
| **RSI** | Relative Strength Index; momentum oscillator measuring speed and magnitude of price changes |
| **Sharpe Ratio** | Risk-adjusted return: (portfolio return − risk-free rate) / portfolio standard deviation |
| **SHAP** | SHapley Additive exPlanations; game-theoretic feature attribution for ML model interpretability |
| **Shrinkage** | Scaling each tree's contribution by learning rate; slows overfitting in boosting |
| **Stacking** | Two-level ensemble: base learners feed predictions to meta-learner for final prediction |
| **Stationarity** | Property of a process whose statistical properties do not change over time |
| **Stop-Loss** | Pre-set price at which a position is automatically closed to limit losses |
| **Survivorship Bias** | Studying only successful entities while ignoring failures; inflates retrospective performance |
| **Take-Profit** | Pre-set price at which a position is automatically closed to lock in gains |
| **Temporal Causality** | Constraint that causes precede effects in time; train on past, test on future only |
| **True Range** | Maximum of: (H−L), |H−C_prev|, |L−C_prev|; measures price range including gap |
| **Vanishing Gradient** | Problem in deep RNNs where gradients decay exponentially through long sequences, impeding learning |
| **Variance** (statistical) | Error from sensitivity to fluctuations in training data; high variance = overfitting |
| **VWAP** | Volume-Weighted Average Price; institutional benchmark for "fair" intraday execution price |
| **Walk-Forward Validation** | Time series evaluation that strictly preserves temporal ordering to prevent lookahead bias |
| **Winsorization** | Replacing extreme outliers with the Nth percentile value; reduces outlier influence on training |
| **XGBoost** | eXtreme Gradient Boosting; state-of-the-art gradient boosted tree algorithm with regularization |

---

## References

1. **Chen, T. & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*. https://arxiv.org/abs/1603.02754

2. **Cho, K. et al. (2014).** "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." *EMNLP 2014*. — Original GRU paper.

3. **Hochreiter, S. & Schmidhuber, J. (1997).** "Long Short-Term Memory." *Neural Computation*, 9(8), 1735–1780. https://www.bioinf.jku.at/publications/older/2604.pdf

4. **Shah, D. et al. (2021).** "Stock Market Prediction Using Bidirectional LSTM." *IEEE Access*. https://ieeexplore.ieee.org/document/9395265

5. **Shrivastav, S. & Kumar, S. (2022).** "Comparison of Random Forest and Gradient Boosting Machines for NSE Stock Price Prediction." *Journal of King Saud University — Computer and Information Sciences*.

6. **Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).** "Why tree-based models still outperform deep learning on tabular data." *NeurIPS 2022*. https://arxiv.org/abs/2207.08815

7. **Wilder, J.W. (1978).** *New Concepts in Technical Trading Systems.* Trend Research. — Original RSI, ATR, ADX.

8. **Bollinger, J. (2002).** *Bollinger on Bollinger Bands.* McGraw-Hill.

9. **Wolpert, D. (1992).** "Stacked Generalization." *Neural Networks*, 5(2), 241–259. — Original stacking paper.

10. **Fama, E.F. (1970).** "Efficient Capital Markets: A Review of Theory and Empirical Work." *Journal of Finance*, 25(2), 383–417.

---

*Document prepared for academic/research purposes — Pandit Deendayal Energy University, School of Technology, 2025–2026.*
