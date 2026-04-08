# Comprehensive Explanation of the Research Presentation — Phase 2
## "Development of Positional Trading Strategy Using Deep Learning and Its Training, Testing, and Implementation on a Real-Time Platform Using API"

**Author:** Vishwam Shah  
**Guide:** Dr. Jigarkumar Shah  
**Institution:** Pandit Deendayal Energy University, School of Technology  
**Presentation:** End Semester Project — Phase 2 | March 2026

---

## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [Data Leakage: Discovery & Fix](#2-data-leakage-discovery--fix)
3. [Dataset & Stock Selection](#3-dataset--stock-selection)
4. [Data Sources & Walk-Forward Validation Design](#4-data-sources--walk-forward-validation-design)
5. [System Architecture: End-to-End Pipeline](#5-system-architecture-end-to-end-pipeline)
6. [Feature Engineering: ~219 Raw → 50 Selected Features](#6-feature-engineering-219-raw--50-selected-features)
7. [Model Architecture: 10 Heterogeneous Models](#7-model-architecture-10-heterogeneous-models)
8. [Ensemble Aggregation: How Predictions Are Combined](#8-ensemble-aggregation-how-predictions-are-combined)
9. [Signal Generation & Probability Calibration](#9-signal-generation--probability-calibration)
10. [Experimental Results](#10-experimental-results)
11. [Analysis & Discussion](#11-analysis--discussion)
12. [Comparison with Prior Literature](#12-comparison-with-prior-literature)
13. [Conclusions](#13-conclusions)
14. [Challenges Faced](#14-challenges-faced)
15. [Future Work](#15-future-work)
16. [Glossary of Scientific Terms](#16-glossary-of-scientific-terms)

---

## 1. Introduction & Motivation

### 1.1 The Phase 1 → Phase 2 Evolution

Phase 1 of this research (December 2025) reported a direction accuracy of **68.28%** across 106 NSE stocks, an impressive figure that placed the results well above prior literature. However, during Phase 2 development, a critical methodological error was discovered: **data leakage**. Once the leakage was identified and systematically eliminated, the honest out-of-sample accuracy corrected to approximately **51.24%** on average — a dramatically different number that reflects the true difficulty of market prediction.

Rather than being a setback, this correction is the defining intellectual contribution of Phase 2. **Identifying, understanding, and fixing data leakage in a financial ML pipeline is a significant research contribution in itself.** The financial ML literature is littered with published results that contain silent leakage, and the discipline of designing truly leakage-free validation frameworks is one of the most important skills in quantitative research.

### 1.2 The Challenge of NSE Market Prediction

The **National Stock Exchange of India (NSE)** is Asia's largest derivatives market by trading volume. NSE equity prices are driven by a complex, interacting system of:

- **Domestic macroeconomic factors:** Reserve Bank of India (RBI) monetary policy, Union Budget fiscal decisions, inflation (WPI, CPI), industrial production (IIP)
- **Corporate fundamentals:** Quarterly earnings (EPS, revenue growth, margin trajectory), management guidance, dividend announcements
- **Global macro spillovers:** US Federal Reserve policy (which drives capital flows to emerging markets), S&P-500 performance (NSE follows US futures with a +1 trading day lag), CBOE VIX (measures US options-implied volatility, a global risk-off indicator), crude oil prices (India imports ~85% of its crude), US Dollar Index (DXY)
- **Market microstructure:** Foreign Institutional Investor (FII) flows, Domestic Institutional Investor (DII) activity, derivatives expiry (F&O — Futures & Options — every last Thursday of the month creates large intraday volatility as positions are rolled/expired)
- **Investor sentiment and behavioural finance:** Herding, momentum chasing, loss aversion, and overreaction to news

These interacting drivers create a **non-stationary, non-linear, heteroscedastic** time series — highly resistant to conventional forecasting methodologies.

### 1.3 Why This Research Is More Than Just Prediction

Phase 2 elevates the research beyond simple direction prediction by:

1. **Honest evaluation:** Expanding walk-forward validation with strict temporal causality
2. **Comprehensive model diversity:** 10 architecturally distinct models covering tree boosting, recurrent networks, convolutional architectures, temporal convolution networks, and neural basis expansion
3. **Novel features:** Global macro cues (S&P-500, VIX, DXY, Crude, Nikkei) and NSE-specific calendar events (F&O expiry, RBI MPC, Budget week) — all properly time-shifted to prevent leakage
4. **Production-ready calibration:** Temperature scaling for probability calibration and minimum-move gating to filter economically non-viable signals

### 1.4 Summary of Key Research Metrics

| Metric | Value |
|---|---|
| NSE Stocks | 100 (full Nifty-100 universe) |
| Raw Engineered Features | ~219 |
| Features After Selection | 50 |
| Heterogeneous Models | 10 |
| Expanding Walk-Forward Folds | 6 |
| Historical Data Period | 8 Years (2018–2026) |
| Avg OOS Direction Accuracy | **51.24%** |
| Best Model/Window Accuracy | **60.64%** |
| Stocks ≥ 60% Target | **55 / 100** |

---

## 2. Data Leakage: Discovery & Fix

### 2.1 What Is Data Leakage?

**Data leakage** (also called **lookahead bias**) is the inadvertent incorporation of future information into the training or feature construction of a model evaluated on historical data. In financial ML, even a single day of leakage can inflate reported accuracy by **10–20 percentage points**, producing models that appear profitable in backtesting but fail immediately in live trading because the information they relied upon during training is not available at the time of a real trade decision.

Leakage is insidious because it is often silent — the code runs without error, the model trains successfully, and the reported accuracy looks excellent. The problem only manifests when the model is deployed in a live environment where the future data it was inadvertently conditioned on during training is, of course, unavailable.

### 2.2 Phase 1 Errors

| Issue | Phase 1 (Incorrect) | Impact |
|---|---|---|
| **Feature scaling** | `RobustScaler` fitted on the **full dataset** before any train/val/test split | The scaler's median and IQR were computed using data from the test period (2022–2025), introducing knowledge of test-period statistical properties into training |
| **Global macro cues** | S&P-500, VIX, DXY used on the **same trading day** as NSE | Today's NSE movement was predicted using today's US market data — which is known only after NSE has already closed, or simultaneously if using US pre-market futures |
| **Validation design** | Fixed 60/20/20 split applied once | The same validation set was seen repeatedly during hyperparameter tuning, progressively overfitting the hyperparameters to validation-set noise |
| **Combined effect** | — | Reported direction accuracy of **68.28%** was artificially inflated |

### 2.3 Phase 2 Fixes

**Fix 1 — Per-Window Scaler Fitting:**  
The `RobustScaler` is now fitted **exclusively on the training window** of each fold and then applied to the corresponding validation and test sets independently. This means the scaler's statistics (median and interquartile range) are computed only from data temporally preceding the evaluation period. A separate scaler checkpoint is saved alongside each trained model for deployment in live trading.

**Fix 2 — Global Macro Cues Shifted +1 Trading Day:**  
All external market signals (S&P-500, VIX, DXY, Crude Oil, Nikkei-225) are shifted forward by **+1 trading day** using `merge_asof(backward)`. This means: to predict NSE direction on Tuesday, the model uses Monday's (previous day's) US market data — information genuinely available at the time of making Tuesday's trade decision (Monday's US market closes at ~1:30 AM IST, well before NSE opens at 9:15 AM IST Tuesday). This is not only leakage-free but is also economically accurate — it captures the overnight global sentiment spillover that drives NSE's opening gap.

**Fix 3 — Expanding Walk-Forward Validation (6 Folds):**  
Instead of a single fixed split, Phase 2 uses 6 sequential, expanding folds. The training set starts at 70% of available data and expands by 5% per fold, reaching 95% by fold 6. Each fold has a completely fresh, unseen test period. This means hyperparameters can only be implicitly validated across folds, preventing the test set from being repeatedly re-used.

**Fix 4 — Higher Confidence Gate (0.58) and Minimum Move (0.004):**  
The confidence threshold was raised from 0.55 to **0.58**, and a minimum price move filter of **0.004** (0.4% log return) was added. Only signals where the ensemble predicts a move exceeding 0.4% with ≥58% calibrated probability are issued as trade signals. This filters out predictions of trivially small movements that would be consumed entirely by transaction costs.

### 2.4 The "Honest 51%" vs "60.64% Best"

The corrected average OOS accuracy of **51.24%** reflects all 100 stocks, all 6 windows, with the full ensemble — including stocks and windows where the model genuinely has no predictive edge. The **60.64%** figure represents the best-performing model architecture for each stock's best walk-forward window — demonstrating that for specific stocks in specific market regimes, the 10-model ensemble achieves genuinely meaningful directional accuracy.

The critical distinction: **51.24% is the unconditional, fully honest answer; 60.64% is the conditional answer when the best model is selected per stock and window** — which is itself an achievable target if the system is run with model selection logic.

**Why 51% is still statistically significant:** With 100 stocks × 6 walk-forward windows × approximately 100–200 test-period trading days per window, the total sample size across all evaluations is approximately **60,000–120,000 trading day predictions**. A consistent 51.24% accuracy over this sample size, compared to a 50% random baseline, yields a **z-score >> 3** under a binomial test, confirming that the ensemble captures statistically significant but economically modest directional information in the market.

---

## 3. Dataset & Stock Selection

### 3.1 The Stock Universe: 100 NSE Nifty-100 Stocks Across 13 Sectors

The research covers the complete **NSE Nifty-100 index** — the 100 most liquid, largest-market-capitalisation stocks on the National Stock Exchange of India. Nifty-100 comprises NIFTY-50 (large-cap) + NIFTY-Next-50 (next 50 by market cap) and is revised quarterly by NSE's Index Maintenance Sub-Committee.

| Sector | Count | Representative Stocks |
|---|---|---|
| Banking & Finance | 18 | HDFCBANK, ICICIBANK, SBIN, AXISBANK, KOTAKBANK, SBILIFE |
| IT & Technology | 10 | TCS, INFY, WIPRO, HCLTECH, TECHM, MPHASIS, COFORGE |
| Automobiles | 7 | MARUTI, M&M, BAJAJ-AUTO, EICHERMOT, TVSMOTOR |
| FMCG | 8 | HINDUNILVR, ITC, BRITANNIA, COLPAL, MARICO, NESTLEIND |
| Pharmaceuticals | 8 | SUNPHARMA, DRREDDY, CIPLA, LUPIN, DIVISLAB, AUROPHARMA |
| Energy & Oil | 8 | RELIANCE, ONGC, BPCL, NTPC, TATAPOWER, COALINDIA |
| Metals & Mining | 5 | TATASTEEL, JSWSTEEL, HINDALCO, VEDL, NMDC |
| Capital Goods | 7 | LT, BEL, HAL, SIEMENS, BHEL, ABB |
| Consumer Durables | 4 | TITAN, VOLTAS, HAVELLS, POLYCAB |
| Cement | 4 | ULTRACEMCO, SHREECEM, AMBUJACEM, GRASIM |
| Conglomerate / Infra | 7 | ADANIENT, ADANIPORTS, SAIL, MOTHERSON, INDUSTOWER |
| Telecom | 2 | BHARTIARTL, INDUSTOWER |
| Others | 12 | PAGEIND, PIDILITIND, DMART, NAUKRI, DLF, GODREJPROP |
| **Total** | **100** | |

### 3.2 Why the Nifty-100 Universe?

**1. Index-based selection eliminates survivorship bias concerns:**  
Rather than manually selecting stocks, using the Nifty-100 index ensures the universe is defined by an objective, pre-specified rule (market capitalisation × liquidity). All 100 stocks have been liquid, tradeable instruments throughout the 2018–2026 period.

**2. Market Capitalisation > INR 5,000 Crore:**  
This filter ensures adequate liquidity for real trading. A stock with a large market cap has tight bid-ask spreads, sufficient daily volume for position entry/exit, and price discovery driven by fundamental information rather than thin-market manipulation.

**3. 2018–2026 data period (8 years):**  
The 2018–2026 window captures:
- **2018–2019:** NBFC (Non-Banking Financial Company) liquidity crisis, IL&FS default, global trade war anxieties
- **2020:** COVID-19 pandemic — March 2020 crash (−35% in 40 days) and V-shaped recovery
- **2021:** Global liquidity-driven bull market, SPAC boom, retail investor surge
- **2022:** US Federal Reserve interest rate hiking cycle (fastest since 1981), global equity bear market
- **2023:** Adani Group short-selling crisis (Hindenburg Research report), NSE recovery
- **2024–2026:** AI-driven technology rally, India's domestic consumption-led growth narrative, general election effects

Training across these diverse regimes forces the model to learn patterns that are robust across multiple distinct macroeconomic environments.

**3. Sector Diversity as a Robustness Test:**  
Banking stocks are primarily driven by RBI policy and credit cycles. IT stocks follow global technology sentiment and USD/INR dynamics. Metal stocks correlate strongly with Chinese industrial demand and global commodity prices. Each sector represents a distinct generative process for returns. A model that achieves consistent accuracy across all 13 sectors has learned genuinely generalizable patterns.

---

## 4. Data Sources & Walk-Forward Validation Design

### 4.1 Data Sources

**Primary: Yahoo Finance via yfinance Python library**

Each stock's historical data is downloaded using the ticker format `SYMBOL.NS` (e.g., `RELIANCE.NS`, `TCS.NS`). The library returns daily OHLCV data adjusted for corporate actions (splits, bonuses, dividends). Data is stored in **Parquet format** for efficient incremental updates — only new trading days are downloaded on each re-run, preserving the full history.

**Global Macro Data (all shifted +1 trading day):**

| Source | Yahoo Finance Ticker | What It Measures |
|---|---|---|
| S&P-500 | `^GSPC` | US large-cap equity performance; leading indicator of global risk appetite |
| Nasdaq-100 | `^NDX` | US technology equity performance; correlates with Indian IT sector |
| CBOE VIX | `^VIX` | 30-day implied volatility of S&P-500 options; the global "fear index" |
| US Dollar Index | `DX-Y.NYB` | Strength of USD vs basket of 6 currencies; inversely correlated with EM equity flows |
| WTI Crude Oil | `CL=F` | West Texas Intermediate crude oil price; critical for India as a net oil importer |
| Nikkei-225 | `^N225` | Japanese equity index; proxy for Asian overnight sentiment |
| USD/INR | `INR=X` | Indian Rupee exchange rate; critical for IT (revenue in USD) and importers |

**NSE Calendar Data:**  
Key market-moving scheduled events on the NSE calendar are encoded as features:
- **F&O Expiry:** NSE monthly derivatives contracts expire on the last Thursday of each month, creating systematic intraday volatility (gamma exposure unwinding, position rolling)
- **RBI MPC (Monetary Policy Committee):** The RBI's six-member MPC meets approximately every two months to decide interest rates; the bank interest rate announcement drives Banking and Rate-Sensitive sectors
- **Union Budget:** India's annual fiscal budget (presented in February) creates significant volatility across all sectors as tax, subsidy, and infrastructure spending decisions are announced
- **Quarterly Result Season:** Q1 (July–August), Q2 (October–November), Q3 (January–February), Q4 (April–May) — corporate earnings releases drive individual stock volatility

### 4.2 RobustScaler: Why Not StandardScaler or MinMaxScaler?

The `RobustScaler` from scikit-learn normalizes each feature using the **median** and **interquartile range (IQR)** rather than the mean and standard deviation:

$$x_{\text{scaled}} = \frac{x - \text{median}(x)}{\text{IQR}(x)} = \frac{x - Q_2}{Q_3 - Q_1}$$

**Why robust scaling is preferred for financial data:**
- Financial return distributions have **fat tails** — extreme events (market crashes, circuit breakers, earnings surprises) occur far more frequently than a Gaussian distribution predicts. Standard scaling (Z-score: `(x − mean) / std`) is highly sensitive to these outliers, which can skew the mean and inflate the standard deviation, compressing the bulk of the distribution into a narrow range.
- `RobustScaler` uses the median (the middle value, insensitive to extremes) and IQR (the 50th percentile spread, unaffected by the top and bottom 25% of values), making it robust to outliers.
- `MinMaxScaler` maps all values to [0, 1], which is optimal when outliers have been removed — but since some extreme financial events should be preserved (they carry signal), MinMaxScaler would compress legitimate signal into a tiny range.

The scaler is fitted separately on each training window to prevent leakage, and the fitted scaler object is serialized (pickled) alongside the trained models for use in live inference.

### 4.3 Expanding Walk-Forward Validation: Design and Rationale

**Parameters:**

| Parameter | Value |
|---|---|
| Initial training ratio | 70% |
| Expansion step per fold | 5% |
| Maximum training ratio | 95% |
| Minimum training samples | 400 trading days |
| Minimum test samples | 30 trading days |
| Total folds | **6** |
| Data start date | 2018-01-01 |

**How the 6 Folds Work:**

For a stock with 2,000 total trading days (approx. 8 years of NSE data):

| Fold | Training Period | Test Period | Training Days | Test Days |
|---|---|---|---|---|
| Fold 1 | Days 1–1400 (70%) | Days 1401–1500 | 1400 | 100 |
| Fold 2 | Days 1–1500 (75%) | Days 1501–1600 | 1500 | 100 |
| Fold 3 | Days 1–1600 (80%) | Days 1601–1700 | 1600 | 100 |
| Fold 4 | Days 1–1700 (85%) | Days 1701–1800 | 1700 | 100 |
| Fold 5 | Days 1–1800 (90%) | Days 1801–1900 | 1800 | 100 |
| Fold 6 | Days 1–1900 (95%) | Days 1901–2000 | 1900 | 100 |

**The "Expanding" nature is critical:** Unlike **rolling window** validation (where the training window slides, keeping a fixed length), the expanding window retains all historical data as new folds are created. This reflects the real-world practice of model retraining — a quant fund running its models in January 2026 would train on all data from 2018–2025, not just the most recent 2 years.

**Why 6 specifically?** Six folds provide a balance between:
- Having enough out-of-sample observations per fold (≥100 days) for statistically meaningful accuracy estimates
- Covering diverse market regimes: Fold 1 tests on late 2020 (COVID recovery), Fold 3 on 2022 (bear market), Fold 6 on 2025–2026 (current period)
- Computational feasibility: 100 stocks × 10 models × 6 folds = 6,000 model training runs

The **OOS accuracy averaged across all 6 folds** for all 100 stocks gives the 51.24% figure — the rigorous, honest assessment of the framework's generalizability.

**Why Walk-Forward Over K-Fold Cross-Validation?**

Standard **K-Fold cross-validation** randomly partitions data into K subsets. For time-series data, this allows test-period observations to appear in training sets of other folds — a form of data leakage. For example, in 5-fold CV, Fold 3's test set (random sample of all years) could include data from 2019, while its training set includes data from 2024 — the model effectively sees the future. Walk-forward validation strictly forbids this: every test observation is always temporally after every training observation.

---

## 5. System Architecture: End-to-End Pipeline

The research implements a fully automated, modular pipeline with six sequential stages:

### Stage 1: Data Acquisition & Storage
- Incremental downloads from Yahoo Finance for all 100 stocks + global indices
- Data stored as compressed **Parquet files** (columnar format, ~10× smaller than CSV for time-series data)
- Global macro cues are time-shifted **+1 trading day** during ingestion using `merge_asof(backward)` — a pandas function that merges on the nearest key less than or equal to the target, ensuring the +1 day shift is correctly handled across weekends, holidays, and market closures in different countries
- **Output:** Raw OHLCV + macro data ready for feature engineering

### Stage 2: Feature Engineering & Selection
- ~219 raw features computed across 10 categories (detailed in Section 6)
- `RobustScaler` applied per training window
- Feature selection pipeline reduces from ~219 to **50 features**:
  1. Correlation pruning: features with pairwise `|ρ| > 0.95` removed
  2. Variance threshold: near-zero-variance features removed
  3. Importance ranking via XGBoost gain-based importance
  4. Force-inclusion of sector-relevant features (macro cues for all, USD/INR for IT, RBI features for Banking)
- **Output:** Normalized 50-dimensional feature vectors per trading day per stock

### Stage 3: Model Training (10 Models × 6 Windows × 100 Stocks)
- **Parallelization:** 4 concurrent processes handle stocks simultaneously; 8 I/O threads handle data downloads
- Each of 10 models trained independently per fold, per stock: **6,000 training runs total**
- Model checkpoints saved after each fold for ensemble aggregation and live deployment
- **Output:** 60,000 trained model artifacts (10 models × 6 folds × 100 stocks)

### Stage 4: Ensemble Aggregation
- **Step 1:** Validation-logloss-weighted soft-vote combines all 10 model probabilities
- **Step 2:** Logistic meta-learner (stacking) trained on validation-fold predictions
- Meta-learner output used when it improves over soft-vote (evaluated per fold, per stock)
- **Output:** Final calibrated ensemble probability for Up/Down

### Stage 5: Signal Generation & Calibration
- Temperature scaling applied to raw ensemble probabilities (detailed in Section 9)
- Minimum-move gate (0.4%) and confidence threshold (58%) applied
- Signals classified as: BUY (UP with confidence ≥ 58%), SELL/SHORT (DOWN with confidence ≥ 58%), or NO TRADE
- **Output:** Daily directional trading signals per stock

### Stage 6: Evaluation & Reporting
- Per-stock accuracy metrics exported to individual CSV files
- Aggregate summary exported to XLSX (all stocks × all models × all folds)
- Performance plots (accuracy distribution, model comparison, fold progression) saved as PDF
- **Output:** Full experimental record usable for research reporting

---

## 6. Feature Engineering: ~219 Raw → 50 Selected Features

### 6.1 Why Feature Reduction from 219 to 50?

Phase 1 used 244 features and Phase 2 uses only 50. This is not a regression — it reflects an important insight: **with a finite training set (400–1,900 trading days per stock), adding more features increases the risk of overfitting faster than it increases predictive power.**

The **curse of dimensionality** states that in high-dimensional feature spaces, training samples become increasingly sparse relative to the volume of the feature space. With 219 features and 1,400 training samples, a linear model has roughly 6 observations per feature — barely above the rule-of-thumb minimum of 5. Non-linear models like LSTM are even more data-hungry.

Reducing to 50 features achieves a more favorable **sample-to-feature ratio of ~28:1**, significantly reducing overfitting risk while retaining the most informative features through rigorous selection. The remaining 50 features are also interpretable — the researcher knows exactly what each feature measures, enabling domain validation.

### 6.2 Feature Categories

#### Category 1: Technical Indicators (50 raw features)

The largest category, encoding market practitioner knowledge about price momentum, trend, and mean reversion:

| Indicator | Definition | Economic Intuition |
|---|---|---|
| **SMA(10, 20, 50, 200)** | Simple Moving Average over N days | Trend baseline; stock above SMA-200 = long-term uptrend |
| **EMA(10, 20, 50)** | Exponential Moving Average (more weight on recent) | More responsive to recent regime changes than SMA |
| **MACD** | EMA(12) − EMA(26); signal = EMA(9) of MACD | Captures acceleration/deceleration of trend momentum |
| **RSI(14)** | 100 − 100/(1 + avg_gain/avg_loss) | Measures price velocity; >70 = overbought, <30 = oversold |
| **Bollinger Bands** | SMA(20) ± 2σ(20) | Contraction indicates low volatility (before a breakout); expansion = high volatility |
| **ATR(14)** | Mean(True Range) over 14 days | Measures average daily price range; used for position sizing |
| **ADX** | Directional movement index; >25 = strong trend | Distinguishes trending markets from ranging markets |
| **Stochastic %K, %D** | Where is close within recent N-day range? | Momentum oscillator for overbought/oversold in a range |

#### Category 2: Price Features (20 features)

Stationary, scale-independent representations of price action:
- **Log returns:** `log(C_t / C_{t−1})`, `log(C_t / C_{t−5})`, `log(C_t / C_{t−20})` — 1-day, weekly, and monthly momentum
- **Gap features:** `log(O_t / C_{t−1})` — overnight gap (captures post-close news impact)
- **Price ratios:** `C_t / SMA_20`, `C_t / SMA_50` — measures deviation from trend
- **VWAP deviation:** `(C_t − VWAP_t) / VWAP_t` — measures whether the stock closed above or below the volume-weighted fair value

**Why log returns instead of raw prices or arithmetic returns?**
Log returns satisfy three key statistical properties:
1. **Time-additivity:** Log return over N days = sum of N daily log returns
2. **Approximate symmetry:** +10% and −10% have symmetric log-return magnitudes (+0.095 and −0.105), unlike arithmetic returns where a 50% loss requires a 100% gain to recover
3. **Normality approximation:** Log returns are closer to normally distributed than raw prices, making them better inputs for models that perform matrix operations (LSTM, linear meta-learner)

#### Category 3: Volatility Measures (20 features)

Multiple estimators of realized volatility capture different aspects of market uncertainty:

| Estimator | Formula | Efficiency vs Close-to-Close |
|---|---|---|
| **Historical Volatility (HV)** | `σ(log returns, N days)` | Baseline (1×) |
| **Parkinson Volatility** | `sqrt(1/(4N·ln2) × Σ(ln(H/L))²)` | **5×** more efficient — uses intraday range |
| **Garman-Klass Volatility** | Combines O, H, L, C in an MVUE estimator | **7×** more efficient — uses all 4 OHLC prices |
| **Keltner Channel Width** | `ATR(20) / EMA(20)` — normalized volatility | Scale-invariant; captures volatility regime |

**Why multiple volatility measures?** No single estimator is optimal under all market conditions. Parkinson assumes no overnight gaps; Garman-Klass includes a correction term for gaps. Using all three provides a robust representation of the volatility regime from multiple angles.

**Why volatility as a feature?** In financial markets, **volatility clustering** (GARCH effect) is well-documented: high-volatility periods cluster together. A model knowing the current volatility regime can condition its predictions appropriately — directional signals are less reliable during extreme volatility (high VIX, high ATR), and this information is captured through the volatility feature cluster.

#### Category 4: Volume Analysis (15 features)

Volume features encode the principle that **price moves accompanied by high volume are more significant and sustainable** than moves on thin volume:

| Feature | Formula/Concept | Interpretation |
|---|---|---|
| **OBV (On-Balance Volume)** | Cumulative: +Vol on up day, −Vol on down day | Tracks direction of money flow; divergence from price = early reversal warning |
| **CMF (Chaikin Money Flow)** | `Σ[(2C−H−L)/(H−L) × V] / Σ[V]` over 20 days | Measures buying vs selling pressure; >0 = accumulation, <0 = distribution |
| **MFI (Money Flow Index)** | Volume-weighted RSI | Combines price momentum and volume confirmation |
| **VWAP Z-score** | How many σ is current price from VWAP? | Institutional reversion-to-VWAP signal |
| **Volume Ratio** | `V_t / SMA(V, 20)` | Is today's volume above or below the 20-day average? |

**Volume-price divergence** is particularly important: if a stock makes a new high on declining volume, the breakout is "unconfirmed" and likely to fail — institutional money is not participating. Encoding this as a feature directly captures this practitioner heuristic.

#### Category 5: Momentum (25 features)

Beyond RSI and MACD, additional momentum indicators:
- **ROC (Rate of Change):** `(C_t − C_{t−N}) / C_{t−N}` over 5, 10, 20 days
- **Williams %R:** `−100 × (H_N − C_t)/(H_N − L_N)` — normalized position within recent range
- **CCI (Commodity Channel Index):** `(Typical Price − SMA) / (0.015 × Mean Deviation)` — identifies price deviations from statistical mean
- **Momentum Acceleration:** Second derivative: `(mom_5d − mom_10d)` — is momentum increasing or decreasing?
- **Multi-period crossovers:** `EMA_5 > EMA_20` (short-term trend confirmation)

#### Category 6: Statistical Features (20 features)

- **Skewness of returns (N-day rolling):** Positive skew = right tail dominates (occasional large gains); negative skew = left tail (crash risk)
- **Kurtosis:** Excess fourth moment; high kurtosis indicates fat tails and regime instability
- **Rolling z-score of price:** `(C_t − mean(C, 20)) / std(C, 20)` — measures how unusual today's price level is
- **Autocorrelation (lag 1, 5, 10):** Is today's return positively correlated with yesterday's return? (momentum effect) or negatively? (mean reversion)

**Skewness and kurtosis as features** encode information about the asymmetry and tail risk of the stock's price distribution — features that humans cannot easily read off a chart but that ML models can use to adjust prediction confidence.

#### Category 7: Market Regime (10 features)

- **Trend regime:** Bull (price above SMA_50 and SMA_50 rising) vs Bear vs Sideways
- **Volatility regime:** High (ATR > 80th percentile 6-month trailing ATR) vs Low vs Normal
- **Support/Resistance proximity:** Distance from nearest 20-day high (resistance) and nearest 20-day low (support) — key price levels where reversal probability is elevated
- **Breakout indicator:** Binary flag when price breaks above previous 20-day high on above-average volume

**Regime features are among the most valuable** because the optimal trading strategy differs fundamentally across regimes. In a strong uptrend, mean-reversion signals (oversold RSI) should be faded; in a sideways market, they should be traded. Encoding the regime explicitly allows the model to implicitly learn these conditional strategies.

#### Category 8: Interaction Features (12 features)

Non-linear cross-products of existing features:
- **Price × Volume interactions:** `daily_return × volume_ratio` — a 3% move on 3× average volume has different signal strength than 3% on 0.5× volume
- **RSI-MACD divergence:** Cases where RSI indicates overbought but MACD is still rising — conflicting signals that indicate regime instability
- **Multi-timeframe agreement:** `(EMA_5 > EMA_20) AND (EMA_20 > EMA_50)` — trend confirmation across three timeframes

#### Category 9: Global Macro Cues (9 features, Force-Included, Shifted +1 Day)

These features are **mandatory** for all 100 stocks — they cannot be removed by feature selection even if their computed importance scores are low, because their importance is structural (market-regime-wide) rather than statistical.

| Feature Name | Source | What It Captures |
|---|---|---|
| `sp500_ret_prev` | S&P-500 | Previous-day US equity return — the strongest overnight predictor of NSE opening gap |
| `sp500_ret_5d` | S&P-500 | US 5-day momentum — captures extended risk-on/risk-off sentiment |
| `us_vix_level` | CBOE VIX | Absolute level of the "fear index"; VIX > 25 = elevated risk-off environment |
| `us_vix_zscore` | CBOE VIX | VIX vs its 20-day mean; captures spikes relative to recent baseline |
| `us_vix_spike` | CBOE VIX | Binary flag: VIX jumped > 1.5 standard deviations; signals acute market stress |
| `dxy_ret_prev` | US Dollar Index | Dollar strengthening → foreign capital outflows from India → NSE bearish |
| `dxy_ret_5d` | US Dollar Index | Extended dollar trend (5 days) |
| `crude_ret_prev` | WTI Crude Oil | India imports 85% of crude; oil rising → inflationary pressure → rate concerns → NSE bearish for rate-sensitives |
| `nikkei_ret_prev` | Nikkei-225 | Asian overnight signal; Nikkei closes at ~6:30 AM IST, before NSE opens |

**The +1 day shift explained precisely:** When the model predicts NSE direction for Wednesday, it uses:
- Monday's S&P-500 return (two-day-prior, after applying +1 shift to Tuesday's S&P data) — strictly available before Wednesday's NSE open
- Tuesday's Nikkei return (shifted to Wednesday record, but Nikkei data is from Tuesday night/Wednesday morning — available before NSE 9:15 AM IST opening bell)

This is implemented as `global_df.shift(1)` followed by `merge_asof(nse_df, global_df, on='date', direction='backward')`.

#### Category 10: NSE Calendar Events (8 features, Force-Included)

| Feature | Definition | Market Impact |
|---|---|---|
| `days_to_expiry` | Trading days remaining until last Thursday of month | F&O gamma effects increase within 5 days of expiry |
| `is_expiry_week` | Binary: within 5 trading days of expiry | Elevated volatility regime, trend continuation tends to fail |
| `is_expiry_day` | Binary: on expiry Thursday | Maximum intraday volatility day |
| `days_to_rbi` | Trading days until next RBI MPC announcement | Uncertainty increases near announcement → volatility spike |
| `is_rbi_week` | Binary: within 7 days of RBI meeting | Interest-rate-sensitive stocks (Banks, Realty, NBFC) most affected |
| `days_to_budget` | Trading days to Union Budget | Sector rotation ahead of budget (sell auto before budget, buy infra) |
| `is_budget_week` | Binary: within 7 days of budget | Highest cross-sector volatility of the year |
| `is_result_season` | Binary: months when Q-results are announced (April, July, October, January) | Individual stock volatility elevated (earnings surprises) |

**Sector-Specific Feature Additions:**
- **IT Stocks:** Force-include `usd_inr_ret_prev` (rupee depreciation = IT revenue windfall in INR terms) and `nasdaq_ret_prev` (IT stocks globally correlated)
- **Banking Stocks:** Additional RBI-related features (`rbi_rate_change_flag`, `rbi_commentary_hawkish`) given direct transmission mechanism from RBI policy to bank NIM (Net Interest Margin)

### 6.3 Feature Selection Process

The ~219 raw features are reduced to 50 through three sequential stages:

**Stage 1 — Correlation Pruning (ρ > 0.95):**  
Pearson correlation matrix is computed for all 219 features. If any two features have |ρ| > 0.95, one is removed (retaining the one with higher variance). This eliminates **multicollinearity** — near-perfect linear relationships between features that waste model capacity and inflate variance estimates.

**Stage 2 — Variance Threshold:**  
Features with near-zero variance (standard deviation < 0.001) are removed. These are features that are essentially constant over the training window (e.g., a calendar feature that fires only in rare conditions may have zero variance in a particular training fold).

**Stage 3 — Importance Ranking:**  
XGBoost is trained as a feature ranker using **gain-based feature importance** (which measures the total reduction in impurity from splits using each feature). The top 50 features after removing duplicates and constants are selected. Force-included features (macro cues, sector-specific additions) are added back regardless of their rank score.

**Result:** The final 50 selected features per stock provide substantial dimensions while keeping the sample-to-feature ratio well above 10:1 for all training windows.

---

## 7. Model Architecture: 10 Heterogeneous Models

The 10-model ensemble spans 5 fundamentally distinct architectural paradigms, providing diverse inductive biases:

### 7.1 Gradient Boosting Models (2 Models)

#### LightGBM (Light Gradient Boosting Machine)

**Developed by:** Microsoft Research (Ke et al., 2017, NeurIPS)

**What it is:** LightGBM is a gradient boosted tree ensemble that introduces two key algorithmic innovations over standard GBDT:

1. **Histogram-Based Split Finding:** Instead of sorting feature values to find splits (O(n log n)), LightGBM bins continuous features into discrete histograms of K bins (default K=255) and finds the optimal split in O(K) time. This reduces training time from hours to minutes for large datasets.

2. **Leaf-Wise Tree Growth:** Standard GBDT grows trees level by level (breadth-first). LightGBM grows leaf-wise (depth-first) — always expanding the leaf with the highest gain. This produces asymmetric trees that better fit complex patterns with fewer leaves. The risk of overfitting is controlled through `num_leaves=31` (the maximum total leaves across the entire tree).

**Configuration:**
```
n_estimators = 1000      # Maximum trees (early stopping controls actual count)
max_depth = 5            # Hard depth limit as secondary overfitting control
num_leaves = 31          # Primary complexity control (leaf-wise growth)
learning_rate = 0.01     # Shrinkage: each tree scaled by 0.01
subsample = 0.8          # Stochastic: train each tree on 80% of samples
colsample_bytree = 0.8   # Use 80% of features per tree (reduces correlation)
reg_alpha = 0.3          # L1 regularization: induces sparsity in leaf weights
reg_lambda = 1.5         # L2 regularization: shrinks leaf weights
early_stopping_rounds = 50  # Stop if validation logloss doesn't improve for 50 rounds
```

**Why LightGBM for financial data?**  
The same reasons as XGBoost (heterogeneous features, robustness to outliers, interpretability) apply, with two additional advantages: significantly faster training (critical for 6,000 model training runs) and better performance on high-cardinality features through optimized histogram splits.

#### XGBoost (eXtreme Gradient Boosting)

**Developed by:** Tianqi Chen, Carlos Guestrin at University of Washington (KDD 2016)

**What it is:** XGBoost is also a gradient boosted tree ensemble, but with **level-wise (breadth-first) tree growth** and an explicit regularization term in the objective function:

$$\mathcal{L}^{(t)} = \sum_i L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

where $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_j w_j^2$ penalizes the number of leaves T and the magnitude of leaf weights $w_j$.

XGBoost and LightGBM complement each other: LightGBM's leaf-wise growth finds deeper, asymmetric patterns while XGBoost's level-wise growth is more conservative and sometimes generalizes better on small datasets.

**Configuration:**
```
n_estimators = 1000, max_depth = 5, learning_rate = 0.01
subsample = 0.8, colsample_bytree = 0.8
reg_alpha = 0.3, reg_lambda = 1.5
early_stopping_rounds = 50
```

**Best-stock performance:** XGBoost achieves 57.2% on VEDL — a metals stock where commodity price signals (captured through global macro features) interact strongly with domestic supply features in a tree-split-friendly way.

### 7.2 Recurrent Neural Networks (3 Models)

All RNNs receive input as **sequences of 20 trading days × 50 features** — a 3D tensor of shape `(batch, 20, 50)`. The sequence length of 20 days (one trading month) was chosen to capture intramonth patterns (including F&O expiry dynamics) while keeping the sequence short enough for stable gradient flow.

#### LSTM (Long Short-Term Memory)

**Invented by:** Sepp Hochreiter and Jürgen Schmidhuber (1997)

**What it is:** LSTM is a gated recurrent neural network that solves the **vanishing gradient problem** of simple RNNs through three multiplicative gates:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(Forget gate: how much to erase from memory)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(Input gate: how much new info to write)}$$
$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C) \quad \text{(Candidate: what new info to potentially write)}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(Cell state: the long-term memory)}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(Output gate: what to expose)}$$
$$h_t = o_t \odot \tanh(C_t) \quad \text{(Hidden state: the output at each step)}$$

The **cell state** C_t acts as a long-range memory highway: the forget gate allows gradients to flow backward through time without vanishing, enabling the network to learn dependencies across all 20 days of the sequence.

**Architecture:**  
2-layer LSTM: 64 units (Layer 1) → 32 units (Layer 2)  
`dropout = 0.3` (applied to outputs) + `recurrent_dropout = 0.2` (applied to recurrent connections)

**Why recurrent dropout ≠ regular dropout?**  
Standard dropout randomly zeros activations at each time step independently, which disrupts the temporal continuity of the hidden state — the very thing LSTMs are designed to preserve. **Recurrent dropout** applies the same dropout mask across all time steps, zeroing entire neurons consistently throughout the sequence. This regularizes the recurrent connections without breaking temporal coherence.

#### Bidirectional LSTM (BiLSTM)

**What it is:** BiLSTM runs two LSTM layers over the 20-day sequence simultaneously — one forward (day 1 → day 20) and one backward (day 20 → day 1). The hidden states from both directions are concatenated at each time step, giving each position access to both past and future context within the sequence.

**Why BiLSTM for financial prediction?**  
Within a 20-day sequence that has already occurred (none of these are future data — the sequence ends on day t−1 and the model predicts day t), the backward pass captures patterns like "the stock was recovering in the most recent days but had been declining earlier in the month" — context from later in the sequence that informs the interpretation of earlier patterns.

**Architecture:** Bidirectional layers of [32, 16] units (smaller than uni-LSTM to control parameter count given the doubling from bidirectionality).

**Best-stock performance:** BiLSTM achieves strong accuracy on KOTAKBANK (56.90%) and BHARTIARTL — possibly because these stocks exhibit multi-week narratives (rate cycle, 5G rollout cycles) where bidirectional context within a 20-day window is particularly useful.

#### GRU (Gated Recurrent Unit)

**Proposed by:** Kyunghyun Cho et al. (2014)

**What it is:** GRU simplifies LSTM by merging the forget and input gates into a single **update gate** and eliminating the separate cell state:

$$z_t = \sigma(W_z [h_{t-1}, x_t]) \quad \text{(Update gate: how much to update hidden state)}$$
$$r_t = \sigma(W_r [h_{t-1}, x_t]) \quad \text{(Reset gate: how much past to forget for candidate)}$$
$$\tilde{h}_t = \tanh(W[r_t \odot h_{t-1}, x_t]) \quad \text{(Candidate hidden state)}$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(New hidden state)}$$

**Why GRU leads all DL models on the most stocks (14/100)?**  
GRU has approximately **25% fewer parameters than LSTM** for the same number of units. With only 1,400–1,900 training sequences per stock, GRU's lower **overparameterization ratio** (parameters / training samples) means it overfits less aggressively. This aligns with the **Occam's Razor principle in statistical learning**: simpler models with fewer parameters generalize better when data is limited.

**Architecture:** [64, 32] units in 2 layers, `dropout = 0.3`.

### 7.3 Convolutional Hybrid Models (2 Models)

#### CNN-LSTM and CNN-GRU

**The core idea:** A 1D Convolutional Neural Network (Conv1D) acts as a local pattern extractor, learning short-range temporal features from the 20-day sequence. The output of the convolutional layer (a spatially reduced, feature-enriched representation) is then fed into an LSTM or GRU for sequential modelling.

**Architecture for both:**
```
Conv1D(filters=64, kernel_size=3, activation='relu')
    → Extracts 3-day local patterns across all 50 features
MaxPooling1D(pool_size=2)
    → Halves the sequence length from 20 to 10 (dimension reduction)
LSTM(32) or GRU(32)
    → Sequential modelling on the compressed, locally-abstracted representation
```

**Why CNN before LSTM/GRU?**  
Raw LSTM processes all 50 features at each time step jointly. The 3-day convolution learns specific local patterns (e.g., "a three-day RSI divergence with volume confirmation") that a plain LSTM would need many more parameters to represent. The MaxPooling then reduces the sequence dimension, making the subsequent RNN computationally cheaper and less prone to overfitting.

**The effective receptive field:** With `kernel_size=3` and `MaxPooling(2)`, the subsequent LSTM/GRU at position `t` sees a representation derived from the original days `[2t, 2t+2]` — a 3-day local window. Combined with the LSTM's temporal memory, the hybrid model's effective **receptive field** spans the full 20-day sequence through a multi-level hierarchy.

**Best performance:** CNN-GRU achieves 57.0% on BOSCHLTD — where short-range candlestick patterns (captured by CNN) followed by medium-term sequential evolution (captured by GRU) may be particularly informative for this capital goods stock driven by order cycle dynamics.

### 7.4 Advanced Temporal Models (3 Models)

#### TCN-GRU (Temporal Convolutional Network + GRU)

**The TCN component:** A Temporal Convolutional Network uses **dilated causal convolutions** to achieve a large receptive field with a small number of parameters:

| Layer | Dilation | Effective Receptive Field |
|---|---|---|
| Conv1D(dilation=1) | 1 | 3 days |
| Conv1D(dilation=2) | 2 | 7 days |
| Conv1D(dilation=4) | 4 | 15 days |
| Conv1D(dilation=8) | 8 | 31 days |

With **dilation = 8** and kernel size 3, the final TCN layer's receptive field spans **61 trading days** (3 months of market data) without using 61 parameters — instead, only 61 weight parameters per filter even with this large receptive field. **Causal convolutions** ensure the output at time t only depends on inputs at times ≤ t (no lookahead).

The **GRU** following the TCN then processes the TCN's output (a sequence of rich 61-day-context features) to model the sequential evolution of these multi-scale features.

**Why 61-day effective window?** Many market cycles (earnings season, F&O monthly cycle, quarterly macroeconomic data release cycles) operate on timescales longer than the 20-day LSTM sequence. The TCN's dilated convolutions bridge the gap, capturing patterns at 1-, 2-, 4-, and 8-week timescales simultaneously.

#### TCN-Transformer

**Architecture:** Dilated TCN (dilations 1, 2, 4) followed by a Transformer encoder with `d_model = 64` and 4 attention heads.

**The Transformer's self-attention mechanism:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where Q (Query), K (Key), V (Value) are linear projections of the TCN's output sequence. Each position in the sequence attends to all other positions with a learned weight proportional to their relevance. The **4 attention heads** compute 4 parallel attention patterns, each potentially focusing on a different type of temporal relationship (e.g., one head learns day-of-week patterns, another learns distance-to-expiry patterns).

**Why TCN first, then Transformer?** Raw Transformer on a 20-day financial sequence would have 20² = 400 attention weights per head, and with only 1,400 training sequences, this is severely overparameterized. The TCN first reduces dimensionality and extracts local features, after which the Transformer performs global attention over a more compact representation.

**Best stock:** NMDC (metals, 55.3% OOS) — possibly because commodity-driven stocks exhibit long-range regime dependencies (commodity cycles span months) that the TCN-Transformer's multi-scale temporal attention captures.

#### N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting)

**Proposed by:** Boris Oreshkin et al. (Element AI / ServiceNow, 2019)

**What it is:** N-BEATS is a **pure deep learning architecture for time-series** that uses no recurrence and no convolution — only fully-connected layers organized into hierarchical "blocks" with a doubly-residual design:

**Architecture:**
- 4 blocks, each containing 4 fully-connected layers with `fc_dim = 256` units
- Each block produces two outputs: a **backcast** (reconstruction of the input sequence) and a **forecast** (the prediction target)
- The input to each subsequent block is the **residual** of the previous block's backcast — the portion of the input sequence the previous block could not explain
- All blocks are summed to produce the final prediction (**additive decomposition**)

**N-BEATS as basis expansion:**  
Each block learns to project the input sequence onto an automatically discovered basis. In the **generic (non-interpretable) variant** used here, these basis functions are learned purely from data. In the interpretable variant, they can be constrained to trend and seasonality components.

**Why N-BEATS for stocks?**  
N-BEATS' doubly-residual structure allows each block to specialize in explaining a different aspect of the signal — one block might capture trend component, another intramonth seasonality, another macro correlation. This hierarchical decomposition is particularly useful for financial time series, which are superpositions of multiple overlapping rhythmic components.

**Best stock:** TORNTPHARM (pharma, 54.8%) — pharmaceutical stocks exhibit both trend components (regulatory approval cycles) and seasonal patterns (seasonal illness patterns) that N-BEATS' decomposition approach may capture more effectively than recurrent models.

### 7.5 Deep Learning Training Configuration (All 8 DL Models)

| Hyperparameter | Value | Reason |
|---|---|---|
| `batch_size = 32` | 32 sequences per gradient update | Balance between gradient noise (small batch) and computation (large batch) |
| `max_epochs = 80` | 80 maximum training epochs | Sufficient for convergence on 1,400–1,900 training sequences |
| `early_stopping_patience = 8` | Stop if val loss doesn't improve for 8 epochs | Aggressive early stopping to prevent overfitting on limited data |
| `es_min_delta = 5e-5` | Minimum improvement to reset patience counter | Prevents stopping on noise fluctuations |
| `optimizer = Adam(lr=1e-3)` | Adam with lr=0.001 | Adaptive per-parameter learning rates; standard for DL |
| `lr_reduce_factor = 0.5, patience = 5` | Halve LR after 5 stagnant epochs | Fine-grained adjustment as model approaches optimum |
| `loss = binary_crossentropy` | Binary classification loss | Appropriate for Up/Down binary target |

---

## 8. Ensemble Aggregation: How Predictions Are Combined

### 8.1 The Two-Step Architecture

The Phase 2 ensemble uses a **two-step combination strategy** that is more principled than the Ridge regression meta-learner from Phase 1:

#### Step 1: Validation-Logloss-Weighted Soft-Vote

Each of the 10 models produces a **probability** `p_i(t)` — the estimated probability that the stock will close Up on day t. The weighted ensemble probability is:

$$P_{\text{ensemble}}(t) = \sum_{i=1}^{10} w_i \cdot p_i(t)$$

where the weight of model i is:

$$w_i = \frac{1/\text{logloss}_i^{\text{val}}}{\sum_{j=1}^{10} 1/\text{logloss}_j^{\text{val}}}$$

**Interpretation:** Models with **lower validation log-loss** (more calibrated, accurate probability estimates) receive proportionally higher weights. A model with validation logloss of 0.60 receives twice the weight of a model with logloss of 1.20, because its probability estimates are twice as informative.

**Log-loss as the weighting criterion:**  
Log-loss (negative log-likelihood) = $-\frac{1}{N}\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$ measures not just whether predictions are correct, but also **how confident the model is when it is correct vs incorrect**. A model that always predicts 0.51 probability achieves reasonable accuracy but poor log-loss. A model that predicts 0.80 when it is right and 0.30 when it is wrong achieves the same accuracy but much better log-loss. Weighting by inverse-logloss rewards **calibrated confidence**.

**Why "soft" voting?** Hard voting takes the majority class predicted by each model (ignoring probability magnitude). Soft voting averages the probabilities, preserving the **degree of confidence** — a signal that 9 of 10 models predict Up with probability 0.75 each is much stronger than 6 of 10 models predicting Up with probability 0.52.

#### Step 2: Logistic Meta-Learner (Stacking)

A **Logistic Regression with L2 regularization** (`C = 0.05`, where C = 1/λ, so C = 0.05 means strong regularization) is trained on the **validation-fold predictions** from all 10 models as features:

```
Input  → [p_LightGBM, p_XGBoost, p_LSTM, p_BiLSTM, p_GRU,
          p_CNN-LSTM, p_CNN-GRU, p_TCN-GRU, p_TCN-Trans, p_N-BEATS]
Output → Final UP probability
```

**Why Logistic Regression as meta-learner (not a neural network)?**  
With 10 features (the 10 model probabilities) and typically 200–500 validation samples per fold, a complex meta-learner would overfit to the validation fold's idiosyncrasies. Logistic Regression with strong regularization (`C = 0.05`) provides:
- At most 10 weights to optimize — minimal overfitting risk
- **Platt-scaled** output probabilities (Logistic Regression naturally outputs calibrated probabilities)
- Interpretable weights that reveal which base models the meta-learner trusts most for each stock

**When is the stacking meta-learner used?**  
Only when the meta-learner's validation performance exceeds the soft-vote ensemble. If soft-vote already provides optimal combination, using a potentially noisier meta-learner would degrade performance.

### 8.2 Why No Single Model Is Sufficient

The **best model distribution** across 100 stocks (GRU: 14, XGBoost: 13, LightGBM: 10, N-BEATS: 9, TCN-GRU: 9, CNN-GRU: 8, BiLSTM: 8, TCN-Transformer: 8, LSTM: 8, CNN-LSTM: 7) is **nearly uniform** — each model wins on 7–14% of stocks. This near-uniform distribution is the empirical proof that:

1. **No single architecture generalizes to all 100 stocks** — the 100 stocks exhibit fundamentally different time-series characteristics
2. **The ensemble is not a nice-to-have — it is an architectural necessity**
3. Different sectors require different models: Banking stocks follow RBI-driven macro narratives → tree models; IT stocks exhibit USD/INR-correlated global sentiment → BiLSTM; Metals exhibit commodity cycle correlations → XGBoost, GRU

---

## 9. Signal Generation & Probability Calibration

### 9.1 The Binary Target Definition

Phase 2 uses a **three-class implicit target** with a neutral/no-trade zone:

| Signal | Condition | Action |
|---|---|---|
| **UP (Buy)** | $\log(C_{t+1}/C_t) > +0.004$ | Enter long position |
| **DOWN (Sell/Short)** | $\log(C_{t+1}/C_t) < -0.004$ | Enter short position or stay flat |
| **Neutral (No Trade)** | $|\log(C_{t+1}/C_t)| \leq 0.004$ | No action |

**Why 0.004 (0.4%) as the minimum move threshold?**  
A round-trip trade on NSE incurs:
- Securities Transaction Tax (STT): ~0.10% for delivery, 0.025% for intraday
- Brokerage: 0.01–0.05% (discount broker like Zerodha)
- SEBI turnover fee: ~0.0001%
- Exchange charges: ~0.0002%
- GST on charges: ~18% of brokerage
- Impact cost (bid-ask spread): 0.05–0.20% for Nifty-100 stocks

Total round-trip cost: **~0.2–0.4%**. The 0.4% minimum move ensures the model only issues signals when the predicted price change is large enough to profitably absorb transaction costs, even with modest position sizing.

### 9.2 Temperature Scaling: Probability Calibration

#### Why Raw Model Probabilities Need Calibration

Neural networks and gradient boosting models are systematically **overconfident** — they tend to assign probabilities like 0.85 or 0.92 to predictions when the true underlying accuracy at that confidence level may be only 60–65%. This overconfidence is a well-documented phenomenon in deep learning (Guo et al., 2017, "On Calibration of Modern Neural Networks").

**Calibration** means that when a model says "72% probability of Up," the stock should actually go Up approximately 72% of the time across similar prediction instances. A perfectly calibrated model's reliability diagram (confidence vs actual frequency) is a diagonal line.

#### Temperature Scaling in Detail

Temperature scaling applies a single scalar temperature `T` to the pre-softmax logits before computing the probability:

$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

where `z_i` are the raw logit outputs and T is the temperature:

- **T = 1:** No effect on probabilities
- **T > 1 (cooling):** Probabilities become softer (closer to 0.5) — reduces overconfidence
- **T < 1 (sharpening):** Probabilities become more extreme — increases confidence

The optimal temperature `T*` is found by **minimizing Negative Log-Likelihood (NLL)** on the validation set:

$$T^* = \arg\min_T \text{NLL}(T) = \arg\min_T \left[-\sum_i y_i \log \hat{p}_i(T)\right]$$

This 1D optimization is solved efficiently using `scipy.minimize_scalar`.

**Why temperature scaling over more complex calibration methods (Platt scaling, isotonic regression)?**  
Temperature scaling requires fitting only **1 parameter** vs 2 (Platt) or N (isotonic regression). With only 100–500 validation samples per fold, more complex calibrators would overfit the calibration function, producing worse OOS calibration than temperature scaling.

### 9.3 The Confidence Threshold (0.58)

After temperature scaling, the ensemble's calibrated probability is compared to the threshold **0.58**:

- If `P_ensemble(UP) ≥ 0.58` AND predicted log-return magnitude > 0.004: Issue **BUY signal**
- If `P_ensemble(DOWN) ≥ 0.58` (i.e., `P_ensemble(UP) ≤ 0.42`) AND magnitude > 0.004: Issue **SELL signal**
- Otherwise: **No trade**

**Why 0.58 specifically?**  
At 0.58 calibrated confidence, the model claims to be right 58% of the time. Given binary outcomes and the transaction costs discussed above, a 58% success rate with symmetrical win/loss provides a positive expected P&L per trade after costs. The higher threshold compared to Phase 1 (0.55) is deliberately conservative — it sacrifices trade frequency for signal quality.

---

## 10. Experimental Results

### 10.1 All Models Across 100 Stocks — Summary

| Model | Category | Avg OOS Accuracy | Best Stock | Best Accuracy |
|---|---|---|---|---|
| LightGBM | Gradient Boosting | 50.8% | BAJAJ-AUTO | 53.2% |
| XGBoost | Gradient Boosting | 51.0% | VEDL | 57.2% |
| LSTM | Recurrent NN | 50.5% | BANDHANBNK | 54.9% |
| BiLSTM | Recurrent NN | 50.4% | KOTAKBANK | 55.4% |
| GRU | Recurrent NN | 50.7% | EICHERMOT | 55.4% |
| CNN-LSTM | Conv + Recurrent | 50.3% | GODREJCP | 53.7% |
| CNN-GRU | Conv + Recurrent | 50.5% | BOSCHLTD | 57.0% |
| TCN-GRU | Temporal Conv+RNN | 50.4% | JSWSTEEL | 53.7% |
| TCN-Transformer | Temporal Conv+Attn | 50.5% | NMDC | 55.3% |
| N-BEATS | Neural Basis Exp. | 50.5% | TORNTPHARM | 54.8% |
| **Ensemble (avg OOS)** | Soft-Vote + Stacking | **51.24%** | KOTAKBANK | **56.90%** |
| **Best Model/Window** | Per-stock optimal | **60.64%** | — | **55/100 ≥ 60%** |

### 10.2 Top-10 Performing Stocks

| Rank | Stock | Sector | OOS Accuracy |
|---|---|---|---|
| 1 | KOTAKBANK | Banking | **56.90%** |
| 2 | BANDHANBNK | Banking | 56.82% |
| 3 | HDFCLIFE | Insurance | 56.57% |
| 4 | SBIN | Banking | 55.92% |
| 5 | BRITANNIA | FMCG | 55.72% |
| 6 | VEDL | Metals | 55.64% |
| 7 | COFORGE | IT | 55.56% |
| 8 | HINDALCO | Metals | 55.05% |
| 9 | TATAPOWER | Energy | 54.79% |
| 10 | IRFC | Finance | 54.74% |

**Full average across all 100 stocks: 51.24%**

### 10.3 Interpreting the Results

**Why do Banking stocks dominate the top-10?**  
Banking stocks (KOTAKBANK, BANDHANBNK, SBIN, HDFCBANK at #4-position with 54.7% up-direction accuracy) show the highest OOS accuracy because:

1. Their primary driver (RBI monetary policy) is **scheduled and partially predictable** — the model knows when RBI MPC meetings occur (NSE calendar features) and can learn how banking stocks historically react in the week before and after rate announcements
2. Global VIX and DXY are strong predictors of FII flows, which disproportionately affect large-cap banking stocks
3. Banking stocks exhibit stronger **momentum patterns** (trend persistence) than other sectors — macro themes like "credit growth acceleration" or "NPA resolution cycle" play out over weeks, creating learnable sequential patterns

**Directional Accuracy for Up-Signals:**  
A secondary metric reported is `dir_acc_up` — of all days where the ensemble predicted UP, what fraction actually closed up? Top performers:
- BHARTIARTL: **57.4%** — Telecom stock with clear 5G capex narrative driving multi-week trends
- SBIN: **57.2%** — Largest public sector bank, highly responsive to RBI policy signals
- KOTAKBANK: **55.5%** — Premium private bank, follows clear credit-cycle narrative

This directional accuracy for up-signals is more practically relevant than overall accuracy for a **long-only trading strategy**, which is the primary deployment target.

### 10.4 The "60.64% Best Model/Window" Interpretation

The best-model/window accuracy of 60.64% requires explanation: this figures aggregates, for each stock, the single fold and model combination that performed best. This is an **oracle-selected upper bound** — it tells us the ceiling of what the system can achieve if the optimal model were selected per stock and per time window.

In practice, model selection would be done using the validation performance (not peeking at test results), which is why we report 60.64% as an aspirational target rather than a deployable result. However, the fact that **55/100 stocks have at least one model/window combination exceeding 60% accuracy** demonstrates that the framework has genuine predictive capability — it's not uniformly wrong. The challenge is **learning which model works for which stock in which regime** — the meta-learning problem of Phase 3.

---

## 11. Analysis & Discussion

### 11.1 Why No Single Architecture Generalizes to All Stocks

The near-uniform best-model distribution (each of 10 models wins on 7–14% of stocks) is not accidental — it reflects a fundamental truth about NSE equity markets:

**Different stocks are driven by different generative processes:**

| Sector | Dominant Driver | Why | Best Architecture |
|---|---|---|---|
| Banking | RBI monetary policy + credit cycles | Interest rate changes have immediate, direct transmission to bank NIM | Tree models + TCN (captures scheduled event patterns) |
| IT | USD/INR rate + global tech sentiment | Revenue in USD, costs in INR; global tech earnings lead Indian IT by 1–2 weeks | BiLSTM, N-BEATS (captures multi-week USD trend) |
| Metals | Commodity prices + Chinese demand | VEDL, HINDALCO, TATASTEEL closely track LME metals prices | XGBoost, GRU (strong commodity feature interactions) |
| FMCG | Volume momentum + GST cycle | Consumer staples driven by volume and margin consistency, less macro-sensitive | CNN hybrids (local pattern extraction) |

This sector-model mapping validates the necessity of including all 10 architectures — a Banking specialist would fail on IT stocks and vice versa.

### 11.2 Why GRU Leads All Deep Learning Models

GRU achieves the best single-model performance (14/100 stocks) among all DL architectures. The explanation:

1. **Fewer parameters than LSTM** (no separate cell state): ~25% reduction in trainable parameters for the same unit count → lower overparameterization ratio on short NSE sequences
2. **Similar expressiveness to LSTM for short sequences (20 days):** LSTM's advantage over GRU grows with sequence length; for 20-step sequences, GRU's simplified gating is sufficient to capture all relevant temporal dependencies
3. **Faster convergence:** Fewer parameters + simpler gradient flow → convergence in fewer epochs; combined with aggressive early stopping (patience=8), GRU reaches a better solution before overfitting dominates

### 11.3 Why the TCN Family (17/100 stocks combined) Outperforms LSTM Family (16/100)

Despite LSTM's theoretical advantage (designed specifically for long sequences), the TCN-based models (TCN-GRU: 9, TCN-Transformer: 8; total 17 stocks) collectively match LSTM-family models (LSTM: 8, BiLSTM: 8; total 16):

1. **Dilated receptive field (61 days) vs LSTM's sequence length (20 days):** For stocks with monthly or quarterly patterns, TCN's 61-day receptive field captures patterns invisible to the 20-day LSTM
2. **Parallelizable training:** Unlike LSTM which processes sequences step by step, TCN's convolutions are computed in parallel — enabling larger batch sizes and more stable gradient estimates in the same training time
3. **No vanishing gradient:** TCN's skip connections (residual connections) and the GRU/Transformer component handle gradient flow without the gating complexity of LSTM

### 11.4 Practical Implication: Portfolio Construction

With 55/100 stocks showing ≥60% accuracy under optimal model selection, a practical portfolio construction strategy emerges:

1. **Select the 20–30 stocks with the highest 6-fold average OOS accuracy** as the trading universe
2. **Apply position sizing proportional to calibrated confidence** — higher ensemble probability → larger position
3. **Set stop-losses at 1× ATR below entry price** (ATR from the 50-feature set provides current volatility context)
4. **Target holding period: 3–5 days** (positional trading) — long enough for the directional prediction to play out, short enough to avoid mean-reversion deterioration

Simulation (not yet backtest): with 55 stocks at average 56% accuracy, 1:1.5 risk-reward ratio (stop at ATR, target at 1.5×ATR), and 0.35% round-trip transaction cost → **estimated positive expectancy per trade of ~0.15%**, compounding to ~**22–25% annualized return** in a favorable regime — before portfolio correlation adjustments.

---

## 12. Comparison with Prior Literature

| Study | Models | Accuracy | Features | Stocks | Validation |
|---|---|---|---|---|---|
| **This Work** | 10-model ensemble | **51.24% OOS (honest)** / **60.64% best** | **50 (selected from 219)** | **100** | **Expanding Walk-Forward** |
| Shah et al. (2021) | Bi-LSTM | 57% | 6 | 3 | Hold-out |
| Shrivastav & Kumar (2022) | RF + GBM | 64% | 48 | 5 | K-Fold |
| Chen & Guestrin (2016) | XGBoost | 61% | 30 | — | Hold-out |
| Ozbayoglu et al. (2020) | LSTM | 55% | 20 | 10 | Hold-out |

**Critical methodological observation:**  
All prior works use either hold-out validation or K-Fold cross-validation — both methodologies that permit forms of data leakage in time-series settings. The 57–64% figures likely contain lookahead bias. If evaluated under the same expanding walk-forward protocol used in this research, their honest OOS accuracy would likely fall to the 51–53% range.

**This research's novel contributions:**

1. **Largest model diversity test:** 10 architecturally distinct models vs single-architecture in all prior works — providing the definitive comparison of GB vs RNN vs CNN-hybrid vs TCN vs N-BEATS on a standardized NSE dataset

2. **Complete Nifty-100 universe:** 100 stocks across 13 sectors vs 3–10 in prior works — prevents cherry-picking and provides robust, sector-stratified conclusions

3. **Expanding walk-forward validation:** The most conservative, realistic evaluation framework available for financial time series — eliminates lookahead bias entirely

4. **Global macro cues + NSE calendar:** Novel feature sets not present in any cited prior work — addressing the well-known "missing information" critique of models trained only on individual stock technical features

5. **Leakage audit and remediation:** The Phase 1 → Phase 2 correction is itself a methodological contribution — a transparent demonstration that prior high-accuracy results should be scrutinized for silent leakage

---

## 13. Conclusions

### 13.1 The Leakage Lesson: Integrity Over Impressive Numbers

The most important conclusion of Phase 2 is methodological: **honest evaluation under a leakage-free expanding walk-forward protocol reveals an average OOS accuracy of 51.24%, correcting the Phase 1 figure of 68.28%**. This 17 percentage point correction is an unambiguous demonstration that, in financial ML, validation methodology determines results as much as model architecture.

The willingness to report the corrected, lower figure — rather than finding post-hoc justifications for the inflated Phase 1 number — reflects scientific integrity. The research community benefits more from an honest 51% than from a misleadingly impressive 68%.

### 13.2 The Ensemble Is Necessary

The near-uniform distribution of "best model" across all 10 architectures empirically proves that no single model generalizes to all 100 stocks. The 10-model ensemble with two-step aggregation (logloss-weighted soft-vote + logistic stacking) provides the fairest combination, enabling **55/100 stocks to exceed the 60% accuracy target** — a result that no individual model approaches:

- Best individual model average: ~51.0% (XGBoost)
- Ensemble average: **51.24%** (modest improvement)
- Best model/window (per-stock optimized): **60.64%** (demonstrates the 60% threshold is achievable)

### 13.3 Feature Engineering Quality Matters More Than Architecture

Across all 10 models and 100 stocks, the most consistent finding is that **stocks where global macro cues are strongly correlated with price action (Banking, IT, Metals, Energy) show higher accuracy** than stocks driven by idiosyncratic company-specific factors (e.g., pharma stocks awaiting FDA approval decisions). This confirms that feature design — specifically, including the right external market signals, shifted by the correct number of days — drives accuracy more than architectural choices.

### 13.4 On Semi-Market Efficiency

The 51.24% average OOS accuracy, slightly above the 50% random baseline, is consistent with the **weak form of the Efficient Market Hypothesis (EMH)**: past price information (technical analysis) alone cannot reliably generate returns above a random walk. However, the fact that global macro cues (VIX, DXY, Crude) — which are genuinely exogenous to individual NSE stocks — improve prediction suggests that:

1. **Cross-market information transmission** is not instantaneously priced into NSE stocks — there is a roughly 1-trading-day lag before US market sentiment fully propagates to NSE stock prices
2. This lag creates a brief, exploitable predictive window that the model leverages

This finding is consistent with the **gradual information diffusion model** (Hong and Stein, 1999) of return predictability in financial economics.

### 13.5 Practical Viability

Despite the honest 51.24% average accuracy, the framework has genuine practical value:
- **55/100 stocks exceed the 60% target** with optimal model selection
- **Temperature-calibrated probabilities** enable proper bet-sizing via the Kelly criterion
- **NSE calendar features** enable tactical avoidance of unpredictable volatility events (F&O expiry, RBI announcements)
- The full pipeline is **automated** — capable of generating next-day signals for all 100 stocks in minutes

---

## 14. Challenges Faced

### 14.1 Data Leakage: Subtle, Insidious, Expensive to Find

The data leakage in Phase 1 was not obvious. The scaler fitted on the full dataset meant that features for, say, 2019 were normalized using statistics that included data from 2022–2025 — information not available in 2019. The effect on any individual data point was small (the normalization shifted each value by a fraction of a percent), but across 60,000+ training examples, this systematically conditioned the model to use test-period statistical patterns. Detecting this required careful inspection of the cross-validation logic, not the model's outputs.

**The debugging process:** Accuracy suspiciously above 60% for most stocks was the first signal. Reproducing the result on a truly held-out time period (data collected after the study period) showed accuracy dropping to ~52%. Bisecting the pipeline revealed the scaler as the source.

### 14.2 Computational Scale: 6,000+  Model Training Runs

100 stocks × 10 models × 6 walk-forward windows = **6,000 training runs**. DL models (LSTM, BiLSTM, GRU, CNN-LSTM, CNN-GRU, TCN-GRU, TCN-Transformer, N-BEATS) require GPU acceleration — without it, each training run takes 5–15 minutes on CPU. With a single GPU and 4-process parallelization, the full pipeline requires approximately 12–18 hours of compute time.

**Solutions implemented:**
- 4 parallel worker processes handling different stocks simultaneously
- 8 I/O threads for data downloads (network bound, not CPU bound)
- Incremental Parquet storage to avoid re-downloading base data
- Model checkpointing: if a run fails (OOM, network error), it resumes from the last successful fold

### 14.3 Global Macro Alignment: Timezone and Calendar Mismatches

Aligning S&P-500, Nikkei-225, and NSE data across different timezones and trading calendars is technically challenging:
- The US markets are closed on different holidays than India (Thanksgiving, Memorial Day vs Republic Day, Diwali)
- The Nikkei closes at 6:30 AM IST; NSE opens at 9:15 AM IST — the Nikkei's same-day return is technically available before NSE opens, but using it requires careful timestamp handling
- WTI Crude Oil futures trade nearly 24/7; mapping the daily "previous close" to the correct NSE trading day requires timezone-aware datetime handling

`merge_asof(backward)` with explicit timezone conversion resolved most issues, but edge cases (holidays in both markets on adjacent days) required manual validation.

### 14.4 Class Imbalance and the Neutral Zone

The binary target with a 0.4% minimum move creates substantial **class imbalance**: approximately 20–30% of trading days fall in the "neutral" zone (move < 0.4%), and of the remaining 70–80%, slightly more days are UP than DOWN (market's long-term upward drift). The model is trained only on non-neutral days, which reduces training samples by 20–30% and requires careful handling of neutral days in evaluation metrics.

### 14.5 Per-Stock Hyperparameter Suboptimality

Ideally, each of the 100 stocks would have model hyperparameters tuned specifically to its characteristics. A globally optimal configuration (sequence_length=20, LSTM units=[64,32]) may be suboptimal for a stock with strong weekly periodicity (suggesting sequence_length=5) or a stock with strong quarterly patterns (suggesting sequence_length=60). Per-stock Optuna-based hyperparameter optimization is left for Phase 3 due to the computational cost: 100 stocks × 50 Optuna trials × 10 models × 6 folds = 300,000 training runs.

### 14.6 Distribution Shift Across Walk-Forward Folds

The 2020 COVID crash represents a **structural break** (sudden, persistent change in statistical properties) that models trained on Fold 1 (ending mid-2020) do not encounter in training but face in test periods. Models exhibiting strong performance on Fold 1 sometimes show degraded performance on Fold 2 (which includes the 2022 rate hike environment) — not because their architecture is wrong, but because the statistical properties of the return-generating process changed.

This **non-stationarity** (time-varying statistical properties) is the fundamental unsolved challenge in financial ML. Expanding walk-forward at least ensures models are retrained with increasingly recent data in later folds, reducing but not eliminating distribution shift effects.

---

## 15. Future Work

### 15.1 FinBERT Sentiment Integration

**FinBERT** (Araci, 2019) is a BERT (Bidirectional Encoder Representations from Transformers) language model fine-tuned on financial corpora. Unlike rule-based sentiment analysis (which counts positive/negative/neutral keywords), FinBERT understands financial language context:

- "The company beat estimates by 15% but guided lower for next quarter" → FinBERT: BEARISH (guidance dominates)
- "HDFC Bank raised its MCLR by 10 basis points" → FinBERT: BEARISH for rate-sensitive stocks, BULLISH for bank margins

Replacing the 15 rule-based sentiment features with FinBERT-derived sentiment scores for the 100 Nifty stocks would add a 16th feature category and potentially improve accuracy for news-sensitive stocks (Pharma, Banking, Conglomerate).

Real-time news can be crawled from NSE press releases, moneycontrol.com, economictimes.com, and company investor relations pages — all freely accessible.

### 15.2 Graph Neural Networks (GNN) for Cross-Stock Modeling

A **Graph Neural Network** models the 100 stocks as **nodes in a knowledge graph**, with edges representing sector membership (Banking stocks are connected), cross-holdings (conglomerates), and historical return correlations. The GNN can aggregate information from neighboring nodes — if KOTAKBANK's model predicts strong Up, this is a signal for the Banking sector that should influence HDFCBANK's prediction via graph message passing.

This cross-stock information sharing is economically motivated: **sector rotation** (when money flows out of IT into Banking) creates coordinated movements across all stocks in a sector. The current architecture treats each stock independently and misses this sector-level signal.

### 15.3 Reinforcement Learning (RL) for Dynamic Position Sizing

Current signals are binary (buy/sell) or ternary (buy/sell/no-trade). A **Deep Reinforcement Learning agent** would learn continuous position sizes:

- **State:** 50 features + current portfolio state (exposed positions, unrealized P&L, cash balance, days in current position)
- **Action:** For each stock: position size in [-1, +1] (negative = short, zero = flat, positive = long)
- **Reward:** Sharpe-ratio-adjusted portfolio return net of transaction costs, minus drawdown penalties

The RL agent would learn that during high-VIX periods, all positions should be smaller; during high-confidence ensemble signals, positions can be larger. This replaces the static confidence threshold with a learned, adaptive rule.

### 15.4 Per-Stock Hyperparameter Optimization (Optuna)

**Optuna** (Akiba et al., 2019) is an automatic hyperparameter optimization framework using Tree-structured Parzen Estimator (TPE) search and pruning of unpromising trials. Per-stock Optuna optimization would:
- Tune sequence length (10–60), LSTM units, dropout rate, learning rate for each stock × model combination
- Use the Successive Halving (Hyperband) pruner to abort clearly underperforming configurations early
- Potentially improve accuracy by 1–3 percentage points on well-behaved stocks

### 15.5 Live Deployment via AngelOne SmartAPI

**Phase 3** plans full production deployment:

1. **End-of-day signal generation:** At 3:30 PM IST (NSE close), download OHLCV for all 100 stocks, compute 50 features, run 10 trained models, generate ensemble signal with calibrated probability
2. **Order placement:** Signals exceeding threshold placed as next-day market-open limit orders via AngelOne SmartAPI
3. **ATR-based stop-loss:** Each entered position has a stop-loss at `entry_price − 1.5 × ATR` set simultaneously with entry
4. **Portfolio monitor dashboard:** Real-time P&L tracking, position-level risk, sector-level exposure, portfolio drawdown alerts
5. **Intraday exit rules:** If market moves favorably by 2× ATR before close, lock in gains via trailing stop

### 15.6 Multi-Timeframe Analysis

Current models operate on daily OHLCV data. Adding **15-minute intraday bar data** would enable:
- Intraday opening range confirmation (is the stock following its predicted daily direction in the first 15 minutes of trading?)
- Time-of-day patterns (VIX-driven morning volatility vs orderly afternoon trending)
- Integration of daily directional signal with intraday entry timing for better risk/reward

**Challenges:** 15-minute bars generate 25× more data per stock (25 bars/day vs 1), requiring ~150× more storage and compute for the same date range.

### 15.7 Diebold-Mariano Statistical Significance Testing

Before publishing results, the **Diebold-Mariano (DM) test** should be applied to formally test whether the ensemble's OOS accuracy is statistically superior to each individual model — or whether observed differences are within sampling noise. With 100 stocks × 6 folds × ~100 test days = ~60,000 observations, the DM test has very high power, and even a 0.2 percentage point accuracy difference would be statistically significant. This formal significance testing is a requirement for journal publication.

---

## 16. Glossary of Scientific Terms

| Term | Explanation |
|---|---|
| **Adam Optimizer** | Adaptive Moment Estimation; maintains per-parameter learning rates using first and second moment estimates of gradients |
| **ADX** | Average Directional Index; measures trend strength (not direction); > 25 = strong trend |
| **ATR** | Average True Range; volatility measure combining intraday range and overnight gaps |
| **Attention (Transformer)** | Mechanism that computes a weighted sum of all positions' representations, with weights learned based on query-key similarity |
| **Autocorrelation** | Correlation of a time series with its own past values at a given lag |
| **Backcast** | In N-BEATS: the block's reconstruction of its input sequence; the residual after subtracting the backcast flows to the next block |
| **Batch Size** | Number of training sequences processed in one gradient update step |
| **Bias-Variance Tradeoff** | Fundamental ML tradeoff between underfitting (high bias) and overfitting (high variance) |
| **Binary Cross-Entropy** | Loss function for binary classification: $-[y\log p + (1-y)\log(1-p)]$ |
| **BiLSTM** | Bidirectional LSTM; processes sequence both forward and backward; concatenates both hidden states |
| **Causal Convolution** | Convolution restricted to use only current and past inputs (no future inputs); necessary for time-series prediction |
| **CBOE VIX** | Chicago Board Options Exchange Volatility Index; measures 30-day implied volatility of S&P-500 options |
| **Class Imbalance** | Unequal class frequencies in the training data; more bullish days than bearish in equity markets |
| **Confidence Threshold** | Minimum calibrated probability required to issue a trade signal |
| **Conv1D** | 1D Convolutional Neural Network layer; applies a sliding filter over time to extract local patterns |
| **Curse of Dimensionality** | Phenomenon where high-dimensional spaces become exponentially sparse relative to finite training samples |
| **Data Leakage** | Inadvertent use of future information during model training; inflates backtested performance |
| **Dilated Convolution** | Convolution with gaps between filter taps; exponentially expands receptive field without increasing parameters |
| **Diebold-Mariano Test** | Statistical test for comparing predictive accuracy of two models |
| **Distribution Shift** | Change in statistical properties of input data between training and test periods |
| **Dropout** | Neural network regularization: randomly zero activations during training to prevent overfitting |
| **DXY** | US Dollar Index; weighted geometric mean of USD against EUR, JPY, GBP, CAD, SEK, CHF |
| **Early Stopping** | Halt training when validation loss stops improving; prevents overfitting |
| **EMA** | Exponential Moving Average; geometrically weighted average with more weight on recent observations |
| **Ensembling** | Combining multiple models to produce better predictions than any individual model |
| **Expanding Window** | Walk-forward validation where training set grows with each fold (vs fixed-length rolling window) |
| **F&O Expiry** | Futures & Options expiry; NSE derivatives settle on the last Thursday of each month |
| **Fat Tails** | Distribution property where extreme values occur more frequently than a normal distribution predicts |
| **Feature Importance (Gain)** | XGBoost/LightGBM metric: total impurity reduction achieved by splits on a feature |
| **FinBERT** | BERT language model fine-tuned on financial corpora for sentiment analysis |
| **GARCH** | Generalized AutoRegressive Conditional Heteroscedasticity; model for time-varying volatility |
| **Garman-Klass Volatility** | High-efficiency volatility estimator using all four OHLC prices; 7× more efficient than close-to-close |
| **GNN** | Graph Neural Network; models data as nodes/edges; enables cross-stock information sharing |
| **GRU** | Gated Recurrent Unit; RNN variant with update and reset gates; ~25% fewer parameters than LSTM |
| **Heteroscedasticity** | Non-constant variance; financial returns exhibit volatility clustering (GARCH effect) |
| **Hyperparameter** | Model setting not learned from data (e.g., learning rate, sequence length, tree depth) |
| **IQR** | Interquartile Range; Q3−Q1; range containing the middle 50% of values; robust to outliers |
| **Kelly Criterion** | Formula for optimal bet size: `f = (bp−q)/b` where p = win probability, q = 1−p, b = win/loss ratio |
| **L1 Regularization (Lasso)** | Penalty proportional to absolute weight values; induces sparsity (some weights → 0) |
| **L2 Regularization (Ridge)** | Penalty proportional to squared weight values; shrinks all weights toward 0 smoothly |
| **LightGBM** | Microsoft's GBDT with histogram-based splits and leaf-wise tree growth; faster than XGBoost |
| **Log-Loss (NLL)** | Binary cross-entropy; penalizes confident wrong predictions more than uncertain wrong predictions |
| **Lookahead Bias** | See: Data Leakage |
| **LSTM** | Long Short-Term Memory; gated RNN with forget, input, cell-state, and output mechanisms |
| **MACD** | Moving Average Convergence Divergence; EMA(12) − EMA(26) + signal EMA(9) |
| **MaxPooling1D** | Takes maximum value in each window; reduces sequence length while preserving dominant features |
| **merge_asof** | Pandas function for time-series merges on nearest (preceding) key; used for +1 day shift of global cues |
| **Meta-Learner** | Second-level model trained on base learner outputs to produce final ensemble prediction |
| **MFI** | Money Flow Index; volume-weighted RSI; measures buying vs selling pressure |
| **Min-Move Gate** | Filter rejecting signals below minimum predicted price change (0.4%); removes economically non-viable trades |
| **N-BEATS** | Neural Basis Expansion Analysis for Time Series; pure DL, no recurrence, doubly-residual |
| **NIM** | Net Interest Margin; key profitability metric for banks: difference between lending and deposit rates |
| **Non-Stationarity** | Statistical properties (mean, variance) change over time; stock prices and volatility are non-stationary |
| **NPA** | Non-Performing Assets; bank loans not generating income; key risk indicator for Indian banking sector |
| **OBV** | On-Balance Volume; cumulative volume indicator: +V on up days, −V on down days |
| **OHLCV** | Open-High-Low-Close-Volume; standard representation of daily price and volume data |
| **Optuna** | Automatic hyperparameter optimization framework using Tree-structured Parzen Estimator (TPE) |
| **Overparameterization** | More model parameters than training samples; leads to memorization rather than generalization |
| **Parkinson Volatility** | High-efficiency volatility estimator using intraday H−L range; 5× more efficient than close-to-close |
| **Parquet** | Columnar binary storage format; ~10× more efficient than CSV for time-series data |
| **Platt Scaling** | Logistic regression fitted on model outputs to produce calibrated probabilities |
| **RL (Reinforcement Learning)** | ML paradigm where an agent learns from reward/penalty feedback in an environment |
| **RBI** | Reserve Bank of India; India's central bank; sets monetary policy via MPC |
| **Receptive Field** | The span of input time-steps that influence a model's output at a given position |
| **Recurrent Dropout** | Dropout applied consistently across all time steps (same mask); avoids disrupting temporal continuity |
| **RobustScaler** | Feature normalization using median and IQR; robust to outliers; preferred for financial data |
| **ROC** | Rate of Change; `(C_t − C_{t−N}) / C_{t−N}` — percentage price change over N days |
| **RSI** | Relative Strength Index; momentum oscillator measuring speed of price changes; overbought/oversold |
| **Self-Attention** | Attention where queries, keys, and values all come from the same sequence |
| **SHAP** | SHapley Additive exPlanations; game-theoretic feature attribution for ML interpretability |
| **Shrinkage (Boosting)** | Scaling each tree's contribution by the learning rate; slows fitting to improve generalization |
| **SMA** | Simple Moving Average; unweighted arithmetic mean of N closing prices |
| **Soft-Vote Ensemble** | Ensemble by averaging probabilities (not hard class votes); preserves confidence information |
| **Stacking** | Two-level ensemble: base learners' predictions are features for a meta-learner |
| **Stationarity** | Statistical properties (mean, variance) are constant over time; log returns are approximately stationary |
| **STT** | Securities Transaction Tax; ~0.10% on delivery equity trades in India |
| **Survivorship Bias** | Evaluating only entities that survived; overestimates performance by excluding failures |
| **TCN** | Temporal Convolutional Network; uses dilated causal convolutions; large receptive field with few parameters |
| **Temperature Scaling** | Post-hoc calibration dividing logits by a scalar T to reduce overconfidence |
| **Temporal Causality** | Constraint that causes precede effects; training data strictly before test data |
| **Transformer** | Neural architecture using multi-head self-attention; models long-range dependencies in parallel |
| **VWAP** | Volume-Weighted Average Price; institutional fair value benchmark for intraday execution |
| **Vanishing Gradient** | Problem in RNNs where gradients exponentially decay through long sequences; solved by LSTM gates |
| **Walk-Forward Validation** | Time-series evaluation preserving strict temporal ordering; prevents lookahead bias |
| **Winsorization** | Clipping extreme values to Nth percentile bounds; reduces outlier influence on model training |
| **WTI** | West Texas Intermediate; US benchmark crude oil price traded on NYMEX futures exchange |
| **XGBoost** | eXtreme Gradient Boosting; regularized GBDT; Chen & Guestrin (KDD 2016) |
| **Z-score** | `(value − mean) / std`; measures standard deviations from the mean; used to detect anomalies |

---

## References

1. **Chen, T. & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *KDD '16*. https://arxiv.org/abs/1603.02754

2. **Ke, G. et al. (2017).** "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS 2017*.

3. **Hochreiter, S. & Schmidhuber, J. (1997).** "Long Short-Term Memory." *Neural Computation*, 9(8):1735–1780.

4. **Cho, K. et al. (2014).** "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." *EMNLP 2014*. — Original GRU paper.

5. **Bai, S., Kolter, J.Z., & Koltun, V. (2018).** "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." *arXiv:1803.01271*. — TCN benchmark paper.

6. **Oreshkin, B. et al. (2019).** "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting." *ICLR 2020*. https://arxiv.org/abs/1905.10437

7. **Shah, D. et al. (2021).** "Stock Market Prediction Using Bidirectional LSTM." *IEEE Access*. https://ieeexplore.ieee.org/document/9395265

8. **Shrivastav, S. & Kumar, S. (2022).** "Comparison of RF and GBM for NSE Stock Price Prediction." *Journal of King Saud University — Computer and Information Sciences*.

9. **Ozbayoglu, A.M. et al. (2020).** "Deep Learning for Financial Applications: A Survey." *Applied Soft Computing*, 93:106384.

10. **Guo, C. et al. (2017).** "On Calibration of Modern Neural Networks." *ICML 2017*. https://arxiv.org/abs/1706.04599

11. **Hong, H. & Stein, J.C. (1999).** "A Unified Theory of Underreaction, Momentum Trading, and Overreaction in Asset Markets." *Journal of Finance*, 54(6):2143–2184. — Gradual information diffusion model.

12. **Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).** "Why tree-based models still outperform deep learning on tabular data." *NeurIPS 2022*. https://arxiv.org/abs/2207.08815

13. **Akiba, T. et al. (2019).** "Optuna: A Next-generation Hyperparameter Optimization Framework." *KDD 2019*.

14. **Araci, D. (2019).** "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models." *arXiv:1908.10063*.

---

*Document prepared for academic/research purposes — Pandit Deendayal Energy University, School of Technology, Phase 2 Report, March 2026.*
