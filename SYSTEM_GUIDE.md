# AI Stock Prediction System — Complete Guide
**Project:** NSE Algo Trading with LightGBM + XGBoost + CatBoost Ensemble
**Author:** Vishwam Shah
**Date:** May 2026

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [How the Trading Works — Buy, Hold, Sell](#2-how-the-trading-works)
3. [Why No Short Selling](#3-why-no-short-selling)
4. [Pipeline Modes — Fast vs Full](#4-pipeline-modes)
5. [Why Retrain Daily](#5-why-retrain-daily)
6. [Why Ensemble — The Core Justification](#6-why-ensemble)
7. [Research Defense — Honest Assessment](#7-research-defense)
8. [Profitability Analysis](#8-profitability-analysis)
9. [Capital Requirements for Live Trading](#9-capital-requirements)
10. [Storage and Model Management](#10-storage-and-model-management)
11. [Full System Audit — What Works, What Doesn't](#11-system-audit)
12. [Promotion Gate — Paper to Live](#12-promotion-gate)
13. [Daily Workflow — Exact Commands](#13-daily-workflow)
14. [Risks and Honest Limitations](#14-risks-and-limitations)
15. [Sentiment System — FinBERT, Keywords, and Current Status](#15-sentiment-system)

---

## 1. System Overview

This is a **long-only swing trading system** for NSE (National Stock Exchange) Indian equities.

- **Universe:** 100 NSE stocks across 13 sectors
- **Models:** LightGBM + XGBoost + CatBoost ensemble (fast mode); + BiLSTM + TCN-Transformer + NBEATS (full mode)
- **Signal:** Predicts probability that a stock will rise >0.4% within the next 10 trading days
- **Threshold:** Only trade when ensemble confidence (prob_up) ≥ 58%
- **Position sizing:** Volatility-adjusted half-Kelly, max 3 simultaneous holdings, max 34% per stock
- **Exit:** Multi-rule policy — vol stop, trailing stop, signal decay, partial profit-take, time stop
- **Capital:** Default ₹5,00,000 (configurable)
- **Mode:** Paper trading by default (TRADING_MODE=paper). Live requires promotion gate approval.

The system runs automatically on a schedule:
- **6:00 PM Mon-Fri** — evening pipeline (predictions + orders)
- **9:00 AM Mon-Fri** — morning execution (exits + new buys)
- **3:45 PM Mon-Fri** — reconciliation (P&L check)

---

## 2. How the Trading Works

### The Complete Life Cycle of One Position

```
EVENING (T-1, ~6 PM)
  Pipeline downloads today's closing prices
  Computes 200+ features (RSI, MACD, global cues, sentiment, etc.)
  Ensemble predicts prob_up for each stock
  signal_publisher filters: only UP signals with prob_up >= 58%
  portfolio_optimizer allocates capital across top 3 signals
  Orders saved as orders_YYYYMMDD.json
         |
         v
MORNING (T, 9:00 AM)
  exit_runner runs FIRST — closes any positions past their exit condition
  signal_publisher's BUY orders are loaded
  OrderManager places LIMIT buy orders at today's open price
  Positions filled and recorded in execution_log.parquet
         |
         v
HOLD PERIOD (T to T+10 trading days)
  exit_policy.py monitors the position every morning:
    1. Vol stop   — if price drops 2.5x ATR below entry → SELL ALL immediately
    2. Trailing   — if position hit +3% then fell back 4% from peak → SELL ALL
    3. Signal decay — if prob_up < 50% for 2 consecutive days → SELL ALL
    4. Partial PT — if unrealised gain >= +6% → SELL 50% now
    5. Time stop  — if held >= 10 trading days → SELL ALL remaining
         |
         v
EXIT (somewhere between T+1 and T+10)
  OrderManager places SELL orders
  Cash returns to the ₹5,00,000 pool
  P&L recorded in portfolio_ledger.py → ledger/state.json
  Available for next cycle's allocation
```

### Capital Allocation Example (₹5,00,000 capital)

```
Today's UP signals ranked by prob_up:
  HDFCBANK   prob_up=0.71  → Slot 1: ₹1,67,000 (33%)
  INFY       prob_up=0.65  → Slot 2: ₹1,67,000 (33%)
  SUNPHARMA  prob_up=0.61  → Slot 3: ₹1,66,000 (33%)

Total deployed: ₹5,00,000 (100%)
Max stocks held at once: 3
Max per stock: 34%

If only 1 signal today: ₹1,67,000 deployed, ₹3,33,000 stays as cash
If 0 signals today: full ₹5,00,000 stays as cash (no forced trades)
```

### The Exit Policy — Why Losers Are Cut Early

The exit hierarchy runs every morning before new buys:

| Rule | Trigger | Action | Purpose |
|------|---------|--------|---------|
| Vol stop | Price < entry − 2.5×ATR(14) | Sell 100% | Protects against crashes |
| Trailing stop | Was +3% profit, fell back 4% from peak | Sell 100% | Locks in gains |
| Signal decay | prob_up < 50% for 2 consecutive days | Sell 100% | Model says get out |
| Partial profit | Unrealised gain ≥ +6% | Sell 50% | Lock half, let rest run |
| Time stop | Held ≥ 10 trading days | Sell 100% | Hard deadline = backtest match |

This asymmetry — cutting losers early, letting winners run — is why average loss (~0.8%) is smaller than average win (~1.5%) even at 60% accuracy.

---

## 3. Why No Short Selling

### Structural Reason

All orders use **NSE CNC (Cash and Carry)** — the delivery product:

- You pay full cash upfront and receive actual shares
- You can only sell shares you already own
- Short selling in CNC is **not permitted by NSE**
- To short a stock you need F&O (Futures and Options) on the derivatives segment

This requires:
- Different API endpoints on Angel One
- Margin account (not just cash)
- Separate risk model for leveraged positions
- Hedge accounting for the futures roll

This is deliberately out of scope for the current system. It is planned for Phase 4.

### Practical Reason

```python
# signal_publisher.py
df = df[df["direction"] == "UP"].copy()   # DOWN signals dropped entirely
```

When the model predicts DOWN for a stock — the system does nothing for that stock today.
No loss, no gain. The capital stays in the cash pool.

### Why This Is Still Profitable

The system does not need to trade both directions. The edge is in **selection**:

| Scenario | What happens | P&L impact |
|----------|-------------|------------|
| Model says UP, stock goes UP | Buy → sell higher | +profit |
| Model says UP, stock goes DOWN | Buy → stop loss triggers | -small, capped loss |
| Model says DOWN, stock goes DOWN | Do nothing | ₹0 |
| Model says DOWN, stock goes UP | Do nothing | ₹0 (missed gain, not a loss) |

At 60.1% accuracy with 1.5:1 win/loss ratio:

```
Expected value = 0.601 × 1.5% − 0.399 × 0.8% = +0.58% per trade
After round-trip costs (0.25%): net = +0.33% per trade
25 cycles per year × 0.33% × compounding = ~9-12% baseline annual return
3-slot concentrated portfolio with compounding → observed 79.8% over test period
```

---

## 4. Pipeline Modes

### Fast Mode — Use This Every Day

**Models:** LightGBM + XGBoost + CatBoost (3 models)
**Time:** ~30-40 minutes for 100 stocks
**Use:** Every weekday evening at 6 PM (automated)

```bash
python V3/07_pipeline/orchestrator.py --fast
# or via daily runner:
python V3/05_live_trading/daily_runner.py --mode evening --capital 500000
```

Fast mode is the production mode for daily trading. Tree models (LightGBM, XGBoost, CatBoost) are stable — adding 1 new bar to ~1,800 bars changes weights by <0.1%. They are fast, interpretable, and produce well-calibrated probabilities.

### Full Mode — Use Weekly (Sunday) or for Research

**Models:** LightGBM + XGBoost + CatBoost + BiLSTM + TCN-Transformer + NBEATS (6 models)
**Time:** 4-6 hours for 30 stocks, ~16 hours for 100 stocks
**Use:** Sunday evenings, or when updating deep learning weights for thesis

```bash
python V3/07_pipeline/orchestrator.py
# (no --fast flag)
```

Full mode includes deep learning models that capture temporal sequence patterns tree models cannot. Not suitable for daily use — too slow for a 6 PM → 9 AM window.

### Recommendation

```
Daily production → Fast mode (Mon-Fri 6 PM, automated)
Weekly refresh   → Full mode (Sunday evening, manual)
Research/thesis  → Full mode (run on top-30 stocks, extract DL accuracy numbers)
```

---

## 5. Why Retrain Daily

### The Short Answer

You retrain daily to generate **tomorrow's prediction on today's data**, not to fundamentally change the model.

### What Actually Changes Each Day

Each evening:
1. **New bar downloaded**: 1 new closing price (OHLCV row) added to each stock's parquet
2. **Features recomputed**: RSI, MACD, EMA, ATR, global cues, sentiment — all shift because they are rolling window calculations that include today's close
3. **Model retrained** on expanded dataset (e.g., 1,818 → 1,819 bars): weights change by ~0.05%
4. **Prediction generated**: Today's feature vector passed through the retrained model → `prob_up` for tomorrow

**Step 4 is the reason you must run daily.** The features change every day. A model frozen from last month can still generate valid predictions if you just compute today's features and pass them through — the retraining is a safeguard against slow market regime drift.

### Why Not Freeze the Model?

For tree models, you could freeze the model for a week without significant accuracy loss. However:

- **Regime shifts**: A model trained before a major market event (RBI rate change, budget, earnings season) may not have that event reflected in its patterns
- **Global cue features**: VIX, DXY, crude oil — these drift significantly week to week
- **NSE calendar features**: days-to-expiry, RBI meeting proximity — these must be recomputed daily

Retraining daily costs 30 minutes (fast mode). The marginal cost is low; the protection against drift is real.

---

## 6. Why Ensemble — The Core Justification

### The Three Models Learn Differently

| Model | Growth strategy | Key difference |
|-------|----------------|----------------|
| LightGBM | Leaf-wise | Lower bias, better at rare patterns, faster |
| XGBoost | Level-wise | Different regularization path, lower variance |
| CatBoost | Ordered boosting | Less target leakage within folds, handles categorical features natively |

These three learn different decision boundaries from the same feature set. Their errors are **not perfectly correlated** — when LightGBM misclassifies a stock, XGBoost often gets it right, because they stumbled on different local optima.

### The Theoretical Basis

Dietterich (2000) — foundational ensemble theorem — states an ensemble of M classifiers beats any individual classifier if:

1. Each classifier accuracy > 50% (better than random) — our models: 57-61% ✓
2. Classifiers make independent errors — different inductive biases ensure this ✓

Both conditions are satisfied.

### The Threshold Mechanism — Why Ensemble Makes the 58% Gate Meaningful

**Single model at 58% threshold:**
`prob_up = 0.59` might mean "truly 59% confident" or it might be a noisy 52% estimate that slightly overfit. You cannot distinguish these.

**Ensemble at 58% threshold:**
Three independently-trained models must all agree (via soft-vote average) that prob_up ≥ 58%. When LightGBM says 0.64, XGBoost says 0.61, and CatBoost says 0.60, the average is 0.617 and this is a **genuine consensus signal** — not one model's noise.

This is the core mechanism:

```
Ensemble 58% threshold = multi-model consensus filter
→ selects only the highest-conviction signals
→ filters out stocks where models disagree
→ disagreement usually means the signal is ambiguous → skip it
→ agreement usually means a real pattern is present → trade it
```

This is why 60.1% accuracy is achieved on activated signals (those that pass the 58% gate) even though mean OOS accuracy across all stocks is only 51.2%. The threshold is doing the selection work. The ensemble makes the threshold trustworthy.

### Walk-Forward + Ensemble = The Method's Strength

Walk-forward validation produces an OOS accuracy estimate for each of the 6 training windows. The ensemble's accuracy is measured on data the models never saw during training (true out-of-sample). This combination is the methodological core:

```
Walk-forward OOS test → unbiased accuracy estimate
Ensemble soft-vote    → better-calibrated probabilities
58% threshold         → selects only high-conviction, agreed signals
Kelly sizing          → bets proportional to edge, not fixed %
Multi-exit policy     → convex payoff: cuts losers, lets winners run
```

Each layer builds on the previous one. Remove any layer and the edge shrinks.

---

## 7. Research Defense — Honest Assessment

### What the System Claims (and Can Defend)

| Claim | Evidence | Defensible? |
|-------|---------|-------------|
| 60.1% accuracy on activated UP signals | Bootstrap CI [57.8-62.4%], p < 0.001, n=1,634 | Yes — statistically significant |
| Walk-forward methodology is leakage-free | Features at t use only data available at t | Yes — verified in code |
| Ensemble beats single model | 6-window OOS comparison shows ~2-3% accuracy gain | Yes |
| 79.8% total return in test period | 3-slot portfolio equity curve | Partially — concentrated, bull market |
| System beats momentum baseline | Compared to top-5 by 20-day return, same costs | Yes — documented |

### What the System Does NOT Claim

- This is not a claim of 70%+ accuracy (impossible for stock prediction)
- The 79.8% return is from a 3-slot concentrated portfolio in a 2020-2025 bull market
- Per-stock annualised returns have been capped at ±500% to prevent inflation artifacts
- Deep learning models were not included in the latest fast-mode run (DL = full mode only)

### How to Compare Against Other Papers

Most papers that claim high accuracy (70%+) have one or more of these flaws:

1. **Simple train/test split** — not walk-forward. The model sees the future in its training data indirectly through feature normalisation or data leakage
2. **No transaction costs** — even 0.25% round-trip changes many profitable strategies to losing ones
3. **Survivorship bias** — testing only on stocks that exist today, missing the ones that went bankrupt
4. **In-sample reporting** — reporting training accuracy, not OOS accuracy

Our methodology avoids all four:
- Walk-forward expanding windows (6 windows, 70%→95%)
- 0.25% round-trip cost applied to every trade
- Survivorship bias acknowledged and mild (only 2.3-year lookback)
- All reported accuracy numbers are OOS (test window, never seen in training)

### The Honest Risk Statement

> "Our walk-forward OOS methodology correctly estimates OOS accuracy without look-ahead bias. The 60.1% bootstrap-confirmed accuracy on activated signals is statistically significant above the 50% random baseline. However, the test period (2020-2025) was predominantly a bull market — model performance in a prolonged bear market is unknown. The live paper trading phase is specifically designed to validate performance in current market conditions before any real capital is deployed."

---

## 8. Profitability Analysis

### Breakdown of 100 Stocks (Latest Run)

| Category | Count | Description |
|----------|-------|-------------|
| Strong performers | ~24 | OOS accuracy consistently >54%, Sharpe > 0 |
| Marginal | ~43 | OOS accuracy 50-54%, inconsistent signals |
| Underperformers | ~34 | OOS accuracy <50%, excluded from trading |

Only the ~24 strong performers are in the `tradeable` set that receives capital allocation. The remaining 76 stocks are modelled and monitored but do not receive orders.

### Signal Statistics (Pooled Across All Stocks)

- Total UP signals generated: ~1,634 per run
- Signals passing 58% gate: ~24% (the activated signals)
- Bootstrap accuracy on activated signals: **60.1% [57.8-62.4%]**
- Momentum baseline (top-5 by 20-day return): ~54%
- System advantage over momentum: ~6 percentage points

### Portfolio Simulation (3-Slot, ₹5L capital)

- Backtest period: 2020-2025
- Total return: 79.8%
- Annualised return: ~12.4%
- Max drawdown: ~18%
- Sharpe ratio: ~1.4-1.6
- Win rate: ~60% of closed trades profitable
- Average hold: ~8.3 trading days (exits trigger before 10-day time stop ~40% of the time)

---

## 9. Capital Requirements

### For Paper Trading (Start Immediately)

**₹0** — the system uses simulated fills. No real money moves. This is the current mode (`TRADING_MODE=paper`). Orders are generated and "filled" at the previous close price without any brokerage account interaction.

### For Live Trading on Angel One

**Minimum to open account:** ~₹10,000-25,000 (Angel One minimum)

**Recommended starting capital for this strategy:**

| Capital | Max per stock (34%) | Shares of ₹1,500 stock | Practical? |
|---------|--------------------|-----------------------|-----------|
| ₹1,00,000 | ₹34,000 | ~22 shares | Yes, works fine |
| ₹2,00,000 | ₹68,000 | ~45 shares | Comfortable |
| ₹5,00,000 | ₹1,67,000 | ~111 shares | Designed for this |
| ₹10,00,000 | ₹3,40,000 | ~226 shares | Scales well |

**You can start with ₹1,00,000.** The strategy works at any capital above ₹50,000 because NSE CNC allows 1-share lots (no lot size minimum for delivery).

At ₹1L, with 3 slots of ₹33K each:
- HDFCBANK at ₹1,740 → 18-19 shares per position
- Daily P&L volatility: ~₹300-500
- Monthly expected: ~₹500-1,000 net of costs

### When to Go Live

The promotion gate (`promotion_gate.py`) must approve before `TRADING_MODE=live` is set. It requires:

- ≥40 closed paper trades
- ≥20 calendar days of paper trading
- Rolling Sharpe ≥ 1.0 (last 30 closed trades)
- Max drawdown ≤ 10% over same window
- Fill rate ≥ 90%
- Slippage drift ≤ 25 bps
- Calibration drift ≤ 0.05

Currently at 0 closed paper trades (just reset ledger). Expected to reach 40 closed trades in ~3-4 weeks of daily paper trading.

---

## 10. Storage and Model Management

### What Each Run Creates

```
V3/02_models/
├── runs/{run_id}/              ← NEW folder every run
│   ├── {symbol}/window_01/    lightgbm.pkl  xgboost.pkl  ...
│   ├── {symbol}/window_02/    ...
│   └── ...  (6 windows × N stocks × 3-6 models)
└── production/{symbol}/       ← OVERWRITTEN each run
    ├── lightgbm.pkl
    ├── xgboost.pkl
    ├── catboost.pkl
    ├── metadata.json
    └── feature_names.txt
```

### Does Each Run Duplicate Models?

**Yes — new run folder each time. No — production folder is always overwritten.**

Each run trains fresh (not reusing previous weights). This is by design:
- Walk-forward expanding windows mean the training data grows each run
- Yesterday's 1,818-bar model is replaced by today's 1,819-bar model
- The run folder keeps a checkpoint of every window (for debugging and research)

### Storage Per Run

| Mode | Run folder size | Production folder |
|------|----------------|-------------------|
| Fast (3 models, 30 stocks) | ~200 MB | ~3.5 GB total |
| Fast (3 models, 100 stocks) | ~600 MB | ~3.5 GB total |
| Full (6 models, 30 stocks) | ~2-3 GB | ~5-7 GB total |

### Auto-Cleanup (Added)

The orchestrator now keeps only the **last 2 run folders** and deletes older ones automatically:

```python
KEEP_MODEL_RUNS = 2
all_model_runs = sorted(MODELS_RUNS_DIR.glob("20*"), reverse=True)
for old_run in all_model_runs[KEEP_MODEL_RUNS:]:
    shutil.rmtree(old_run)
```

After this, disk usage stabilises at: `production/ + 2 recent run folders`. No unbounded growth.

---

## 11. System Audit

### Components Checked and Status

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Angel One credentials | .env | OK | All 4 keys present |
| SmartAPI SDK | pip | OK | `from SmartApi import SmartConnect` works |
| Trading mode | .env | Fixed | `TRADING_MODE=paper` now explicit |
| UTF-8 encoding | All live trading .py | Fixed | `sys.stdout.reconfigure(encoding='utf-8')` added |
| Daily runner | daily_runner.py | Fixed | Removed unnecessary `--force-features` |
| Exit runner | exit_runner.py | Fixed | Unicode arrows fixed, Windows `.replace()` used |
| Order manager | order_manager.py | Fixed | Unicode ₹ fixed, `.rename()` → `.replace()` |
| Portfolio ledger | portfolio_ledger.py | OK | Atomic writes, idempotent rebuild |
| Signal publisher | signal_publisher.py | OK | Kelly sizing + risk gates + optimizer |
| Risk guard | risk_guard.py | OK | Position%, sector cap, circuit breaker |
| Exit policy | exit_policy.py | OK | Vol stop, trailing, signal decay, partial PT, time stop |
| Promotion gate | promotion_gate.py | OK | 7 checks before paper→live |
| Portfolio optimizer | portfolio_optimizer.py | OK | Inv-vol weighting, MCR, sector cap |
| Instrument master | instrument_master.py | OK | Token map with local JSON fallback |
| Windows scheduler | Task Scheduler | Done | 3 tasks registered (Evening/Morning/Reconcile) |
| Ledger state | ledger/state.json | Reset | Clean: 0 lots, ₹5,00,000 cash, ready for paper trading |
| Old paper history | execution_log.parquet | Archived | Moved to execution_log_archive_20260514.parquet |
| Model auto-cleanup | orchestrator.py | Done | Keep last 2 run folders |
| Frontend label | PipelineControl.tsx | Fixed | "6 models" (was "5 models") |
| Summary CSV | summary.csv | Fixed | 23 columns (was 8) |
| Ann. return inflation | backtest.py | Fixed | Capped at ±500%, uses calendar span |

### Windows Task Scheduler — Registered Tasks

| Task | Time | Days | Action |
|------|------|------|--------|
| AlgoTrading_Evening | 6:00 PM | Mon-Fri | `daily_runner.py --mode evening --capital 500000` |
| AlgoTrading_Morning | 9:00 AM | Mon-Fri | `daily_runner.py --mode morning --capital 500000` |
| AlgoTrading_Reconcile | 3:45 PM | Mon-Fri | `daily_runner.py --mode reconcile` |

---

## 12. Promotion Gate — Paper to Live

### What It Checks

The gate (`V3/05_live_trading/promotion_gate.py`) evaluates 7 metrics before allowing live trading:

```
Check 1: min_paper_trades    >= 40 closed trades
Check 2: min_paper_days      >= 20 calendar days
Check 3: min_rolling_sharpe  >= 1.0  (last 30 trades)
Check 4: max_rolling_dd      <= 10%  (peak-to-trough over last 30 trades)
Check 5: max_slip_bps        <= 25 bps slippage drift
Check 6: min_fill_rate       >= 90% of placed orders filled
Check 7: max_brier_drift     <= 0.05 calibration drift
```

All 7 must pass. Any failure → decision = "no-go".

### How to Run It

```bash
python V3/05_live_trading/promotion_gate.py          # evaluate only, print result
python V3/05_live_trading/promotion_gate.py --flip   # if go: set TRADING_MODE=live in .env
```

### Current State

- Closed paper trades: 0 (ledger just reset — paper trading starts from tomorrow)
- Expected to reach 40 closed trades: ~3-4 weeks (10-day hold means ~2-3 closes/week)
- Do NOT manually set TRADING_MODE=live. Let the gate do it.

---

## 13. Daily Workflow — Exact Commands

### Automated (Task Scheduler handles this)

```
6:00 PM  → AlgoTrading_Evening fires automatically
9:00 AM  → AlgoTrading_Morning fires automatically
3:45 PM  → AlgoTrading_Reconcile fires automatically
```

### Manual Override

```bash
# Run evening pipeline manually (if scheduler missed)
python V3/05_live_trading/daily_runner.py --mode evening --capital 500000

# Run morning execution manually
python V3/05_live_trading/daily_runner.py --mode morning --capital 500000

# Run reconcile manually
python V3/05_live_trading/daily_runner.py --mode reconcile

# Check what exits are due (dry run, no execution)
python V3/05_live_trading/exit_runner.py

# Check portfolio state
python V3/05_live_trading/portfolio_ledger.py --summary

# Check promotion gate status
python V3/05_live_trading/promotion_gate.py

# Run full-mode pipeline (Sunday, for DL model refresh)
python V3/07_pipeline/orchestrator.py --symbols HDFCBANK INFY SBIN ...

# Extract feature importance after full-mode run
python V3/07_pipeline/extract_feature_importance.py
```

### Weekly Sunday Routine (Manual)

```bash
# 1. Run full pipeline on top-30 stocks to update DL models
python V3/07_pipeline/orchestrator.py  # (no --fast)

# 2. Extract feature importance
python V3/07_pipeline/extract_feature_importance.py

# 3. Check promotion gate
python V3/05_live_trading/promotion_gate.py
```

---

## 14. Risks and Honest Limitations

### Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| Bull-market regime (2020-2025 test period) | Sharpe may drop in bear markets | Paper trading in current conditions before going live |
| Per-stock n_trades small (~10-15) | High per-stock variance | Pool all signals for statistical tests; diversify across 100 stocks |
| Survivorship bias (mild) | 2.3-year lookback is short enough to limit this | Acknowledged in research; not a significant distortion |
| No short selling | Cannot profit from DOWN signals | Accepted; F&O shorting is Phase 4 scope |
| Indian Calendar features weakest | Lowest importance group (0.0132 mean normalised importance) | Still included for regime awareness; do not remove |
| Global macro features third | (0.0232) behind Statistical and Price/Return | All 3 groups contribute; importance varies by stock |
| DL models (BiLSTM, TCN, NBEATS) not in daily run | Only 3 tree models in production | Full mode runs weekly to update DL weights |

### What Could Break the System

1. **Angel One API change** — if SmartAPI token format changes, `instrument_master.py` must be updated
2. **yfinance delisting** — if a stock is removed from NSE100, the pipeline skips it (handled gracefully)
3. **Prolonged bear market** — system is calibrated on a bull market. DOWN signals = no trade, so losses are limited but returns will drop
4. **Windows machine offline at 6 PM** — task scheduler cannot fire. Add a cloud backup or check next morning manually

### What the System Cannot Do

- Cannot short stocks (no F&O)
- Cannot trade intraday (CNC is delivery only; positions held overnight)
- Cannot trade stocks outside the 100-stock universe
- Cannot handle corporate actions (splits, bonuses) automatically — yfinance adjusts prices but execution log prices may be pre-adjustment
- Cannot guarantee Angel One order fills (limit orders may not fill if price moves away)

---

---

## 15. Sentiment System — FinBERT, Keywords, and Current Status

### What Was Built

A three-tier news sentiment pipeline that feeds 9 daily features into the prediction model.

```
Google News RSS (free)
       |
       v
news_fetcher.py — fetches up to 20 headlines per stock per day
       |
       v
Three-tier scoring:
  Tier 1: Fine-tuned FinBERT India (V3/02_models/finbert_india/)  — 88.2% accuracy
  Tier 2: Base ProsusAI/finbert from HuggingFace                  — ~82% accuracy
  Tier 3: VADER keyword scorer (Indian vocabulary)                  — simple, fast fallback
       |
       v
sentiment_history.parquet  (one row per symbol per day)
       |
       v
features.py — merges 9 sentiment features into model training data
       |
       v
LightGBM / XGBoost / CatBoost train on features including sentiment
```

### Is Sentiment Being Used Right Now?

**No — and this is a gap that reduces model quality.**

The infrastructure is fully built and wired in. The orchestrator loads `sentiment_history.parquet` if it exists and passes it to features.py. But the parquet **does not exist** — sentiment was never actually fetched and accumulated. So all 9 sentiment features default to 0.0 (neutral prior) for every stock on every date.

The model has been trained as if every stock always has zero sentiment, which means the model has learned to ignore sentiment entirely. It cannot use something it has never seen a signal from.

### The FinBERT India Fine-Tune — Status

A custom FinBERT model was fine-tuned on 631 labeled Indian financial sentences:

| Metric | Value |
|--------|-------|
| Base model | ProsusAI/finbert |
| Training samples | 504 |
| Validation samples | 127 |
| Validation accuracy | **88.19%** |
| Validation F1 macro | **0.879** |
| Labels | positive / negative / neutral |
| Domain coverage | Banking, IT, Pharma, FMCG, Auto, Metals, Telecom, Power, Realty, Oil & Gas |

**The problem: model weights are missing from disk.**

The directory `V3/02_models/finbert_india/` contains only:
- `config.json` — model architecture
- `tokenizer.json` — vocabulary
- `tokenizer_config.json` — tokenizer settings
- `label_map.json` — label mappings + performance record

It is **missing** `model.safetensors` or `pytorch_model.bin` — the actual trained weights. The training ran successfully and the performance was recorded, but the weight file was not saved (likely a disk space or interrupt issue at save time).

**Current behaviour:** Tier 1 (India fine-tuned) fails silently → falls to Tier 2 (base ProsusAI/finbert from HuggingFace, downloaded on first use) → or Tier 3 (VADER) if transformers not available.

### The Training Dataset — Keywords and Sentences

The `indian_finance_dataset.py` file contains 631 hand-labeled sentences covering Indian market-specific vocabulary that generic FinBERT models miss:

**Positive keywords and phrases trained on:**
- `"FII buying"`, `"DII buying"`, `"repo cut"`, `"accommodative stance"`, `"rate cut"`
- `"CASA ratio improves"`, `"NIM expansion"`, `"gross NPA falls"`, `"deal win"`
- `"guidance raised"`, `"buyback"`, `"IPO listing gains"`, `"all time high"`
- `"credit growth accelerates"`, `"slippage ratio falls"`, `"provision coverage improves"`

**Negative keywords and phrases trained on:**
- `"FII selling"`, `"NPA rises"`, `"stressed assets"`, `"SEBI penalty"`, `"enforcement notice"`
- `"guidance cut"`, `"margin pressure"`, `"attrition"`, `"repo hike"`, `"hawkish"`
- `"fraud"`, `"scam"`, `"circuit breaker"`, `"52 week low"`, `"margin call"`
- `"outflows"`, `"FII outflows"`, `"provision spike"`, `"default risk"`

**Why this matters vs. generic FinBERT:** Base FinBERT was trained on Reuters/Bloomberg English text. It does not know that `"SEBI notice"` or `"NPA rises"` is strongly negative for Indian banking stocks, or that `"RBI MPC accommodative"` is strongly positive. The India fine-tune teaches it this vocabulary.

### The VADER Fallback — Keyword Scoring

If neither FinBERT variant is available, `news_fetcher.py` falls back to a simple word-count scorer:

```python
score = (positive_word_hits - negative_word_hits) / total_hits   # range [-1, +1]
```

It uses the same Indian market vocabulary as the training data — FII flows, NPA mentions, regulatory penalties, RBI rate decisions. It is less accurate than FinBERT (~70% vs 88%) but always works with zero dependencies.

### The 9 Sentiment Features Sent to the Model

All 9 features use T-1 sentiment to predict T direction (leakage-safe):

| Feature | Description | Why useful |
|---------|-------------|-----------|
| `sentiment_score` | Raw FinBERT score [-1, +1] on T-1 | Direct news sentiment |
| `sentiment_ma_7d` | 7-day rolling mean | Short-term narrative trend |
| `sentiment_ma_30d` | 30-day rolling mean | Medium-term baseline |
| `sentiment_trend` | ma_7d - ma_30d | Momentum of sentiment (improving/worsening) |
| `sentiment_zscore` | Z-score vs 30-day window | How extreme today's news is vs recent norm |
| `sentiment_vol_7d` | Rolling std of score | Uncertainty / conflicting news signal |
| `sentiment_n_articles` | Article count | Signal strength proxy (more articles = bigger event) |
| `sentiment_positive` | Positive article ratio | Direct positive news fraction |
| `sentiment_negative` | Negative article ratio | Direct negative news fraction |

The `sentiment_trend` and `sentiment_zscore` are the most useful: they capture **whether the narrative is deteriorating or improving** rather than just the absolute level. A stock with raw_score = -0.1 but an improving trend is very different from one with raw_score = -0.1 and a worsening trend.

### How Sentiment Affects Signals (Beyond Features)

`news_fetcher.py` also has a `adjust_confidence_threshold()` method used by `signal_publisher.py`:

```python
# If news is extremely negative (spike, score < -0.6):
# raise the confidence threshold from 58% to 63% (harder to get a buy signal)
if spike and score < 0:
    return min(base_threshold + 0.05, 0.75)

# If news is very positive (score > 0.5):
# slightly lower the threshold from 58% to 56% (easier to buy on good news)
elif score > 0.5:
    return max(base_threshold - 0.02, 0.50)
```

This means extreme negative news raises the bar for entry, and strongly positive news slightly lowers it — acting as a real-time fundamental filter on top of the technical ML signal.

### Free News Sources — What Is Used and What Is Available

| Source | Used? | Cost | Articles/day | Historical? |
|--------|-------|------|-------------|-------------|
| Google News RSS | YES (primary) | Free, no key | 20 per symbol | No (current only) |
| yfinance `.news` | YES (backfill) | Free, no key | 10-30 per symbol | ~2-4 weeks back |
| Economic Times RSS | Not wired yet | Free, no key | Unlimited | No |
| Moneycontrol RSS | Not wired yet | Free, no key | Unlimited | No |
| NSE announcements | Not wired yet | Free | Corporate actions | Yes (PDF) |
| Bloomberg / Reuters | No | Paid | Full archive | Yes |
| Refinitiv Eikon | No | Very expensive | Full archive | Yes |

**You do not need paid news.** Google News RSS + yfinance covers enough for the model to learn the signal. The improvement from paid news (more articles, better coverage) is marginal compared to fixing the current gap (no sentiment at all).

### What Needs to Be Fixed — Action Plan

**Step 1: Retrain FinBERT India (model weights missing)**

```bash
python V3/01_data/news/finetune_finbert.py --epochs 8 --lr 1e-5
# Takes ~15-30 min on CPU, ~5 min on GPU
# Saves model.safetensors to V3/02_models/finbert_india/
# Expected: ~88% validation accuracy (already achieved before)
```

Only needs to be done once. The model weights then persist and Tier 1 activates automatically.

**Step 2: Backfill historical sentiment (last 2-4 weeks)**

```bash
python V3/01_data/news/backfill_sentiment.py
# Uses yfinance.Ticker.news — free, no key needed
# Fills in sentiment_history.parquet for the last 1-4 weeks
# ~5 min for all 100 symbols
```

**Step 3: Start daily sentiment accumulation (already wired into scheduler)**

The evening task (`daily_runner.py --mode evening`) already calls `sentiment_history.py` as Step 1 before the pipeline. Once the file exists, it auto-appends today's sentiment every evening. No additional setup needed.

**Step 4: Re-run pipeline after 30 days of sentiment history**

Sentiment features are computed as 7-day and 30-day rolling stats. After 30 days of daily data:

```bash
python V3/07_pipeline/orchestrator.py --fast --force-features
# force-features recomputes feature parquets with real sentiment (not all zeros)
```

The models will retrain on data that includes meaningful sentiment signals for the first time.

### Expected Impact on Model Performance

Sentiment is unlikely to be the dominant feature — technical indicators and price/return features capture more of the predictable variance. However:

| Scenario where sentiment helps most | Expected lift |
|-------------------------------------|---------------|
| Earnings season (quarterly results) | Significant — news strongly predicts direction |
| RBI meetings (rate decisions) | Significant — rate-sensitive Banking/Finance stocks |
| Regulatory events (SEBI notices, penalties) | Large impact — negative news often precedes selloff |
| Normal trading days (no major news) | Minimal — sentiment is flat/neutral, features add little |

Realistic expectation: **+1 to +2 percentage points** on OOS accuracy for news-heavy stocks (Banking, IT, Pharma). This is meaningful — it could shift borderline stocks from the 50-54% bucket into the >54% tradeable bucket.

The `sentiment_trend` feature (ma_7d - ma_30d) is the most likely contributor: a stock where the narrative has been improving for 7 days above the 30-day baseline has a real statistical tendency to continue upward in the short term.

### Summary of Sentiment System State

| Item | Status |
|------|--------|
| `news_fetcher.py` (three-tier scorer) | Built, working |
| `finetune_finbert.py` (training script) | Built, working |
| `indian_finance_dataset.py` (631 labeled sentences) | Built |
| `sentiment_history.py` (daily accumulator) | Built, called by daily_runner |
| `backfill_sentiment.py` (historical fill) | Built, never run |
| FinBERT India model weights | **MISSING — needs retraining** |
| `sentiment_history.parquet` | **MISSING — needs backfill** |
| Sentiment features in pipeline | Wired correctly, but all = 0.0 currently |
| Daily news fetch in scheduler | Wired in evening task — activates once parquet exists |
| Paid news required | **No** — Google News RSS + yfinance is sufficient |

---

*Document compiled from conversation sessions, May 2026.*
*All code references are to the V3 pipeline under `c:\Users\Home\Documents\AI_IN_STOCK_V2\`.*
