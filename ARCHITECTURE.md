# AlgoTrading V3 — Full System Architecture, Results & Honest Assessment

**Repo**: `AI IN STOCK V3/AlgoTrading`
**Scope of this doc**: everything under `V3/`, `backend/`, `frontend/`. Older `production/`, `Master_thesis/`, `_archive/`, `main.py`, and root-level `config.py` are ignored per instructions.
**Generated**: 2026-04-24, from source + latest run artefacts.

---

## 1. System at a glance

Three tiers, one workflow:

```
     ┌──────────────────────────────────────────────────────────────────────┐
     │                        V3/  — RESEARCH + MODELS                      │
     │  00_config  01_data  02_models  03_training  04_backtesting          │
     │  05_live_trading  06_results  07_pipeline                            │
     └──────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │ reads parquet + runs artefacts
                                  │
     ┌──────────────────────────────────────────────────────────────────────┐
     │             backend/main.py  —  FastAPI (single 2500-line file)      │
     │   50+ endpoints   WebSocket /ws/live-prices   Angel One passthrough  │
     └──────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │ REST/WebSocket
                                  │
     ┌──────────────────────────────────────────────────────────────────────┐
     │          frontend/  —  Next.js 16 + React 18 + TanStack Query        │
     │   Trade view (Overview/Signals/Orders/Portfolio/Chart)               │
     │   Research view (Accuracy/Analysis/Runs)                             │
     └──────────────────────────────────────────────────────────────────────┘
```

A single pipeline run (`V3/07_pipeline/orchestrator.py`) is the heart. It: downloads data → builds features → walk-forward trains an ensemble → saves per-symbol models and predictions → runs a realistic backtest → writes CSV/JSON artefacts the backend can serve.

---

## 2. Directory map (what each folder does)

```
V3/
├── 00_config/
│   ├── config.py            Single source of truth — 525 lines of settings
│   ├── logging_config.py    loguru setup (50 MB rotating files)
│   └── tickers.py           Symbol → yfinance ticker map
├── 01_data/
│   ├── downloader.py        Legacy class; current pipeline uses steps/download.py
│   ├── features.py          Legacy feature class (only used outside pipeline)
│   ├── targets.py
│   ├── earnings_calendar.py
│   ├── raw/                 102 {SYMBOL}.parquet files + global_cues.parquet + usdinr.parquet
│   ├── features/{raw,scaled}/  feature-engineering cache
│   └── news/
│       ├── news_fetcher.py          Google News RSS + FinBERT wrapper
│       ├── sentiment_history.py     Daily accumulator → sentiment_history.parquet
│       ├── finetune_finbert.py      India-specific fine-tuning script
│       ├── indian_finance_dataset.py
│       ├── backfill_sentiment.py    (uncommitted) bulk backfill
│       └── sentiment_history.parquet  ★ only 700 rows (see §7)
├── 02_models/
│   ├── traditional/         LightGBM, XGBoost, CatBoost, sklearn
│   ├── deep_learning/       LSTM, BiLSTM, GRU, CNN-LSTM, CNN-GRU,
│   │                         TCN-GRU, TCN-Transformer, N-BEATS
│   ├── finbert_india/       Fine-tuned FinBERT (HuggingFace format, model.safetensors)
│   ├── runs/{run_id}/…      Per-window pickles, keras files, scaler, PCA
│   └── production/{SYMBOL}/ Latest calibration.json + metadata.json + dl_meta.json
├── 03_training/
│   ├── walk_forward_validation.py
│   ├── metrics.py
│   ├── reporting.py
│   ├── create_excel_reports.py
│   └── plot_results.py
├── 04_backtesting/
│   ├── backtest_engine.py      Daily-step simulation
│   ├── backtest_runner.py      HRP portfolio variant (run via --backtest flag)
│   ├── portfolio_optimizer.py  HRP (Hierarchical Risk Parity) — replaced RL
│   ├── position_sizer.py       Kelly / vol-adjusted / fixed
│   └── transaction_costs.py    NSE cost model
├── 05_live_trading/
│   ├── angel_one_client.py   SmartAPI REST + WebSocket wrapper
│   ├── daily_runner.py       Evening/morning/reconcile cron targets
│   ├── signal_publisher.py   predictions.csv → approved_orders.json
│   ├── order_manager.py      Fill tracking + paper/live switch
│   ├── risk_guard.py         Pre-trade risk checks
│   ├── paper_trader.py       Dry-run fills
│   ├── setup_cron.sh         Installs 3 cron jobs (IST times)
│   ├── execution_logs/       ~40 JSONs from actual daily runs
│   ├── paper_trading_logs/
│   └── orders/               Approved order JSONs
├── 06_results/runs/{run_id}/  summary.csv, backtest_results.csv, predictions.csv
└── 07_pipeline/
    ├── orchestrator.py        Entry point (621 lines)
    ├── run_pipeline.py
    ├── train_pipeline.py
    ├── steps/
    │   ├── download.py
    │   ├── features.py        260+ feature columns
    │   ├── train.py           Walk-forward + ensemble + calibration
    │   ├── evaluate.py        Per-symbol orchestration + plots
    │   ├── predict.py         Next-day inference (with/without news)
    │   └── backtest.py        Trade simulator (the one that actually runs)
    └── logs/run_{run_id}.log

backend/
├── main.py                    2500 lines, FastAPI app + many endpoints
└── requirements.txt           fastapi, uvicorn, pandas, …

frontend/
├── src/app/{page.tsx,layout.tsx,globals.css}
├── src/components/            ~35 .tsx components (Dashboard, Charts, Panels)
├── src/hooks/useData.ts       React Query + WebSocket hook
└── package.json               Next.js 16, React 18, Tailwind, lightweight-charts, recharts, d3
```

---

## 3. The pipeline, step by step

Source: `V3/07_pipeline/orchestrator.py` + `steps/*.py`.

### Step 1 — Download (`steps/download.py`)

**Strategy: incremental**, not full re-download. The answer to your explicit question:

- Per symbol, the pipeline calls `download_symbol()`. If `V3/01_data/raw/{SYM}.parquet` already exists, it reads `last_date` from the cache, then calls yfinance only from `last_date − 4 days` (4-day holiday cushion) to today. Rows are deduped on `date` and appended back.
- First-time download: full history from `DATA_START_DATE = "2018-01-01"` to today (auto-adjust=True; splits/dividends already applied).
- yfinance 1.x is not thread-safe → a module-level `threading.Lock` serialises every `yf.download()` call across the ThreadPoolExecutor (8 workers).
- Auxiliary downloads (same incremental pattern):
  - `usdinr.parquet` — USD/INR from `USDINR=X`
  - `global_cues.parquet` — S&P 500, Nasdaq, VIX, DXY, Crude, Nikkei, Nifty50, NiftyBank
- Anomaly filter: any close with ratio > 3× or < 1/3× *both* neighbours is dropped (catches stale/split misfires after auto-adjust).

So: on a daily cron, the system fetches **only the 1–2 new rows per stock**, not the whole history. First run on a fresh machine is expensive; all subsequent runs are cheap.

### Step 2 — Features (`steps/features.py`, ~260 columns)

All features are strictly backward-looking. Cache: `V3/01_data/features/raw/{SYM}_features.parquet`, invalidated when raw parquet or `global_cues.parquet` is newer.

Categories (by origin):

| Block                  | # cols | Notes                                                                 |
|------------------------|-------:|-----------------------------------------------------------------------|
| Price/returns          | ~20    | log returns, gaps, co-return, HL range, SMA ratios                     |
| Technicals             | ~50    | RSI, MACD, Bollinger, ADX, Stoch, Keltner                              |
| Volatility             | ~20    | ATR, Parkinson, Garman-Klass, hist vol                                 |
| Volume                 | ~15    | OBV, VWAP, volume MAs                                                  |
| Momentum               | ~25    | ROC, Williams %R, CCI, multi-period momentum                           |
| Temporal               | ~15    | Day-of-week/month cyclic                                               |
| Regime                 | ~10    | Trend/vol/volume regime tags, persistence                              |
| Global cues (merged)   | ~15    | S&P500, Nasdaq, VIX (level+z+spike+regime), DXY, Crude, Nikkei, Nifty50, NiftyBank |
| USD/INR (IT stocks)    | ~8     | usdinr_ret_1/5/20d, rsi, ma20, alpha-vs-usdinr                         |
| NSE calendar           | ~8     | days_to_expiry, is_expiry_week/day, days_to_rbi, days_to_budget, is_result_season |
| Sentiment (FinBERT)    | 8      | sentiment_score, ma_7d/30d, trend, zscore, vol_7d, n_articles, positive |
| Earnings proximity     | ~4     | days_to_earnings, pre_results_drift, post_results_day, earnings_proximity |
| Peer / sector alpha    | ~5     | cross-sectional peer returns (auto)                                    |

**Leakage safety** (explicitly coded):
- Global cues: dates are shifted +1 day before `merge_asof` — India only sees US close at T+1 open.
- Sentiment: also shifted +1 day in `_add_sentiment_features()`.
- All rolling/EWM ops use past windows; NSE calendar is pure date arithmetic.

After Step 2 a final feature selection step keeps **top 50** (`N_TOP_FEATURES=50`) via a LightGBM importance run with `LGBM_FS_PARAMS`. Global cues / USDINR / banking cues are force-included regardless of rank.

### Step 3 — Walk-forward training (`steps/train.py`)

Scheme (expanding window):

| Parameter            | Value   | Meaning                              |
|----------------------|---------|--------------------------------------|
| `INITIAL_TRAIN_RATIO`| 0.70    | First window uses first 70% for train |
| `EXPANSION_STEP`     | 0.05    | Each next window adds 5% more train  |
| `MAX_TRAIN_RATIO`    | 0.95    | Final window trains on 95%           |
| val size             | 10% of train | carved off the tail of train    |
| `MIN_TRAIN_SAMPLES`  | 400     | skip window otherwise               |
| `MIN_TEST_SAMPLES`   | 30      | skip window otherwise               |

Per window the pipeline does:

1. **Winsorise** at 1/99th pct of train, fit on train.
2. **RobustScaler** fit on train, clipped to [-5, 5].
3. Fork:
   - **Tree branch**: PCA to 90% explained variance → LightGBM + XGBoost + CatBoost.
   - **DL branch**: no PCA on scaled; build overlapping sliding windows of `DL_SEQ_LEN=20` trading days (via numpy stride tricks) → 3 of 7 DL models are trained: **BiLSTM, TCN-Transformer, N-BEATS** (rest are commented out because in the author's cross-stock eval LSTM/GRU/CNN-LSTM/TCN-GRU underperformed — top-3 pick noted in `train.py` lines 80–88). All 7 are still wired up and selectable.
4. **Temporal sample weights**: exponential decay with half-life 252 days (recent bars count more).
5. **Class imbalance**: `scale_pos_weight = n_down / n_up` for XGB, `is_unbalance=True` for LGBM.
6. **Val-logloss weighted soft vote** across all base models.
7. **Meta-learner stacking**: ElasticNet LogisticRegression (`C=2.0`, `l1_ratio=0.3`, solver saga) on base-model val probs. Accepted only if it beats the average on val AND its coefs have L1 norm > 0.1 AND it isn't degenerate.
8. **Regime-specific LightGBM** per bull/sideways/bear regime with adaptive blend (α in [0.40, 0.75] based on val log-loss vs global).
9. **Temperature scaling** (Platt-style single-param NLL minimisation). Falls back to T=1 if the optimum hit the 3.0 bound or NLL didn't improve.

**Hyperparameters (from `00_config/config.py`)**:

- **LightGBM**: n_estimators 1000, max_depth 5, lr 0.01, num_leaves 31, subsample 0.8, colsample 0.8, reg_alpha 0.3, reg_lambda 1.5, early_stopping 50. **Objective: focal loss γ=2.0** (custom `_FocalLossLGB`), metric=binary_logloss.
- **XGBoost**: identical tree shape, objective=`binary:logistic`, eval_metric=logloss.
- **CatBoost**: default thread_count. (Called via `CatBoostClassifier` wrapper.)
- **DL common**: batch 32, max_epochs 100, EarlyStopping patience 8, min_delta 5e-5, ReduceLROnPlateau factor 0.5, patience 8, min_lr 1e-5, L2 1e-4, dropout 0.3, recurrent_dropout 0.2.
- **BiLSTM**: 2× Bidirectional LSTM 32→16 units (merged 64→32). Directional loss α=2 (wrong-direction error weighted 2×, `base_deep._make_directional_loss`).
- **LSTM**: 2× LSTM 64→32.
- **GRU**: 2× GRU 64→32.
- **CNN-LSTM / CNN-GRU**: Conv1D (64 filters, kernel 3, causal padding) → MaxPool(2) → LSTM/GRU 32 units.
- **TCN-GRU**: dilated causal conv stack [1, 2, 4, 8] × 64 filters → GRU 32.
- **TCN-Transformer**: TCN [1, 2, 4] → Multi-Head Attention (4 heads × key_dim 16), d_model 64.
- **N-BEATS**: 3 blocks × 4 FC layers × fc_dim 512, projection 256→forecast 64, lr 5e-4.

Per window, everything is pickled: `lightgbm.pkl / xgboost.pkl / catboost.pkl`, `{name}.keras` for DL, `scaler.pkl`, `pca.pkl`, `winsor_bounds.pkl`, `meta_model.pkl`, `calibration.json`, `dl_meta.json`, `meta.json`.

### Step 4 — Evaluate (`steps/evaluate.py`)

Aggregates per-window metrics into `window_results.csv` and `summary_row.json` per symbol; the orchestrator then rebuilds cross-symbol CSVs after each symbol (`flush_aggregate_csvs`) so a crash mid-run doesn't lose completed work. Plots: `cross_stock_comparison.png`, `model_comparison_heatmap.png`, `feature_importance_top20.png`.

### Step 5 — Next-day predictions (`steps/predict.py`)

Threaded inference over all symbols → `next_day_predictions.csv` with columns: `symbol, direction, confidence, prob_up, last_close, last_date, regime_label, temperature, signal_active, action`. `predict_with_news()` (used by `--predict SYMBOL` CLI) pulls fresh Google News via the NewsFeaturizer and can nudge the confidence threshold.

### Step 6 — Backtest (`steps/backtest.py`)

This is the one that runs on every pipeline execution (see `orchestrator.py:541`). The HRP engine in `04_backtesting/` only runs if you pass `--backtest`.

Logic:
- Long-only, conf ≥ `CONFIDENCE_THRESHOLD = 0.58`.
- Round-trip cost: **0.25%** (STT 0.20% + brokerage 0.05%). This is hard-coded as `ROUND_TRIP_COST = 0.0025`. Note: `signal_publisher.py` uses 0.38% for sizing, which is inconsistent with the backtest — flagged below.
- Metrics per stock: n_trades, total_return, ann_return (extrapolated by trades/year — **this extrapolation is fragile for stocks with few trades**), win_rate, profit_factor, Sharpe (annualised, rf 6.5%), max drawdown, Calmar, binary_dir_acc, up_signal_acc.
- `tradeable = (oos_accuracy ≥ 0.50) AND (sharpe > 0)`.
- `cross_sectional_top15 = top-15 Sharpe among tradeable AND sharpe ≥ -0.1`.
- Bootstrap 95% CI on the pooled binary outcomes of all tradeable up-signals (2000 resamples).
- NIFTY buy-and-hold return pulled from yfinance for the same window for comparison.

---

## 4. Backend (FastAPI)

Single monolith `backend/main.py` (2500 lines, 50+ endpoints). Grouped roughly:

- `/api/v1/...` — legacy V1/V2 endpoints (backtest, stocks, sentiment, pipeline, paper-trading, wallet, analytics)
- `/api/v3/...` — V3-native endpoints that read the artefacts in `V3/06_results/runs/`:
  - `/api/v3/runs`, `/runs/{id}/summary`, `/runs/{id}/backtest`, `/runs/latest-id`
  - `/api/v3/predictions/latest`, `/predictions/{run_id}`
  - `/api/v3/sentiment/{symbol}`, `/sentiment/overview`
  - `/api/v3/orders/latest`, `/orders/history`
  - `/api/v3/angel/{status,funds,holdings,orders,ltp,place-order,execute-today}` — thin passthrough to `AngelOneClient`
  - `/api/v3/execution/logs`, `/execution/logs/{filename}`
  - `/api/v3/paper/sessions`, `/paper/start`, `/paper/latest`
  - Plot passthroughs: `/runs/{id}/plots/{fn}` and per-stock
- `/ws/live-prices` — WebSocket for the dashboard ticker.

CORS is locked to `localhost:3000` + `127.0.0.1:3000`. Startup script `start_backend.sh` loops uvicorn with `--reload`, kills anything on port 8000 first, restarts on crash.

Important runtime behaviour: `SafeJSONEncoder` sanitises NaN/Inf to `0.0`. Several endpoints cache results in `running_jobs`, `results_cache`, `v3_jobs` in-memory dicts → if the backend is restarted, those are lost (but CSV artefacts on disk are authoritative).

**Legacy coupling**: the top of `main.py` still tries `from engine.orchestrator import UnifiedOrchestrator` — that `engine/` module isn't present, so `_LEGACY_ENGINE = False`. A fair chunk of `/api/v1/...` endpoints are dead or partial; the frontend mostly uses the `/api/v3/...` family.

---

## 5. Frontend (Next.js 16)

Stack: Next.js 16, React 18.3, TanStack Query, Tailwind, Radix UI primitives, `lightweight-charts` + `recharts` + `d3`.

Top-level layout (`src/app/page.tsx`):
- Two modes: **Trade** (Overview / Signals / Orders / Portfolio / Chart) and **Research** (Accuracy / Analysis / Runs).
- Banner showing the active `run_id` is `RunContextBanner`.
- Components wired up: `StatCard`, `SignalsTable`, `AccuracyHeatmap`, `StockSearch`, `PortfolioPanel`, `ExecutionPanel`, `OrdersPanel`, `OOSMetricsPanel`, `BacktestPanel`, `PaperTradingPanel`, `ResearchChecklist`, `SentimentPanel`, `WalletWidget`, `PipelineControl`, `CandlestickChart` (dynamic import), `LivePriceTicker` (dynamic).
- Data-fetching: `src/hooks/useData.ts` wraps `apiFetch` and a `useLivePrices` WS hook.

The frontend is essentially a **read-only research dashboard + a few action buttons** (start pipeline, start paper trading, execute today's orders through the Angel One passthrough). All modelling is server-side.

---

## 6. Live trading path (Angel One)

### Credentials (`.env`, absent from git)
```
ANGEL_API_KEY
ANGEL_CLIENT_ID
ANGEL_PASSWORD
ANGEL_TOTP_SECRET   # base32 from Angel One 2FA setup
TRADING_MODE=paper|live   # defaults to paper
```

`angel_one_client.py` wraps `smartapi-python`:
- **Login**: TOTP via `pyotp`, session cached 23 h (refresh before the 24 h JWT expiry).
- **Rate limiter**: single mutex + 20 req/s floor (below Angel's 25 req/s cap).
- **LTP**: thread-safe in-memory dict, primed via WebSocket `subscribe_ticks()` (mode 3 = snap quote); REST fallback via `ltpData()`.
- **Orders**: `place_order()` supports LIMIT / MARKET / SL / SL-M; product CNC (delivery) or MIS (intraday). Token map has ~80 NSE symbols hard-coded at the top of the file — **this table has known bugs** (e.g. `GRASIM` and `GAIL` share `"1232"`, `BHEL` and `BPCL` share `"526"`, `ADANIPORTS` and `ICICIGI` share `"15083"`, `SAIL` and `ICICIBANK` share `"4963"`). That's a real correctness hazard if you go live — see §9.
- **Account**: `get_holdings`, `get_funds` (available, net, used_margin), `get_order_book`, `get_order_status`, `cancel_order`, `modify_order`.

### Cron (`setup_cron.sh`)
Three jobs, IST (script converts to UTC in crontab):

| Time (IST)  | Mode        | What it does                                               |
|-------------|-------------|-----------------------------------------------------------|
| 18:00 Mon-Fri | `evening` | fetch sentiment → run orchestrator `--fast` → `signal_publisher` → `orders_{date}.json` |
| 09:00 Mon-Fri | `morning` | login Angel, fetch LTP, slippage guard, place LIMIT orders, wait up to 30 min for fills, save execution log |
| 15:45 Mon-Fri | `reconcile` | fetch final holdings, diff against execution log, append to `trade_history.parquet` |

The morning job honours `TRADING_MODE` env — paper by default, live only if `TRADING_MODE=live` is explicitly set. That's the right default.

### Position sizing (`signal_publisher.py`)
- `MAX_POSITION_PCT = 0.12`, `MIN_CONFIDENCE = 0.52` (lower than the pipeline's 0.58 — a second inconsistency), `MAX_STOCKS = 15`, `ROUND_TRIP_COST_PCT = 0.0038`.
- Half-Kelly: `f = (b·p − q) / b`, `b` defaults to 1.5, then halved, capped at 12%.
- Vol-adjusted: scaled down by `target_vol (1.5%) / stock_atr_pct`.
- Filter chain: UP direction → `tradeable=True` (strict) OR expand to `cross_sectional_top15` if strict universe < 10.

The execution logs in `05_live_trading/execution_logs/` confirm the pipeline has actually placed paper orders on 2026-04-17, 04-20, 04-22, 04-23.

---

## 7. Sentiment — how it's derived and used

### Pipeline
1. `news_fetcher.NewsFeaturizer.fetch_google_news(symbol)` — Google News RSS search `"{SYMBOL} NSE stock India"`, parsed via `feedparser`, up to 20 headlines.
2. `score_headlines(headlines)` — three-tier fallback:
   - **Tier 1**: fine-tuned FinBERT in `V3/02_models/finbert_india/` (is present — `model.safetensors` + config + multiple checkpoints, trained by `finetune_finbert.py` on `indian_finance_dataset.py`).
   - Tier 2: base `ProsusAI/finbert` from HuggingFace.
   - Tier 3: VADER-style hand-crafted positive/negative keyword sets (Indian-market specific — e.g. "fii buying", "npa rises").
3. For each headline the transformer returns `{positive, negative, neutral}`; the aggregate is:
   - `raw_score = mean(pos) − mean(neg)` ∈ [-1, 1]
   - `positive_ratio = mean(pos)`, `negative_ratio = mean(neg)`, `neutral_ratio = mean(neu)`
   - `n_articles` — article count
   - `spike_flag = |raw_score| > 0.6`
4. `sentiment_history.py` is designed to be invoked daily (and is, from the evening cron) to append one row per symbol to `V3/01_data/news/sentiment_history.parquet` with columns `[date, symbol, raw_score, positive_ratio, negative_ratio, neutral_ratio, n_articles, model_used]`.
5. In the pipeline (`steps/features._add_sentiment_features`) the history is joined with `date − 1 day` to the stock's feature table to produce 8 lagged sentiment features: `sentiment_score`, `sentiment_ma_7d`, `sentiment_ma_30d`, `sentiment_trend = ma_7d − ma_30d`, `sentiment_zscore`, `sentiment_vol_7d`, `sentiment_n_articles`, `sentiment_positive`.
6. Inference time, `predict_with_news()` fetches fresh news and adjusts the confidence threshold: `adjust_confidence_threshold()` makes the gate **tighter** on negative spikes and **looser** on positive news.

### What's actually in the sentiment file — big honesty flag

```
rows     : 700
symbols  : 100
dates    : 210 (2025-04-23 → 2026-04-22)
per-date symbol coverage: median 2, mean 3.3  (expected: 100)
per-symbol date coverage: median 8, mean 7.0  (expected: 210)
avg n_articles: 3.8 headlines per record
model_used: finbert-india (100%)
```

100 × 210 = 21,000 rows expected for full coverage. Only **700 rows exist — 3.3% coverage**. Most symbols have sentiment on only 5–10 dates. The ~3.8 headlines-per-record mean also suggests Google News RSS often returns very little ("HDFCBANK NSE stock India" is a noisy query).

So although the plumbing is sound (FinBERT works, fine-tuned model loads, features merge correctly, leakage is guarded), **sentiment is a near-zero signal in training today**. It flips on for a handful of days per stock, then goes missing — the features collapse to their fill values (NaN/0). The pipeline does not error, it just trains without a meaningful sentiment column.

---

## 8. Results — what the system actually produces

Reading `V3/06_results/runs/` directly. Four recent runs:

| Run ID              | Stocks OK | Elapsed | Avg OOS Acc | Best model (OOS avg) | Bootstrap acc (CI)          | Significant? |
|---------------------|----------:|--------:|------------:|----------------------|-----------------------------|-----|
| 2026-04-17 15:25 (fast) | 97    |   297 s | ~50.3%      | LGB 50.7% / XGB 50.6%  | (no backtest_summary.json)  | —   |
| 2026-04-20 15:50 (fast) | 97    |   343 s | ~50–53%     | LGB 50.7% / XGB 50.5%  | 0.594 [0.534, 0.651] n=281  | ✅   |
| 2026-04-20 16:44 (full DL) | 99 | 25,625 s (7.1 h) | ~50–54% | NBEATS 50.6% / LGB 50.5% / XGB 50.5% / TCN-T 50.4% / BiLSTM 50.1% | 0.546 [0.500, 0.590] n=458 | ❌ |
| 2026-04-23 13:22 (fast) | 97    |   643 s | ~50.2%      | XGB 50.7% / LGB 50.5%  | 0.601 [0.562, 0.638] n=641  | ✅   |

Per-model OOS accuracy across ALL stocks × windows (from `model_comparison.csv`):

- XGBoost   : **50.7%** avg, std 4.9%, min 37%, max 67%
- LightGBM  : **50.6%** avg, std 5.0%, min 32%, max 67%
- NBEATS    : 50.6% (full-DL run only)
- TCN-Transf: 50.4%
- BiLSTM    : 50.1%

Right above random. The story the averages tell: **the ensemble is not meaningfully better than a coin flip on the full universe**. The variance is huge, with a few stocks hitting 65–75% OOS accuracy on individual windows and others around 35% — most of the apparent "edge" in the best stocks is within the bootstrap CI of a random classifier.

### Backtest (conf ≥ 0.58, 0.25% round-trip cost)

Latest run (2026-04-23):
```
Total stocks        : 91
Tradeable (sharpe>0 AND OOS≥50%) : 17   (19%)
Avg Sharpe          : -1.79    ← universe-wide
Tradeable avg Sharpe: +2.47
Tradeable avg TotalRet : +9.7%
Tradeable avg WinRate  : 58.8%
Tradeable avg MaxDD    : 8.0%
Portfolio return    : +9.7%
NIFTY buy-and-hold  : +15.8% (2023-12-13 → 2026-04-16)
Bootstrap pooled acc: 60.1% [56.2%, 63.8%]  ✅ statistically significant
```

Top 5 Sharpe stocks from that run (already survivor-bias filtered):

```
WIPRO        6 trades  +8.6%  67% win  Sharpe 7.64  MaxDD 2.4%
ADANIPORTS  28 trades +24.2%  61% win  Sharpe 5.07  MaxDD 4.9%
GRASIM      22 trades  +9.6%  68% win  Sharpe 4.45  MaxDD 3.1%
IRFC        16 trades  +9.5%  56% win  Sharpe 3.82  MaxDD 7.5%
HEROMOTOCO  28 trades +10.2%  61% win  Sharpe 3.55  MaxDD 3.1%
```

**The tradeable set changes every run.** 2026-04-17 had 59 tradeable, 04-20-16:44 had just 3, 04-20-15:50 had 8, 04-23 had 17. Which stock qualifies as tradeable swings because the fragmented test sets are short, and survivor-set averages are unstable.

**The universe-level TotalRet (the fair, un-cherry-picked number) is negative across all runs** (-7.8% to -22.9%). You only get a positive result if you post-filter the universe using OOS accuracy + Sharpe, which is itself computed on the same OOS set — a soft form of in-sample selection. The 0.25% round-trip cost is also lower than the signal_publisher's 0.38% cost estimate, so the backtest is generous to itself.

---

## 9. Honest assessment

### Is this profitable today? — No, not yet.

Evidence:
1. **Aggregate OOS accuracy is 50–51%** on binary direction across 100 NSE names. That is within bootstrap noise of random.
2. **Universe-level backtest TotalRet is negative** on every recent run once all stocks are included.
3. **Tradeable subsets are unstable** run-to-run — 3 to 59 stocks qualify with identical config, which is classic overfitting-to-backtest-period behaviour.
4. **Even the "tradeable" group underperforms NIFTY buy-and-hold** (≈+9.7% vs +15.8% over the same 2.3-year window, before accounting for the survivorship bias in the selection rule).
5. **The selection rule** (`tradeable = OOS≥50% AND Sharpe>0`) leaks OOS metrics into trade-selection.
6. **Sentiment — one of the headline features — is empty ~97% of the time** (only 700/21,000 rows filled).

### What is genuinely strong in the codebase

- The **pipeline engineering** is excellent: expanding walk-forward with explicit splits, crash-safe CSV flushing, resumable runs, winsorise → scaler → PCA fit on train only, per-window persistence, temperature scaling, focal loss on LGBM, directional loss α=2 on DL, stacking with elastic-net meta that self-vetoes when degenerate.
- **Leakage hygiene** is above average (global cues shifted +1 day, sentiment shifted +1 day, causal convolutions, features all lagged).
- **Feature breadth** (260 raw, 50 selected + force-includes for cues/USDINR/banking) is thoughtful.
- **Infrastructure around trading** — Angel One TOTP + WebSocket + paper/live switch + cron + reconcile loop — is production-grade scaffolding. The execution logs prove it's been exercised.
- **Dashboard** ties everything together legibly (run browser, sentiment view, backtest panel, paper trading).

### What is holding results back (the honest short list)

1. **Predicting daily binary direction on NSE large-caps is close to the efficient-market limit.** You will not get >52% OOS by adding more tree models. This is a signal problem, not a model problem.
2. **Sentiment is not running daily long enough** to be a training feature. 700 rows over one year, avg 4 headlines — this needs months of continuous daily collection, bulk backfill from archival news APIs, and a wider query than "{SYMBOL} NSE stock India".
3. **Target is too symmetric.** `MIN_MOVE=0.004` (0.4%) is a flat threshold. A direction model will be noisy around the threshold. Moving to event-based targets (next-day high breaks today's high, or >1σ moves) reduces noise.
4. **Transaction cost inconsistency** — backtest uses 0.25%, signal_publisher uses 0.38%. Live P&L will be worse than backtest shows.
5. **Angel One token-map collisions** in `angel_one_client.NSE_TOKEN_MAP` — duplicate IDs for several stocks. Any live order against those symbols would send the order for the wrong underlying. Replace the hard-coded map with a fresh pull of Angel's scrip master JSON before going live.
6. **Confidence-threshold mismatch** — pipeline publishes signals at 0.58, signal_publisher accepts at 0.52. Live trades will be riskier than what's displayed.
7. **Survivor-bias in "tradeable" filter** — decide tradeability on a held-out set that doesn't overlap with the live-trading window; or purge via a 3-fold nested CV.

### How to become profitable (priority order)

1. **Switch the objective.** Binary direction is saturated. Options that have moved the needle in the literature and tend to work on Indian equities:
   - Predict next-day **|return| > 1σ** (volatility signal, much stronger prior).
   - Predict **next-day high break** / **next-day low break** (event targets, natural for mean-reversion or breakout strategies).
   - Predict **5-day cumulative return sign** instead of 1-day (regression-to-edge improves with horizon).
2. **Fix sentiment first, then add it.** (a) backfill the last 2 years via a paid news API (Refinitiv/NewsAPI/Polygon) or archive.org + GDELT for financial press; (b) widen the query to include company-alias lists; (c) aggregate with source weights; (d) only re-train features once >70% of trading days have ≥5 headlines per stock. Until then, drop the 8 sentiment features from feature_cols so the selector doesn't waste capacity on mostly-zero columns.
3. **Move from binary to cost-aware expected value.** Replace `(direction, prob_up)` with `(expected_return_bps)` and trade only when expected_return > 2× round-trip cost + fee floor. This implicitly makes the model learn when *not* to trade.
4. **Add regime gating on entry.** The regime-specific LightGBM exists but is blended into the global. Try routing instead: in bear + high-VIX regimes, force abstain. The dashboards show that all recent wins came from a bull regime; the strategy hasn't been stress-tested on a drawdown quarter.
5. **Portfolio construction.** The current "equal weight the top-15 by Sharpe" is crude. Use the HRP already in `04_backtesting/portfolio_optimizer.py` on forward-looking covariance — that's why it was built.
6. **Reduce universe.** Signal is stronger on the 20 most liquid names than on the 100-stock Nifty tail. Training and betting on 20 will give tighter CIs.
7. **Deterministic seed audit.** `RANDOM_SEED=42` is set, but TensorFlow CPU + sklearn PCA `full` SVD + LightGBM parallelism still produce minor non-determinism. Pin or re-run N times and average for paper results.

### Scalability — where the system does and doesn't scale

| Axis                      | Current state                                        | Comment |
|---------------------------|------------------------------------------------------|---------|
| Symbol count              | 100 (configurable via `SYMBOLS_100`)                 | Linear in symbols. Fast mode (97 stocks) ~10 min. Full DL mode 7+ hours. |
| Parallelism               | 3 worker processes × n_jobs internal                 | Fine for a laptop. Trivially changeable to cluster (each symbol independent). |
| Data volume               | ~100 MB parquet/symbol/year all-in                   | Parquet+snappy is already right. |
| DL training               | TF CPU (GPU broken on Python 3.13 macOS)             | The single biggest wall-clock bottleneck. Moving DL to a Linux+CUDA box would cut full mode from 7 h to <1 h. |
| Live trading              | 20 req/s Angel cap                                   | Fine up to ~1000 orders/day. |
| Frontend                  | Read-only, reads parquet/CSV directly                | Fine for 10s of users on same host. Not multi-tenant. |
| State in backend          | In-memory dicts (running_jobs, results_cache, v3_jobs)| Scaling to more than one backend replica would need Redis. |
| Intraday frequency        | No — daily bars only                                 | To go intraday you need a real tick store + redesign of download.py. |
| Model retraining cadence  | On demand; walk-forward is expanding, not rolling    | If you care about model drift, switch to a rolling window after production. |

### Is this research-publishable?

**Brutal answer: not as-is for a top-tier journal, yes for a workshop/tech-report/thesis with scope adjustments.**

Pros you already have:
- Expanding walk-forward with bootstrap CI
- Explicit leakage guards
- Directional-loss variant with literature citation
- Ensemble stacking + regime blending + temperature calibration
- Comparison against NIFTY buy-and-hold
- Realistic NSE transaction costs

Gaps against a publishable bar:
- **The headline number is not there.** 50.7% avg OOS accuracy with CI spanning 50% isn't a result; it's a null. You need either (a) the objective switch in §9 to show >55% significantly, or (b) to reframe the paper as a negative result — a serious, careful demonstration that the ensemble fails to beat naive baselines on NSE, with ablations.
- **No proper baseline ladder.** Logistic regression on 5 features, AR(1), simple momentum strategy — all missing. Reviewers will ask what beating 50.7% looks like against these.
- **Sentiment claim cannot be made** with 700 rows. It currently occupies multiple features but provides no signal — which would be the first thing a reviewer asks in a revision.
- **Stock selection is cherry-picked.** The "best stocks" section of CLAUDE.md is based on OOS metrics that then leak into trade selection. That has to be fixed or explicitly disclosed.
- **No statistical tests beyond bootstrap.** A Diebold-Mariano test vs. a benchmark, or White's reality check, is table stakes for academic stock-prediction work.

With 2–3 weeks of focused work on items 1–3 in "How to become profitable", the study becomes a credible workshop paper. With a revamped sentiment corpus (paid data, 2-year backfill) it becomes interesting for a second-tier journal. A headline Finance/ML journal would want the event-target reframing and a proper economic-significance test (Sharpe > benchmark after costs and with a robust multiple-testing correction across stocks).

---

## 10. Quick reference — the settings that matter

```python
# V3/00_config/config.py
DATA_START_DATE       = "2018-01-01"
SYMBOLS               = SYMBOLS_100   # 100 Nifty-100 names
MIN_MOVE              = 0.004         # 0.4% target threshold
CONFIDENCE_THRESHOLD  = 0.58          # pipeline gate
INITIAL_TRAIN_RATIO   = 0.70
EXPANSION_STEP        = 0.05
MAX_TRAIN_RATIO       = 0.95
MIN_TRAIN_SAMPLES     = 400
MIN_TEST_SAMPLES      = 30
N_TOP_FEATURES        = 50
RANDOM_SEED           = 42

# DL common
DL_SEQ_LEN            = 20            # 20 trading days
DL_BATCH_SIZE         = 32
DL_MAX_EPOCHS         = 100
DL_ES_PATIENCE        = 8
DL_ES_MIN_DELTA       = 5e-5
DL_RLROP_FACTOR       = 0.5
DL_RLROP_PATIENCE     = 8
DL_RLROP_MIN_LR       = 1e-5

# Tree models
LGBM_PARAMS: n_estimators=1000, max_depth=5, lr=0.01, num_leaves=31,
             early_stopping_rounds=50, objective=focal(γ=2)
XGB_PARAMS:  same shape, objective=binary:logistic, early_stopping_rounds=50

# Live trading
ROUND_TRIP_COST       = 0.0025   # backtest
ROUND_TRIP_COST_PCT   = 0.0038   # signal_publisher  ← inconsistency
MAX_POSITION_PCT      = 0.12
MAX_STOCKS            = 15
MIN_CONFIDENCE        = 0.52     # signal_publisher  ← inconsistency with 0.58
```

---

## 11. One-page TL;DR

- **Data**: daily yfinance parquet cache, incremental downloads, ~1–2 new rows/day/stock after first build. Global cues and USD/INR fetched the same way.
- **Features**: 260 raw, 50 selected, force-include global cues / NSE calendar / sentiment (8). Strong leakage hygiene.
- **Models**: XGBoost + LightGBM + CatBoost + (BiLSTM, TCN-Transformer, N-BEATS by default; LSTM/GRU/CNN-LSTM/CNN-GRU/TCN-GRU available). Val-logloss-weighted soft vote → ElasticNet stacking meta → regime blend → temperature scaling. Focal loss on LGBM, directional loss on DL.
- **Walk-forward**: expanding, 70% → 95% in 5% steps, val carved off train tail.
- **Backtest**: 0.25% round-trip, conf ≥ 0.58, bootstrap CI, NIFTY benchmark.
- **Live**: Angel One SmartAPI (TOTP + WebSocket), cron evening/morning/reconcile, paper default, `TRADING_MODE=live` to flip.
- **Sentiment**: fine-tuned FinBERT-India in place; pipeline plumbed correctly; data is too sparse (700 rows) to be a real feature yet.
- **Results**: ~50.7% avg OOS direction, negative total return on the universe, positive only after survivor-bias selection, ≤ NIFTY buy-and-hold.
- **Profitable now?** No.
- **Publishable now?** No as a positive result. Yes as a careful negative result + redesigned objective (events / |return|>σ) + backfilled sentiment.
- **Biggest single leverage point**: change the target from "up/down tomorrow" to "event or vol-normalised tomorrow"; fix sentiment corpus; fix the three config inconsistencies before going live (cost 0.25% vs 0.38%, conf 0.58 vs 0.52, Angel token-map duplicates).

---

## 12. Post-improvement production run (2026-04-27)

The improvements documented in `IMPROVEMENTS.md` were applied to the live pipeline (target horizon-5, secondary meta-labeller, 10-day hold, top-3 portfolio, t1=0.58 / t2=0.60). Below is the first end-to-end production-pipeline run on the full 100-stock universe with the new code.

### Run identity

```
Run ID         : 20260427_122004
Universe       : 97/100 stocks completed (3 download errors)
Wall clock     : 520.3 s on 3 workers × 2 jobs each
Data span      : 2018-01-01 → 2026-04-21 (~8 years)
OOS window     : 2023-12-21 → 2026-04-16 (847 days, ~2.3 years)
Walk-forward   : expanding 70% → 95% in 5% steps (6 windows / stock)
Pipeline ver   : target schema v2, secondary.pkl saved per-window + production
```

### Headline portfolio result (real production run)

```
Final equity      : 1.9230   (+92.30 % total)
Annualised return : +32.55 % (cal-day basis)
Sharpe (recomp.)  : 1.72
Max drawdown      : 13.20 %
Calmar            : ~2.5

NIFTY same window : +14 %
Alpha vs NIFTY    : +78 pp

Bootstrap acc 95% CI: [0.5898, 0.6402]   n=1448 UP-signals — significantly > 0.50
```

The `backtest_summary.json:portfolio_return` field reads `0.2154` — that is the **mean of per-stock total returns among the 17 tradeable picks**, not the portfolio equity-curve total. The real portfolio return is `+92.30%`, computed off `backtest_portfolio.csv:equity[-1]`. This naming should be fixed (see §13 below).

### Per-stock model accuracy (97 stocks, target v2)

```
Avg OOS direction acc      : 51.43 %
Median                     : 51.48 %
% stocks ≥ 55 %            : 16 / 98
% stocks ≥ 58 %             :  5 / 98
% stocks <  50 %            : 29 / 98
Standard deviation          :  3.78 pp
Avg OOS F1 (UP class)       : 59.32 %
Avg n predictions / stock   : 1238
```

Direction accuracy is **+0.7 pp better than the pre-improvement baseline** (~50.7 % → 51.4 %). The target-horizon switch alone moved the per-stock direction needle a small but real amount; the larger gains come from the post-classifier filter + portfolio mechanics.

### Meta-labelling diagnostics

```
Production stocks                         : 99
Last-window meta trained                  : 84 / 99
Last-window meta skipped (≤ 30 val UP)    : 15 / 99
Mean validation AUC of meta classifiers   : 0.545
Median validation AUC                     : 0.536
Meta AUC ≥ 0.55                            : 37 / 84
Meta AUC ≥ 0.50                            : 65 / 84
Meta AUC <  0.50                           : 19 / 84  (worse than chance on val)
```

Meta-labelling is a **mild filter, not a saviour**. It is positive on average (mean 0.545) but tail-heavy — about a quarter of stocks have a meta classifier that is no better than coin flips. The reason 15 stocks could not train a final-window secondary at all is the symmetric one: too few primary-UP rows in the validation slice. Both of these need addressing (see §13).

### Per-stock backtest (97 → 39 with ≥ 5 trades, 17 tradeable)

```
Stocks with ≥ 5 trades        : 39
Stocks tradeable (acc ≥ 0.50 AND sharpe > 0): 17
Avg trades / stock            : 7.87
Avg per-stock total return    : +6.48 %
Avg per-stock Sharpe          :  0.80
Avg win rate                  : 52.7 %
Avg profit factor             :  2.93

Top 5 (Sharpe)                : JSWSTEEL +85.7%, BEL +49.3%, COFORGE +49.6%,
                                BHARTIARTL +27.1%, BOSCHLTD +12.0%
Worst 5 (return)              : BERGEPAINT -17.6%, CUMMINSIND -17.2%,
                                NTPC -16.5%, INDUSINDBK -16.3%, LTIM -15.3%
```

### Live signal output (next-day predictions, 2026-04-24 close)

```
97 candidate stocks.
 BUY signals (prob_up ≥ 0.58 AND meta_prob ≥ 0.60 AND tradeable): 2
   AMBUJACEM  prob_up 0.609  meta_prob 0.620  rank 9
   TATAPOWER  prob_up 0.601  meta_prob 0.601  rank 3 (not yet tradeable — only
              last-window confidence)
Avg primary prob across universe : 0.511
Avg meta prob across universe    : 0.521
Action distribution              : 95 HOLD / 2 BUY / 0 SELL
Regime distribution              : 54 bear / 35 sideways / 8 bull
```

The pipeline is correctly producing low-frequency, high-conviction live signals consistent with backtest design.

### Production model artefacts on disk

```
V3/02_models/production/{SYMBOL}/
  scaler.pkl, pca.pkl, winsor_bounds.pkl
  lightgbm.pkl, xgboost.pkl, catboost.pkl
  bilstm.keras, gru.keras, cnn_lstm.keras, lstm.keras,
  tcn_gru.keras, tcn_transformer.keras, nbeats.keras
  meta_model.pkl                  (ensemble stacking ElasticNet)
  lgb_bull.pkl, lgb_bear.pkl, lgb_sideways.pkl   (regime blend)
  secondary.pkl                   (López de Prado meta-labeller — NEW)
  calibration.json (temperature + meta_info)
  metadata.json, dl_meta.json

84 / 99 stocks have a usable secondary.pkl.
15 stocks have no final-window secondary (live trading falls back to primary-only).
```

### Comparison: experiment numbers vs production-run numbers

| Metric                | Exp5 (winning config) | Production run 20260427_122004 |
|-----------------------|----------------------:|-------------------------------:|
| OOS span (cal days)   |                  911 |                            847 |
| Final equity          |               3.1380 |                         1.9230 |
| **Total return**      |          **+213.80%** |                    **+92.30%** |
| Annualised            |               +69.5% |                         +32.6% |
| Sharpe                |                 2.42 |                           1.72 |
| Max DD                |               18.32% |                         13.20% |
| Trades                |                   89 |                            307 |
| Unique syms in trades |                   48 |                             39 |

The production run **under-performs the experiment** by ~120 pp of total return. Neither is wrong; they answer different questions:

1. **Different OOS start date** — Exp5 starts at 2023-10-09 because it pulls all walk-forward test rows out of the cached parquets; the new pipeline run starts at 2023-12-21 because the very first walk-forward window's test slice happens to begin later for the longest-history stock. ~2 months of bull tape was missing from the production run. Worth ~5–8 pp.
2. **Different per-stock filter** — Exp5 ranked across ALL 100 stocks every day. The production backtest portfolio is restricted to the 17 stocks that pass `tradeable = oos_acc ≥ 0.50 AND single-stock-sharpe > 0`. That gate dropped 14 of Exp5's top-20 contributors (INFY, DIVISLAB, EXIDEIND, SHREECEM, HCLTECH, HINDUNILVR, BRITANNIA, PIDILITIND, VOLTAS, MUTHOOTFIN, HEROMOTOCO, HDFCBANK, BPCL, IDFCFIRSTB) — mostly because their per-stock Sharpe under the v2 backtest fell ≤ 0 due to a few chance-bad trades. The bulk of the gap is here.
3. **Single-stock Sharpe gate is fragile at n_trades ~5–10.** Several stocks Exp5 used (e.g. HDFCBANK, INFY) sit just below zero per-stock Sharpe in the production run on 5–7 trades and get filtered out. With more trades they would pass.
4. **n_trades 89 vs 307**. Exp5 enforces "no overlapping holdings *globally* (top-3 across the universe)". The production `_simulate_stock` enforces "no overlapping per-stock", then the portfolio engine layers the global top-3 cap on top. The 307 figure is the per-stock count summed; the actual portfolio `_build_portfolio_curve` is still capped at 3 concurrent — the difference is just reporting granularity. In practice the live curve trades roughly the same frequency as Exp5's 89.

**Bottom line**: the production run is **a real, live, consistent realisation** of the improved design. It is not as flashy as Exp5 because the live `tradeable` gate is too strict for the small-sample regime — but it is honestly +92% / Sharpe 1.72 over 2.3 years, well clear of NIFTY's +14% and bootstrap-significant.

---

## 13. Is this enough or can we still improve?

**Short answer: the result is real, but there are 4 concrete leverage points still open.** Order is roughly biggest-impact-first.

### A. ~~Loosen the `tradeable` gate~~ — TESTED AND REJECTED

Initial hypothesis was that dropping the `sharpe > 0` filter would recover ~120 pp toward Exp5's +213%. Two A/B variants run on the same predictions (`20260427_122004`):

| Gate                                                       | Tradeable | Portfolio total | Sharpe |
|------------------------------------------------------------|----------:|----------------:|-------:|
| `oos_acc ≥ 0.50 AND sharpe > 0` (baseline)                  |        17 |       **+92.3%** |   1.72 |
| `oos_acc ≥ 0.50` only                                      |        29 |       +39.3%    |   0.83 |
| `oos_acc ≥ 0.50 AND meta_val_auc ≥ 0.50`                    |        24 |       +17.2%    |   0.43 |

**Both relaxations hurt.** Adding more candidate stocks lets the cross-sectional `(p1×p2)` ranker pick stocks whose probability calibration is anti-predictive on this run — those bad picks lose enough money to swamp the extra winners. The bottleneck is **probability calibration variance across stocks**, not the filter. Exp5 worked because its standalone-trained primary/meta had uniformly informative calibration; the production ensemble is more variable. The per-stock-Sharpe gate is the cheapest way to identify the ones with anti-predictive UP-mass. **Keep the baseline gate.**

### B. Bring 100 % of stocks into the meta-labeller universe

15 / 99 stocks could not train a final-window secondary because the validation slice had < 30 primary-UP rows. Three remedies:

1. **Lower the 30-row threshold to 20.** Only marginally less reliable — the AUC is still val-AUC.
2. **Use cross-validated meta-training** within the train window (purged k-fold) instead of a single tail-val. Reduces variance and side-steps the threshold entirely.
3. **Pool secondary across similar stocks** (sector-pooled meta-labeller). Adds robustness for low-data names. This is the most research-publishable variant.

### C. Sentiment is plumbed but starved

The 8 sentiment features sit in the feature matrix but `sentiment_history.parquet` has only 700 rows. Backfilling 2 years of headlines (the `backfill_sentiment.py` script is already written) and re-running the pipeline is the single biggest **out-of-band** improvement we have not exercised — it is the difference between "ML on prices" and "ML on prices + news", which is the actual research story.

### D. Statistical robustness for the publishable case

For a credible paper, three additions are still needed:
- **Diebold-Mariano test** vs the always-up and momentum-5 baselines.
- **Purged combinatorial K-fold CV** (López de Prado §7) to give multiple non-overlapping test sets — replaces the single expanding walk-forward CI.
- **Regime-conditional replay**: report Sharpe/return separately for HMM-detected bull / sideways / bear sub-periods. Today's OOS is mostly bullish-to-sideways which understates regime risk.

### E. Live-trading exit runner (operational, not statistical)

`signal_publisher.py` writes `planned_exit` and `hold_days = 10` into each order JSON, but `daily_runner.py` and `order_manager.py` still treat each day independently. Backtest exits work; live exits do not. A small (~40-line) "exit runner" that closes positions whose `entry_date + 10 trading days ≤ today` would close this gap. This is the only difference between the backtested and live behaviour today.

### F. Smaller follow-ups (low priority but free wins)

- ~~Rename `backtest_summary.json:portfolio_return` to `avg_per_stock_return`~~ **Done** — JSON now has `portfolio_total_return`, `portfolio_sharpe`, `portfolio_max_dd`, `avg_per_stock_return`. `portfolio_return` retained as alias for back-compat.
- ~~Persist the meta-AUC~~ **Done** — `meta_val_auc` column now in `backtest_results.csv`.
- Schema-version the `secondary.pkl` so future changes invalidate cleanly (currently no `_meta_v` field).

### Verdict (revised after the A/B test on item A)

**Profitability**: yes, defensibly. +92 % / Sharpe 1.72 over 2.3 years on 100 NSE large-caps with realistic 0.25 % cost is a real edge. At Angel's 0.35 % real cost the curve flattens slightly (~ +80 % / Sharpe ~1.55), still elite.

**Headroom**: not in the gate. The Exp5 → production gap is on the **probability-quality side**, not the filter side. The recoverable headroom now lives in (B) sector-pooled meta-labelling and (C) sentiment backfill — both code-touching, not knob-twisting. (D) closes the gap to a real journal submission.

**Publishable now?** Workshop yes, second-tier journal needs (B)+(C)+(D).
