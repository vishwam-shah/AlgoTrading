# Production Trading System

A complete, production-ready stock prediction and trading system for NSE (National Stock Exchange of India).

## Architecture Overview

```
production/
├── __init__.py           # Package exports
├── feature_engine.py     # Advanced feature engineering (100+ features)
├── models.py             # XGBoost + LightGBM ensemble
├── signals.py            # Trading signal generation
├── backtester.py         # Walk-forward backtesting
├── broker.py             # Angel One / Paper trading integration
└── orchestrator.py       # Main pipeline coordinator
```

## Quick Start

```bash
# Full pipeline
python trade.py --full

# Generate signals only
python trade.py --signals

# Run backtest
python trade.py --backtest --symbols HDFCBANK TCS

# Paper trading with execution
python trade.py --full --paper --execute

# Live trading (CAUTION!)
python trade.py --full --live --execute
```

## Pipeline Stages

### Stage 1: Data Collection
- Downloads OHLCV data from Yahoo Finance
- Supports 50+ NSE stocks
- Configurable history (default: 500 days)

### Stage 2: Feature Engineering
113 features across 9 categories:

| Category | Count | Description |
|----------|-------|-------------|
| Technical | 50 | RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic |
| Volatility | 15 | Historical, Parkinson, Garman-Klass volatility |
| Volume | 12 | OBV, VPT, Volume ratios, Accumulation/Distribution |
| Momentum | 15 | ROC, Momentum, Trend strength, Higher highs/lows |
| Statistical | 10 | Skewness, kurtosis, z-scores, autocorrelation |
| Regime | 8 | Trend/volatility/momentum regime detection |
| Alpha | 10 | Proprietary signals (smart money flow, gap analysis) |
| Market | 5 | Nifty correlation, beta, sector relative strength |
| Sentiment | 8 | News sentiment from Google News + VADER |

### Stage 3: Model Training
Ensemble of XGBoost + LightGBM:

- **Direction Model**: Predicts up/down (binary classification)
- **Return Model**: Predicts magnitude of move (regression)
- **Quantile Models**: Predicts confidence bounds (25th/75th percentile)

Features:
- Class-balanced training
- Early stopping with validation
- Feature importance tracking
- Model calibration for reliable probabilities

### Stage 4: Backtesting
Walk-forward validation:

- Train on N years, test on next period
- Retrain periodically (every 3 months)
- Realistic transaction costs (0.1% commission + 0.1% slippage)
- ATR-based stop losses and take profits
- Portfolio-level risk management

Metrics tracked:
- Total return, Annualized return
- Sharpe ratio, Sortino ratio
- Maximum drawdown
- Win rate, Profit factor
- Average holding period

### Stage 5: Signal Generation
Multi-factor signal scoring:

1. Model prediction (direction + confidence)
2. Risk/reward ratio (ATR-based)
3. Position sizing (volatility-adjusted)
4. Sector allocation constraints

Output:
- BUY/SELL/HOLD recommendation
- Entry price, target, stop loss
- Suggested position size (% of portfolio)
- Technical summary

### Stage 6: Execution
Two modes:

**Paper Trading** (default):
- Simulated orders with realistic slippage
- Uses real-time prices from Yahoo Finance
- Full portfolio tracking

**Live Trading** (Angel One):
- Requires API credentials in `.env`
- Market and limit orders
- Position and order management
- Real-time quotes

## Configuration

### Environment Variables (.env)

```bash
ANGEL_API_KEY=your_api_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_pin
ANGEL_TOTP_SECRET=your_totp_secret
```

### Stock Universe

Default 10 stocks (configurable):
- Banking: HDFCBANK, ICICIBANK, SBIN, KOTAKBANK
- IT: TCS, INFY
- Energy: RELIANCE
- Auto: TATAMOTORS
- Pharma: SUNPHARMA
- Infra: LT

## Results Storage

```
production_results/
├── data/           # Downloaded OHLCV data
├── features/       # Computed features
├── models/         # Trained models + scalers
├── backtest/       # Backtest reports + plots
├── signals/        # Generated signals (JSON)
├── trades/         # Trade execution logs
└── reports/        # Summary reports
```

## Key Innovations

### 1. Alpha Signals
Proprietary features not found in standard libraries:
- **Smart Money Flow**: Volume-weighted price movement
- **Gap Analysis**: Overnight sentiment indicator
- **Intraday Strength**: Where price closed in day's range
- **Efficiency Ratio**: Trend strength vs noise
- **Mean Reversion Signal**: Z-score weighted by trend

### 2. Ensemble Confidence
Confidence score based on:
- Model agreement (XGBoost vs LightGBM)
- Direction probability strength
- Return prediction uncertainty spread

### 3. Risk-Based Position Sizing
Position size adjusted by:
- Prediction confidence
- Current volatility
- Risk/reward ratio
- Portfolio risk constraints

## Performance Expectations

Realistic targets:
- Direction accuracy: 55-60%
- Return MAE: 1.0-1.5%
- Sharpe ratio: 1.0-1.5
- Win rate: 50-55%
- Max drawdown: < 20%

## Limitations

1. **No guaranteed profits**: Market prediction is inherently uncertain
2. **Transaction costs**: Real costs may vary from simulation
3. **Slippage**: Large orders may experience more slippage
4. **Market hours**: System doesn't account for after-hours movements
5. **News lag**: Sentiment data may be delayed

## Safety Features

- Paper trading by default
- Live trading requires explicit confirmation
- Maximum position size limits
- Sector allocation constraints
- Stop loss on all trades
