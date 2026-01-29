# Long-Term Equity Portfolio Framework

## Philosophy: Evidence-Based Investing

This framework is built on 50+ years of academic research, not hype.

### What Actually Works (Peer-Reviewed Research)

| Factor | Seminal Paper | Annual Premium | Why It Works |
|--------|---------------|----------------|--------------|
| Value | Fama-French (1992) | 3-5% | Behavioral: investors overpay for "glamour" |
| Momentum | Jegadeesh-Titman (1993) | 4-8% | Behavioral: underreaction to news |
| Quality | Novy-Marx (2013) | 3-4% | Investors undervalue stable earners |
| Low Volatility | Ang et al. (2006) | 2-3% | Leverage constraints, lottery preference |

### What Doesn't Work (Despite Claims)

| Approach | Why It Fails |
|----------|--------------|
| Pure price prediction | Markets are ~85% efficient at daily level |
| Complex deep learning on prices | Overfits, no edge after costs |
| High-frequency rebalancing | Transaction costs destroy returns |
| Concentrated portfolios | Single stock risk, no diversification benefit |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    LONG-TERM EQUITY SYSTEM                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   DATA      │───▶│   FACTOR    │───▶│   STOCK     │          │
│  │   LAYER     │    │   ENGINE    │    │   RANKING   │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│        │                                      │                  │
│        │                                      ▼                  │
│        │            ┌─────────────┐    ┌─────────────┐          │
│        │            │   REGIME    │───▶│  PORTFOLIO  │          │
│        └───────────▶│  DETECTOR   │    │  OPTIMIZER  │          │
│                     └─────────────┘    └─────────────┘          │
│                                              │                   │
│                                              ▼                   │
│                     ┌─────────────┐    ┌─────────────┐          │
│                     │    RISK     │◀───│  REBALANCE  │          │
│                     │  MANAGER    │    │   ENGINE    │          │
│                     └─────────────┘    └─────────────┘          │
│                                              │                   │
│                                              ▼                   │
│                                       ┌─────────────┐           │
│                                       │   SIGNALS   │           │
│                                       │  BUY/SELL   │           │
│                                       └─────────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Factor Definitions (NSE-Specific)

### 1. Value Factor

```python
VALUE_METRICS = {
    # Price ratios (lower is better)
    "earnings_yield": "1 / PE_ratio",  # Inverse of P/E
    "book_yield": "1 / PB_ratio",      # Inverse of P/B
    "dividend_yield": "Annual dividend / Price",
    "fcf_yield": "Free cash flow / Market cap",

    # Composite score
    "value_score": "rank(earnings_yield) + rank(book_yield) + rank(dividend_yield)",
}

# NSE data sources: Screener.in, Moneycontrol, BSE/NSE filings
```

### 2. Momentum Factor

```python
MOMENTUM_METRICS = {
    # Price momentum
    "momentum_12_1": "(Price_now / Price_12m_ago) - (Price_now / Price_1m_ago)",
    # Skip last month to avoid short-term reversal

    # Alternatives
    "momentum_6m": "6-month return",
    "momentum_3m": "3-month return",

    # Risk-adjusted momentum
    "sharpe_12m": "12-month return / 12-month volatility",

    # Composite
    "momentum_score": "rank(momentum_12_1) + rank(sharpe_12m)",
}
```

### 3. Quality Factor

```python
QUALITY_METRICS = {
    # Profitability
    "roe": "Net income / Shareholder equity",
    "roa": "Net income / Total assets",
    "gross_margin": "Gross profit / Revenue",

    # Stability
    "earnings_stability": "1 / std(earnings growth, 5 years)",
    "revenue_growth": "CAGR(revenue, 3 years)",

    # Financial health
    "debt_to_equity": "Total debt / Equity (lower is better)",
    "interest_coverage": "EBIT / Interest expense",
    "current_ratio": "Current assets / Current liabilities",

    # Composite
    "quality_score": "rank(roe) + rank(gross_margin) - rank(debt_to_equity)",
}
```

### 4. Low Volatility Factor

```python
LOW_VOL_METRICS = {
    # Historical volatility
    "volatility_1y": "std(daily returns) * sqrt(252)",
    "beta": "Covariance(stock, nifty) / Variance(nifty)",

    # Downside risk
    "max_drawdown_1y": "Maximum peak-to-trough decline",
    "downside_deviation": "std(negative returns only)",

    # Composite (LOWER is better, so we invert)
    "low_vol_score": "-rank(volatility_1y) - rank(beta)",
}
```

---

## Portfolio Construction

### Method 1: Equal-Weight Factor Portfolio (Simple)

```python
def simple_factor_portfolio(stocks, n_holdings=20):
    """
    Simple approach: Rank by combined factor score, equal weight top N.
    """
    # Compute combined score
    stocks['combined_score'] = (
        0.25 * stocks['value_score'] +
        0.25 * stocks['momentum_score'] +
        0.25 * stocks['quality_score'] +
        0.25 * stocks['low_vol_score']
    )

    # Select top N
    selected = stocks.nlargest(n_holdings, 'combined_score')

    # Equal weight
    weights = {stock: 1/n_holdings for stock in selected.index}

    return weights
```

### Method 2: Mean-Variance Optimization (Better)

```python
def mean_variance_portfolio(expected_returns, cov_matrix,
                            risk_aversion=1.0, constraints=None):
    """
    Markowitz optimization with constraints.

    Maximize: expected_return - (risk_aversion/2) * variance
    Subject to: weights sum to 1, position limits, sector limits
    """
    from scipy.optimize import minimize

    n = len(expected_returns)

    def objective(weights):
        port_return = weights @ expected_returns
        port_variance = weights @ cov_matrix @ weights
        return -(port_return - risk_aversion/2 * port_variance)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
    ]

    bounds = [(0, 0.10) for _ in range(n)]  # Max 10% per stock

    result = minimize(objective, x0=np.ones(n)/n,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x
```

### Method 3: Risk Parity (Most Robust)

```python
def risk_parity_portfolio(cov_matrix):
    """
    Each position contributes equal risk to portfolio.
    More robust than mean-variance (doesn't need return estimates).
    """
    from scipy.optimize import minimize

    n = cov_matrix.shape[0]

    def risk_contribution(weights):
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_risk = cov_matrix @ weights / port_vol
        risk_contrib = weights * marginal_risk
        return risk_contrib

    def objective(weights):
        rc = risk_contribution(weights)
        target_rc = np.ones(n) / n  # Equal risk contribution
        return np.sum((rc - target_rc * np.sum(rc))**2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 0.20) for _ in range(n)]

    result = minimize(objective, x0=np.ones(n)/n,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x
```

---

## Rebalancing Strategy

### When to Rebalance

```python
REBALANCE_TRIGGERS = {
    # Time-based
    "scheduled": "Monthly (last trading day)",

    # Drift-based
    "position_drift": "Any position deviates >5% from target",
    "sector_drift": "Any sector deviates >10% from target",

    # Event-based
    "earnings": "After major earnings surprises",
    "regime_change": "When market regime shifts",

    # Threshold
    "min_trade_size": "Rs 10,000 (to avoid small trades)",
}
```

### Rebalancing Algorithm

```python
def calculate_rebalance_trades(current_weights, target_weights,
                                portfolio_value, min_trade_pct=0.01):
    """
    Calculate trades needed to move from current to target.
    """
    trades = {}

    for stock in set(current_weights.keys()) | set(target_weights.keys()):
        current = current_weights.get(stock, 0)
        target = target_weights.get(stock, 0)

        diff = target - current

        # Only trade if difference exceeds threshold
        if abs(diff) >= min_trade_pct:
            trade_value = diff * portfolio_value
            trades[stock] = {
                'action': 'BUY' if diff > 0 else 'SELL',
                'weight_change': diff,
                'value': abs(trade_value),
            }

    return trades
```

---

## Regime Detection (Where Deep Learning Helps)

### Market Regimes

```python
MARKET_REGIMES = {
    "BULL": {
        "characteristics": "Rising prices, low VIX, positive breadth",
        "strategy": "Full equity allocation, tilt to momentum",
    },
    "BEAR": {
        "characteristics": "Falling prices, high VIX, negative breadth",
        "strategy": "Reduce equity, tilt to quality/low-vol",
    },
    "SIDEWAYS": {
        "characteristics": "Range-bound, moderate VIX",
        "strategy": "Balanced allocation, tilt to value",
    },
    "CRISIS": {
        "characteristics": "VIX spike, correlation breakdown",
        "strategy": "Defensive, raise cash, quality only",
    },
}
```

### Regime Detection Model

```python
REGIME_MODEL = {
    "type": "Hidden Markov Model or LSTM",

    "inputs": [
        "nifty_50_return_20d",
        "india_vix",
        "advance_decline_ratio",
        "new_highs_vs_lows",
        "yield_curve_slope",
        "fii_dii_flows",
    ],

    "output": "Probability of each regime",

    "use_in_portfolio": {
        "BULL": {"equity": 1.0, "momentum_tilt": 0.3},
        "BEAR": {"equity": 0.5, "quality_tilt": 0.3},
        "SIDEWAYS": {"equity": 0.8, "value_tilt": 0.2},
        "CRISIS": {"equity": 0.3, "cash": 0.5},
    }
}
```

---

## Sentiment Analysis (Deep Learning Sweet Spot)

### Data Sources

```python
SENTIMENT_SOURCES = {
    # News
    "google_news_rss": "Free, real-time, your current implementation",
    "moneycontrol_news": "Scrape headlines",
    "economic_times": "Business news",

    # Social
    "twitter_cashtags": "$RELIANCE, $TCS mentions",
    "reddit_india": "r/IndiaInvestments sentiment",

    # Filings (highest alpha)
    "quarterly_results": "BSE/NSE filings",
    "earnings_calls": "Transcripts from company websites",
    "annual_reports": "Management discussion section",
}
```

### Sentiment Model Architecture

```python
SENTIMENT_MODEL = {
    "base_model": "FinBERT or fine-tuned BERT",

    "pipeline": [
        "1. Collect text (news, filings, calls)",
        "2. Clean and preprocess",
        "3. Run through FinBERT",
        "4. Aggregate to stock-level sentiment",
        "5. Create features: sentiment_score, sentiment_momentum, sentiment_dispersion",
    ],

    "output_features": {
        "sentiment_score": "Average sentiment (-1 to 1)",
        "sentiment_momentum": "Change in sentiment over 30 days",
        "sentiment_volume": "Number of mentions (attention)",
        "sentiment_dispersion": "Disagreement (high = uncertainty)",
    },

    "use_in_model": "Add as additional factor in ranking model",
}
```

---

## Risk Management

### Position Limits

```python
RISK_LIMITS = {
    # Position level
    "max_position_weight": 0.10,       # 10% max per stock
    "min_position_weight": 0.02,       # 2% min (avoid tiny positions)

    # Sector level
    "max_sector_weight": 0.30,         # 30% max per sector

    # Concentration
    "max_top5_weight": 0.40,           # Top 5 positions max 40%

    # Correlation
    "max_pairwise_correlation": 0.80,  # Avoid highly correlated pairs
}
```

### Drawdown Management

```python
DRAWDOWN_RULES = {
    # Warning levels
    "level_1": {
        "threshold": -0.10,  # -10%
        "action": "Review positions, no new buys",
    },
    "level_2": {
        "threshold": -0.15,  # -15%
        "action": "Reduce position sizes by 20%",
    },
    "level_3": {
        "threshold": -0.20,  # -20%
        "action": "Move to 50% cash, quality stocks only",
    },
    "level_4": {
        "threshold": -0.30,  # -30%
        "action": "Full defensive mode, 70% cash",
    },
}
```

---

## Experiment Tracking

### Experiment Template

```python
EXPERIMENT = {
    "id": "LT_EXP_001",
    "name": "Momentum + Quality Factor Test",
    "date": "2026-01-25",
    "hypothesis": "Combining momentum and quality outperforms single factors",

    # Configuration
    "config": {
        "universe": "NIFTY100",
        "factors": ["momentum_12_1", "quality_score"],
        "weights": {"momentum": 0.5, "quality": 0.5},
        "n_holdings": 20,
        "rebalance": "monthly",
        "backtest_period": "2019-01-01 to 2025-12-31",
    },

    # Results
    "results": {
        "total_return": 0.156,           # 15.6% annual
        "benchmark_return": 0.122,       # NIFTY50: 12.2%
        "excess_return": 0.034,          # +3.4% alpha

        "volatility": 0.18,
        "sharpe_ratio": 0.87,
        "sortino_ratio": 1.12,
        "max_drawdown": -0.22,

        "win_rate_monthly": 0.58,        # 58% profitable months
        "best_month": 0.12,
        "worst_month": -0.09,

        "turnover_annual": 1.2,          # 120% annual turnover
        "transaction_costs": 0.012,      # 1.2% annual
    },

    # Statistical validation
    "validation": {
        "t_stat_vs_benchmark": 2.34,
        "p_value": 0.019,
        "is_significant": True,
        "information_ratio": 0.68,
    },

    "conclusion": "Momentum+Quality shows significant alpha. Proceed to paper trading.",
}
```

---

## Performance Expectations (Realistic)

### What to Expect

| Metric | Conservative | Moderate | Aggressive |
|--------|--------------|----------|------------|
| Annual Return | 12-15% | 15-20% | 18-25% |
| Volatility | 12-15% | 15-20% | 18-25% |
| Sharpe Ratio | 0.7-0.9 | 0.9-1.1 | 1.0-1.3 |
| Max Drawdown | 15-20% | 20-30% | 25-40% |
| Turnover | 50-80% | 100-150% | 150-250% |

### Comparison to Alternatives

| Strategy | Expected Return | Risk | Effort |
|----------|----------------|------|--------|
| NIFTY50 Index Fund | 12% | Medium | None |
| Your Factor Portfolio | 16% | Medium | Medium |
| Active Stock Picking | 10-20% | High | Very High |
| Day Trading | -10% to +5% | Very High | Extreme |

---

## Implementation Roadmap

### Phase 1: Data & Factors (Week 1-2)
- [ ] Collect fundamental data (P/E, P/B, ROE, etc.)
- [ ] Implement factor calculations
- [ ] Backtest individual factors
- [ ] Validate factor premiums exist in NSE data

### Phase 2: Portfolio Construction (Week 3-4)
- [ ] Implement equal-weight portfolio
- [ ] Implement mean-variance optimization
- [ ] Implement risk parity
- [ ] Compare methods in backtest

### Phase 3: Rebalancing & Risk (Week 5-6)
- [ ] Implement monthly rebalancing
- [ ] Add drawdown management rules
- [ ] Test different rebalancing frequencies
- [ ] Optimize for transaction costs

### Phase 4: ML Enhancement (Week 7-8)
- [ ] Add sentiment as factor
- [ ] Implement regime detection
- [ ] Test ML factor combination
- [ ] Validate out-of-sample

### Phase 5: Paper Trading (Week 9-12)
- [ ] Run live paper trading for 1 month
- [ ] Track all signals and execution
- [ ] Compare paper vs backtest results
- [ ] Iterate based on findings
