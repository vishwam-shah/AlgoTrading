"""
Long-Term Equity Portfolio System
=================================
Factor-based portfolio management for NSE stocks.

Structure:
---------
production/
├── longterm/       # Long-term strategy components
│   ├── factor_engine.py       - Value, Momentum, Quality, Low-Vol factors
│   ├── portfolio_optimizer.py - Mean-Variance, Risk Parity optimization
│   ├── experiment_tracker.py  - Research experiment tracking
│   └── longterm_orchestrator.py - Main pipeline coordinator
├── utils/          # Shared utilities
│   ├── logger.py              - Logging setup
│   ├── metrics.py             - Performance metrics
│   ├── validators.py          - Data validation
│   └── fast_sentiment.py      - News sentiment (RSS-based)
├── broker.py       # Paper/Live trading integration
└── __init__.py

Usage:
------
    # CLI usage
    python production/longterm/longterm_orchestrator.py --mode backtest
    python production/longterm/longterm_orchestrator.py --compare

    # Python usage
    from production.longterm import LongTermOrchestrator

    orchestrator = LongTermOrchestrator(
        n_holdings=20,
        optimization_method='risk_parity'
    )
    signals = orchestrator.run_pipeline(mode='backtest')
"""

# Long-term strategy components
from .longterm import (
    LongTermOrchestrator,
    FactorEngine,
    FactorScores,
    PortfolioOptimizer,
    PortfolioAllocation,
    LongTermBacktester,
    ExperimentTracker,
    ExperimentConfig,
    ExperimentResults,
    Experiment,
)

# Broker
from .broker import PaperBroker

# Utilities
from .utils import setup_logger

__all__ = [
    # Main orchestrator
    'LongTermOrchestrator',
    # Factor components
    'FactorEngine',
    'FactorScores',
    # Portfolio optimization
    'PortfolioOptimizer',
    'PortfolioAllocation',
    'LongTermBacktester',
    # Experiment tracking
    'ExperimentTracker',
    'ExperimentConfig',
    'ExperimentResults',
    'Experiment',
    # Broker
    'PaperBroker',
    # Utilities
    'setup_logger',
]
