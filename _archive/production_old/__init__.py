"""
Production Trading System
=========================
Unified, clean, production-ready trading system for NSE stocks.

Structure:
---------
production/
├── core/           # Core components
│   ├── data_loader.py      - Data collection utilities
│   ├── feature_engine.py   - 100+ technical features
│   ├── model_trainer.py    - XGBoost ensemble models
│   ├── signal_generator.py - Confidence-based signals
│   └── backtester.py       - Walk-forward testing
├── utils/          # Shared utilities
│   ├── logger.py           - Logging setup
│   ├── metrics.py          - Performance metrics
│   └── validators.py       - Data/signal validation
├── runners/        # Entry points
│   ├── cli.py              - Unified CLI
│   ├── paper_trading.py    - Paper trading mode
│   └── backtest.py         - Backtest runner
├── analysis/       # Analytics
│   └── optimizer.py        - Parameter optimization
├── broker.py       # Angel One integration
├── orchestrator.py # Main coordinator
└── __init__.py

Usage:
------
    # CLI usage
    python -m production.runners.cli paper --symbols HDFCBANK TCS
    python -m production.runners.cli backtest --symbols HDFCBANK TCS
    python -m production.runners.cli optimize --symbols HDFCBANK
    
    # Python usage
    from production import TradingOrchestrator
    orchestrator = TradingOrchestrator()
    orchestrator.run_pipeline()
"""

# Core components (backward compatible)
from .feature_engine import AdvancedFeatureEngine
from .models import ProductionModel
from .signals import SignalGenerator
from .backtester import WalkForwardBacktester
from .orchestrator import TradingOrchestrator
from .broker import PaperBroker

# New modular imports
from .core import DataLoader, FeatureEngine
from .utils import (
    setup_logger, 
    PerformanceMetrics, 
    DataValidator, 
    SignalValidator
)

__all__ = [
    # Main orchestrator
    'TradingOrchestrator',
    # Core components
    'AdvancedFeatureEngine',
    'ProductionModel',
    'SignalGenerator',
    'WalkForwardBacktester',
    'PaperBroker',
    # New modular components
    'DataLoader',
    'FeatureEngine',
    # Utilities
    'setup_logger',
    'PerformanceMetrics',
    'DataValidator',
    'SignalValidator',
]
