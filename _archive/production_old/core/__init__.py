"""
Production Core Components
==========================
- data_loader: Data collection from yfinance
- feature_engine: Feature engineering
- model_trainer: Model training and prediction
- signal_generator: Trading signal generation
- backtester: Walk-forward backtesting
"""

from .data_loader import DataLoader
from .feature_engine import AdvancedFeatureEngine
from .model_trainer import ProductionModel, ModelEvaluator
from .signal_generator import SignalGenerator, TradingSignal, PortfolioSignals
from .backtester import WalkForwardBacktester

# Aliases for convenience
FeatureEngine = AdvancedFeatureEngine

__all__ = [
    'DataLoader',
    'AdvancedFeatureEngine', 
    'FeatureEngine',  # Alias
    'ProductionModel',
    'ModelEvaluator',
    'SignalGenerator',
    'TradingSignal',
    'PortfolioSignals',
    'WalkForwardBacktester'
]
