"""
================================================================================
UNIFIED ENGINE - AI Stock Trading System
================================================================================
Merged from production/ and pipeline/ into a single unified engine.

Modules:
- data_collector: Data download and validation (yfinance)
- feature_engine: 130+ features (technical + volume + momentum + volatility + regime)
- sentiment: Google News RSS sentiment (VADER + TextBlob)
- factor_analyzer: 5-factor model (Value, Momentum, Quality, Low-Vol, Sentiment)
- ml_models: XGBoost, LightGBM, LSTM, GRU, Transformer, Ensemble
- portfolio_optimizer: Risk parity, Max Sharpe, Min Vol, Equal Weight, HRP
- backtester: Per-stock trades + portfolio-level rebalancing
- signal_generator: BUY/SELL/HOLD with confidence scoring
- broker: Paper trading + Angel One live trading
- orchestrator: 8-step unified pipeline with progress callbacks
================================================================================
"""

__version__ = "4.0.0"
__author__ = "AI Stock Trading Research"

from engine.data_collector import DataCollector
from engine.feature_engine import AdvancedFeatureEngine, FeatureEngineer
from engine.sentiment import FastSentimentEngine
from engine.factor_analyzer import FactorAnalyzer
from engine.ml_models import (
    MLModelTrainer, XGBoostModel, LightGBMModel,
    LSTMModel, GRUModel, EnsembleModel,
    ProductionModel, ModelEvaluator
)
from engine.portfolio_optimizer import PortfolioOptimizer
from engine.backtester import WalkForwardBacktester, BacktestValidator
from engine.signal_generator import SignalGenerator
from engine.broker import create_broker, BaseBroker, PaperBroker
from engine.orchestrator import UnifiedOrchestrator

__all__ = [
    'DataCollector',
    'AdvancedFeatureEngine',
    'FeatureEngineer',
    'FastSentimentEngine',
    'FactorAnalyzer',
    'MLModelTrainer',
    'XGBoostModel',
    'LightGBMModel',
    'LSTMModel',
    'GRUModel',
    'EnsembleModel',
    'ProductionModel',
    'ModelEvaluator',
    'PortfolioOptimizer',
    'WalkForwardBacktester',
    'BacktestValidator',
    'SignalGenerator',
    'create_broker',
    'BaseBroker',
    'PaperBroker',
    'UnifiedOrchestrator',
]
