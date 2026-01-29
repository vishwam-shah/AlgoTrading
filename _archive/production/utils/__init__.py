"""
Production Utilities
====================
Shared utilities for the production trading system.
"""

from .logger import setup_logger, TradingLogger
from .metrics import (
    PerformanceMetrics,
    PerformanceTracker,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_cagr,
    calculate_calmar_ratio,
)
from .validators import (
    ValidationResult,
    DataValidator,
    SignalValidator,
    TradeValidator,
    validate_market_hours,
    validate_position_limits,
)

__all__ = [
    # Logger
    'setup_logger',
    'TradingLogger',
    # Metrics
    'PerformanceMetrics',
    'PerformanceTracker',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_cagr',
    'calculate_calmar_ratio',
    # Validators
    'ValidationResult',
    'DataValidator',
    'SignalValidator',
    'TradeValidator',
    'validate_market_hours',
    'validate_position_limits',
]
