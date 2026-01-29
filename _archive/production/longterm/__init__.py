"""
Long-Term Equity Portfolio Module

Components:
- LongTermOrchestrator: Main pipeline coordinator
- FactorEngine: Compute Value, Momentum, Quality, Low-Vol factors
- PortfolioOptimizer: Mean-Variance, Risk Parity optimization
- ExperimentTracker: Track all research experiments
"""

from .factor_engine import FactorEngine, FactorScores
from .portfolio_optimizer import PortfolioOptimizer, PortfolioAllocation, LongTermBacktester
from .experiment_tracker import ExperimentTracker, ExperimentConfig, ExperimentResults, Experiment
from .longterm_orchestrator import LongTermOrchestrator

__all__ = [
    'LongTermOrchestrator',
    'FactorEngine',
    'FactorScores',
    'PortfolioOptimizer',
    'PortfolioAllocation',
    'LongTermBacktester',
    'ExperimentTracker',
    'ExperimentConfig',
    'ExperimentResults',
    'Experiment',
]
