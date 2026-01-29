"""
Reinforcement Learning Trading System for AI Stock Prediction
============================================================

This module provides:
- Trading environments (Gym-compatible)
- RL agents (DQN, PPO, SAC)
- Backtesting engine
- Angel One broker integration
- Risk management

Author: AI_IN_STOCK_V2 Project
"""

from .config.rl_config import RLConfig
from .config.trading_config import TradingConfig

__version__ = '1.0.0'
__all__ = ['RLConfig', 'TradingConfig']
