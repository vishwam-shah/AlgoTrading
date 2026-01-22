"""
================================================================================
PERFORMANCE METRICS
================================================================================
Calculate and track trading performance metrics.
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    
    # Risk
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # Trading
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Portfolio
    initial_value: float = 0.0
    final_value: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'benchmark_return': self.benchmark_return,
            'excess_return': self.excess_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'initial_value': self.initial_value,
            'final_value': self.final_value
        }


def calculate_sharpe_ratio(
    returns: np.ndarray, 
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily)
        
    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio (uses downside deviation only).
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(downside_returns)


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int]:
    """
    Calculate maximum drawdown and its duration.
    
    Returns:
        (max_drawdown_pct, max_duration_periods)
    """
    if len(equity_curve) < 2:
        return 0.0, 0
    
    # Running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Drawdown at each point
    drawdowns = (equity_curve - running_max) / running_max
    
    # Max drawdown
    max_dd = np.min(drawdowns)
    
    # Duration
    in_drawdown = drawdowns < 0
    
    if not in_drawdown.any():
        return 0.0, 0
    
    # Find longest drawdown period
    drawdown_periods = []
    current_period = 0
    
    for dd in in_drawdown:
        if dd:
            current_period += 1
        else:
            if current_period > 0:
                drawdown_periods.append(current_period)
            current_period = 0
    
    if current_period > 0:
        drawdown_periods.append(current_period)
    
    max_duration = max(drawdown_periods) if drawdown_periods else 0
    
    return max_dd, max_duration


def calculate_win_rate(pnls: List[float]) -> float:
    """Calculate win rate from P&L list."""
    if not pnls:
        return 0.0
    
    winners = sum(1 for p in pnls if p > 0)
    return winners / len(pnls)


def calculate_profit_factor(pnls: List[float]) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    if not pnls:
        return 0.0
    
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_cagr(
    initial_value: float,
    final_value: float,
    years: float
) -> float:
    """Calculate Compound Annual Growth Rate."""
    if initial_value <= 0 or years <= 0:
        return 0.0
    
    return (final_value / initial_value) ** (1 / years) - 1


def calculate_calmar_ratio(
    annualized_return: float,
    max_drawdown: float
) -> float:
    """Calculate Calmar Ratio (return / max drawdown)."""
    if max_drawdown == 0:
        return 0.0
    
    return annualized_return / abs(max_drawdown)


class PerformanceTracker:
    """
    Track and calculate performance metrics over time.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.equity_history = [initial_capital]
        self.return_history = []
        self.trade_pnls = []
        self.timestamps = [datetime.now()]
    
    def update(self, portfolio_value: float, trade_pnl: float = None):
        """Update with new portfolio value."""
        self.equity_history.append(portfolio_value)
        
        # Calculate return
        prev_value = self.equity_history[-2]
        if prev_value > 0:
            daily_return = (portfolio_value - prev_value) / prev_value
            self.return_history.append(daily_return)
        
        # Record trade P&L
        if trade_pnl is not None:
            self.trade_pnls.append(trade_pnl)
        
        self.timestamps.append(datetime.now())
    
    def get_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        equity = np.array(self.equity_history)
        returns = np.array(self.return_history) if self.return_history else np.array([0])
        
        # Calculate drawdown
        max_dd, max_dd_duration = calculate_max_drawdown(equity)
        
        # Trading metrics
        winning_trades = [p for p in self.trade_pnls if p > 0]
        losing_trades = [p for p in self.trade_pnls if p < 0]
        
        return PerformanceMetrics(
            total_return=(equity[-1] - equity[0]) / equity[0] if equity[0] > 0 else 0,
            annualized_return=calculate_cagr(equity[0], equity[-1], len(equity) / 252) if len(equity) > 1 else 0,
            volatility=np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0,
            sharpe_ratio=calculate_sharpe_ratio(returns),
            sortino_ratio=calculate_sortino_ratio(returns),
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            total_trades=len(self.trade_pnls),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=calculate_win_rate(self.trade_pnls),
            profit_factor=calculate_profit_factor(self.trade_pnls),
            avg_win=np.mean(winning_trades) if winning_trades else 0,
            avg_loss=np.mean(losing_trades) if losing_trades else 0,
            initial_value=self.initial_capital,
            final_value=equity[-1]
        )
