"""
Backtesting Engine — Core portfolio simulation.

Simulates daily trading:
1. Load predictions for each day
2. Compute target allocations (HRP + Kelly)
3. Execute trades with transaction costs
4. Track portfolio value, positions, P&L
5. Calculate metrics (Sharpe, Sortino, MaxDD, etc.)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TradeRecord:
    """Single trade execution record."""
    date: str
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    price: float
    cost: float  # Transaction costs (₹)
    position_value: float  # Shares × price


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time."""
    date: str
    cash: float
    holdings: Dict[str, int]  # {symbol: shares}
    position_values: Dict[str, float]  # {symbol: shares × price}
    portfolio_value: float  # cash + sum(positions)
    daily_return: float  # (value_today - value_yesterday) / value_yesterday
    drawdown: float  # (peak - current) / peak


class BacktestEngine:
    """Simulate portfolio trading over a backtest period."""

    def __init__(
        self,
        initial_cash: float = 100_000,
        max_leverage: float = 1.0,
        min_trade_value: float = 10_000,
    ):
        """
        Initialize backtest engine.

        Args:
            initial_cash: Starting portfolio value (default ₹100k)
            max_leverage: Maximum portfolio leverage (default 1.0 = no leverage)
            min_trade_value: Minimum trade value to execute (default ₹10k)
        """
        self.initial_cash = initial_cash
        self.max_leverage = max_leverage
        self.min_trade_value = min_trade_value

        # State
        self.cash = initial_cash
        self.holdings: Dict[str, int] = {}  # {symbol: shares}
        self.trades: List[TradeRecord] = []
        self.portfolio_values: List[float] = [initial_cash]
        self.snapshots: List[PortfolioSnapshot] = []

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Current portfolio value = cash + position values."""
        position_value = sum(
            self.holdings.get(s, 0) * prices.get(s, 0) for s in self.holdings
        )
        return self.cash + position_value

    def step(
        self,
        date: str,
        prices: Dict[str, float],
        target_weights: Dict[str, float],
        transaction_cost_calculator=None,
        rebalance: bool = True,
    ) -> PortfolioSnapshot:
        """
        Execute one day of trading.

        Args:
            date: Trading date (YYYY-MM-DD format)
            prices: {symbol: current_price}
            target_weights: {symbol: desired_allocation_weight}
            transaction_cost_calculator: Transaction cost calculator
            rebalance: If False, just mark-to-market without any trades

        Returns:
            PortfolioSnapshot with updated portfolio state
        """
        # Mark-to-market current portfolio value
        current_portfolio_value = self.get_portfolio_value(prices)
        previous_portfolio_value = self.portfolio_values[-1]
        daily_return = (
            (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
            if previous_portfolio_value > 0
            else 0
        )

        # Calculate drawdown
        max_portfolio_value = max(self.portfolio_values)
        drawdown = (
            (max_portfolio_value - current_portfolio_value) / max_portfolio_value
            if max_portfolio_value > 0
            else 0
        )

        # Skip all trading if mark-to-market only mode
        if not rebalance or not target_weights:
            self.portfolio_values.append(current_portfolio_value)
            snapshot = PortfolioSnapshot(
                date=date,
                cash=self.cash,
                holdings=self.holdings.copy(),
                position_values={s: self.holdings.get(s, 0) * prices.get(s, 0)
                               for s in self.holdings},
                portfolio_value=current_portfolio_value,
                daily_return=daily_return,
                drawdown=drawdown,
            )
            self.snapshots.append(snapshot)
            return snapshot

        # Rebalance to target weights
        for symbol, target_weight in target_weights.items():
            target_value = target_weight * current_portfolio_value
            current_position = self.holdings.get(symbol, 0)
            current_value = current_position * prices.get(symbol, 0)

            # Calculate rebalancing trade
            value_diff = target_value - current_value

            if abs(value_diff) < self.min_trade_value:
                continue  # Skip small trades

            # Determine action and quantity
            price = prices.get(symbol, 0)
            if price <= 0:
                continue

            if value_diff > 0:  # BUY
                shares_to_buy = int(value_diff / price)
                if shares_to_buy > 0:
                    cost = self._execute_trade(
                        date, symbol, shares_to_buy, price, "BUY",
                        transaction_cost_calculator
                    )
                    self.cash -= cost

            else:  # SELL
                shares_to_sell = min(int(abs(value_diff) / price), current_position)
                if shares_to_sell > 0:
                    proceeds = self._execute_trade(
                        date, symbol, shares_to_sell, price, "SELL",
                        transaction_cost_calculator
                    )
                    self.cash += proceeds

        # Close positions for stocks no longer in target weights
        for symbol in list(self.holdings.keys()):
            if symbol not in target_weights or target_weights[symbol] == 0:
                shares = self.holdings[symbol]
                price = prices.get(symbol, 0)
                if shares > 0 and price > 0:
                    proceeds = self._execute_trade(
                        date, symbol, shares, price, "SELL",
                        transaction_cost_calculator
                    )
                    self.cash += proceeds

        # Record portfolio state
        current_value = self.get_portfolio_value(prices)
        self.portfolio_values.append(current_value)

        snapshot = PortfolioSnapshot(
            date=date,
            cash=self.cash,
            holdings=self.holdings.copy(),
            position_values={s: self.holdings.get(s, 0) * prices.get(s, 0)
                           for s in self.holdings},
            portfolio_value=current_value,
            daily_return=daily_return,
            drawdown=drawdown,
        )
        self.snapshots.append(snapshot)

        return snapshot

    def _execute_trade(
        self,
        date: str,
        symbol: str,
        quantity: int,
        price: float,
        action: str,
        transaction_cost_calculator=None,
    ) -> float:
        """
        Execute a single trade (BUY or SELL).

        Returns:
            Cost (BUY) or proceeds net of costs (SELL)
        """
        gross_value = quantity * price

        # Calculate transaction costs
        if transaction_cost_calculator:
            cost_breakdown = transaction_cost_calculator.calculate_one_way_cost(
                symbol, price, quantity, is_buy=(action == "BUY")
            )
            transaction_cost = cost_breakdown["total_cost_rupees"]
        else:
            # Default: 0.08% one-way (0.16% round-trip)
            transaction_cost = gross_value * 0.0008

        # Update holdings
        if action == "BUY":
            self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
            result = gross_value + transaction_cost
        else:  # SELL
            self.holdings[symbol] = max(0, self.holdings.get(symbol, 0) - quantity)
            result = gross_value - transaction_cost

        # Record trade
        self.trades.append(
            TradeRecord(
                date=date,
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                cost=transaction_cost,
                position_value=gross_value,
            )
        )

        return result

    def calculate_metrics(self, risk_free_rate: float = 0.06) -> Dict[str, float]:
        """
        Calculate backtest performance metrics.

        Args:
            risk_free_rate: Annual risk-free rate (default 6%)

        Returns:
            Dict with metrics: Sharpe, Sortino, CAGR, MaxDD, CalmarRatio, WinRate
        """
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        n_days = len(returns)
        n_years = n_days / 252.0

        # CAGR
        cagr = (portfolio_values[-1] / portfolio_values[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Volatility
        annual_vol = np.std(returns) * np.sqrt(252)

        # Sharpe Ratio
        daily_rfr = (1 + risk_free_rate) ** (1 / 252) - 1
        excess_returns = returns - daily_rfr
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

        # Sortino Ratio (downside volatility)
        downside_returns = np.minimum(excess_returns, 0)
        downside_vol = np.std(downside_returns) * np.sqrt(252)
        sortino = np.mean(excess_returns) / downside_vol * np.sqrt(252) if downside_vol > 0 else 0

        # Max Drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # Calmar Ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0

        # Win Rate (% of days with positive return)
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        # Profit Factor (from trades)
        trade_df = pd.DataFrame([(t.date, t.symbol, t.action, t.position_value) for t in self.trades],
                               columns=['date', 'symbol', 'action', 'value'])
        if len(trade_df) > 0:
            # Simplified: buy-sell pairs
            buys = trade_df[trade_df['action'] == 'BUY']['value'].sum()
            sells = trade_df[trade_df['action'] == 'SELL']['value'].sum()
            profit_factor = sells / buys if buys > 0 else 0
        else:
            profit_factor = 0

        return {
            "cagr": cagr,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "total_return": (portfolio_values[-1] / portfolio_values[0] - 1) if portfolio_values[0] > 0 else 0,
        }
