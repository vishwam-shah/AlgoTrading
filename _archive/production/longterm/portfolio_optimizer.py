"""
================================================================================
PORTFOLIO OPTIMIZER - Long-Term Equity
================================================================================
Portfolio construction and optimization methods:
- Equal Weight
- Mean-Variance Optimization (Markowitz)
- Risk Parity
- Black-Litterman (with views)

Plus rebalancing logic and transaction cost management.
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from scipy.optimize import minimize
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str
    date: datetime = field(default_factory=datetime.now)

    # Risk metrics
    max_position: float = 0
    concentration_top5: float = 0
    n_positions: int = 0

    def to_dict(self) -> Dict:
        return {
            'weights': self.weights,
            'expected_return': self.expected_return,
            'expected_volatility': self.expected_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'method': self.method,
            'date': str(self.date),
            'max_position': self.max_position,
            'concentration_top5': self.concentration_top5,
            'n_positions': self.n_positions,
        }


@dataclass
class RebalanceOrder:
    """Single rebalance trade."""
    symbol: str
    action: str  # 'BUY' or 'SELL'
    current_weight: float
    target_weight: float
    weight_change: float
    estimated_value: float
    estimated_shares: int = 0
    reason: str = ''


@dataclass
class RebalanceResult:
    """Complete rebalance result."""
    orders: List[RebalanceOrder]
    total_buy_value: float
    total_sell_value: float
    turnover: float  # As fraction of portfolio
    estimated_cost: float
    should_rebalance: bool
    reason: str


class PortfolioOptimizer:
    """
    Portfolio optimization with multiple methods.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.06,  # 6% for India
        max_position: float = 0.10,     # 10% max per stock
        min_position: float = 0.02,     # 2% min per stock
        max_sector: float = 0.30,       # 30% max per sector
        min_stocks: int = 10,
        max_stocks: int = 30,
        transaction_cost: float = 0.001  # 0.1% per trade
    ):
        self.risk_free_rate = risk_free_rate
        self.max_position = max_position
        self.min_position = min_position
        self.max_sector = max_sector
        self.min_stocks = min_stocks
        self.max_stocks = max_stocks
        self.transaction_cost = transaction_cost

        logger.info("PortfolioOptimizer initialized")

    def optimize(
        self,
        symbols: List[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        method: str = 'risk_parity',
        sector_map: Optional[Dict[str, str]] = None
    ) -> PortfolioAllocation:
        """
        Optimize portfolio weights.

        Args:
            symbols: List of stock symbols
            expected_returns: Array of expected returns
            cov_matrix: Covariance matrix
            method: 'equal', 'mean_variance', 'risk_parity', 'max_sharpe', 'min_variance'
            sector_map: Optional {symbol: sector} for sector constraints

        Returns:
            PortfolioAllocation with optimized weights
        """
        n = len(symbols)

        if method == 'equal':
            weights = self._equal_weight(n)
        elif method == 'mean_variance':
            weights = self._mean_variance(expected_returns, cov_matrix)
        elif method == 'risk_parity':
            weights = self._risk_parity(cov_matrix)
        elif method == 'max_sharpe':
            weights = self._max_sharpe(expected_returns, cov_matrix)
        elif method == 'min_variance':
            weights = self._min_variance(cov_matrix)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply constraints
        weights = self._apply_constraints(weights, symbols, sector_map)

        # Calculate portfolio metrics
        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        # Create weights dict
        weights_dict = {symbols[i]: float(weights[i]) for i in range(n) if weights[i] > 0.001}

        # Risk metrics
        sorted_weights = sorted(weights, reverse=True)
        max_pos = sorted_weights[0]
        top5_conc = sum(sorted_weights[:5])
        n_positions = sum(w > 0.001 for w in weights)

        return PortfolioAllocation(
            weights=weights_dict,
            expected_return=float(port_return),
            expected_volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            method=method,
            max_position=float(max_pos),
            concentration_top5=float(top5_conc),
            n_positions=n_positions
        )

    def _equal_weight(self, n: int) -> np.ndarray:
        """Equal weight allocation."""
        return np.ones(n) / n

    def _mean_variance(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 1.0
    ) -> np.ndarray:
        """
        Mean-Variance Optimization (Markowitz).

        Maximize: expected_return - (risk_aversion/2) * variance
        """
        n = len(expected_returns)

        def objective(weights):
            port_return = np.dot(weights, expected_returns)
            port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            return -(port_return - risk_aversion / 2 * port_var)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = [(0, self.max_position) for _ in range(n)]

        result = minimize(
            objective,
            x0=np.ones(n) / n,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        return result.x if result.success else np.ones(n) / n

    def _risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Risk Parity: Equal risk contribution from each asset.

        More robust than mean-variance as it doesn't need return estimates.
        """
        n = cov_matrix.shape[0]

        def risk_contribution(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if port_vol < 1e-10:
                return np.ones(n) / n
            marginal_risk = np.dot(cov_matrix, weights) / port_vol
            risk_contrib = weights * marginal_risk
            return risk_contrib

        def objective(weights):
            rc = risk_contribution(weights)
            target_rc = np.ones(n) / n
            # Minimize deviation from equal risk contribution
            return np.sum((rc - target_rc * np.sum(rc)) ** 2)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = [(0.01, self.max_position) for _ in range(n)]

        result = minimize(
            objective,
            x0=np.ones(n) / n,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        return result.x if result.success else np.ones(n) / n

    def _max_sharpe(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Maximum Sharpe Ratio portfolio.
        """
        n = len(expected_returns)

        def neg_sharpe(weights):
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(port_return - self.risk_free_rate) / (port_vol + 1e-10)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = [(0, self.max_position) for _ in range(n)]

        result = minimize(
            neg_sharpe,
            x0=np.ones(n) / n,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        return result.x if result.success else np.ones(n) / n

    def _min_variance(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Minimum Variance portfolio.
        """
        n = cov_matrix.shape[0]

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = [(0, self.max_position) for _ in range(n)]

        result = minimize(
            portfolio_variance,
            x0=np.ones(n) / n,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        return result.x if result.success else np.ones(n) / n

    def _apply_constraints(
        self,
        weights: np.ndarray,
        symbols: List[str],
        sector_map: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """Apply position and sector constraints."""
        weights = weights.copy()

        # Clip to max/min position
        weights = np.clip(weights, 0, self.max_position)

        # Remove tiny positions
        weights[weights < self.min_position / 2] = 0

        # Apply sector constraints if map provided
        if sector_map:
            sector_weights = {}
            for i, sym in enumerate(symbols):
                sector = sector_map.get(sym, 'Other')
                sector_weights[sector] = sector_weights.get(sector, 0) + weights[i]

            # Scale down over-allocated sectors
            for sector, total_weight in sector_weights.items():
                if total_weight > self.max_sector:
                    scale = self.max_sector / total_weight
                    for i, sym in enumerate(symbols):
                        if sector_map.get(sym, 'Other') == sector:
                            weights[i] *= scale

        # Renormalize
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)

        return weights

    def calculate_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        min_trade_pct: float = 0.01,
        max_turnover: float = 0.25
    ) -> RebalanceResult:
        """
        Calculate rebalance trades needed.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            prices: Current prices per symbol
            min_trade_pct: Minimum trade size as % of portfolio
            max_turnover: Maximum turnover allowed

        Returns:
            RebalanceResult with orders
        """
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        orders = []
        total_buy = 0
        total_sell = 0

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            diff = target - current

            # Skip small changes
            if abs(diff) < min_trade_pct:
                continue

            trade_value = abs(diff) * portfolio_value
            price = prices.get(symbol, 0)
            shares = int(trade_value / price) if price > 0 else 0

            order = RebalanceOrder(
                symbol=symbol,
                action='BUY' if diff > 0 else 'SELL',
                current_weight=current,
                target_weight=target,
                weight_change=diff,
                estimated_value=trade_value,
                estimated_shares=shares,
                reason='drift' if current > 0 else 'new_position'
            )
            orders.append(order)

            if diff > 0:
                total_buy += trade_value
            else:
                total_sell += trade_value

        # Calculate turnover
        turnover = (total_buy + total_sell) / (2 * portfolio_value)

        # Check if we should rebalance
        should_rebalance = len(orders) > 0 and turnover <= max_turnover
        reason = ""

        if not orders:
            reason = "No significant drift"
        elif turnover > max_turnover:
            reason = f"Turnover {turnover:.1%} exceeds max {max_turnover:.1%}"
            should_rebalance = False

        # Estimated transaction costs
        estimated_cost = (total_buy + total_sell) * self.transaction_cost

        return RebalanceResult(
            orders=orders,
            total_buy_value=total_buy,
            total_sell_value=total_sell,
            turnover=turnover,
            estimated_cost=estimated_cost,
            should_rebalance=should_rebalance,
            reason=reason
        )

    def check_drift(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        position_threshold: float = 0.05,
        total_threshold: float = 0.10
    ) -> Tuple[bool, str]:
        """
        Check if portfolio has drifted enough to warrant rebalancing.

        Returns:
            (should_rebalance, reason)
        """
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        max_drift = 0
        total_drift = 0
        drifted_positions = []

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            drift = abs(current - target)

            total_drift += drift
            if drift > max_drift:
                max_drift = drift

            if drift > position_threshold:
                drifted_positions.append(symbol)

        if max_drift > position_threshold:
            return True, f"Position drift: {', '.join(drifted_positions)} exceeded {position_threshold:.0%}"

        if total_drift > total_threshold:
            return True, f"Total drift {total_drift:.1%} exceeded {total_threshold:.0%}"

        return False, "No significant drift"


class LongTermBacktester:
    """
    Backtester for long-term portfolio strategies.
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        rebalance_frequency: str = 'monthly',  # 'monthly', 'quarterly'
        transaction_cost: float = 0.001
    ):
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost

    def run_backtest(
        self,
        price_data: Dict[str, pd.DataFrame],
        factor_scores: pd.DataFrame,
        optimizer: PortfolioOptimizer,
        start_date: str,
        end_date: str,
        n_holdings: int = 20,
        method: str = 'risk_parity'
    ) -> Dict:
        """
        Run backtest with monthly rebalancing.

        Returns:
            Dict with backtest results
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")

        # Combine all price data
        prices = {}
        for symbol, df in price_data.items():
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            prices[symbol] = df['close']

        price_df = pd.DataFrame(prices)
        price_df = price_df.loc[start_date:end_date]

        # Get rebalance dates
        if self.rebalance_frequency == 'monthly':
            rebalance_dates = price_df.resample('M').last().index
        else:
            rebalance_dates = price_df.resample('Q').last().index

        # Initialize
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions = {}  # {symbol: shares}
        weights = {}

        # Track results
        equity_curve = []
        trades_log = []

        for i, rebal_date in enumerate(rebalance_dates):
            # Get current prices
            try:
                current_prices = price_df.loc[rebal_date].to_dict()
            except:
                continue

            # Calculate current portfolio value
            position_value = sum(
                positions.get(sym, 0) * current_prices.get(sym, 0)
                for sym in positions
            )
            portfolio_value = cash + position_value

            # Get top stocks by factor score
            available_symbols = [s for s in factor_scores['symbol'].unique()
                                if s in current_prices and not pd.isna(current_prices[s])]

            if len(available_symbols) < n_holdings:
                continue

            # Get scores for available symbols
            latest_scores = factor_scores[factor_scores['symbol'].isin(available_symbols)]
            top_stocks = latest_scores.nlargest(n_holdings, 'combined_score')['symbol'].tolist()

            # Calculate expected returns and covariance
            returns_data = price_df[top_stocks].pct_change().dropna()
            if len(returns_data) < 60:
                continue

            expected_returns = returns_data.mean().values * 252
            cov_matrix = returns_data.cov().values * 252

            # Optimize
            allocation = optimizer.optimize(
                symbols=top_stocks,
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                method=method
            )

            # Execute rebalance
            target_weights = allocation.weights

            # Sell positions not in target
            for sym in list(positions.keys()):
                if sym not in target_weights:
                    shares = positions[sym]
                    price = current_prices.get(sym, 0)
                    sell_value = shares * price * (1 - self.transaction_cost)
                    cash += sell_value
                    trades_log.append({
                        'date': rebal_date,
                        'symbol': sym,
                        'action': 'SELL',
                        'shares': shares,
                        'price': price
                    })
                    del positions[sym]

            # Buy target positions
            for sym, weight in target_weights.items():
                target_value = portfolio_value * weight
                current_value = positions.get(sym, 0) * current_prices.get(sym, 0)
                diff = target_value - current_value

                if abs(diff) > portfolio_value * 0.01:  # Min 1% change
                    price = current_prices.get(sym, 0)
                    if price > 0:
                        if diff > 0:  # Buy
                            buy_value = diff * (1 + self.transaction_cost)
                            if buy_value <= cash:
                                shares = int(diff / price)
                                positions[sym] = positions.get(sym, 0) + shares
                                cash -= shares * price * (1 + self.transaction_cost)
                                trades_log.append({
                                    'date': rebal_date,
                                    'symbol': sym,
                                    'action': 'BUY',
                                    'shares': shares,
                                    'price': price
                                })
                        else:  # Sell
                            shares_to_sell = int(abs(diff) / price)
                            shares_to_sell = min(shares_to_sell, positions.get(sym, 0))
                            if shares_to_sell > 0:
                                positions[sym] -= shares_to_sell
                                cash += shares_to_sell * price * (1 - self.transaction_cost)
                                trades_log.append({
                                    'date': rebal_date,
                                    'symbol': sym,
                                    'action': 'SELL',
                                    'shares': shares_to_sell,
                                    'price': price
                                })

            weights = target_weights

            # Record equity
            position_value = sum(
                positions.get(sym, 0) * current_prices.get(sym, 0)
                for sym in positions
            )
            equity_curve.append({
                'date': rebal_date,
                'portfolio_value': cash + position_value,
                'cash': cash,
                'positions_value': position_value
            })

        # Calculate final metrics
        equity_df = pd.DataFrame(equity_curve)
        if len(equity_df) < 2:
            return {'error': 'Insufficient data for backtest'}

        equity_df = equity_df.set_index('date')
        returns = equity_df['portfolio_value'].pct_change().dropna()

        total_return = (equity_df['portfolio_value'].iloc[-1] / self.initial_capital) - 1

        # Calculate years from actual dates
        start = equity_df.index[0]
        end = equity_df.index[-1]
        years = (end - start).days / 365.25
        years = max(years, 0.5)  # At least 6 months

        # Annualized return (CAGR)
        annual_return = (1 + total_return) ** (1 / years) - 1
        volatility = returns.std() * np.sqrt(12)  # Monthly to annual
        sharpe = (annual_return - 0.06) / volatility if volatility > 0 else 0

        # Max drawdown
        cummax = equity_df['portfolio_value'].cummax()
        drawdown = (equity_df['portfolio_value'] - cummax) / cummax
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_df,
            'trades': trades_log,
            'n_rebalances': len(rebalance_dates),
            'n_trades': len(trades_log),
            'final_value': equity_df['portfolio_value'].iloc[-1]
        }


if __name__ == "__main__":
    # Test optimization
    np.random.seed(42)

    symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'SBIN']
    expected_returns = np.array([0.15, 0.12, 0.10, 0.14, 0.11])
    cov_matrix = np.array([
        [0.04, 0.01, 0.01, 0.01, 0.02],
        [0.01, 0.03, 0.01, 0.02, 0.01],
        [0.01, 0.01, 0.025, 0.01, 0.015],
        [0.01, 0.02, 0.01, 0.035, 0.01],
        [0.02, 0.01, 0.015, 0.01, 0.045]
    ])

    optimizer = PortfolioOptimizer()

    for method in ['equal', 'mean_variance', 'risk_parity', 'max_sharpe', 'min_variance']:
        allocation = optimizer.optimize(symbols, expected_returns, cov_matrix, method=method)
        print(f"\n{method.upper()}:")
        print(f"  Expected Return: {allocation.expected_return:.2%}")
        print(f"  Volatility: {allocation.expected_volatility:.2%}")
        print(f"  Sharpe Ratio: {allocation.sharpe_ratio:.2f}")
        print(f"  Weights: {allocation.weights}")
