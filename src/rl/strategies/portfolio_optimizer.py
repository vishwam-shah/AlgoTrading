"""
Portfolio Optimizer with RL-Enhanced Allocation
================================================

Advanced portfolio optimization combining:
- Mean-Variance Optimization (Markowitz)
- Risk Parity
- ML Prediction-weighted allocation
- RL-based dynamic rebalancing
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy.optimize import minimize
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import config


@dataclass
class AssetAllocation:
    """Asset allocation result"""
    symbol: str
    weight: float  # Portfolio weight (0-1)
    target_value: float  # Target value in INR
    current_value: float  # Current value
    action: str  # 'BUY', 'SELL', 'HOLD'
    trade_value: float  # Amount to trade
    expected_return: float
    risk_contribution: float


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    diversification_ratio: float
    effective_positions: int


class MeanVarianceOptimizer:
    """
    Mean-Variance (Markowitz) Portfolio Optimization.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.06,  # 6% annual risk-free rate (India)
        min_weight: float = 0.0,
        max_weight: float = 0.25,  # Max 25% per asset
        target_volatility: float = None
    ):
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target_volatility = target_volatility

    def optimize(
        self,
        returns: pd.DataFrame,
        predictions: Dict[str, float] = None,
        method: str = 'max_sharpe'
    ) -> Tuple[np.ndarray, PortfolioMetrics]:
        """
        Optimize portfolio weights.

        Args:
            returns: Historical returns DataFrame (symbols as columns)
            predictions: Optional ML predictions for expected returns
            method: 'max_sharpe', 'min_variance', 'risk_parity'

        Returns:
            Tuple of (weights array, PortfolioMetrics)
        """
        n_assets = len(returns.columns)
        symbols = returns.columns.tolist()

        # Calculate mean returns and covariance
        if predictions is not None:
            # Use ML predictions as expected returns
            mean_returns = np.array([predictions.get(s, returns[s].mean() * 252) for s in symbols])
        else:
            mean_returns = returns.mean().values * 252  # Annualized

        cov_matrix = returns.cov().values * 252  # Annualized covariance

        # Initial weights
        init_weights = np.ones(n_assets) / n_assets

        # Bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        if method == 'max_sharpe':
            # Maximize Sharpe ratio
            def neg_sharpe(weights):
                port_return = np.dot(weights, mean_returns)
                port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(port_return - self.risk_free_rate) / port_vol

            result = minimize(neg_sharpe, init_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)

        elif method == 'min_variance':
            # Minimize variance
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))

            result = minimize(portfolio_variance, init_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)

        elif method == 'risk_parity':
            # Equal risk contribution
            def risk_parity_objective(weights):
                port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights)
                risk_contrib = weights * marginal_contrib / port_vol
                target_risk = port_vol / n_assets
                return np.sum((risk_contrib - target_risk) ** 2)

            result = minimize(risk_parity_objective, init_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
        else:
            raise ValueError(f"Unknown method: {method}")

        weights = result.x

        # Calculate metrics
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol

        # Diversification ratio
        weighted_vols = np.sqrt(np.diag(cov_matrix)) * weights
        diversification = np.sum(weighted_vols) / port_vol

        # Effective number of positions
        effective_n = 1 / np.sum(weights ** 2)

        # VaR 95%
        var_95 = -port_return / 252 + 1.65 * port_vol / np.sqrt(252)

        metrics = PortfolioMetrics(
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            max_drawdown=0.0,  # Would need historical simulation
            var_95=var_95,
            diversification_ratio=diversification,
            effective_positions=int(effective_n)
        )

        return weights, metrics


class PredictionWeightedOptimizer:
    """
    Portfolio optimizer that weights assets by ML prediction confidence.
    """

    def __init__(
        self,
        base_optimizer: MeanVarianceOptimizer = None,
        prediction_weight: float = 0.5,  # How much to weight predictions vs. historical
        min_confidence: float = 0.55
    ):
        self.base_optimizer = base_optimizer or MeanVarianceOptimizer()
        self.prediction_weight = prediction_weight
        self.min_confidence = min_confidence

    def optimize(
        self,
        returns: pd.DataFrame,
        predictions: Dict[str, Dict],
        current_prices: Dict[str, float]
    ) -> Tuple[np.ndarray, PortfolioMetrics]:
        """
        Optimize using ML predictions.

        Args:
            returns: Historical returns
            predictions: Dict of {symbol: {direction, confidence, close_return}}
            current_prices: Current prices

        Returns:
            Tuple of (weights, metrics)
        """
        symbols = returns.columns.tolist()

        # Filter by confidence
        valid_symbols = []
        expected_returns = {}

        for symbol in symbols:
            if symbol in predictions:
                pred = predictions[symbol]
                if pred.get('confidence', 0) >= self.min_confidence:
                    valid_symbols.append(symbol)
                    # Expected return = predicted return * direction signal
                    direction_signal = 1 if pred.get('direction', 0) == 1 else -1
                    expected_returns[symbol] = pred.get('close_return', 0) * 252 * direction_signal

        if not valid_symbols:
            # No valid predictions, return equal weights
            weights = np.ones(len(symbols)) / len(symbols)
            return weights, PortfolioMetrics(0, 0, 0, 0, 0, 1, len(symbols))

        # Optimize with predictions
        valid_returns = returns[valid_symbols]
        weights, metrics = self.base_optimizer.optimize(
            valid_returns,
            predictions=expected_returns,
            method='max_sharpe'
        )

        # Map back to full symbol list
        full_weights = np.zeros(len(symbols))
        for i, symbol in enumerate(valid_symbols):
            full_idx = symbols.index(symbol)
            full_weights[full_idx] = weights[i]

        # Normalize
        full_weights = full_weights / full_weights.sum() if full_weights.sum() > 0 else full_weights

        return full_weights, metrics


class RLEnhancedOptimizer:
    """
    Portfolio optimizer enhanced with RL for dynamic allocation.
    """

    def __init__(
        self,
        base_optimizer: PredictionWeightedOptimizer = None,
        rl_weight: float = 0.3,
        momentum_factor: float = 0.1
    ):
        self.base_optimizer = base_optimizer or PredictionWeightedOptimizer()
        self.rl_weight = rl_weight
        self.momentum_factor = momentum_factor

        # State tracking
        self.previous_weights: Optional[np.ndarray] = None
        self.weight_history: List[np.ndarray] = []
        self.performance_history: List[float] = []

    def optimize(
        self,
        returns: pd.DataFrame,
        predictions: Dict[str, Dict],
        current_prices: Dict[str, float],
        rl_signals: Dict[str, Dict] = None,
        portfolio_state: Dict = None
    ) -> Tuple[np.ndarray, List[AssetAllocation]]:
        """
        RL-enhanced portfolio optimization.

        Args:
            returns: Historical returns
            predictions: ML predictions
            current_prices: Current prices
            rl_signals: RL agent signals per symbol
            portfolio_state: Current portfolio state

        Returns:
            Tuple of (weights, allocations)
        """
        symbols = returns.columns.tolist()

        # Get base optimization
        base_weights, metrics = self.base_optimizer.optimize(
            returns, predictions, current_prices
        )

        # Adjust with RL signals
        adjusted_weights = base_weights.copy()

        if rl_signals:
            for i, symbol in enumerate(symbols):
                if symbol in rl_signals:
                    signal = rl_signals[symbol]
                    action = signal.get('action', 0)
                    confidence = signal.get('confidence', 0.5)

                    # Adjust weight based on RL signal
                    if action == 1:  # Buy signal
                        adjustment = self.rl_weight * confidence
                        adjusted_weights[i] *= (1 + adjustment)
                    elif action == 2:  # Sell signal
                        adjustment = self.rl_weight * confidence
                        adjusted_weights[i] *= (1 - adjustment)

        # Apply momentum smoothing (prevent drastic changes)
        if self.previous_weights is not None:
            adjusted_weights = (
                (1 - self.momentum_factor) * adjusted_weights +
                self.momentum_factor * self.previous_weights
            )

        # Normalize
        adjusted_weights = adjusted_weights / adjusted_weights.sum()

        # Store history
        self.previous_weights = adjusted_weights.copy()
        self.weight_history.append(adjusted_weights.copy())

        # Create allocation objects
        total_capital = portfolio_state.get('total_capital', 100000) if portfolio_state else 100000
        allocations = []

        for i, symbol in enumerate(symbols):
            weight = adjusted_weights[i]
            target_value = total_capital * weight
            current_value = portfolio_state.get('positions', {}).get(symbol, {}).get('value', 0)
            trade_value = target_value - current_value

            # Determine action
            if abs(trade_value) < total_capital * 0.01:  # Less than 1% change
                action = 'HOLD'
                trade_value = 0
            elif trade_value > 0:
                action = 'BUY'
            else:
                action = 'SELL'
                trade_value = abs(trade_value)

            pred = predictions.get(symbol, {})

            allocations.append(AssetAllocation(
                symbol=symbol,
                weight=weight,
                target_value=target_value,
                current_value=current_value,
                action=action,
                trade_value=trade_value,
                expected_return=pred.get('close_return', 0) * 252,
                risk_contribution=0  # Would need calculation
            ))

        return adjusted_weights, allocations


class DynamicRebalancer:
    """
    Handles dynamic portfolio rebalancing.
    """

    def __init__(
        self,
        rebalance_threshold: float = 0.05,  # 5% deviation triggers rebalance
        min_rebalance_interval_days: int = 1,
        transaction_cost_pct: float = 0.001
    ):
        self.rebalance_threshold = rebalance_threshold
        self.min_rebalance_interval_days = min_rebalance_interval_days
        self.transaction_cost_pct = transaction_cost_pct

        self.last_rebalance: Optional[datetime] = None
        self.target_weights: Optional[np.ndarray] = None

    def check_rebalance_needed(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        current_time: datetime = None
    ) -> Tuple[bool, str]:
        """
        Check if rebalancing is needed.

        Returns:
            Tuple of (should_rebalance, reason)
        """
        current_time = current_time or datetime.now()

        # Check time since last rebalance
        if self.last_rebalance:
            days_since = (current_time - self.last_rebalance).days
            if days_since < self.min_rebalance_interval_days:
                return False, f"Only {days_since} days since last rebalance"

        # Check weight deviation
        weight_deviation = np.abs(current_weights - target_weights).max()

        if weight_deviation >= self.rebalance_threshold:
            return True, f"Weight deviation {weight_deviation:.2%} exceeds threshold"

        return False, "No rebalancing needed"

    def calculate_rebalance_trades(
        self,
        current_holdings: Dict[str, float],  # symbol -> value
        target_weights: np.ndarray,
        symbols: List[str],
        prices: Dict[str, float]
    ) -> List[Dict]:
        """
        Calculate trades needed for rebalancing.

        Returns:
            List of trade orders
        """
        total_value = sum(current_holdings.values())
        trades = []

        for i, symbol in enumerate(symbols):
            current_value = current_holdings.get(symbol, 0)
            target_value = total_value * target_weights[i]

            diff = target_value - current_value

            # Account for transaction costs
            if abs(diff) > total_value * 0.01:  # Only trade if > 1% of portfolio
                trade_value = abs(diff)
                cost = trade_value * self.transaction_cost_pct

                if diff > 0:
                    # Buy
                    quantity = int((trade_value - cost) / prices.get(symbol, 1))
                    if quantity > 0:
                        trades.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': quantity,
                            'price': prices.get(symbol, 0),
                            'value': quantity * prices.get(symbol, 0),
                            'cost': cost
                        })
                else:
                    # Sell
                    quantity = int(trade_value / prices.get(symbol, 1))
                    if quantity > 0:
                        trades.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'quantity': quantity,
                            'price': prices.get(symbol, 0),
                            'value': quantity * prices.get(symbol, 0),
                            'cost': cost
                        })

        # Sort: sells first, then buys (to free up capital)
        trades.sort(key=lambda x: 0 if x['action'] == 'SELL' else 1)

        return trades


class PortfolioManager:
    """
    High-level portfolio management combining all components.
    """

    def __init__(
        self,
        symbols: List[str],
        initial_capital: float = 100000,
        optimization_method: str = 'rl_enhanced'
    ):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.capital = initial_capital

        # Initialize optimizers
        self.mv_optimizer = MeanVarianceOptimizer()
        self.pred_optimizer = PredictionWeightedOptimizer(self.mv_optimizer)
        self.rl_optimizer = RLEnhancedOptimizer(self.pred_optimizer)
        self.rebalancer = DynamicRebalancer()

        # State
        self.holdings: Dict[str, Dict] = {}  # symbol -> {quantity, avg_price, value}
        self.cash = initial_capital
        self.weights: Optional[np.ndarray] = None
        self.trade_history: List[Dict] = []

        logger.info(f"PortfolioManager initialized with {len(symbols)} symbols")

    def get_historical_returns(self, lookback_days: int = 252) -> pd.DataFrame:
        """Get historical returns for all symbols"""
        returns_data = {}

        for symbol in self.symbols:
            try:
                filepath = os.path.join(config.RAW_DATA_DIR, f"{symbol}.csv")
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    df['return'] = df['close'].pct_change()
                    returns_data[symbol] = df['return'].tail(lookback_days).values
            except Exception as e:
                logger.warning(f"Could not load returns for {symbol}: {e}")

        if returns_data:
            # Align lengths
            min_len = min(len(v) for v in returns_data.values())
            for symbol in returns_data:
                returns_data[symbol] = returns_data[symbol][-min_len:]

            return pd.DataFrame(returns_data)

        return pd.DataFrame()

    def optimize_portfolio(
        self,
        predictions: Dict[str, Dict],
        current_prices: Dict[str, float],
        rl_signals: Dict[str, Dict] = None
    ) -> List[AssetAllocation]:
        """
        Optimize portfolio allocation.

        Args:
            predictions: ML predictions per symbol
            current_prices: Current prices
            rl_signals: RL agent signals

        Returns:
            List of allocation recommendations
        """
        # Get historical returns
        returns = self.get_historical_returns()

        if returns.empty:
            logger.warning("No historical returns available")
            return []

        # Portfolio state
        portfolio_state = {
            'total_capital': self.capital + sum(h['value'] for h in self.holdings.values()),
            'positions': self.holdings
        }

        # Optimize
        weights, allocations = self.rl_optimizer.optimize(
            returns=returns,
            predictions=predictions,
            current_prices=current_prices,
            rl_signals=rl_signals,
            portfolio_state=portfolio_state
        )

        self.weights = weights

        return allocations

    def execute_rebalance(
        self,
        allocations: List[AssetAllocation],
        prices: Dict[str, float],
        broker_api = None
    ) -> List[Dict]:
        """
        Execute rebalancing trades.

        Args:
            allocations: Target allocations
            prices: Current prices
            broker_api: Broker API for execution

        Returns:
            List of executed trades
        """
        executed_trades = []

        # First process sells
        for alloc in allocations:
            if alloc.action == 'SELL' and alloc.trade_value > 0:
                quantity = int(alloc.trade_value / prices.get(alloc.symbol, 1))
                if quantity > 0:
                    trade = {
                        'symbol': alloc.symbol,
                        'action': 'SELL',
                        'quantity': quantity,
                        'price': prices[alloc.symbol],
                        'timestamp': datetime.now()
                    }

                    if broker_api:
                        order_id = broker_api.place_market_order(
                            alloc.symbol, quantity, 'SELL'
                        )
                        trade['order_id'] = order_id

                    # Update holdings
                    if alloc.symbol in self.holdings:
                        self.holdings[alloc.symbol]['quantity'] -= quantity
                        if self.holdings[alloc.symbol]['quantity'] <= 0:
                            del self.holdings[alloc.symbol]
                        self.cash += quantity * prices[alloc.symbol]

                    executed_trades.append(trade)
                    self.trade_history.append(trade)

        # Then process buys
        for alloc in allocations:
            if alloc.action == 'BUY' and alloc.trade_value > 0:
                quantity = int(min(alloc.trade_value, self.cash) / prices.get(alloc.symbol, 1))
                if quantity > 0 and self.cash >= quantity * prices[alloc.symbol]:
                    trade = {
                        'symbol': alloc.symbol,
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': prices[alloc.symbol],
                        'timestamp': datetime.now()
                    }

                    if broker_api:
                        order_id = broker_api.place_market_order(
                            alloc.symbol, quantity, 'BUY'
                        )
                        trade['order_id'] = order_id

                    # Update holdings
                    if alloc.symbol not in self.holdings:
                        self.holdings[alloc.symbol] = {
                            'quantity': 0,
                            'avg_price': 0,
                            'value': 0
                        }

                    old_qty = self.holdings[alloc.symbol]['quantity']
                    old_avg = self.holdings[alloc.symbol]['avg_price']

                    new_qty = old_qty + quantity
                    new_avg = (old_qty * old_avg + quantity * prices[alloc.symbol]) / new_qty

                    self.holdings[alloc.symbol]['quantity'] = new_qty
                    self.holdings[alloc.symbol]['avg_price'] = new_avg
                    self.holdings[alloc.symbol]['value'] = new_qty * prices[alloc.symbol]

                    self.cash -= quantity * prices[alloc.symbol]

                    executed_trades.append(trade)
                    self.trade_history.append(trade)

        self.rebalancer.last_rebalance = datetime.now()

        return executed_trades

    def get_portfolio_summary(self, prices: Dict[str, float] = None) -> Dict:
        """Get portfolio summary"""
        prices = prices or {}

        total_value = self.cash
        positions_value = 0

        for symbol, holding in self.holdings.items():
            price = prices.get(symbol, holding['avg_price'])
            holding['value'] = holding['quantity'] * price
            positions_value += holding['value']
            total_value += holding['value']

        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'num_positions': len(self.holdings),
            'holdings': self.holdings.copy(),
            'weights': self.weights.tolist() if self.weights is not None else None,
            'return_pct': (total_value / self.initial_capital - 1) * 100,
            'num_trades': len(self.trade_history)
        }


def demo_optimization():
    """Demo portfolio optimization"""
    symbols = ['HDFCBANK', 'TCS', 'RELIANCE', 'INFY', 'SBIN']

    # Mock predictions
    predictions = {
        'HDFCBANK': {'direction': 1, 'confidence': 0.72, 'close_return': 0.01},
        'TCS': {'direction': 1, 'confidence': 0.68, 'close_return': 0.008},
        'RELIANCE': {'direction': 0, 'confidence': 0.55, 'close_return': -0.005},
        'INFY': {'direction': 1, 'confidence': 0.60, 'close_return': 0.006},
        'SBIN': {'direction': 1, 'confidence': 0.65, 'close_return': 0.012}
    }

    # Mock prices
    prices = {
        'HDFCBANK': 1650.0,
        'TCS': 3800.0,
        'RELIANCE': 2500.0,
        'INFY': 1450.0,
        'SBIN': 620.0
    }

    # RL signals
    rl_signals = {
        'HDFCBANK': {'action': 1, 'confidence': 0.70},
        'TCS': {'action': 1, 'confidence': 0.65},
        'SBIN': {'action': 1, 'confidence': 0.60}
    }

    # Create manager
    manager = PortfolioManager(symbols, initial_capital=100000)

    # Optimize
    allocations = manager.optimize_portfolio(predictions, prices, rl_signals)

    # Print allocations
    print("\n" + "=" * 70)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("=" * 70)

    for alloc in allocations:
        print(f"\n{alloc.symbol}:")
        print(f"  Weight: {alloc.weight:.2%}")
        print(f"  Target Value: Rs {alloc.target_value:,.2f}")
        print(f"  Action: {alloc.action}")
        print(f"  Trade Value: Rs {alloc.trade_value:,.2f}")
        print(f"  Expected Return: {alloc.expected_return:.2%}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    demo_optimization()
