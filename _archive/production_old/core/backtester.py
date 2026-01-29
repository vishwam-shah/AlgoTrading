"""
================================================================================
WALK-FORWARD BACKTESTING ENGINE
================================================================================
Production-grade backtesting with:
- Walk-forward validation (no look-ahead bias)
- Transaction cost modeling
- Realistic slippage simulation
- Portfolio-level metrics
- Detailed trade logging
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: int = 0
    direction: str = 'LONG'  # 'LONG' or 'SHORT'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0
    exit_reason: str = ''  # 'TARGET', 'STOP_LOSS', 'TIME_EXIT', 'SIGNAL_REVERSAL'


@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    # Returns
    total_return: float
    annualized_return: float
    benchmark_return: float
    excess_return: float  # Alpha

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_holding_days: float

    # Portfolio metrics
    final_value: float
    initial_value: float

    # Time series
    equity_curve: pd.Series = None
    drawdown_curve: pd.Series = None
    trades: List[Trade] = field(default_factory=list)


class WalkForwardBacktester:
    """
    Walk-forward backtesting engine.

    Features:
    1. Proper train/test splits with no look-ahead bias
    2. Realistic transaction costs and slippage
    3. Position sizing based on volatility
    4. Portfolio-level risk management
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_pct: float = 0.001,  # 0.1% commission
        slippage_pct: float = 0.001,  # 0.1% slippage
        max_position_pct: float = 0.10,  # Max 10% per position
        stop_loss_atr_mult: float = 2.5,  # Stop loss at 2.5x ATR (wider to reduce whipsaws)
        take_profit_atr_mult: float = 4.0,  # Take profit at 4x ATR (better R:R ratio)
        max_holding_days: int = 20,  # Maximum holding period
        use_trailing_stop: bool = True,  # Enable trailing stops
        trailing_stop_activation: float = 0.02,  # Activate trailing after 2% profit
        trailing_stop_distance: float = 0.015  # Trail by 1.5%
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.max_position_pct = max_position_pct
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.max_holding_days = max_holding_days
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_activation = trailing_stop_activation
        self.trailing_stop_distance = trailing_stop_distance

    def run_backtest(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        confidences: np.ndarray,
        expected_returns: np.ndarray,
        min_confidence: float = 0.55
    ) -> BacktestResult:
        """
        Run backtest on historical data with predictions.

        Args:
            data: DataFrame with OHLCV and features
            predictions: Array of direction predictions (0/1)
            confidences: Array of prediction confidences
            expected_returns: Array of expected return predictions
            min_confidence: Minimum confidence to take a trade

        Returns:
            BacktestResult with all metrics
        """
        logger.info("Running backtest...")

        # Initialize portfolio
        cash = self.initial_capital
        positions = {}  # symbol -> Trade
        equity = [self.initial_capital]
        dates = []
        trades = []

        # Get ATR for position sizing
        atr = data['atr_14'].values if 'atr_14' in data.columns else data['close'].values * 0.02

        # Simulate day by day
        for i in range(len(data)):
            current_date = data.index[i] if isinstance(data.index[i], datetime) else datetime.now()
            dates.append(current_date)

            current_price = data['close'].iloc[i]
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]

            # Check existing positions for exit conditions
            positions_to_close = []
            for symbol, trade in positions.items():
                # Initialize trailing stop price if not set
                if not hasattr(trade, 'trailing_stop_price'):
                    trade.trailing_stop_price = trade.entry_price - atr[trade.entry_date_idx] * self.stop_loss_atr_mult
                    trade.highest_price = trade.entry_price
                
                # Update highest price and trailing stop
                if self.use_trailing_stop and trade.direction == 'LONG':
                    if high > trade.highest_price:
                        trade.highest_price = high
                        # Check if profit threshold reached to activate trailing stop
                        profit_pct = (trade.highest_price - trade.entry_price) / trade.entry_price
                        if profit_pct >= self.trailing_stop_activation:
                            # Update trailing stop
                            new_trailing_stop = trade.highest_price * (1 - self.trailing_stop_distance)
                            trade.trailing_stop_price = max(trade.trailing_stop_price, new_trailing_stop)
                
                # Check trailing stop (replaces fixed stop loss when active)
                if trade.direction == 'LONG' and low <= trade.trailing_stop_price:
                    trade.exit_price = trade.trailing_stop_price
                    if trade.trailing_stop_price > trade.entry_price:
                        trade.exit_reason = 'TRAILING_STOP'
                    else:
                        trade.exit_reason = 'STOP_LOSS'
                    positions_to_close.append(symbol)

                # Check take profit
                elif trade.direction == 'LONG' and high >= trade.entry_price + atr[trade.entry_date_idx] * self.take_profit_atr_mult:
                    trade.exit_price = trade.entry_price + atr[trade.entry_date_idx] * self.take_profit_atr_mult
                    trade.exit_reason = 'TARGET'
                    positions_to_close.append(symbol)

                # Check time-based exit
                elif (i - trade.entry_date_idx) >= self.max_holding_days:
                    trade.exit_price = current_price
                    trade.exit_reason = 'TIME_EXIT'
                    positions_to_close.append(symbol)

                # Check signal reversal (only if in profit or minimal loss)
                elif predictions[i] != (1 if trade.direction == 'LONG' else 0):
                    unrealized_pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                    if unrealized_pnl_pct > -0.01:  # Only exit on reversal if not down more than 1%
                        trade.exit_price = current_price
                        trade.exit_reason = 'SIGNAL_REVERSAL'
                        positions_to_close.append(symbol)

            # Close positions
            for symbol in positions_to_close:
                trade = positions[symbol]
                trade.exit_date = current_date
                trade.holding_days = i - trade.entry_date_idx

                # Calculate P&L with costs
                exit_value = trade.exit_price * trade.quantity
                exit_cost = exit_value * (self.commission_pct + self.slippage_pct)
                cash += exit_value - exit_cost

                trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity - exit_cost - trade.entry_cost
                trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity) * 100

                trades.append(trade)
                del positions[symbol]

            # Consider new positions
            if len(positions) == 0 and confidences[i] >= min_confidence:
                # Volume confirmation: only trade if volume > 1.2x 20-day average
                volume_confirmed = True
                if 'volume_sma_20' in data.columns:
                    avg_vol = data['volume_sma_20'].iloc[i]
                    if avg_vol > 0:
                        volume_ratio = data['volume'].iloc[i] / avg_vol
                        volume_confirmed = volume_ratio >= 1.2
                
                if predictions[i] == 1 and volume_confirmed:  # Buy signal with volume confirmation
                    # Position sizing based on volatility
                    position_value = min(
                        cash * self.max_position_pct,
                        cash * 0.9  # Keep some cash reserve
                    )

                    # Adjust for volatility
                    vol_20 = data['volatility_20d'].iloc[i] if 'volatility_20d' in data.columns else 0.20
                    vol_adjustment = min(0.25 / (vol_20 + 0.01), 1.0)
                    position_value *= vol_adjustment

                    # Calculate entry with slippage
                    entry_price = current_price * (1 + self.slippage_pct)
                    quantity = int(position_value / entry_price)

                    if quantity > 0:
                        entry_cost = entry_price * quantity * self.commission_pct
                        cash -= (entry_price * quantity + entry_cost)

                        trade = Trade(
                            symbol='STOCK',
                            entry_date=current_date,
                            entry_price=entry_price,
                            quantity=quantity,
                            direction='LONG'
                        )
                        trade.entry_date_idx = i
                        trade.entry_cost = entry_cost

                        positions['STOCK'] = trade

            # Calculate portfolio value
            position_value = sum(
                t.quantity * current_price for t in positions.values()
            )
            portfolio_value = cash + position_value
            equity.append(portfolio_value)

        # Close any remaining positions at the end
        for symbol, trade in positions.items():
            final_price = data['close'].iloc[-1]
            trade.exit_date = dates[-1]
            trade.exit_price = final_price
            trade.exit_reason = 'END_OF_BACKTEST'
            trade.holding_days = len(data) - trade.entry_date_idx

            exit_value = trade.exit_price * trade.quantity
            exit_cost = exit_value * (self.commission_pct + self.slippage_pct)

            trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity - exit_cost - trade.entry_cost
            trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity) * 100

            trades.append(trade)

        # Calculate metrics
        result = self._calculate_metrics(equity, trades, data)

        return result

    def _calculate_metrics(
        self,
        equity: List[float],
        trades: List[Trade],
        data: pd.DataFrame
    ) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""

        equity_series = pd.Series(equity[1:])  # Exclude initial value

        # Returns
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        n_days = len(equity) - 1
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # Benchmark return (buy and hold)
        benchmark_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        excess_return = total_return - benchmark_return

        # Risk metrics
        daily_returns = equity_series.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Sortino ratio (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0

        # Drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()

        # Drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()
        drawdown_durations = in_drawdown.groupby(drawdown_periods).sum()
        max_drawdown_duration = int(drawdown_durations.max()) if len(drawdown_durations) > 0 else 0

        # Trade statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0

        avg_holding_days = np.mean([t.holding_days for t in trades]) if trades else 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_holding_days=avg_holding_days,
            final_value=equity[-1],
            initial_value=self.initial_capital,
            equity_curve=equity_series,
            drawdown_curve=drawdown,
            trades=trades
        )

    def walk_forward_backtest(
        self,
        data: pd.DataFrame,
        model,
        feature_engine,
        train_periods: int = 252 * 3,  # 3 years training
        test_periods: int = 63,  # 3 months testing
        step_size: int = 63  # Retrain every 3 months
    ) -> List[BacktestResult]:
        """
        Run walk-forward backtest with periodic retraining.

        Args:
            data: Full historical data
            model: Production model instance
            feature_engine: Feature engine instance
            train_periods: Number of days for training
            test_periods: Number of days for testing
            step_size: Days between retraining

        Returns:
            List of BacktestResult for each test period
        """
        logger.info("Running walk-forward backtest...")

        results = []
        n = len(data)

        for start in range(train_periods, n - test_periods, step_size):
            train_start = start - train_periods
            train_end = start
            test_start = start
            test_end = min(start + test_periods, n)

            logger.info(f"Period: Train [{train_start}:{train_end}], Test [{test_start}:{test_end}]")

            # Get training data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]

            # Compute features
            train_features = feature_engine.compute_all_features(train_data)
            test_features = feature_engine.compute_all_features(test_data)

            # Prepare training targets
            train_df = feature_engine.compute_targets(train_features.df)
            test_df = feature_engine.compute_targets(test_features.df)

            # Drop NaN
            train_df = train_df.dropna()
            test_df = test_df.dropna()

            if len(train_df) < 100 or len(test_df) < 10:
                continue

            # Get feature columns
            feature_cols = [c for c in train_df.columns
                           if c not in ['open', 'high', 'low', 'close', 'volume',
                                       'timestamp', 'date', 'symbol'] and
                           not c.startswith('target_')]

            # Prepare data
            X_train = train_df[feature_cols].values
            y_train = {
                'direction': train_df['target_direction'].values,
                'close_return': train_df['target_close_return'].values,
                'close_return_5d': train_df['target_close_return'].rolling(5).sum().values
                    if 'target_close_return' in train_df else None
            }

            X_test = test_df[feature_cols].values

            # Split training for validation
            val_size = int(len(X_train) * 0.2)
            X_val = X_train[-val_size:]
            X_train = X_train[:-val_size]

            y_val = {k: v[-val_size:] if v is not None else None for k, v in y_train.items()}
            y_train = {k: v[:-val_size] if v is not None else None for k, v in y_train.items()}

            # Train model
            model.train(X_train, y_train, X_val, y_val, feature_names=feature_cols)

            # Get predictions
            predictions = model.predict(X_test)
            directions = np.array([p.direction for p in predictions])
            confidences = np.array([p.confidence for p in predictions])
            expected_returns = np.array([p.expected_return for p in predictions])

            # Run backtest on test period
            result = self.run_backtest(
                test_df.reset_index(drop=True),
                directions,
                confidences,
                expected_returns
            )

            results.append(result)

        return results

    def generate_report(self, result: BacktestResult, symbol: str = None) -> str:
        """Generate formatted backtest report."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"  BACKTEST RESULTS" + (f" - {symbol}" if symbol else ""))
        lines.append("=" * 80)

        lines.append(f"\nPERFORMANCE:")
        lines.append(f"  Total Return:        {result.total_return:+.2%}")
        lines.append(f"  Annualized Return:   {result.annualized_return:+.2%}")
        lines.append(f"  Benchmark Return:    {result.benchmark_return:+.2%}")
        lines.append(f"  Excess Return:       {result.excess_return:+.2%}")

        lines.append(f"\nRISK METRICS:")
        lines.append(f"  Volatility:          {result.volatility:.2%}")
        lines.append(f"  Sharpe Ratio:        {result.sharpe_ratio:.2f}")
        lines.append(f"  Sortino Ratio:       {result.sortino_ratio:.2f}")
        lines.append(f"  Max Drawdown:        {result.max_drawdown:.2%}")
        lines.append(f"  Max DD Duration:     {result.max_drawdown_duration} days")

        lines.append(f"\nTRADE STATISTICS:")
        lines.append(f"  Total Trades:        {result.total_trades}")
        lines.append(f"  Winning Trades:      {result.winning_trades}")
        lines.append(f"  Losing Trades:       {result.losing_trades}")
        lines.append(f"  Win Rate:            {result.win_rate:.2%}")
        lines.append(f"  Avg Win:             Rs {result.avg_win:,.2f}")
        lines.append(f"  Avg Loss:            Rs {result.avg_loss:,.2f}")
        lines.append(f"  Profit Factor:       {result.profit_factor:.2f}")
        lines.append(f"  Avg Holding Days:    {result.avg_holding_days:.1f}")

        lines.append(f"\nPORTFOLIO:")
        lines.append(f"  Initial Value:       Rs {result.initial_value:,.2f}")
        lines.append(f"  Final Value:         Rs {result.final_value:,.2f}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def plot_results(self, result: BacktestResult, save_path: str = None):
        """Plot backtest results."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Equity curve
        ax1 = axes[0]
        ax1.plot(result.equity_curve.values, label='Portfolio Value', color='blue')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (Rs)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = axes[1]
        ax2.fill_between(range(len(result.drawdown_curve)),
                         result.drawdown_curve.values * 100, 0,
                         color='red', alpha=0.3)
        ax2.plot(result.drawdown_curve.values * 100, color='red', linewidth=0.5)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # Trade P&L distribution
        ax3 = axes[2]
        if result.trades:
            pnls = [t.pnl_pct for t in result.trades]
            colors = ['green' if p > 0 else 'red' for p in pnls]
            ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
            ax3.axhline(y=0, color='black', linewidth=0.5)
            ax3.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Trade Number')
            ax3.set_ylabel('P&L (%)')
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.close()
