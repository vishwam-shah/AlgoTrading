"""
================================================================================
UNIFIED BACKTESTING ENGINE
================================================================================
Merged backtesting from production/ and pipeline/ systems.

Dual mode:
1. WalkForwardBacktester (production): Per-stock trade simulation with entry
   filters, position sizing, exit strategies, trailing stops
2. BacktestValidator (pipeline): Portfolio-level simulation with periodic
   rebalancing, sector allocation, comprehensive metrics
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class Trade:
    """Individual trade record."""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: int = 0
    direction: str = 'LONG'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0
    exit_reason: str = ''


@dataclass
class BacktestResult:
    """Comprehensive backtest results (production-style)."""
    total_return: float
    annualized_return: float
    benchmark_return: float
    excess_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_holding_days: float
    final_value: float
    initial_value: float
    equity_curve: pd.Series = None
    drawdown_curve: pd.Series = None
    trades: List[Trade] = field(default_factory=list)
    avg_trade_return: float = None


@dataclass
class BacktestResults:
    """Portfolio backtest results (pipeline-style)."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    win_rate: float
    information_ratio: float
    var_95: float
    cvar_95: float
    total_trades: int
    turnover: float
    final_value: float
    initial_value: float
    rebalance_count: int
    equity_curve: pd.Series = None
    drawdown_curve: pd.Series = None


# ============================================================================
# WALK-FORWARD BACKTESTER (Per-Stock Trades)
# ============================================================================

class WalkForwardBacktester:
    """
    Per-stock trade backtesting engine from production system.

    Features:
    - Walk-forward validation
    - Entry filters (volume, trend, momentum, MACD, sentiment)
    - Exit strategies (trailing stop, take profit, time exit, trend breakdown)
    - Position sizing based on volatility
    - Transaction costs and slippage
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.001,
        max_position_pct: float = 0.15,
        stop_loss_atr_mult: float = 3.0,
        take_profit_atr_mult: float = 5.0,
        max_holding_days: int = 30,
        use_trailing_stop: bool = True,
        trailing_stop_activation: float = 0.03,
        trailing_stop_distance: float = 0.02
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

    def run_backtest(self, data, predictions, confidences, expected_returns,
                     min_confidence=0.55):
        """Run per-stock backtest with predictions."""
        logger.info("Running backtest...")

        cash = self.initial_capital
        positions = {}
        equity = [self.initial_capital]
        dates = []
        trades = []

        atr = data['atr_14'].values if 'atr_14' in data.columns else data['close'].values * 0.02

        for i in range(len(data)):
            current_date = data.index[i] if isinstance(data.index[i], datetime) else datetime.now()
            dates.append(current_date)
            current_price = data['close'].iloc[i]
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]

            # Check existing positions for exits
            positions_to_close = []
            for symbol, trade in positions.items():
                if not hasattr(trade, 'trailing_stop_price'):
                    trade.trailing_stop_price = trade.entry_price - atr[trade.entry_date_idx] * self.stop_loss_atr_mult
                    trade.highest_price = trade.entry_price

                if self.use_trailing_stop and trade.direction == 'LONG':
                    if high > trade.highest_price:
                        trade.highest_price = high
                        profit_pct = (trade.highest_price - trade.entry_price) / trade.entry_price
                        if profit_pct >= self.trailing_stop_activation:
                            trail_distance = max(self.trailing_stop_distance * 0.7, 0.01)
                            new_trailing_stop = trade.highest_price * (1 - trail_distance)
                            trade.trailing_stop_price = max(trade.trailing_stop_price, new_trailing_stop)

                unrealized_pnl_pct = (current_price - trade.entry_price) / trade.entry_price

                if trade.direction == 'LONG' and low <= trade.trailing_stop_price:
                    trade.exit_price = trade.trailing_stop_price
                    trade.exit_reason = 'TRAILING_STOP' if trade.trailing_stop_price > trade.entry_price else 'STOP_LOSS'
                    positions_to_close.append(symbol)
                elif trade.direction == 'LONG' and high >= trade.entry_price + atr[trade.entry_date_idx] * self.take_profit_atr_mult:
                    trade.exit_price = trade.entry_price + atr[trade.entry_date_idx] * self.take_profit_atr_mult
                    trade.exit_reason = 'TARGET'
                    positions_to_close.append(symbol)
                elif (i - trade.entry_date_idx) >= self.max_holding_days:
                    trade.exit_price = current_price
                    trade.exit_reason = 'TIME_EXIT'
                    positions_to_close.append(symbol)
                elif unrealized_pnl_pct < 0 and 'sma_20' in data.columns:
                    if current_price < data['sma_20'].iloc[i] * 0.98:
                        trade.exit_price = current_price
                        trade.exit_reason = 'TREND_BREAKDOWN'
                        positions_to_close.append(symbol)
                elif predictions[i] != (1 if trade.direction == 'LONG' else 0):
                    if unrealized_pnl_pct < 0.02:
                        trade.exit_price = current_price
                        trade.exit_reason = 'SIGNAL_REVERSAL'
                        positions_to_close.append(symbol)

            # Close positions
            for symbol in positions_to_close:
                trade = positions[symbol]
                trade.exit_date = current_date
                trade.holding_days = i - trade.entry_date_idx
                exit_value = trade.exit_price * trade.quantity
                exit_cost = exit_value * (self.commission_pct + self.slippage_pct)
                cash += exit_value - exit_cost
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity - exit_cost - trade.entry_cost
                trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity) * 100
                trades.append(trade)
                del positions[symbol]

            # Consider new entries
            if len(positions) == 0 and confidences[i] >= min_confidence:
                volume_confirmed = True
                if 'volume_sma_20' in data.columns:
                    avg_vol = data['volume_sma_20'].iloc[i]
                    if avg_vol > 0:
                        volume_confirmed = data['volume'].iloc[i] / avg_vol >= 1.0

                trend_confirmed = True
                if 'sma_20' in data.columns:
                    trend_confirmed = current_price > data['sma_20'].iloc[i]

                momentum_ok = True
                if 'rsi_14' in data.columns:
                    rsi = data['rsi_14'].iloc[i]
                    momentum_ok = 30 < rsi < 70

                macd_bullish = True
                if 'macd' in data.columns and 'macd_signal' in data.columns:
                    macd = data['macd'].iloc[i]
                    macd_signal = data['macd_signal'].iloc[i]
                    if i > 0:
                        macd_prev = data['macd'].iloc[i-1]
                        macd_bullish = (macd > macd_signal) or (macd > macd_prev)
                    else:
                        macd_bullish = macd > macd_signal

                sentiment_ok = True
                if 'news_sentiment' in data.columns:
                    sent = data['news_sentiment'].iloc[i]
                    sentiment_ok = sent >= -0.2 or pd.isna(sent) or sent == 0

                core_conditions = predictions[i] == 1 and momentum_ok
                supporting_score = sum([volume_confirmed, trend_confirmed, macd_bullish, sentiment_ok])

                if core_conditions and supporting_score >= 2:
                    position_value = min(cash * self.max_position_pct, cash * 0.9)
                    vol_20 = data['volatility_20d'].iloc[i] if 'volatility_20d' in data.columns else 0.20
                    vol_adjustment = min(0.20 / (vol_20 + 0.01), 1.0)
                    position_value *= vol_adjustment
                    entry_price = current_price * (1 + self.slippage_pct)
                    quantity = int(position_value / entry_price)

                    if quantity > 0:
                        entry_cost = entry_price * quantity * self.commission_pct
                        cash -= (entry_price * quantity + entry_cost)
                        trade = Trade(symbol='STOCK', entry_date=current_date,
                                    entry_price=entry_price, quantity=quantity, direction='LONG')
                        trade.entry_date_idx = i
                        trade.entry_cost = entry_cost
                        positions['STOCK'] = trade

            position_value = sum(t.quantity * current_price for t in positions.values())
            equity.append(cash + position_value)

        # Close remaining positions
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

        return self._calculate_metrics(equity, trades, data)

    def _calculate_metrics(self, equity, trades, data):
        """Calculate comprehensive metrics."""
        if 'timestamp' in data.columns:
            dates = pd.to_datetime(data['timestamp'].values)
        elif isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        else:
            dates = pd.date_range(start='2020-01-01', periods=len(equity)-1, freq='D')

        equity_dates = dates[:len(equity)-1] if len(dates) >= len(equity)-1 else dates
        equity_series = pd.Series(equity[1:len(equity_dates)+1], index=equity_dates)

        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        n_days = len(equity) - 1
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        benchmark_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]

        daily_returns = equity_series.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0

        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()

        in_drawdown = drawdown < 0
        drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()
        drawdown_durations = in_drawdown.groupby(drawdown_periods).sum()
        max_drawdown_duration = int(drawdown_durations.max()) if len(drawdown_durations) > 0 else 0

        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 0
        if total_losses > 0:
            profit_factor = total_wins / total_losses
        elif total_wins > 0:
            profit_factor = 99.99
        else:
            profit_factor = 0.0

        avg_holding_days = np.mean([t.holding_days for t in trades]) if trades else 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            benchmark_return=benchmark_return,
            excess_return=total_return - benchmark_return,
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

    def generate_report(self, result: BacktestResult, symbol: str = None) -> str:
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
        lines.append(f"  Win Rate:            {result.win_rate:.2%}")
        lines.append(f"  Profit Factor:       {result.profit_factor:.2f}")
        lines.append(f"  Avg Holding Days:    {result.avg_holding_days:.1f}")
        lines.append(f"\nPORTFOLIO:")
        lines.append(f"  Initial Value:       Rs {result.initial_value:,.2f}")
        lines.append(f"  Final Value:         Rs {result.final_value:,.2f}")
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def plot_results(self, result: BacktestResult, save_path: str = None):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        ax1 = axes[0]
        ax1.plot(result.equity_curve.values, label='Portfolio Value', color='blue')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (Rs)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.fill_between(range(len(result.drawdown_curve)),
                         result.drawdown_curve.values * 100, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

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
        plt.close()


# ============================================================================
# PORTFOLIO BACKTEST VALIDATOR (Pipeline-style)
# ============================================================================

class BacktestValidator:
    """
    Portfolio-level backtest from pipeline system.

    Features:
    - Periodic rebalancing (weekly/monthly/quarterly)
    - Drift-based rebalancing (>2% from target)
    - Transaction costs and slippage
    - Comprehensive risk metrics (Sharpe, Sortino, Calmar, VaR, CVaR)
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        rebalance_frequency: str = 'monthly',
        rebalance_threshold: float = 0.02,
        risk_free_rate: float = 0.05
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.rebalance_frequency = rebalance_frequency
        self.rebalance_threshold = rebalance_threshold
        self.risk_free_rate = risk_free_rate

    def run_backtest(
        self,
        price_data: Dict[str, pd.DataFrame],
        allocation,
        start_date: str = None,
        end_date: str = None,
        initial_capital: float = None,
        benchmark: pd.DataFrame = None
    ) -> BacktestResults:
        """Run portfolio-level backtest with periodic rebalancing."""
        logger.info("=" * 60)
        logger.info("PORTFOLIO BACKTEST")
        logger.info("=" * 60)

        capital = initial_capital or self.initial_capital

        # Get target weights
        if hasattr(allocation, 'weights'):
            weights = allocation.weights
        elif isinstance(allocation, dict):
            weights = allocation
        else:
            logger.error("Invalid allocation format")
            return None

        # Get common date range
        all_dates = None
        close_prices = {}

        for symbol, df in price_data.items():
            if symbol not in weights:
                continue
            if isinstance(df.index, pd.DatetimeIndex):
                dates = df.index
            elif 'timestamp' in df.columns:
                dates = pd.to_datetime(df['timestamp'])
                df = df.set_index(dates)
            else:
                continue

            close_prices[symbol] = df['close']
            if all_dates is None:
                all_dates = set(dates)
            else:
                all_dates = all_dates.intersection(set(dates))

        if not all_dates:
            logger.error("No common dates found")
            return None

        all_dates = sorted(all_dates)

        # Filter date range
        if start_date:
            all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
        if end_date:
            all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]

        logger.info(f"Backtest period: {all_dates[0]} to {all_dates[-1]}")
        logger.info(f"Trading days: {len(all_dates)}")
        logger.info(f"Stocks: {len(weights)}")

        # Initialize portfolio
        portfolio_value = capital
        equity = [capital]
        rebalance_count = 0
        total_turnover = 0

        # Current holdings (shares)
        holdings = {}
        cash = capital

        # Initial allocation
        for symbol, weight in weights.items():
            if symbol in close_prices and weight > 0:
                price = close_prices[symbol].loc[all_dates[0]] if all_dates[0] in close_prices[symbol].index else None
                if price is not None and price > 0:
                    alloc_value = capital * weight
                    shares = int(alloc_value / price)
                    cost = shares * price * (1 + self.commission_pct + self.slippage_pct)
                    if cost <= cash:
                        holdings[symbol] = shares
                        cash -= cost

        # Simulate
        last_rebalance = all_dates[0]

        for i, date in enumerate(all_dates[1:], 1):
            # Calculate portfolio value
            portfolio_value = cash
            for symbol, shares in holdings.items():
                if date in close_prices[symbol].index:
                    portfolio_value += shares * close_prices[symbol].loc[date]
            equity.append(portfolio_value)

            # Check if rebalance needed
            should_rebalance = False
            if self.rebalance_frequency == 'monthly':
                should_rebalance = date.month != last_rebalance.month
            elif self.rebalance_frequency == 'weekly':
                should_rebalance = (date - last_rebalance).days >= 5
            elif self.rebalance_frequency == 'quarterly':
                should_rebalance = date.quarter != last_rebalance.quarter

            # Also check drift
            if not should_rebalance:
                current_weights = {}
                for symbol, shares in holdings.items():
                    if date in close_prices[symbol].index:
                        current_weights[symbol] = (shares * close_prices[symbol].loc[date]) / portfolio_value
                max_drift = max(abs(current_weights.get(s, 0) - weights.get(s, 0))
                              for s in set(list(current_weights.keys()) + list(weights.keys())))
                should_rebalance = max_drift > self.rebalance_threshold

            if should_rebalance:
                # Sell all
                for symbol, shares in holdings.items():
                    if date in close_prices[symbol].index:
                        sell_value = shares * close_prices[symbol].loc[date]
                        cash += sell_value * (1 - self.commission_pct - self.slippage_pct)
                        total_turnover += sell_value

                holdings = {}

                # Buy new allocation
                for symbol, weight in weights.items():
                    if symbol in close_prices and weight > 0:
                        if date in close_prices[symbol].index:
                            price = close_prices[symbol].loc[date]
                            if price > 0:
                                alloc_value = portfolio_value * weight
                                shares = int(alloc_value / price)
                                cost = shares * price * (1 + self.commission_pct + self.slippage_pct)
                                if cost <= cash:
                                    holdings[symbol] = shares
                                    cash -= cost
                                    total_turnover += cost

                last_rebalance = date
                rebalance_count += 1

        # Calculate metrics
        equity_series = pd.Series(equity, index=all_dates[:len(equity)])
        daily_returns = equity_series.pct_change().dropna()

        total_return = (equity[-1] - capital) / capital
        n_days = len(equity) - 1
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        volatility = daily_returns.std() * np.sqrt(252)

        rf_daily = self.risk_free_rate / 252
        excess_returns = daily_returns - rf_daily
        sharpe = excess_returns.mean() / (excess_returns.std() + 1e-10) * np.sqrt(252)

        neg_returns = daily_returns[daily_returns < 0]
        downside_std = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else 1e-10
        sortino = (annual_return - self.risk_free_rate) / downside_std

        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_dd = abs(drawdown.min())
        calmar = annual_return / max_dd if max_dd > 0 else 0

        in_dd = drawdown < 0
        dd_periods = (in_dd != in_dd.shift()).cumsum()
        dd_durations = in_dd.groupby(dd_periods).sum()
        max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

        winning_days = (daily_returns > 0).sum()
        win_rate = winning_days / len(daily_returns) if len(daily_returns) > 0 else 0

        var_95 = daily_returns.quantile(0.05)
        cvar_95 = daily_returns[daily_returns <= var_95].mean() if (daily_returns <= var_95).sum() > 0 else var_95

        info_ratio = 0
        if benchmark is not None:
            bench_ret = benchmark['close'].pct_change().reindex(daily_returns.index).dropna()
            if len(bench_ret) > 0:
                active_ret = daily_returns.reindex(bench_ret.index) - bench_ret
                info_ratio = active_ret.mean() / (active_ret.std() + 1e-10) * np.sqrt(252)

        turnover = total_turnover / (capital * max(rebalance_count, 1))

        logger.success(f"Backtest complete: Return={total_return:.2%}, Sharpe={sharpe:.2f}")

        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            volatility=volatility,
            win_rate=win_rate,
            information_ratio=info_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=rebalance_count * len(weights),
            turnover=turnover,
            final_value=equity[-1],
            initial_value=capital,
            rebalance_count=rebalance_count,
            equity_curve=equity_series,
            drawdown_curve=drawdown
        )
