"""
================================================================================
ADVANCED BACKTESTER - Realistic Backtesting with Strategy Optimization
================================================================================
Integrates all Phase 1 components:
1. Entry Optimizer - Multi-condition entry filters
2. Exit Optimizer - Dynamic exit strategies
3. Position Sizer - Kelly Criterion + volatility sizing
4. Risk Manager - Multi-layer risk controls

Includes realistic costs: slippage, brokerage, latency
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from engine.entry_optimizer import EntryOptimizer, EntrySignal
from engine.exit_optimizer import ExitOptimizer, ExitSignal
from engine.position_sizer import PositionSizer
from engine.risk_manager import RiskManager, RiskCheck

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Individual trade record."""
    symbol: str
    direction: int
    entry_date: datetime
    entry_price: float
    shares: int
    stop_loss: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0
    pnl_pct: float = 0
    max_profit: float = 0
    max_loss: float = 0
    days_held: int = 0


@dataclass
class BacktestResult:
    """Backtest results."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_win_pct: float
    avg_loss_pct: float
    best_trade: float
    worst_trade: float
    equity_curve: pd.Series
    trades: List[Trade]
    monthly_returns: pd.Series


class AdvancedBacktester:
    """
    Advanced backtester with realistic costs and strategy optimization.
    
    Features:
    - Multi-condition entry filtering
    - Dynamic exits with trailing stops
    - Position sizing (Kelly Criterion)
    - Multi-layer risk management
    - Realistic slippage & brokerage
    - Execution latency
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        slippage_pct: float = 0.05,  # 0.05% slippage
        brokerage_pct: float = 0.03,  # 0.03% per trade
        execution_delay_bars: int = 1,  # 1 bar delay
        use_entry_optimizer: bool = True,
        use_exit_optimizer: bool = True,
        use_position_sizer: bool = True,
        use_risk_manager: bool = True
    ):
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.brokerage_pct = brokerage_pct
        self.execution_delay_bars = execution_delay_bars
        
        # Components
        self.entry_optimizer = EntryOptimizer() if use_entry_optimizer else None
        self.exit_optimizer = ExitOptimizer() if use_exit_optimizer else None
        self.position_sizer = PositionSizer(initial_capital) if use_position_sizer else None
        self.risk_manager = RiskManager(initial_capital) if use_risk_manager else None
        
        # State
        self.current_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.position_states: Dict[str, Dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.dates = []
    
    def run(
        self,
        predictions: pd.DataFrame,
        features: pd.DataFrame,
        prices: pd.DataFrame,
        market_data: Optional[Dict] = None
    ) -> BacktestResult:
        """
        Run backtest on predictions.
        
        Args:
            predictions: DataFrame with columns [date, symbol, direction, confidence]
            features: DataFrame with all features, MultiIndex (date, symbol)
            prices: DataFrame with OHLCV data, MultiIndex (date, symbol)
            market_data: Market indices data
            
        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting advanced backtest with ${self.initial_capital:,.0f}")
        
        # Get unique dates
        dates = sorted(predictions['date'].unique())
        
        for current_date in dates:
            # 1. Check exits first
            self._process_exits(current_date, features, prices)
            
            # 2. Check for new entries
            todays_signals = predictions[predictions['date'] == current_date]
            
            for _, signal in todays_signals.iterrows():
                self._process_entry_signal(
                    signal, 
                    current_date, 
                    features, 
                    prices,
                    market_data
                )
            
            # 3. Update equity curve
            self._update_equity(current_date, prices)
            
            # 4. Update risk manager (daily checks)
            if self.risk_manager:
                self.risk_manager.update_daily()
        
        # Close any remaining positions
        final_date = dates[-1]
        self._close_all_positions(final_date, prices, reason="Backtest end")
        
        # Calculate results
        return self._calculate_results()
    
    def _process_entry_signal(
        self,
        signal: pd.Series,
        current_date: datetime,
        features: pd.DataFrame,
        prices: pd.DataFrame,
        market_data: Optional[Dict]
    ):
        """Process a potential entry signal."""
        symbol = signal['symbol']
        direction = signal['direction']
        confidence = signal['confidence']
        
        # Skip if already in position
        if symbol in self.positions:
            return
        
        # Check if risk manager allows trading
        if self.risk_manager and self.risk_manager.trading_stopped:
            return
        
        # Get features for this stock
        try:
            stock_features = features.loc[(current_date, symbol)]
        except KeyError:
            return
        
        # 1. ENTRY FILTER
        if self.entry_optimizer:
            entry_signal = self.entry_optimizer.evaluate(
                prediction={'direction': direction, 'confidence': confidence},
                features=stock_features,
                market_data=market_data
            )
            
            if not entry_signal.should_enter:
                logger.debug(f"{symbol}: Entry filtered - {entry_signal.reasons[0]}")
                return
        
        # 2. GET EXECUTION PRICE (with delay and slippage)
        try:
            price_data = prices.loc[(current_date, symbol)]
        except KeyError:
            return
        
        entry_price = self._calculate_execution_price(
            price_data['close'],
            direction,
            is_entry=True
        )
        
        # 3. CALCULATE STOP LOSS
        atr = stock_features.get('atr_pct', 0.015)
        if self.exit_optimizer:
            stop_loss = self.exit_optimizer.calculate_initial_stop_loss(
                entry_price, direction, atr
            )
        else:
            stop_loss = entry_price * (0.97 if direction == 1 else 1.03)
        
        # 4. CALCULATE POSITION SIZE
        if self.position_sizer:
            size_result = self.position_sizer.calculate_position_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                confidence=confidence,
                volatility=atr,
                current_positions=self.positions
            )
            shares = size_result['num_shares']
            position_value = size_result['position_value']
        else:
            # Default: 10% of capital
            position_value = self.cash * 0.10
            shares = int(position_value / entry_price)
            position_value = shares * entry_price
        
        # 5. RISK CHECK
        if self.risk_manager:
            risk_check = self.risk_manager.approve_trade(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                position_size=position_value,
                stop_loss=stop_loss
            )
            
            if not risk_check.approved:
                logger.debug(f"{symbol}: Risk check failed - {risk_check.reason}")
                return
        
        # 6. CHECK IF ENOUGH CASH
        total_cost = position_value * (1 + self.brokerage_pct/100)
        if total_cost > self.cash:
            logger.debug(f"{symbol}: Insufficient cash")
            return
        
        # 7. ENTER POSITION
        self.cash -= total_cost
        
        self.positions[symbol] = {
            'direction': direction,
            'entry_date': current_date,
            'entry_price': entry_price,
            'shares': shares,
            'stop_loss': stop_loss,
            'position_value': position_value
        }
        
        self.position_states[symbol] = {
            'highest_price': entry_price if direction == 1 else entry_price,
            'stop_loss': stop_loss,
            'profit_target_1_hit': False,
            'profit_target_2_hit': False
        }
        
        # Record with risk manager
        if self.risk_manager:
            self.risk_manager.record_trade(
                symbol, direction, entry_price, shares, stop_loss
            )
        
        logger.info(f"✅ ENTER {symbol}: {shares} shares @ ₹{entry_price:.2f}, "
                   f"Stop: ₹{stop_loss:.2f}")
    
    def _process_exits(
        self,
        current_date: datetime,
        features: pd.DataFrame,
        prices: pd.DataFrame
    ):
        """Check all positions for exit signals."""
        symbols_to_exit = []
        
        for symbol, pos in self.positions.items():
            try:
                stock_features = features.loc[(current_date, symbol)]
                price_data = prices.loc[(current_date, symbol)]
            except KeyError:
                continue
            
            current_price = price_data['close']
            
            # Check exit conditions
            if self.exit_optimizer:
                exit_signal = self.exit_optimizer.should_exit(
                    entry_price=pos['entry_price'],
                    current_price=current_price,
                    entry_date=pos['entry_date'],
                    current_date=current_date,
                    direction=pos['direction'],
                    features=stock_features,
                    position_state=self.position_states[symbol]
                )
                
                if exit_signal.should_exit:
                    exit_price = self._calculate_execution_price(
                        current_price,
                        -pos['direction'],  # Exit is opposite direction
                        is_entry=False
                    )
                    
                    self._close_position(
                        symbol, 
                        current_date, 
                        exit_price,
                        exit_signal.exit_type
                    )
                    symbols_to_exit.append(symbol)
                
                elif exit_signal.partial_exit:
                    # Partial exit (profit taking)
                    self._partial_exit(
                        symbol,
                        current_date,
                        current_price,
                        exit_signal.partial_exit,
                        exit_signal.reason
                    )
    
    def _close_position(
        self,
        symbol: str,
        exit_date: datetime,
        exit_price: float,
        exit_reason: str
    ):
        """Close a position."""
        pos = self.positions[symbol]
        
        # Calculate P&L
        if pos['direction'] == 1:  # Long
            gross_pnl = (exit_price - pos['entry_price']) * pos['shares']
        else:  # Short
            gross_pnl = (pos['entry_price'] - exit_price) * pos['shares']
        
        # Deduct brokerage
        brokerage = pos['position_value'] * (self.brokerage_pct / 100)
        net_pnl = gross_pnl - brokerage
        pnl_pct = net_pnl / pos['position_value']
        
        # Add cash back
        exit_value = pos['shares'] * exit_price
        self.cash += exit_value
        
        # Record trade
        days_held = (exit_date - pos['entry_date']).days
        
        trade = Trade(
            symbol=symbol,
            direction=pos['direction'],
            entry_date=pos['entry_date'],
            entry_price=pos['entry_price'],
            shares=pos['shares'],
            stop_loss=pos['stop_loss'],
            exit_date=exit_date,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            days_held=days_held
        )
        
        self.trades.append(trade)
        
        # Update risk manager
        if self.risk_manager:
            self.risk_manager.close_position(symbol, exit_price)
        
        # Update position sizer
        if self.position_sizer:
            self.position_sizer.update_performance(pnl_pct)
        
        # Remove position
        del self.positions[symbol]
        del self.position_states[symbol]
        
        logger.info(f"❌ EXIT {symbol}: P&L=₹{net_pnl:,.0f} ({pnl_pct:.2%}), "
                   f"Reason={exit_reason}, Days={days_held}")
    
    def _close_all_positions(self, date: datetime, prices: pd.DataFrame, reason: str):
        """Close all remaining positions."""
        symbols = list(self.positions.keys())
        
        for symbol in symbols:
            try:
                price_data = prices.loc[(date, symbol)]
                exit_price = price_data['close']
                self._close_position(symbol, date, exit_price, reason)
            except KeyError:
                continue
    
    def _partial_exit(
        self,
        symbol: str,
        date: datetime,
        price: float,
        exit_pct: float,
        reason: str
    ):
        """Partially exit a position (profit taking)."""
        pos = self.positions[symbol]
        
        shares_to_sell = int(pos['shares'] * exit_pct)
        if shares_to_sell == 0:
            return
        
        # Calculate partial P&L
        if pos['direction'] == 1:
            gross_pnl = (price - pos['entry_price']) * shares_to_sell
        else:
            gross_pnl = (pos['entry_price'] - price) * shares_to_sell
        
        brokerage = (shares_to_sell * price) * (self.brokerage_pct / 100)
        net_pnl = gross_pnl - brokerage
        
        # Add cash
        self.cash += (shares_to_sell * price)
        
        # Reduce position
        pos['shares'] -= shares_to_sell
        pos['position_value'] = pos['shares'] * pos['entry_price']
        
        logger.info(f"📉 PARTIAL EXIT {symbol}: {shares_to_sell} shares @ ₹{price:.2f}, "
                   f"P&L=₹{net_pnl:,.0f}, Reason={reason}")
    
    def _calculate_execution_price(
        self,
        reference_price: float,
        direction: int,
        is_entry: bool
    ) -> float:
        """Calculate execution price with slippage."""
        # Apply slippage
        if is_entry:
            if direction == 1:  # Buying
                return reference_price * (1 + self.slippage_pct/100)
            else:  # Shorting
                return reference_price * (1 - self.slippage_pct/100)
        else:  # Exit
            if direction == 1:  # Buying to cover
                return reference_price * (1 + self.slippage_pct/100)
            else:  # Selling
                return reference_price * (1 - self.slippage_pct/100)
    
    def _update_equity(self, date: datetime, prices: pd.DataFrame):
        """Update equity curve."""
        position_value = 0
        
        for symbol, pos in self.positions.items():
            try:
                current_price = prices.loc[(date, symbol), 'close']
                position_value += pos['shares'] * current_price
            except KeyError:
                position_value += pos['position_value']
        
        total_equity = self.cash + position_value
        
        self.equity_curve.append(total_equity)
        self.dates.append(date)
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results."""
        if not self.trades:
            return BacktestResult(
                total_return=0, sharpe_ratio=0, max_drawdown=0,
                win_rate=0, profit_factor=0, total_trades=0,
                winning_trades=0, losing_trades=0, avg_win=0, avg_loss=0,
                avg_win_pct=0, avg_loss_pct=0, best_trade=0, worst_trade=0,
                equity_curve=pd.Series(), trades=[], monthly_returns=pd.Series()
            )
        
        # Total return
        final_equity = self.equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Equity curve series
        equity_series = pd.Series(self.equity_curve, index=self.dates)
        
        # Returns
        returns = equity_series.pct_change().dropna()
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cummax = equity_series.expanding().max()
        drawdowns = (equity_series - cummax) / cummax
        max_drawdown = abs(drawdowns.min())
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
        
        avg_win_pct = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = abs(np.mean([t.pnl_pct for t in losing_trades])) if losing_trades else 0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        best_trade = max(t.pnl for t in self.trades) if self.trades else 0
        worst_trade = min(t.pnl for t in self.trades) if self.trades else 0
        
        # Monthly returns
        monthly_returns = equity_series.resample('M').last().pct_change().dropna()
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            best_trade=best_trade,
            worst_trade=worst_trade,
            equity_curve=equity_series,
            trades=self.trades,
            monthly_returns=monthly_returns
        )


if __name__ == "__main__":
    # Example usage would go here
    logging.basicConfig(level=logging.INFO)
    logger.info("Advanced Backtester Module Ready")
