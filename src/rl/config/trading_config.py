"""
Trading Configuration - Parameters for Algorithmic Trading
===========================================================

Contains all trading-specific parameters including:
- Capital and position limits
- Risk management rules
- Transaction costs
- Strategy parameters
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import time

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class CapitalConfig:
    """Capital and Position Sizing Configuration"""

    # Initial capital (INR)
    initial_capital: float = 100000  # Rs 1 Lakh

    # Position sizing
    max_positions: int = 2  # Maximum concurrent positions
    position_size_pct: float = 0.40  # 40% per position
    cash_buffer_pct: float = 0.20  # Keep 20% in cash always

    # Per-trade limits
    min_trade_value: float = 5000  # Minimum Rs 5,000 per trade
    max_trade_value: float = 50000  # Maximum Rs 50,000 per trade (50% of capital)

    # Lot size (for F&O if needed)
    use_lot_sizes: bool = False
    default_lot_size: int = 1


@dataclass
class RiskConfig:
    """Risk Management Configuration"""

    # Per-trade risk
    max_loss_per_trade_pct: float = 0.01  # 1% of capital per trade
    stop_loss_pct: float = 0.02  # 2% stop loss from entry
    trailing_stop_pct: float = 0.03  # 3% trailing stop after profit
    trailing_stop_activation_pct: float = 0.05  # Activate trailing after 5% profit

    # Take profit
    take_profit_pct: float = 0.05  # 5% take profit
    partial_exit_pct: float = 0.50  # Exit 50% at first target

    # Daily limits
    max_daily_loss_pct: float = 0.02  # 2% max daily loss
    max_daily_trades: int = 5  # Maximum 5 trades per day

    # Weekly limits
    max_weekly_loss_pct: float = 0.05  # 5% max weekly loss

    # Drawdown limits
    max_drawdown_pct: float = 0.10  # 10% max drawdown - pause trading
    drawdown_recovery_pct: float = 0.05  # Resume after 5% recovery

    # Time-based exits
    max_holding_days: int = 5  # Maximum 5 days hold
    force_exit_before_expiry: bool = True  # For derivatives

    # Correlation limits
    max_correlated_positions: int = 2  # Max positions in same sector


@dataclass
class TransactionConfig:
    """Transaction Costs Configuration"""

    # Brokerage (Angel One rates)
    brokerage_pct: float = 0.0003  # 0.03% or Rs 20 per order (whichever lower)
    brokerage_flat: float = 20.0  # Flat fee if applicable
    use_flat_brokerage: bool = False  # Use percentage

    # STT (Securities Transaction Tax)
    stt_delivery_pct: float = 0.001  # 0.1% on sell side
    stt_intraday_pct: float = 0.00025  # 0.025% on sell side

    # Other charges
    exchange_txn_pct: float = 0.0000345  # NSE charges
    sebi_charges_pct: float = 0.000001  # SEBI charges
    gst_pct: float = 0.18  # 18% GST on brokerage
    stamp_duty_pct: float = 0.00015  # Stamp duty on buy

    # Slippage estimation
    slippage_pct: float = 0.0005  # 0.05% slippage

    # Total estimated cost (for quick calculations)
    @property
    def total_roundtrip_cost_pct(self) -> float:
        """Estimated total cost for round trip trade"""
        return 0.002  # ~0.2% total for delivery, ~0.1% for intraday


@dataclass
class IntradayConfig:
    """Intraday Trading Configuration"""

    # Trading hours
    market_open: time = time(9, 15)
    market_close: time = time(15, 30)
    entry_start: time = time(9, 30)  # Start entries 15 min after open
    entry_end: time = time(14, 0)  # Stop entries by 2 PM
    square_off_time: time = time(15, 15)  # Square off by 3:15 PM

    # Product type
    product_type: str = 'MIS'  # Margin Intraday Square-off

    # Leverage
    leverage: float = 5.0  # 5x leverage for intraday

    # Execution intervals
    check_interval_minutes: int = 5  # Check every 5 minutes

    # Risk adjustments for intraday
    stop_loss_pct: float = 0.01  # Tighter 1% stop loss
    take_profit_pct: float = 0.02  # 2% target (2:1 RR)

    # Volume requirements
    min_volume_multiplier: float = 1.5  # Trade only if volume > 1.5x average


@dataclass
class SwingConfig:
    """Swing Trading Configuration"""

    # Product type
    product_type: str = 'CNC'  # Cash and Carry (Delivery)

    # Holding period
    min_holding_days: int = 2
    max_holding_days: int = 10

    # Entry conditions
    entry_after_close: bool = True  # Place orders after market close for next day

    # Risk adjustments for swing
    stop_loss_pct: float = 0.03  # 3% stop loss
    take_profit_pct: float = 0.06  # 6% target (2:1 RR)
    trailing_stop_pct: float = 0.03  # 3% trailing after 5% profit

    # Partial exits
    partial_profit_pct: float = 0.04  # Book 50% at 4%
    partial_exit_qty_pct: float = 0.50  # Exit 50% quantity

    # Weekend/Holiday handling
    reduce_before_weekend: bool = False  # Reduce exposure before weekends
    close_before_holidays: bool = False  # Close before long holidays


@dataclass
class TradingConfig:
    """Master Trading Configuration"""

    # Sub-configurations
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    transaction: TransactionConfig = field(default_factory=TransactionConfig)
    intraday: IntradayConfig = field(default_factory=IntradayConfig)
    swing: SwingConfig = field(default_factory=SwingConfig)

    # Strategy selection
    default_strategy: str = 'swing'  # 'intraday', 'swing', 'both'

    # Paper trading mode
    paper_trading: bool = True  # Start in paper trading mode

    # Stock universe
    tradeable_stocks: List[str] = field(default_factory=lambda: [
        'HDFCBANK', 'ICICIBANK', 'SBIN', 'TCS', 'INFY',
        'RELIANCE', 'TATASTEEL', 'ITC', 'LT', 'BHARTIARTL'
    ])

    # Sector limits
    max_sector_exposure_pct: float = 0.60  # Max 60% in one sector

    # Predictions directory
    predictions_dir: str = os.path.join(BASE_DIR, 'evaluation_results', 'multi_target')

    # Trading logs
    trade_log_dir: str = os.path.join(BASE_DIR, 'logs', 'trades')

    # Confidence thresholds
    min_direction_confidence: float = 0.60  # Minimum 60% confidence
    min_return_prediction: float = 0.005  # Minimum 0.5% expected return

    def get_strategy_config(self, strategy: str):
        """Get configuration for specific strategy"""
        if strategy.lower() == 'intraday':
            return self.intraday
        elif strategy.lower() == 'swing':
            return self.swing
        else:
            return self.swing  # Default to swing

    def calculate_position_size(self, capital: float, price: float) -> int:
        """Calculate number of shares to buy"""
        position_value = capital * self.capital.position_size_pct
        position_value = min(position_value, self.capital.max_trade_value)
        position_value = max(position_value, self.capital.min_trade_value)

        shares = int(position_value / price)
        return max(1, shares)

    def calculate_stop_loss(self, entry_price: float, strategy: str = 'swing') -> float:
        """Calculate stop loss price"""
        config = self.get_strategy_config(strategy)
        return entry_price * (1 - config.stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, strategy: str = 'swing') -> float:
        """Calculate take profit price"""
        config = self.get_strategy_config(strategy)
        return entry_price * (1 + config.take_profit_pct)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            'initial_capital': self.capital.initial_capital,
            'max_positions': self.capital.max_positions,
            'default_strategy': self.default_strategy,
            'paper_trading': self.paper_trading,
            'max_drawdown': self.risk.max_drawdown_pct
        }


# Global instance
trading_config = TradingConfig()


# Create trade log directory
os.makedirs(trading_config.trade_log_dir, exist_ok=True)
