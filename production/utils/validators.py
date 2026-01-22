"""
================================================================================
DATA AND SIGNAL VALIDATORS
================================================================================
Validate data quality, signals, and trading conditions.
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    data_quality_score: float = 1.0
    
    def __bool__(self):
        return self.is_valid


class DataValidator:
    """
    Validate data quality for trading.
    """
    
    def __init__(
        self,
        min_rows: int = 100,
        max_missing_pct: float = 0.05,
        max_gap_days: int = 5
    ):
        self.min_rows = min_rows
        self.max_missing_pct = max_missing_pct
        self.max_gap_days = max_gap_days
        self.logger = logging.getLogger(__name__)
    
    def validate(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> ValidationResult:
        """
        Validate a dataframe for trading.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol for logging
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        quality_score = 1.0
        
        # Check if dataframe exists and has data
        if df is None or df.empty:
            return ValidationResult(
                is_valid=False,
                errors=[f"No data for {symbol}"],
                warnings=[],
                data_quality_score=0.0
            )
        
        # Check minimum rows
        if len(df) < self.min_rows:
            errors.append(f"Insufficient data: {len(df)} rows, need {self.min_rows}")
            quality_score *= 0.5
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
            quality_score *= 0.5
        
        # Check for missing values
        if not missing_cols:
            total_cells = len(df) * len(required_cols)
            missing_cells = df[required_cols].isna().sum().sum()
            missing_pct = missing_cells / total_cells if total_cells > 0 else 0
            
            if missing_pct > self.max_missing_pct:
                errors.append(f"Too many missing values: {missing_pct:.1%}")
                quality_score *= (1 - missing_pct)
            elif missing_pct > 0:
                warnings.append(f"Some missing values: {missing_pct:.1%}")
        
        # Check for data gaps
        if isinstance(df.index, pd.DatetimeIndex):
            date_diffs = df.index.to_series().diff()
            # Exclude weekends (2-3 days gap is normal)
            large_gaps = date_diffs[date_diffs > pd.Timedelta(days=self.max_gap_days)]
            
            if len(large_gaps) > 0:
                warnings.append(f"Found {len(large_gaps)} gaps > {self.max_gap_days} days")
                quality_score *= 0.9
        
        # Check for price anomalies
        if 'Close' in df.columns:
            # Check for zero prices
            zero_prices = (df['Close'] <= 0).sum()
            if zero_prices > 0:
                errors.append(f"Found {zero_prices} zero/negative prices")
                quality_score *= 0.5
            
            # Check for extreme daily moves (>50%)
            returns = df['Close'].pct_change()
            extreme_moves = (abs(returns) > 0.5).sum()
            if extreme_moves > 0:
                warnings.append(f"Found {extreme_moves} extreme moves (>50%)")
        
        # Check for volume anomalies
        if 'Volume' in df.columns:
            zero_volume = (df['Volume'] == 0).sum()
            zero_pct = zero_volume / len(df)
            if zero_pct > 0.1:
                warnings.append(f"High zero volume days: {zero_pct:.1%}")
                quality_score *= 0.95
        
        # Check OHLC relationship
        if all(c in df.columns for c in ['Open', 'High', 'Low', 'Close']):
            invalid_ohlc = (
                (df['High'] < df['Low']) |
                (df['High'] < df['Open']) |
                (df['High'] < df['Close']) |
                (df['Low'] > df['Open']) |
                (df['Low'] > df['Close'])
            ).sum()
            
            if invalid_ohlc > 0:
                warnings.append(f"Found {invalid_ohlc} invalid OHLC rows")
                quality_score *= 0.9
        
        is_valid = len(errors) == 0
        
        self.logger.info(
            f"Validation {symbol}: valid={is_valid}, "
            f"quality={quality_score:.2f}, "
            f"errors={len(errors)}, warnings={len(warnings)}"
        )
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            data_quality_score=quality_score
        )


class SignalValidator:
    """
    Validate trading signals before execution.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.55,
        max_position_size_pct: float = 0.2,
        min_volume_ratio: float = 1.0,
        max_volatility_threshold: float = 0.05
    ):
        self.min_confidence = min_confidence
        self.max_position_size_pct = max_position_size_pct
        self.min_volume_ratio = min_volume_ratio
        self.max_volatility_threshold = max_volatility_threshold
        self.logger = logging.getLogger(__name__)
    
    def validate_signal(
        self,
        signal: Dict,
        market_data: pd.DataFrame = None,
        portfolio_value: float = None
    ) -> ValidationResult:
        """
        Validate a trading signal.
        
        Args:
            signal: Signal dictionary with keys like 'direction', 'confidence', 'symbol'
            market_data: Recent market data for context
            portfolio_value: Current portfolio value
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['symbol', 'direction', 'confidence']
        for field in required_fields:
            if field not in signal:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Check confidence
        confidence = signal.get('confidence', 0)
        if confidence < self.min_confidence:
            errors.append(
                f"Confidence too low: {confidence:.2%} < {self.min_confidence:.2%}"
            )
        
        # Check direction
        direction = signal.get('direction')
        if direction not in [-1, 0, 1]:
            errors.append(f"Invalid direction: {direction}")
        
        # Check position size if provided
        if 'position_size' in signal and portfolio_value:
            size_pct = signal['position_size'] / portfolio_value
            if size_pct > self.max_position_size_pct:
                warnings.append(
                    f"Large position: {size_pct:.1%} > {self.max_position_size_pct:.1%}"
                )
        
        # Market context validation
        if market_data is not None and len(market_data) > 0:
            # Volume check
            if 'Volume' in market_data.columns:
                recent_vol = market_data['Volume'].iloc[-1]
                avg_vol = market_data['Volume'].rolling(20).mean().iloc[-1]
                
                if avg_vol > 0:
                    vol_ratio = recent_vol / avg_vol
                    if vol_ratio < self.min_volume_ratio:
                        warnings.append(f"Low volume: {vol_ratio:.2f}x average")
            
            # Volatility check
            if 'Close' in market_data.columns:
                returns = market_data['Close'].pct_change()
                volatility = returns.std()
                
                if volatility > self.max_volatility_threshold:
                    warnings.append(f"High volatility: {volatility:.2%}")
        
        is_valid = len(errors) == 0
        
        self.logger.info(
            f"Signal validation {signal.get('symbol')}: "
            f"valid={is_valid}, confidence={confidence:.2%}"
        )
        
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


class TradeValidator:
    """
    Validate trades before execution.
    """
    
    def __init__(
        self,
        max_daily_trades: int = 10,
        max_daily_loss_pct: float = 0.05,
        min_trade_interval_seconds: int = 60
    ):
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss_pct = max_daily_loss_pct
        self.min_trade_interval_seconds = min_trade_interval_seconds
        
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        self.trade_date = None
        self.logger = logging.getLogger(__name__)
    
    def reset_daily(self):
        """Reset daily counters."""
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.trade_date = datetime.now().date()
    
    def record_trade(self, pnl: float):
        """Record a completed trade."""
        today = datetime.now().date()
        if self.trade_date != today:
            self.reset_daily()
        
        self.daily_trades += 1
        self.daily_pnl += pnl
        self.last_trade_time = datetime.now()
    
    def validate_trade(
        self,
        symbol: str,
        quantity: int,
        price: float,
        portfolio_value: float
    ) -> ValidationResult:
        """
        Validate if a trade can be executed.
        """
        errors = []
        warnings = []
        
        today = datetime.now().date()
        if self.trade_date != today:
            self.reset_daily()
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            errors.append(f"Daily trade limit reached: {self.daily_trades}")
        
        # Check daily loss limit
        if portfolio_value > 0:
            daily_loss_pct = abs(min(0, self.daily_pnl)) / portfolio_value
            if daily_loss_pct >= self.max_daily_loss_pct:
                errors.append(f"Daily loss limit reached: {daily_loss_pct:.2%}")
        
        # Check trade interval
        if self.last_trade_time:
            seconds_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if seconds_since_last < self.min_trade_interval_seconds:
                warnings.append(
                    f"Trading too frequently: {seconds_since_last:.0f}s since last trade"
                )
        
        # Check quantity
        if quantity <= 0:
            errors.append(f"Invalid quantity: {quantity}")
        
        # Check price
        if price <= 0:
            errors.append(f"Invalid price: {price}")
        
        # Check trade value vs portfolio
        if portfolio_value > 0 and price > 0:
            trade_value = quantity * price
            trade_pct = trade_value / portfolio_value
            
            if trade_pct > 0.5:
                errors.append(f"Trade too large: {trade_pct:.1%} of portfolio")
            elif trade_pct > 0.25:
                warnings.append(f"Large trade: {trade_pct:.1%} of portfolio")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def validate_market_hours(check_time: datetime = None) -> bool:
    """
    Check if currently within market hours (IST).
    NSE: 9:15 AM - 3:30 PM IST
    """
    if check_time is None:
        check_time = datetime.now()
    
    # Check weekday (0=Monday, 6=Sunday)
    if check_time.weekday() >= 5:
        return False
    
    # Check time (assuming IST)
    market_open = check_time.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = check_time.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= check_time <= market_close


def validate_position_limits(
    current_positions: Dict[str, int],
    max_positions: int = 10,
    max_per_symbol: int = 1
) -> ValidationResult:
    """
    Validate current position limits.
    """
    errors = []
    warnings = []
    
    total_positions = len([p for p in current_positions.values() if p != 0])
    
    if total_positions >= max_positions:
        errors.append(f"Max positions reached: {total_positions}/{max_positions}")
    elif total_positions >= max_positions * 0.8:
        warnings.append(f"Near position limit: {total_positions}/{max_positions}")
    
    # Check per-symbol limits
    for symbol, qty in current_positions.items():
        if abs(qty) > max_per_symbol:
            warnings.append(f"Large position in {symbol}: {qty}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
