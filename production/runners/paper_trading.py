"""
================================================================================
PAPER TRADING RUNNER
================================================================================
Run paper trading with performance tracking and optimization analysis.
================================================================================
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from production.core.data_loader import DataLoader
from production.core.feature_engine import AdvancedFeatureEngine as FeatureEngine
from production.broker import PaperBroker
from production.signals import SignalGenerator
from production.utils.logger import setup_logger
from production.utils.metrics import PerformanceMetrics, PerformanceTracker
from production.utils.validators import DataValidator, SignalValidator


class PaperTradingRunner:
    """
    Paper trading runner with comprehensive tracking and analysis.
    """
    
    DEFAULT_SYMBOLS = [
        'HDFCBANK', 'ICICIBANK', 'TCS', 'INFY', 'RELIANCE',
        'SBIN', 'KOTAKBANK', 'WIPRO', 'SUNPHARMA', 'LT'
    ]
    
    def __init__(
        self,
        symbols: List[str] = None,
        initial_capital: float = 100000,
        output_dir: str = None
    ):
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.initial_capital = initial_capital
        self.output_dir = Path(output_dir or "production_results/paper_trading")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger("paper_trading", self.output_dir / "logs")
        
        # Initialize components
        self.data_loader = DataLoader()
        self.feature_engine = FeatureEngine()
        self.broker = PaperBroker(initial_capital=initial_capital)
        self.signal_generator = None  # Loaded per symbol
        
        # Validators
        self.data_validator = DataValidator()
        self.signal_validator = SignalValidator(min_confidence=0.60)
        
        # Performance tracking
        self.tracker = PerformanceTracker(initial_capital)
        self.signals_log = []
        self.trades_log = []
        
        self.logger.info(f"Initialized paper trading for {len(self.symbols)} symbols")
    
    def load_models(self, symbol: str) -> bool:
        """Load models for a symbol."""
        try:
            model_dir = Path("models/fast")
            
            # Try to load signal generator with existing models
            self.signal_generator = SignalGenerator(
                model_dir=str(model_dir),
                min_confidence=0.60,
                volume_threshold=1.2,
                use_regime_aware=True
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models for {symbol}: {e}")
            return False
    
    def run_day(self, date: datetime = None) -> Dict:
        """
        Run paper trading for a single day.
        
        Returns:
            Dictionary with day's results
        """
        date = date or datetime.now()
        self.logger.info(f"Running paper trading for {date.date()}")
        
        day_signals = []
        day_trades = []
        
        for symbol in self.symbols:
            try:
                # Get data (60 trading days for feature calculation) - force fresh download
                df = self.data_loader.download_stock(
                    symbol, 
                    days=60,
                    force_download=True
                )
                
                # Validate data
                validation = self.data_validator.validate(df, symbol)
                if not validation.is_valid:
                    self.logger.warning(f"Data validation failed for {symbol}: {validation.errors}")
                    continue
                
                # Compute features
                feature_result = self.feature_engine.compute_all_features(df, symbol=symbol)
                
                # Handle FeatureSet object
                if hasattr(feature_result, 'df'):
                    features_df = feature_result.df
                else:
                    features_df = feature_result
                
                if features_df is None or features_df.empty:
                    continue
                
                # Generate signal
                if not self.load_models(symbol):
                    continue
                
                signal = self.signal_generator.generate_signal(
                    features_df,
                    symbol=symbol,
                    current_position=self.broker.positions.get(symbol, 0)
                )
                
                # Validate signal
                signal_validation = self.signal_validator.validate_signal(
                    signal,
                    market_data=df,
                    portfolio_value=self.broker.portfolio_value
                )
                
                if signal and signal.get('direction', 0) != 0:
                    signal['timestamp'] = datetime.now()
                    signal['symbol'] = symbol
                    signal['validated'] = signal_validation.is_valid
                    signal['validation_errors'] = signal_validation.errors
                    
                    day_signals.append(signal)
                    self.signals_log.append(signal)
                    
                    # Execute if valid
                    if signal_validation.is_valid:
                        trade = self._execute_signal(signal, df)
                        if trade:
                            day_trades.append(trade)
                            self.trades_log.append(trade)
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Update tracker
        self.tracker.update(self.broker.portfolio_value)
        
        # Save day's results
        self._save_day_results(date, day_signals, day_trades)
        
        return {
            'date': date,
            'signals': len(day_signals),
            'trades': len(day_trades),
            'portfolio_value': self.broker.portfolio_value,
            'positions': dict(self.broker.positions)
        }
    
    def _execute_signal(self, signal: Dict, market_data: pd.DataFrame) -> Optional[Dict]:
        """Execute a trading signal."""
        try:
            symbol = signal['symbol']
            direction = signal['direction']
            confidence = signal['confidence']
            
            current_price = market_data['Close'].iloc[-1]
            
            # Calculate position size
            max_position_value = self.broker.portfolio_value * 0.1  # Max 10% per position
            position_size = int(max_position_value / current_price)
            
            # Adjust for confidence
            position_size = int(position_size * confidence)
            
            if position_size < 1:
                return None
            
            # Execute trade
            if direction == 1:  # Buy
                success = self.broker.buy(symbol, position_size, current_price)
            elif direction == -1:  # Sell/Short
                current_pos = self.broker.positions.get(symbol, 0)
                if current_pos > 0:
                    success = self.broker.sell(symbol, min(position_size, current_pos), current_price)
                else:
                    success = False  # No shorting in paper trading for now
            else:
                return None
            
            if success:
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'direction': 'BUY' if direction == 1 else 'SELL',
                    'quantity': position_size,
                    'price': current_price,
                    'confidence': confidence,
                    'portfolio_value': self.broker.portfolio_value
                }
                self.logger.info(f"Executed trade: {trade}")
                return trade
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to execute signal: {e}")
            return None
    
    def _save_day_results(self, date: datetime, signals: List, trades: List):
        """Save day's results to files."""
        date_str = date.strftime("%Y-%m-%d")
        day_dir = self.output_dir / date_str
        day_dir.mkdir(exist_ok=True)
        
        # Save signals
        if signals:
            signals_df = pd.DataFrame(signals)
            signals_df.to_csv(day_dir / "signals.csv", index=False)
        
        # Save trades
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(day_dir / "trades.csv", index=False)
        
        # Save portfolio state
        portfolio_state = {
            'date': date_str,
            'cash': self.broker.cash,
            'portfolio_value': self.broker.portfolio_value,
            'positions': self.broker.positions
        }
        with open(day_dir / "portfolio.json", 'w') as f:
            json.dump(portfolio_state, f, indent=2, default=str)
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        metrics = self.tracker.get_metrics()
        
        return {
            'metrics': metrics.to_dict(),
            'total_signals': len(self.signals_log),
            'total_trades': len(self.trades_log),
            'signals_by_symbol': self._count_by_symbol(self.signals_log),
            'trades_by_symbol': self._count_by_symbol(self.trades_log),
            'positions': dict(self.broker.positions),
            'cash': self.broker.cash,
            'portfolio_value': self.broker.portfolio_value
        }
    
    def _count_by_symbol(self, records: List[Dict]) -> Dict[str, int]:
        """Count records by symbol."""
        counts = {}
        for record in records:
            symbol = record.get('symbol', 'UNKNOWN')
            counts[symbol] = counts.get(symbol, 0) + 1
        return counts
    
    def run_continuous(self, check_interval_minutes: int = 5):
        """
        Run continuous paper trading during market hours.
        """
        from production.utils.validators import validate_market_hours
        import time
        
        self.logger.info("Starting continuous paper trading...")
        
        while True:
            try:
                if validate_market_hours():
                    self.run_day()
                    self.logger.info(f"Portfolio value: ₹{self.broker.portfolio_value:,.2f}")
                else:
                    self.logger.info("Outside market hours, waiting...")
                
                time.sleep(check_interval_minutes * 60)
                
            except KeyboardInterrupt:
                self.logger.info("Stopping paper trading...")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous trading: {e}")
                time.sleep(60)  # Wait before retrying


def run_paper_trading(
    symbols: List[str] = None,
    capital: float = 100000,
    days: int = 1
):
    """
    Convenience function to run paper trading.
    """
    runner = PaperTradingRunner(symbols=symbols, initial_capital=capital)
    
    for _ in range(days):
        result = runner.run_day()
        print(f"Day result: {result}")
    
    report = runner.get_performance_report()
    print("\n=== Performance Report ===")
    print(json.dumps(report, indent=2, default=str))
    
    return runner


if __name__ == "__main__":
    run_paper_trading()
