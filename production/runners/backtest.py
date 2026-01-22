"""
================================================================================
BACKTEST RUNNER
================================================================================
Run backtests with comprehensive analysis and reporting.
================================================================================
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from production.orchestrator import TradingOrchestrator
from production.utils.logger import setup_logger
from production.utils.metrics import (
    PerformanceMetrics, 
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_profit_factor
)


class BacktestRunner:
    """
    Run backtests with comprehensive analysis and reporting.
    Uses the TradingOrchestrator for the full pipeline.
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
        self.output_dir = Path(output_dir or "production_results/backtest")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger("backtest", str(self.output_dir / "logs"))
        
        # Results storage
        self.results = {}
        
        self.logger.info(f"Initialized backtest for {len(self.symbols)} symbols")
    
    def run_backtest(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        train_window: int = 252,
        test_window: int = 21
    ) -> Dict:
        """
        Run backtest for a single symbol using the orchestrator.
        """
        self.logger.info(f"Running backtest for {symbol}")
        
        try:
            # Create orchestrator for this symbol
            orchestrator = TradingOrchestrator(
                symbols=[symbol],
                paper_trading=True,
                initial_capital=self.initial_capital
            )
            
            # Run the full pipeline
            # Stage 1: Collect data - force download to get latest data
            orchestrator.collect_data(force_download=True)
            
            # Stage 2: Compute features
            orchestrator.compute_features()
            
            # Stage 3: Train model
            orchestrator.train_model(symbol)
            
            # Stage 4: Run backtest
            result = orchestrator.run_backtest(symbol)
            
            result['symbol'] = symbol
            self.results[symbol] = result
            
            # Save individual result
            self._save_result(symbol, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {'symbol': symbol, 'error': str(e)}
    
    def run_all(
        self,
        train_window: int = 252,
        test_window: int = 21
    ) -> Dict:
        """
        Run backtests for all symbols.
        """
        self.logger.info(f"Running backtests for {len(self.symbols)} symbols")
        
        all_results = []
        
        for symbol in self.symbols:
            result = self.run_backtest(
                symbol,
                train_window=train_window,
                test_window=test_window
            )
            all_results.append(result)
        
        # Generate summary
        summary = self._generate_summary(all_results)
        
        # Save summary
        self._save_summary(summary)
        
        return summary
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics from all results."""
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results'}
        
        summary = {
            'total_symbols': len(results),
            'successful': len(valid_results),
            'failed': len(results) - len(valid_results),
            'timestamp': datetime.now().isoformat()
        }
        
        # Aggregate metrics
        metrics_keys = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades']
        
        for key in metrics_keys:
            values = [r[key] for r in valid_results if key in r]
            if values:
                summary[f'avg_{key}'] = float(np.mean(values))
                summary[f'std_{key}'] = float(np.std(values))
                summary[f'min_{key}'] = float(np.min(values))
                summary[f'max_{key}'] = float(np.max(values))
        
        # Per-symbol summary
        summary['by_symbol'] = {}
        for result in valid_results:
            symbol = result.get('symbol', 'UNKNOWN')
            summary['by_symbol'][symbol] = {k: v for k, v in result.items() if k != 'symbol'}
        
        return summary
    
    def _save_result(self, symbol: str, result: Dict):
        """Save individual backtest result."""
        result_file = self.output_dir / f"{symbol}_backtest.json"
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        self.logger.info(f"Saved result for {symbol}")
    
    def _save_summary(self, summary: Dict):
        """Save backtest summary."""
        summary_file = self.output_dir / "summary.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Also save as CSV for easy viewing
        if 'by_symbol' in summary:
            df = pd.DataFrame(summary['by_symbol']).T
            df.to_csv(self.output_dir / "summary.csv")
        
        self.logger.info("Saved backtest summary")
    
    def print_summary(self):
        """Print summary of backtest results."""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        
        for symbol, result in self.results.items():
            if 'error' in result:
                print(f"\n{symbol}: ERROR - {result['error']}")
                continue
            
            print(f"\n{symbol}:")
            print(f"  Total Return: {result.get('total_return', 0)*100:.2f}%")
            print(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {result.get('max_drawdown', 0)*100:.2f}%")
            print(f"  Win Rate: {result.get('win_rate', 0)*100:.1f}%")
            print(f"  Total Trades: {result.get('total_trades', 0)}")


def run_backtest(
    symbols: List[str] = None,
    capital: float = 100000,
    train_window: int = 252,
    test_window: int = 21
):
    """
    Convenience function to run backtests.
    """
    runner = BacktestRunner(symbols=symbols, initial_capital=capital)
    summary = runner.run_all(train_window=train_window, test_window=test_window)
    runner.print_summary()
    return runner


if __name__ == "__main__":
    run_backtest()
