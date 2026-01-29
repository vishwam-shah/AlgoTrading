"""
================================================================================
CONTINUOUS OPTIMIZER
================================================================================
Parameter optimization and gap analysis for trading strategies.
================================================================================
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from itertools import product

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from production.core.data_loader import DataLoader
from production.core.feature_engine import AdvancedFeatureEngine as FeatureEngine
from production.utils.logger import setup_logger
from production.utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown


class ContinuousOptimizer:
    """
    Continuously optimize trading strategy parameters.
    """
    
    DEFAULT_SYMBOLS = ['HDFCBANK', 'TCS', 'INFY', 'RELIANCE', 'ICICIBANK']
    
    # Parameter grids for optimization
    PARAM_GRIDS = {
        'stop_loss_atr_mult': [1.5, 2.0, 2.5, 3.0],
        'take_profit_atr_mult': [3.0, 3.5, 4.0, 4.5, 5.0],
        'min_confidence': [0.55, 0.60, 0.65, 0.70],
        'volume_threshold': [1.0, 1.2, 1.5, 2.0],
        'position_size_pct': [0.05, 0.10, 0.15, 0.20],
        'trailing_stop_activation': [0.01, 0.02, 0.03],
        'trailing_stop_distance': [0.01, 0.015, 0.02]
    }
    
    def __init__(
        self,
        symbols: List[str] = None,
        output_dir: str = None
    ):
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.output_dir = Path(output_dir or "production_results/optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger("optimizer", self.output_dir / "logs")
        
        # Initialize components
        self.data_loader = DataLoader()
        self.feature_engine = FeatureEngine()
        
        # Results storage
        self.optimization_results = []
        
        self.logger.info(f"Initialized optimizer for {len(self.symbols)} symbols")
    
    def _run_single_backtest(
        self,
        features_df: pd.DataFrame,
        params: Dict
    ) -> Dict:
        """Run a single backtest with given parameters."""
        from production.backtester import WalkForwardBacktester
        
        backtester = WalkForwardBacktester(
            initial_capital=100000,
            position_size_pct=params.get('position_size_pct', 0.1),
            stop_loss_atr_mult=params.get('stop_loss_atr_mult', 2.5),
            take_profit_atr_mult=params.get('take_profit_atr_mult', 4.0),
            use_trailing_stop=params.get('use_trailing_stop', True),
            trailing_stop_activation=params.get('trailing_stop_activation', 0.02),
            trailing_stop_distance=params.get('trailing_stop_distance', 0.015),
            volume_confirmation=params.get('volume_confirmation', True),
            min_confidence=params.get('min_confidence', 0.60)
        )
        
        result = backtester.run_walk_forward(
            features_df,
            train_window=252,
            test_window=21
        )
        
        return result
    
    def optimize_parameter(
        self,
        param_name: str,
        fixed_params: Dict = None
    ) -> Dict:
        """
        Optimize a single parameter.
        
        Args:
            param_name: Name of parameter to optimize
            fixed_params: Fixed values for other parameters
            
        Returns:
            Optimization results
        """
        if param_name not in self.PARAM_GRIDS:
            self.logger.error(f"Unknown parameter: {param_name}")
            return {'error': f'Unknown parameter: {param_name}'}
        
        self.logger.info(f"Optimizing {param_name}")
        
        param_values = self.PARAM_GRIDS[param_name]
        base_params = fixed_params or {}
        
        results = []
        
        # Load data for all symbols
        symbol_data = {}
        for symbol in self.symbols:
            df = self.data_loader.download_stock(symbol, period="2y", interval="1d")
            if df is not None:
                features_df = self.feature_engine.compute_features(df)
                if features_df is not None and len(features_df) > 300:
                    symbol_data[symbol] = features_df
        
        if not symbol_data:
            return {'error': 'No valid data'}
        
        # Test each parameter value
        for value in param_values:
            params = base_params.copy()
            params[param_name] = value
            
            value_results = []
            
            for symbol, features_df in symbol_data.items():
                try:
                    result = self._run_single_backtest(features_df, params)
                    if result and 'total_return' in result:
                        value_results.append({
                            'symbol': symbol,
                            'return': result.get('total_return', 0),
                            'sharpe': result.get('sharpe_ratio', 0),
                            'win_rate': result.get('win_rate', 0),
                            'trades': result.get('total_trades', 0)
                        })
                except Exception as e:
                    self.logger.error(f"Backtest failed for {symbol}: {e}")
            
            if value_results:
                avg_return = np.mean([r['return'] for r in value_results])
                avg_sharpe = np.mean([r['sharpe'] for r in value_results])
                avg_win_rate = np.mean([r['win_rate'] for r in value_results])
                
                results.append({
                    'value': value,
                    'avg_return': avg_return,
                    'avg_sharpe': avg_sharpe,
                    'avg_win_rate': avg_win_rate,
                    'symbols_tested': len(value_results)
                })
        
        # Find best value
        if results:
            best = max(results, key=lambda x: x['avg_sharpe'])
            
            optimization_result = {
                'parameter': param_name,
                'best_value': best['value'],
                'best_sharpe': best['avg_sharpe'],
                'best_return': best['avg_return'],
                'all_results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save results
            self._save_optimization_result(param_name, optimization_result)
            
            return optimization_result
        
        return {'error': 'No valid results'}
    
    def run_full_optimization(self) -> Dict:
        """
        Run full parameter sweep optimization.
        """
        self.logger.info("Starting full optimization")
        
        all_results = {}
        
        for param_name in self.PARAM_GRIDS:
            self.logger.info(f"Optimizing {param_name}...")
            result = self.optimize_parameter(param_name)
            all_results[param_name] = result
        
        # Generate optimal configuration
        optimal_config = {}
        for param_name, result in all_results.items():
            if 'best_value' in result:
                optimal_config[param_name] = result['best_value']
        
        summary = {
            'optimal_config': optimal_config,
            'by_parameter': all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_file = self.output_dir / "optimization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Optimization complete. Optimal config: {optimal_config}")
        
        return summary
    
    def analyze_gaps(self, backtest_results: List[Dict]) -> Dict:
        """
        Analyze gaps and areas for improvement.
        
        Args:
            backtest_results: List of backtest result dictionaries
            
        Returns:
            Gap analysis report
        """
        gaps = {
            'issues': [],
            'recommendations': []
        }
        
        if not backtest_results:
            return gaps
        
        # Analyze win rates
        win_rates = [r.get('win_rate', 0) for r in backtest_results if 'win_rate' in r]
        if win_rates:
            avg_win_rate = np.mean(win_rates)
            if avg_win_rate < 0.45:
                gaps['issues'].append({
                    'type': 'low_win_rate',
                    'value': avg_win_rate,
                    'severity': 'high'
                })
                gaps['recommendations'].append(
                    "Increase confidence threshold or improve signal quality"
                )
        
        # Analyze returns
        returns = [r.get('total_return', 0) for r in backtest_results if 'total_return' in r]
        if returns:
            avg_return = np.mean(returns)
            if avg_return < 0:
                gaps['issues'].append({
                    'type': 'negative_returns',
                    'value': avg_return,
                    'severity': 'critical'
                })
                gaps['recommendations'].append(
                    "Review stop-loss settings and position sizing"
                )
        
        # Analyze drawdowns
        drawdowns = [r.get('max_drawdown', 0) for r in backtest_results if 'max_drawdown' in r]
        if drawdowns:
            avg_drawdown = np.mean([abs(d) for d in drawdowns])
            if avg_drawdown > 0.15:
                gaps['issues'].append({
                    'type': 'high_drawdown',
                    'value': avg_drawdown,
                    'severity': 'medium'
                })
                gaps['recommendations'].append(
                    "Consider tighter risk management or position limits"
                )
        
        # Trade frequency
        trade_counts = [r.get('total_trades', 0) for r in backtest_results if 'total_trades' in r]
        if trade_counts:
            avg_trades = np.mean(trade_counts)
            if avg_trades < 5:
                gaps['issues'].append({
                    'type': 'low_trade_frequency',
                    'value': avg_trades,
                    'severity': 'low'
                })
                gaps['recommendations'].append(
                    "Lower confidence threshold or add more signals"
                )
        
        return gaps
    
    def generate_report(self) -> str:
        """Generate optimization report."""
        report_lines = [
            "=" * 60,
            "OPTIMIZATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            f"Symbols: {', '.join(self.symbols)}",
            "",
        ]
        
        # Load latest optimization results
        summary_file = self.output_dir / "optimization_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            report_lines.append("OPTIMAL CONFIGURATION:")
            report_lines.append("-" * 40)
            for param, value in summary.get('optimal_config', {}).items():
                report_lines.append(f"  {param}: {value}")
            
            report_lines.append("")
            report_lines.append("PARAMETER DETAILS:")
            report_lines.append("-" * 40)
            
            for param, result in summary.get('by_parameter', {}).items():
                if 'best_value' in result:
                    report_lines.append(f"\n{param}:")
                    report_lines.append(f"  Best value: {result['best_value']}")
                    report_lines.append(f"  Best Sharpe: {result.get('best_sharpe', 'N/A'):.3f}")
                    report_lines.append(f"  Best Return: {result.get('best_return', 'N/A')*100:.2f}%")
        
        report = "\n".join(report_lines)
        
        # Save report
        report_file = self.output_dir / "optimization_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        return report
    
    def _save_optimization_result(self, param_name: str, result: Dict):
        """Save individual optimization result."""
        result_file = self.output_dir / f"opt_{param_name}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        self.optimization_results.append(result)


def run_optimization(symbols: List[str] = None):
    """
    Convenience function to run optimization.
    """
    optimizer = ContinuousOptimizer(symbols=symbols)
    summary = optimizer.run_full_optimization()
    
    print("\n" + optimizer.generate_report())
    
    return optimizer


if __name__ == "__main__":
    run_optimization()
