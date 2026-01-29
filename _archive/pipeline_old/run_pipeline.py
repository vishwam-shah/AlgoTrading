"""
================================================================================
AI STOCK TRADING PIPELINE - MAIN RUNNER
================================================================================

Run the complete 5-step algorithmic trading pipeline:

1. Data Collection       - Fetch OHLCV data with validation  (1_data_collection.py)
2. Feature Engineering   - Compute 82+ technical indicators  (2_feature_engineering.py)
3. Factor Analysis       - 5-factor model scoring            (3_factor_analysis.py)
4. Portfolio Optimization - Risk parity & other methods      (4_portfolio_optimization.py)
5. Backtest Validation   - Performance metrics & validation  (5_backtest_validation.py)

Usage:
    python pipeline/run_pipeline.py                    # Full NIFTY50 backtest
    python pipeline/run_pipeline.py --symbols 10      # Top 10 stocks only
    python pipeline/run_pipeline.py --quick           # Quick test with 5 stocks

================================================================================
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from importlib import import_module

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Import from numbered modules
_step1 = import_module('pipeline.1_data_collection')
_step2 = import_module('pipeline.2_feature_engineering')
_step3 = import_module('pipeline.3_factor_analysis')
_step4 = import_module('pipeline.4_portfolio_optimization')
_step5 = import_module('pipeline.5_backtest_validation')

DataCollector = _step1.DataCollector
FeatureEngineer = _step2.FeatureEngineer
FactorAnalyzer = _step3.FactorAnalyzer
PortfolioOptimizer = _step4.PortfolioOptimizer
BacktestValidator = _step5.BacktestValidator


def run_full_pipeline(
    symbols: list = None,
    start_date: str = '2022-01-01',
    end_date: str = None,
    n_holdings: int = 15,
    initial_capital: float = 1000000,
    optimization_method: str = 'risk_parity',
    output_dir: str = None,
    verbose: bool = True
):
    """
    Run the complete 5-step pipeline.
    
    Args:
        symbols: List of stock symbols (defaults to ALL_STOCKS from config)
        start_date: Backtest start date
        end_date: Backtest end date (defaults to today)
        n_holdings: Number of stocks in portfolio
        initial_capital: Starting capital
        optimization_method: Portfolio optimization method
        output_dir: Directory to save results
        verbose: Print progress
    """
    
    # Setup
    if symbols is None:
        symbols = config.ALL_STOCKS
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if output_dir is None:
        output_dir = os.path.join(config.RESULTS_DIR, 'pipeline')
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Print header
    if verbose:
        print("\n" + "=" * 80)
        print("AI STOCK TRADING PIPELINE")
        print("=" * 80)
        print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Symbols:    {len(symbols)} stocks")
        print(f"Period:     {start_date} to {end_date}")
        print(f"Holdings:   {n_holdings}")
        print(f"Capital:    Rs {initial_capital:,.0f}")
        print(f"Method:     {optimization_method}")
        print("=" * 80)
    
    results = {
        'config': {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'n_holdings': n_holdings,
            'initial_capital': initial_capital,
            'optimization_method': optimization_method
        },
        'steps': {},
        'errors': []
    }
    
    try:
        # ======================================================================
        # STEP 1: DATA COLLECTION
        # ======================================================================
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 1/5: DATA COLLECTION")
            print("=" * 60)
        
        collector = DataCollector()
        price_data, market_data = collector.collect_all(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        results['steps']['data_collection'] = {
            'symbols_collected': len(price_data),
            'market_data_collected': len(market_data),
            'quality_report': collector.get_quality_report()
        }
        
        if len(price_data) < 5:
            raise ValueError(f"Insufficient data: only {len(price_data)} symbols collected")
        
        # ======================================================================
        # STEP 2: FEATURE ENGINEERING
        # ======================================================================
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 2/5: FEATURE ENGINEERING")
            print("=" * 60)
        
        engineer = FeatureEngineer()
        features = engineer.compute_all_features(price_data, market_data)
        
        results['steps']['feature_engineering'] = {
            'features_computed': engineer.get_feature_count(),
            'symbols_processed': len(features),
            'summary': engineer.get_summary()
        }
        
        # ======================================================================
        # STEP 3: FACTOR ANALYSIS
        # ======================================================================
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 3/5: FACTOR ANALYSIS")
            print("=" * 60)
        
        analyzer = FactorAnalyzer()
        benchmark = market_data.get('NIFTY50')
        factor_scores = analyzer.compute_factors(price_data, features, benchmark)
        
        # Get top stocks
        top_stocks = analyzer.get_top_stocks(factor_scores, n=n_holdings)
        
        results['steps']['factor_analysis'] = {
            'stocks_analyzed': len(factor_scores),
            'factors': ['value', 'momentum', 'quality', 'low_vol', 'sentiment'],
            'weights': analyzer.factor_weights,
            'top_stocks': [
                {
                    'symbol': s.symbol,
                    'combined_score': s.combined_score,
                    'combined_rank': s.combined_rank,
                    'value_score': s.value_score,
                    'momentum_score': s.momentum_score,
                    'quality_score': s.quality_score,
                    'low_vol_score': s.low_vol_score,
                    'sentiment_score': s.sentiment_score
                }
                for s in top_stocks
            ]
        }
        
        # ======================================================================
        # STEP 4: PORTFOLIO OPTIMIZATION
        # ======================================================================
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 4/5: PORTFOLIO OPTIMIZATION")
            print("=" * 60)
        
        optimizer = PortfolioOptimizer()
        allocation = optimizer.optimize(
            price_data=price_data,
            factor_scores=factor_scores,
            n_holdings=n_holdings,
            method=optimization_method,
            sector_map=config.STOCK_SECTOR_MAP
        )
        
        results['steps']['portfolio_optimization'] = {
            'method': optimization_method,
            'expected_return': allocation.expected_return,
            'expected_volatility': allocation.expected_volatility,
            'sharpe_ratio': allocation.sharpe_ratio,
            'n_positions': allocation.n_positions,
            'weights': allocation.weights,
            'risk_contributions': allocation.risk_contributions
        }
        
        # ======================================================================
        # STEP 5: BACKTEST & VALIDATION
        # ======================================================================
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 5/5: BACKTEST & VALIDATION")
            print("=" * 60)
        
        validator = BacktestValidator(
            transaction_cost=0.001,
            slippage=0.0005,
            rebalance_frequency='monthly'
        )
        
        backtest_results = validator.run_backtest(
            price_data=price_data,
            allocation=allocation.weights,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            benchmark=benchmark
        )
        
        results['steps']['backtest'] = backtest_results.to_dict()
        results['success'] = True
        
    except Exception as e:
        results['success'] = False
        results['errors'].append(str(e))
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # ======================================================================
    # FINAL SUMMARY
    # ======================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE - FINAL RESULTS")
        print("=" * 80)
        
        br = backtest_results
        print(f"\n{'PERFORMANCE':-^40}")
        print(f"  Total Return:        {br.total_return:>10.2%}")
        print(f"  Annual Return:       {br.annual_return:>10.2%}")
        print(f"  Benchmark Return:    {br.benchmark_return:>10.2%}")
        print(f"  Excess Return:       {br.excess_return:>10.2%}")
        
        print(f"\n{'RISK':-^40}")
        print(f"  Volatility:          {br.volatility:>10.2%}")
        print(f"  Max Drawdown:        {br.max_drawdown:>10.2%}")
        print(f"  VaR (95%):           {br.var_95:>10.2%}")
        
        print(f"\n{'RISK-ADJUSTED':-^40}")
        print(f"  Sharpe Ratio:        {br.sharpe_ratio:>10.2f}")
        print(f"  Sortino Ratio:       {br.sortino_ratio:>10.2f}")
        print(f"  Calmar Ratio:        {br.calmar_ratio:>10.2f}")
        
        print(f"\n{'TRADING':-^40}")
        print(f"  Total Trades:        {br.n_trades:>10}")
        print(f"  Win Rate:            {br.win_rate:>10.2%}")
        print(f"  Profit Factor:       {br.profit_factor:>10.2f}")
        
        print(f"\n{'CAPITAL':-^40}")
        print(f"  Initial:             Rs {br.initial_capital:>12,.0f}")
        print(f"  Final:               Rs {br.final_value:>12,.0f}")
        print(f"  Profit:              Rs {br.final_value - br.initial_capital:>12,.0f}")
        
        print("\n" + "=" * 80)
        
        # Grade the strategy
        grade = _grade_strategy(br)
        print(f"\nSTRATEGY GRADE: {grade}")
        print("=" * 80)
    
    # Save results
    results_file = os.path.join(output_dir, f'pipeline_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    if verbose:
        print(f"\nResults saved to: {results_file}")
    
    return results


def _grade_strategy(results):
    """Grade the trading strategy based on metrics."""
    score = 0
    
    # Annual return (max 25 points)
    if results.annual_return > 0.25:
        score += 25
    elif results.annual_return > 0.15:
        score += 20
    elif results.annual_return > 0.10:
        score += 15
    elif results.annual_return > 0.05:
        score += 10
    elif results.annual_return > 0:
        score += 5
    
    # Sharpe ratio (max 25 points)
    if results.sharpe_ratio > 2.0:
        score += 25
    elif results.sharpe_ratio > 1.5:
        score += 20
    elif results.sharpe_ratio > 1.0:
        score += 15
    elif results.sharpe_ratio > 0.5:
        score += 10
    elif results.sharpe_ratio > 0:
        score += 5
    
    # Max drawdown (max 25 points)
    if results.max_drawdown < 0.10:
        score += 25
    elif results.max_drawdown < 0.15:
        score += 20
    elif results.max_drawdown < 0.20:
        score += 15
    elif results.max_drawdown < 0.30:
        score += 10
    elif results.max_drawdown < 0.50:
        score += 5
    
    # Win rate (max 25 points)
    if results.win_rate > 0.60:
        score += 25
    elif results.win_rate > 0.55:
        score += 20
    elif results.win_rate > 0.50:
        score += 15
    elif results.win_rate > 0.45:
        score += 10
    elif results.win_rate > 0.40:
        score += 5
    
    # Grade assignment
    if score >= 90:
        return f"A+ ({score}/100) - EXCELLENT"
    elif score >= 80:
        return f"A ({score}/100) - VERY GOOD"
    elif score >= 70:
        return f"B+ ({score}/100) - GOOD"
    elif score >= 60:
        return f"B ({score}/100) - ABOVE AVERAGE"
    elif score >= 50:
        return f"C ({score}/100) - AVERAGE"
    elif score >= 40:
        return f"D ({score}/100) - BELOW AVERAGE"
    else:
        return f"F ({score}/100) - NEEDS IMPROVEMENT"


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run AI Stock Trading Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py                        # Full NIFTY50 test
    python run_pipeline.py --symbols 10           # Top 10 stocks
    python run_pipeline.py --quick                # Quick test (5 stocks)
    python run_pipeline.py --holdings 20          # 20 stock portfolio
    python run_pipeline.py --capital 500000       # Rs 5 lakh capital
        """
    )
    
    parser.add_argument('--symbols', type=int, default=None,
                        help='Number of stocks to use (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 5 stocks')
    parser.add_argument('--start', type=str, default='2022-01-01',
                        help='Backtest start date (default: 2022-01-01)')
    parser.add_argument('--holdings', type=int, default=15,
                        help='Number of portfolio holdings (default: 15)')
    parser.add_argument('--capital', type=float, default=1000000,
                        help='Initial capital (default: 1000000)')
    parser.add_argument('--method', type=str, default='risk_parity',
                        choices=['equal', 'risk_parity', 'max_sharpe', 'min_volatility'],
                        help='Optimization method (default: risk_parity)')
    
    args = parser.parse_args()
    
    # Determine symbols
    if args.quick:
        symbols = config.ALL_STOCKS[:5]
    elif args.symbols:
        symbols = config.ALL_STOCKS[:args.symbols]
    else:
        symbols = config.ALL_STOCKS
    
    # Run pipeline
    run_full_pipeline(
        symbols=symbols,
        start_date=args.start,
        n_holdings=args.holdings,
        initial_capital=args.capital,
        optimization_method=args.method
    )


if __name__ == "__main__":
    main()
