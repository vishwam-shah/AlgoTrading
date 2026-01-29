"""
================================================================================
AI STOCK TRADING SYSTEM - MAIN ENTRY POINT
================================================================================
Unified entry point for the ML-enhanced stock trading pipeline.

Commands:
---------
    python main.py pipeline --sectors Banking IT --holdings 15
    python main.py backtest --symbols SBIN HDFCBANK --capital 100000
    python main.py compare --holdings 15
    python main.py signals --holdings 10

Features:
---------
- 8-step unified pipeline (engine.orchestrator)
- Factor-based portfolio optimization (Fama-French 5-factor)
- ML models: XGBoost, LightGBM, LSTM, GRU, Ensemble
- Walk-forward backtesting with realistic assumptions
- Comprehensive reporting

================================================================================
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import config


def run_pipeline(args):
    """Run the full 8-step unified pipeline."""
    from engine.orchestrator import UnifiedOrchestrator

    # Determine symbols from sectors or direct list
    symbols = []
    if args.sectors:
        for sector in args.sectors:
            sector_stocks = [s for s, sec in config.STOCK_SECTOR_MAP.items() if sec == sector]
            symbols.extend(sector_stocks)
        symbols = list(set(symbols))
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = config.ALL_STOCKS[:args.num_symbols]

    print(f"Running pipeline for {len(symbols)} symbols...")

    orchestrator = UnifiedOrchestrator(
        symbols=symbols,
        initial_capital=args.capital,
        paper_trading=True
    )

    status = orchestrator.run_pipeline(
        optimization_method=args.method,
        n_holdings=args.holdings,
        start_date=args.start,
        force_download=args.fresh
    )

    print(f"\nPipeline completed with status: {status.status}")
    print(f"Steps completed: {sum(1 for s in status.steps if s.status == 'completed')}/{len(status.steps)}")

    # Print backtest summary
    backtest_results = orchestrator.get_backtest_results()
    if backtest_results:
        print("\nBacktest Results:")
        print("-" * 60)
        for symbol, result in backtest_results.items():
            if 'error' not in result:
                print(f"  {symbol}: Return={result.get('total_return', 0)*100:.2f}%, "
                      f"Sharpe={result.get('sharpe_ratio', 0):.2f}, "
                      f"Trades={result.get('total_trades', 0)}")

    # Print signals
    signals = orchestrator.get_signals()
    if signals:
        buy_signals = [s for s, sig in signals.items() if sig['action'] == 'BUY']
        sell_signals = [s for s, sig in signals.items() if sig['action'] == 'SELL']
        print(f"\nSignals: {len(buy_signals)} BUY, {len(sell_signals)} SELL, "
              f"{len(signals) - len(buy_signals) - len(sell_signals)} HOLD")

    return status


def run_backtest(args):
    """Run backtest for specific symbols."""
    from engine.orchestrator import UnifiedOrchestrator

    symbols = args.symbols if args.symbols else config.ALL_STOCKS[:args.num_symbols]

    orchestrator = UnifiedOrchestrator(
        symbols=symbols,
        initial_capital=args.capital,
        paper_trading=True
    )

    status = orchestrator.run_pipeline(
        optimization_method=args.method,
        n_holdings=min(args.holdings, len(symbols)),
        start_date=args.start,
        force_download=args.fresh
    )

    return orchestrator.get_backtest_results()


def run_comparison(args):
    """Compare all optimization methods."""
    from engine import DataCollector, FeatureEngineer, FactorAnalyzer, PortfolioOptimizer, BacktestValidator
    import pandas as pd

    symbols = config.ALL_STOCKS[:args.num_symbols] if args.num_symbols else config.ALL_STOCKS

    print("=" * 80)
    print("COMPARING OPTIMIZATION METHODS")
    print("=" * 80)

    # Collect data
    print("\n[1/4] Collecting data...")
    collector = DataCollector()
    price_data, market_data = collector.collect_all(symbols, start_date=args.start)

    # Compute features
    print("[2/4] Computing features...")
    engineer = FeatureEngineer()
    features = engineer.compute_all_features(price_data, market_data)

    # Factor analysis
    print("[3/4] Analyzing factors...")
    analyzer = FactorAnalyzer()
    factor_scores = analyzer.compute_factors(price_data, features)

    # Compare methods
    print("[4/4] Running backtests...\n")
    methods = ['equal_weight', 'risk_parity', 'max_sharpe', 'min_volatility']
    results = []

    for method in methods:
        print(f"  Testing {method}...")
        optimizer = PortfolioOptimizer()
        allocation = optimizer.optimize(
            price_data=price_data,
            factor_scores=factor_scores,
            n_holdings=args.holdings,
            method=method
        )

        validator = BacktestValidator()
        backtest = validator.run_backtest(
            price_data=price_data,
            allocation=allocation.weights,
            start_date=args.start,
            initial_capital=args.capital,
            benchmark=market_data.get('NIFTY50')
        )

        if backtest:
            results.append({
                'Method': method,
                'Total Return': f"{backtest.total_return:.1%}",
                'Annual Return': f"{backtest.annual_return:.1%}",
                'Sharpe Ratio': f"{backtest.sharpe_ratio:.2f}",
                'Max Drawdown': f"{backtest.max_drawdown:.1%}",
                'Win Rate': f"{backtest.win_rate:.1%}"
            })

    # Print comparison
    comparison_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)

    return comparison_df


def generate_signals(args):
    """Generate current trading signals."""
    from engine import DataCollector, FeatureEngineer, FactorAnalyzer, PortfolioOptimizer

    symbols = config.ALL_STOCKS[:args.num_symbols] if args.num_symbols else config.ALL_STOCKS

    print("=" * 80)
    print("GENERATING TRADING SIGNALS")
    print("=" * 80)

    # Collect recent data
    collector = DataCollector()
    price_data, market_data = collector.collect_all(symbols)

    # Compute features
    engineer = FeatureEngineer()
    features = engineer.compute_all_features(price_data, market_data)

    # Factor analysis
    analyzer = FactorAnalyzer()
    factor_scores = analyzer.compute_factors(price_data, features)

    # Get top stocks
    top_stocks = analyzer.get_top_stocks(factor_scores, n=args.holdings)

    # Optimize portfolio
    optimizer = PortfolioOptimizer()
    allocation = optimizer.optimize(
        price_data=price_data,
        factor_scores=factor_scores,
        n_holdings=args.holdings,
        method=args.method
    )

    # Generate signals
    print("\n" + "=" * 80)
    print("CURRENT TRADING SIGNALS")
    print("=" * 80)

    print(f"\nTOP {args.holdings} STOCKS BY FACTOR SCORE:")
    print("-" * 50)
    for i, stock in enumerate(top_stocks, 1):
        sector = config.STOCK_SECTOR_MAP.get(stock.symbol, 'Other')
        weight = allocation.weights.get(stock.symbol, 0) * 100
        print(f"{i:2}. {stock.symbol:12} | Score: {stock.combined_score:.3f} | "
              f"Sector: {sector:8} | Weight: {weight:.1f}%")

    print("\n" + "=" * 80)
    print(f"Recommended Method: {args.method}")
    print(f"Expected Sharpe: {allocation.sharpe_ratio:.2f}")
    print(f"Expected Return: {allocation.expected_return:.1%}")
    print("=" * 80)

    return allocation


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI Stock Trading System - ML Enhanced',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py pipeline --sectors Banking IT --holdings 15
  python main.py backtest --symbols SBIN HDFCBANK --capital 100000
  python main.py compare --holdings 15
  python main.py signals --holdings 10
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Pipeline command (new unified 8-step)
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full 8-step pipeline')
    pipeline_parser.add_argument('--sectors', nargs='+', help='Sectors to include (Banking, IT, Auto, etc.)')
    pipeline_parser.add_argument('--symbols', nargs='+', help='Specific symbols to use')
    pipeline_parser.add_argument('--holdings', type=int, default=15, help='Number of stocks to hold')
    pipeline_parser.add_argument('--method', default='risk_parity',
                                choices=['equal_weight', 'risk_parity', 'max_sharpe', 'min_volatility'],
                                help='Portfolio optimization method')
    pipeline_parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    pipeline_parser.add_argument('--start', default='2022-01-01', help='Start date')
    pipeline_parser.add_argument('--num-symbols', type=int, default=10, help='Number of symbols if no sectors/symbols specified')
    pipeline_parser.add_argument('--fresh', action='store_true', help='Force fresh data download')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest for specific stocks')
    backtest_parser.add_argument('--holdings', type=int, default=15, help='Number of stocks to hold')
    backtest_parser.add_argument('--method', default='risk_parity',
                                choices=['equal_weight', 'risk_parity', 'max_sharpe', 'min_volatility'],
                                help='Portfolio optimization method')
    backtest_parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    backtest_parser.add_argument('--start', default='2022-01-01', help='Start date')
    backtest_parser.add_argument('--num-symbols', type=int, default=10, help='Number of symbols')
    backtest_parser.add_argument('--symbols', nargs='+', help='Specific symbols to use')
    backtest_parser.add_argument('--fresh', action='store_true', help='Force fresh data download')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare optimization methods')
    compare_parser.add_argument('--holdings', type=int, default=15, help='Number of stocks to hold')
    compare_parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    compare_parser.add_argument('--start', default='2022-01-01', help='Start date')
    compare_parser.add_argument('--num-symbols', type=int, default=30, help='Number of symbols')

    # Signals command
    signals_parser = subparsers.add_parser('signals', help='Generate current signals')
    signals_parser.add_argument('--holdings', type=int, default=15, help='Number of stocks to hold')
    signals_parser.add_argument('--method', default='risk_parity',
                               choices=['equal_weight', 'risk_parity', 'max_sharpe', 'min_volatility'],
                               help='Portfolio optimization method')
    signals_parser.add_argument('--num-symbols', type=int, default=50, help='Number of symbols')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    print("=" * 80)
    print(f"AI STOCK TRADING SYSTEM")
    print(f"Command: {args.command}")
    print(f"Time: {datetime.now()}")
    print("=" * 80)

    if args.command == 'pipeline':
        run_pipeline(args)
    elif args.command == 'backtest':
        run_backtest(args)
    elif args.command == 'compare':
        run_comparison(args)
    elif args.command == 'signals':
        generate_signals(args)


if __name__ == "__main__":
    main()
