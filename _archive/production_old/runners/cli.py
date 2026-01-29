"""
================================================================================
UNIFIED CLI
================================================================================
Command-line interface for all production trading operations.
================================================================================
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Production Trading Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run paper trading
  python -m production.runners.cli paper --symbols HDFCBANK TCS --capital 100000
  
  # Run backtest
  python -m production.runners.cli backtest --symbols HDFCBANK TCS --train-window 252
  
  # Run full pipeline (data + features + train + backtest)
  python -m production.runners.cli pipeline --symbols HDFCBANK TCS
  
  # Optimize parameters
  python -m production.runners.cli optimize --symbols HDFCBANK
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Paper trading
    paper_parser = subparsers.add_parser('paper', help='Run paper trading')
    paper_parser.add_argument('--symbols', nargs='+', help='Stock symbols')
    paper_parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    paper_parser.add_argument('--continuous', action='store_true', help='Run continuously')
    paper_parser.add_argument('--interval', type=int, default=5, help='Check interval (minutes)')
    
    # Backtest
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--symbols', nargs='+', help='Stock symbols')
    backtest_parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    backtest_parser.add_argument('--train-window', type=int, default=252, help='Training window (days)')
    backtest_parser.add_argument('--test-window', type=int, default=21, help='Test window (days)')
    
    # Full pipeline
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--symbols', nargs='+', help='Stock symbols')
    pipeline_parser.add_argument('--fresh', action='store_true', help='Force fresh data download')
    pipeline_parser.add_argument('--retrain', action='store_true', help='Force model retraining')
    pipeline_parser.add_argument('--backtest', action='store_true', help='Run backtest after training')
    
    # Optimize
    optimize_parser = subparsers.add_parser('optimize', help='Optimize parameters')
    optimize_parser.add_argument('--symbols', nargs='+', help='Stock symbols')
    optimize_parser.add_argument('--param', help='Parameter to optimize')
    optimize_parser.add_argument('--output', default='production_results/optimization', help='Output directory')
    
    # Data
    data_parser = subparsers.add_parser('data', help='Download/update data')
    data_parser.add_argument('--symbols', nargs='+', help='Stock symbols')
    data_parser.add_argument('--period', default='2y', help='Data period')
    data_parser.add_argument('--fresh', action='store_true', help='Force fresh download')
    
    args = parser.parse_args()
    
    if args.command == 'paper':
        run_paper_trading(args)
    elif args.command == 'backtest':
        run_backtest(args)
    elif args.command == 'pipeline':
        run_pipeline(args)
    elif args.command == 'optimize':
        run_optimization(args)
    elif args.command == 'data':
        run_data_download(args)
    else:
        parser.print_help()


def run_paper_trading(args):
    """Run paper trading."""
    from production.runners.paper_trading import PaperTradingRunner
    
    print("=" * 60)
    print("PAPER TRADING")
    print("=" * 60)
    
    runner = PaperTradingRunner(
        symbols=args.symbols,
        initial_capital=args.capital
    )
    
    if args.continuous:
        runner.run_continuous(check_interval_minutes=args.interval)
    else:
        result = runner.run_day()
        print(f"\nResult: {result}")
        
        report = runner.get_performance_report()
        print("\nPerformance Report:")
        for key, value in report['metrics'].items():
            print(f"  {key}: {value}")


def run_backtest(args):
    """Run backtest."""
    from production.runners.backtest import BacktestRunner
    
    print("=" * 60)
    print("BACKTEST")
    print("=" * 60)
    
    runner = BacktestRunner(
        symbols=args.symbols,
        initial_capital=args.capital
    )
    
    summary = runner.run_all(
        train_window=args.train_window,
        test_window=args.test_window
    )
    
    runner.print_summary()


def run_pipeline(args):
    """Run full pipeline."""
    from production.orchestrator import TradingOrchestrator
    
    print("=" * 60)
    print("FULL PIPELINE")
    print("=" * 60)
    
    symbols = args.symbols or [
        'HDFCBANK', 'ICICIBANK', 'TCS', 'INFY', 'RELIANCE',
        'SBIN', 'KOTAKBANK', 'WIPRO', 'SUNPHARMA', 'LT'
    ]
    
    # Create orchestrator with specified symbols
    orchestrator = TradingOrchestrator(symbols=symbols)
    
    # Run pipeline steps - always force download for fresh data
    print("\n1. Downloading data...")
    force_download = args.fresh if hasattr(args, 'fresh') else True  # Default to fresh download
    orchestrator.collect_data(force_download=force_download)
    
    print("\n2. Computing features...")
    orchestrator.compute_features()
    
    if args.retrain:
        print("\n3. Training models...")
        orchestrator.train_model()
    
    if args.backtest:
        print("\n4. Running backtest...")
        from production.runners.backtest import BacktestRunner
        runner = BacktestRunner(symbols=symbols)
        runner.run_all()
        runner.print_summary()
    
    print("\nPipeline complete!")


def run_optimization(args):
    """Run parameter optimization."""
    print("=" * 60)
    print("PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Import optimizer
    try:
        from production.analysis.optimizer import ContinuousOptimizer
        
        symbols = args.symbols or ['HDFCBANK', 'TCS', 'INFY']
        
        optimizer = ContinuousOptimizer(
            symbols=symbols,
            output_dir=args.output
        )
        
        if args.param:
            # Optimize specific parameter
            results = optimizer.optimize_parameter(args.param)
        else:
            # Full optimization sweep
            results = optimizer.run_full_optimization()
        
        print("\nOptimization Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
            
    except ImportError:
        print("Optimizer module not found. Creating basic optimization...")
        # Fallback to basic backtest with parameter sweep
        from production.runners.backtest import BacktestRunner
        
        runner = BacktestRunner(symbols=args.symbols)
        runner.run_all()
        runner.print_summary()


def run_data_download(args):
    """Download/update data."""
    from production.core.data_loader import DataLoader
    
    print("=" * 60)
    print("DATA DOWNLOAD")
    print("=" * 60)
    
    symbols = args.symbols or [
        'HDFCBANK', 'ICICIBANK', 'TCS', 'INFY', 'RELIANCE',
        'SBIN', 'KOTAKBANK', 'WIPRO', 'SUNPHARMA', 'LT'
    ]
    
    # Convert period to days (approximate)
    period_map = {'1y': 365, '2y': 730, '6mo': 180, '3mo': 90, '1mo': 30}
    days = period_map.get(args.period, 500)
    
    loader = DataLoader()
    
    for symbol in symbols:
        print(f"\nDownloading {symbol}...")
        df = loader.download_stock(
            symbol,
            days=days,
            force_download=args.fresh
        )
        
        if df is not None and not df.empty:
            print(f"  Downloaded {len(df)} rows")
            if 'timestamp' in df.columns:
                print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            elif isinstance(df.index, pd.DatetimeIndex):
                print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        else:
            print(f"  Failed to download {symbol}")
    
    print("\nData download complete!")


if __name__ == "__main__":
    main()
