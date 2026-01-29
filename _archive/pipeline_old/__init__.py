"""
================================================================================
AI STOCK TRADING PIPELINE - ML-ENHANCED WORKFLOW
================================================================================

This package provides a comprehensive ML-enhanced pipeline for algorithmic trading:

STEP 1: Data Collection       (1_data_collection.py)
STEP 2: Feature Engineering   (2_feature_engineering.py)
STEP 3: Factor Analysis       (3_factor_analysis.py)
STEP 4: Portfolio Optimization (4_portfolio_optimization.py)
STEP 5: Backtest & Validation (5_backtest_validation.py)
STEP 6: Experiment Runner     (6_experiment_runner.py)
STEP 7: ML Models             (7_ml_models.py)
STEP 8: ML Experiment Runner  (8_ml_experiment_runner.py)

Each step is modular, testable, and builds upon the previous step.

Usage:
    # Factor-based pipeline
    from pipeline import run_full_pipeline
    results = run_full_pipeline(symbols=['HDFCBANK', 'ICICIBANK', ...])

    # ML-based pipeline
    from pipeline import MLExperimentRunner
    runner = MLExperimentRunner()
    summary = runner.run_experiment(symbols=['HDFCBANK', ...])

Author: AI Stock Trading Research
Version: 3.0.0
================================================================================
"""

__version__ = "3.0.0"
__author__ = "AI Stock Trading Research"

# Import from renamed modules (1_, 2_, etc.)
from importlib import import_module

# Dynamic imports to handle numbered module names
_data_collection = import_module('.1_data_collection', 'pipeline')
_feature_engineering = import_module('.2_feature_engineering', 'pipeline')
_factor_analysis = import_module('.3_factor_analysis', 'pipeline')
_portfolio_optimization = import_module('.4_portfolio_optimization', 'pipeline')
_backtest_validation = import_module('.5_backtest_validation', 'pipeline')

DataCollector = _data_collection.DataCollector
FeatureEngineer = _feature_engineering.FeatureEngineer
FactorAnalyzer = _factor_analysis.FactorAnalyzer
PortfolioOptimizer = _portfolio_optimization.PortfolioOptimizer
BacktestValidator = _backtest_validation.BacktestValidator

# ML Models (optional - may require tensorflow)
try:
    _ml_models = import_module('.7_ml_models', 'pipeline')
    MLModelTrainer = _ml_models.MLModelTrainer
    XGBoostModel = _ml_models.XGBoostModel
    LSTMModel = _ml_models.LSTMModel
    GRUModel = _ml_models.GRUModel
    LightGBMModel = _ml_models.LightGBMModel
    EnsembleModel = _ml_models.EnsembleModel
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    MLModelTrainer = None

# ML Experiment Runner (optional)
try:
    _ml_experiment = import_module('.8_ml_experiment_runner', 'pipeline')
    MLExperimentRunner = _ml_experiment.MLExperimentRunner
except ImportError:
    MLExperimentRunner = None

__all__ = [
    'DataCollector',
    'FeatureEngineer',
    'FactorAnalyzer',
    'PortfolioOptimizer',
    'BacktestValidator',
    'MLModelTrainer',
    'MLExperimentRunner',
    'run_full_pipeline',
    'run_ml_experiment',
    'ML_AVAILABLE'
]


def run_full_pipeline(
    symbols: list = None,
    start_date: str = '2022-01-01',
    end_date: str = None,
    n_holdings: int = 15,
    initial_capital: float = 1000000,
    verbose: bool = True
) -> dict:
    """
    Run the complete 5-step pipeline.
    
    Args:
        symbols: List of stock symbols (defaults to NIFTY50)
        start_date: Backtest start date
        end_date: Backtest end date (defaults to today)
        n_holdings: Number of stocks in portfolio
        initial_capital: Starting capital
        verbose: Print progress
        
    Returns:
        dict: Complete results from all pipeline steps
    """
    from datetime import datetime
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    if symbols is None:
        symbols = config.ALL_STOCKS
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    results = {
        'symbols': symbols,
        'start_date': start_date,
        'end_date': end_date,
        'steps': {}
    }
    
    if verbose:
        print("=" * 80)
        print("AI STOCK TRADING PIPELINE - FULL RUN")
        print("=" * 80)
        print(f"Symbols: {len(symbols)} stocks")
        print(f"Period: {start_date} to {end_date}")
        print(f"Holdings: {n_holdings}")
        print(f"Capital: Rs {initial_capital:,.0f}")
        print("=" * 80)
    
    # Step 1: Data Collection
    if verbose:
        print("\n[STEP 1/5] DATA COLLECTION")
    collector = DataCollector()
    price_data, market_data = collector.collect_all(symbols, start_date, end_date)
    results['steps']['data_collection'] = {
        'symbols_collected': len(price_data),
        'data_quality': collector.get_quality_report()
    }
    
    # Step 2: Feature Engineering
    if verbose:
        print("\n[STEP 2/5] FEATURE ENGINEERING")
    engineer = FeatureEngineer()
    features = engineer.compute_all_features(price_data, market_data)
    results['steps']['feature_engineering'] = {
        'features_computed': engineer.get_feature_count(),
        'feature_summary': engineer.get_summary()
    }
    
    # Step 3: Factor Analysis
    if verbose:
        print("\n[STEP 3/5] FACTOR ANALYSIS")
    analyzer = FactorAnalyzer()
    factor_scores = analyzer.compute_factors(price_data, features)
    results['steps']['factor_analysis'] = {
        'factors': ['value', 'momentum', 'quality', 'low_vol', 'sentiment'],
        'top_stocks': analyzer.get_top_stocks(factor_scores, n=n_holdings)
    }
    
    # Step 4: Portfolio Optimization
    if verbose:
        print("\n[STEP 4/5] PORTFOLIO OPTIMIZATION")
    optimizer = PortfolioOptimizer()
    allocation = optimizer.optimize(
        price_data=price_data,
        factor_scores=factor_scores,
        n_holdings=n_holdings
    )
    results['steps']['portfolio_optimization'] = {
        'allocation': allocation,
        'method': 'risk_parity'
    }
    
    # Step 5: Backtest & Validation
    if verbose:
        print("\n[STEP 5/5] BACKTEST & VALIDATION")
    validator = BacktestValidator()
    backtest_results = validator.run_backtest(
        price_data=price_data,
        allocation=allocation,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    results['steps']['backtest_validation'] = backtest_results
    
    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        if hasattr(backtest_results, 'total_return'):
            print(f"Total Return: {backtest_results.total_return:.2%}")
            print(f"Annual Return: {backtest_results.annual_return:.2%}")
            print(f"Sharpe Ratio: {backtest_results.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {backtest_results.max_drawdown:.2%}")
        else:
            print(f"Total Return: {backtest_results.get('total_return', 0):.2%}")
            print(f"Annual Return: {backtest_results.get('annual_return', 0):.2%}")
            print(f"Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {backtest_results.get('max_drawdown', 0):.2%}")
        print("=" * 80)

    return results


def run_ml_experiment(
    symbols: list = None,
    start_date: str = '2020-01-01',
    end_date: str = None,
    initial_capital: float = 1000000,
    verbose: bool = True
) -> dict:
    """
    Run the ML-enhanced experiment pipeline.

    This pipeline trains ML models (XGBoost, LightGBM, LSTM, GRU) for each stock,
    creates an ensemble, and backtests predictions.

    Args:
        symbols: List of stock symbols (defaults to ALL_STOCKS)
        start_date: Backtest start date
        end_date: Backtest end date (defaults to today)
        initial_capital: Starting capital
        verbose: Print progress

    Returns:
        dict: Summary with experiment results
    """
    if MLExperimentRunner is None:
        raise ImportError("ML Experiment Runner not available. Check dependencies.")

    runner = MLExperimentRunner()
    summary = runner.run_experiment(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        verbose=verbose
    )

    return {
        'summary': summary,
        'stock_results': runner.stock_results,
        'predictions': runner.all_predictions,
        'output_dir': runner.run_dir
    }
