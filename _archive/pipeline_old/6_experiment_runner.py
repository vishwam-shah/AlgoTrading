"""
================================================================================
COMPREHENSIVE EXPERIMENT RUNNER - LOG ALL COMBINATIONS TO CSV
================================================================================

This module runs systematic experiments across all parameter combinations and
logs detailed results to CSV files for analysis.

Tracks:
- Factor combinations (1 to 5 factors)
- Feature counts and types
- Sample sizes and time periods
- Stock universe sizes
- Training/Validation/Test splits
- Model performance metrics
- Directional accuracy
- Execution times

Output CSV Files:
- experiment_log.csv - Master log of all experiments
- factor_analysis_log.csv - Factor combination results
- feature_importance_log.csv - Feature contribution analysis
- model_performance_log.csv - Individual model results
- directional_accuracy_log.csv - Prediction accuracy analysis

================================================================================
"""

import os
import sys
import csv
import json
import time
import itertools
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from importlib import import_module

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Import pipeline modules
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

from loguru import logger


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: str
    timestamp: str
    
    # Factor configuration
    factors_used: List[str]
    num_factors: int
    factor_weights: Dict[str, float]
    
    # Data configuration
    num_stocks: int
    stock_symbols: List[str]
    start_date: str
    end_date: str
    total_trading_days: int
    
    # Feature configuration
    num_features: int
    feature_categories: List[str]
    
    # Portfolio configuration
    n_holdings: int
    optimization_method: str
    initial_capital: float
    rebalance_frequency: str
    transaction_cost: float
    slippage: float
    
    # Split configuration
    train_pct: float = 0.6
    val_pct: float = 0.2
    test_pct: float = 0.2
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    experiment_id: str
    
    # Performance metrics
    total_return: float
    annual_return: float
    benchmark_return: float
    excess_return: float
    alpha: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    max_dd_duration: int
    var_95: float
    cvar_95: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Trading metrics
    total_trades: int
    num_rebalances: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    
    # Capital metrics
    initial_capital: float
    final_value: float
    total_profit: float
    
    # Directional accuracy
    daily_direction_accuracy: float
    weekly_direction_accuracy: float
    monthly_direction_accuracy: float
    
    # Execution metrics
    execution_time_seconds: float
    data_collection_time: float
    feature_engineering_time: float
    factor_analysis_time: float
    optimization_time: float
    backtest_time: float
    
    # Grade
    strategy_grade: str
    strategy_score: int
    
    # Status
    status: str
    error_message: str = ""


@dataclass
class FactorResult:
    """Results for factor analysis."""
    experiment_id: str
    factor_name: str
    factor_weight: float
    factor_enabled: bool
    
    # Factor performance
    top_stock: str
    top_score: float
    avg_score: float
    score_std: float
    
    # Factor contribution to returns
    factor_return_contribution: float
    factor_risk_contribution: float


@dataclass
class DirectionalAccuracyResult:
    """Directional accuracy tracking."""
    experiment_id: str
    symbol: str
    
    # Daily accuracy
    daily_up_correct: int
    daily_up_total: int
    daily_down_correct: int
    daily_down_total: int
    daily_accuracy: float
    
    # Weekly accuracy
    weekly_up_correct: int
    weekly_up_total: int
    weekly_down_correct: int
    weekly_down_total: int
    weekly_accuracy: float
    
    # Monthly accuracy
    monthly_up_correct: int
    monthly_up_total: int
    monthly_down_correct: int
    monthly_down_total: int
    monthly_accuracy: float


class ExperimentLogger:
    """Handles logging of all experiment data to CSV files."""
    
    def __init__(self, output_dir: str = None):
        """Initialize logger with output directory."""
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'results', 'experiments'
            )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped subdirectory for this run
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / f"run_{self.run_timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV files
        self._init_csv_files()
        
        logger.info(f"ExperimentLogger initialized")
        logger.info(f"  Output directory: {self.run_dir}")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        
        # Master experiment log
        self.experiment_log_path = self.run_dir / 'experiment_log.csv'
        self.experiment_log_headers = [
            'experiment_id', 'timestamp', 'status',
            # Factor config
            'num_factors', 'factors_used', 'factor_weights',
            # Data config
            'num_stocks', 'start_date', 'end_date', 'total_trading_days',
            # Feature config
            'num_features', 'feature_categories',
            # Portfolio config
            'n_holdings', 'optimization_method', 'initial_capital',
            'rebalance_frequency', 'transaction_cost', 'slippage',
            # Split config
            'train_pct', 'val_pct', 'test_pct',
            'train_samples', 'val_samples', 'test_samples',
            # Performance
            'total_return', 'annual_return', 'benchmark_return', 'excess_return', 'alpha',
            # Risk
            'volatility', 'max_drawdown', 'max_dd_duration', 'var_95', 'cvar_95',
            # Risk-adjusted
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'information_ratio',
            # Trading
            'total_trades', 'num_rebalances', 'win_rate', 'profit_factor', 'avg_trade_return',
            # Capital
            'final_value', 'total_profit',
            # Directional accuracy
            'daily_direction_accuracy', 'weekly_direction_accuracy', 'monthly_direction_accuracy',
            # Execution times
            'execution_time_seconds', 'data_collection_time', 'feature_engineering_time',
            'factor_analysis_time', 'optimization_time', 'backtest_time',
            # Grade
            'strategy_grade', 'strategy_score',
            # Error
            'error_message'
        ]
        
        # Factor analysis log
        self.factor_log_path = self.run_dir / 'factor_analysis_log.csv'
        self.factor_log_headers = [
            'experiment_id', 'factor_name', 'factor_weight', 'factor_enabled',
            'top_stock', 'top_score', 'avg_score', 'score_std',
            'factor_return_contribution', 'factor_risk_contribution'
        ]
        
        # Feature importance log
        self.feature_log_path = self.run_dir / 'feature_importance_log.csv'
        self.feature_log_headers = [
            'experiment_id', 'feature_name', 'feature_category',
            'mean_value', 'std_value', 'min_value', 'max_value',
            'correlation_with_return', 'importance_rank'
        ]
        
        # Model performance log (per stock)
        self.model_log_path = self.run_dir / 'model_performance_log.csv'
        self.model_log_headers = [
            'experiment_id', 'symbol', 'sector',
            'weight', 'factor_score_combined',
            'factor_score_value', 'factor_score_momentum',
            'factor_score_quality', 'factor_score_low_vol', 'factor_score_sentiment',
            'stock_return', 'stock_volatility', 'contribution_to_portfolio'
        ]
        
        # Directional accuracy log
        self.accuracy_log_path = self.run_dir / 'directional_accuracy_log.csv'
        self.accuracy_log_headers = [
            'experiment_id', 'symbol',
            'daily_up_correct', 'daily_up_total', 'daily_down_correct', 'daily_down_total', 'daily_accuracy',
            'weekly_up_correct', 'weekly_up_total', 'weekly_down_correct', 'weekly_down_total', 'weekly_accuracy',
            'monthly_up_correct', 'monthly_up_total', 'monthly_down_correct', 'monthly_down_total', 'monthly_accuracy'
        ]
        
        # Combination results log
        self.combination_log_path = self.run_dir / 'combination_results_log.csv'
        self.combination_log_headers = [
            'experiment_id', 'combination_type', 'combination_description',
            'num_factors', 'factors_list', 'num_stocks', 'n_holdings', 'optimization_method',
            'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate',
            'daily_direction_accuracy', 'execution_time_seconds', 'status'
        ]
        
        # Write headers to all files
        self._write_headers(self.experiment_log_path, self.experiment_log_headers)
        self._write_headers(self.factor_log_path, self.factor_log_headers)
        self._write_headers(self.feature_log_path, self.feature_log_headers)
        self._write_headers(self.model_log_path, self.model_log_headers)
        self._write_headers(self.accuracy_log_path, self.accuracy_log_headers)
        self._write_headers(self.combination_log_path, self.combination_log_headers)
    
    def _write_headers(self, path: Path, headers: List[str]):
        """Write headers to CSV file."""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_experiment(self, config: ExperimentConfig, result: ExperimentResult):
        """Log experiment to master CSV."""
        row = [
            result.experiment_id, config.timestamp, result.status,
            config.num_factors, '|'.join(config.factors_used), json.dumps(config.factor_weights),
            config.num_stocks, config.start_date, config.end_date, config.total_trading_days,
            config.num_features, '|'.join(config.feature_categories),
            config.n_holdings, config.optimization_method, config.initial_capital,
            config.rebalance_frequency, config.transaction_cost, config.slippage,
            config.train_pct, config.val_pct, config.test_pct,
            config.train_samples, config.val_samples, config.test_samples,
            result.total_return, result.annual_return, result.benchmark_return,
            result.excess_return, result.alpha,
            result.volatility, result.max_drawdown, result.max_dd_duration,
            result.var_95, result.cvar_95,
            result.sharpe_ratio, result.sortino_ratio, result.calmar_ratio, result.information_ratio,
            result.total_trades, result.num_rebalances, result.win_rate,
            result.profit_factor, result.avg_trade_return,
            result.final_value, result.total_profit,
            result.daily_direction_accuracy, result.weekly_direction_accuracy, result.monthly_direction_accuracy,
            result.execution_time_seconds, result.data_collection_time, result.feature_engineering_time,
            result.factor_analysis_time, result.optimization_time, result.backtest_time,
            result.strategy_grade, result.strategy_score,
            result.error_message
        ]
        
        with open(self.experiment_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_factor_result(self, result: FactorResult):
        """Log factor analysis result."""
        row = [
            result.experiment_id, result.factor_name, result.factor_weight, result.factor_enabled,
            result.top_stock, result.top_score, result.avg_score, result.score_std,
            result.factor_return_contribution, result.factor_risk_contribution
        ]
        
        with open(self.factor_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_feature_importance(self, experiment_id: str, features_df: pd.DataFrame, returns: pd.Series):
        """Log feature importance analysis."""
        rows = []
        
        for col in features_df.columns:
            # Calculate statistics
            mean_val = features_df[col].mean()
            std_val = features_df[col].std()
            min_val = features_df[col].min()
            max_val = features_df[col].max()
            
            # Calculate correlation with returns
            try:
                corr = features_df[col].corr(returns)
                if pd.isna(corr):
                    corr = 0.0
            except:
                corr = 0.0
            
            # Determine category
            category = self._get_feature_category(col)
            
            rows.append([
                experiment_id, col, category,
                mean_val, std_val, min_val, max_val,
                corr, 0  # importance_rank filled later
            ])
        
        # Sort by absolute correlation and assign ranks
        rows.sort(key=lambda x: abs(x[7]), reverse=True)
        for i, row in enumerate(rows):
            row[8] = i + 1
        
        with open(self.feature_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    
    def _get_feature_category(self, feature_name: str) -> str:
        """Determine feature category from name."""
        if any(x in feature_name.lower() for x in ['sma', 'ema', 'macd', 'trend']):
            return 'trend'
        elif any(x in feature_name.lower() for x in ['rsi', 'momentum', 'stoch', 'roc', 'williams']):
            return 'momentum'
        elif any(x in feature_name.lower() for x in ['volatility', 'atr', 'bb_', 'keltner', 'parkinson']):
            return 'volatility'
        elif any(x in feature_name.lower() for x in ['volume', 'obv', 'mfi', 'vwap', 'vpt']):
            return 'volume'
        elif any(x in feature_name.lower() for x in ['body', 'shadow', 'gap', 'doji', 'high', 'low']):
            return 'price_pattern'
        elif any(x in feature_name.lower() for x in ['beta', 'rs_vs', 'corr_nifty']):
            return 'market_relative'
        elif any(x in feature_name.lower() for x in ['day_of', 'month', 'quarter', 'is_month']):
            return 'time_based'
        elif any(x in feature_name.lower() for x in ['ad_line', 'cmf', 'fear_greed', 'sentiment']):
            return 'sentiment'
        else:
            return 'other'
    
    def log_model_performance(self, experiment_id: str, symbol: str, sector: str,
                               weight: float, factor_scores: Dict[str, float],
                               stock_return: float, stock_volatility: float,
                               contribution: float):
        """Log individual model/stock performance."""
        row = [
            experiment_id, symbol, sector, weight,
            factor_scores.get('combined', 0),
            factor_scores.get('value', 0),
            factor_scores.get('momentum', 0),
            factor_scores.get('quality', 0),
            factor_scores.get('low_vol', 0),
            factor_scores.get('sentiment', 0),
            stock_return, stock_volatility, contribution
        ]
        
        with open(self.model_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_directional_accuracy(self, result: DirectionalAccuracyResult):
        """Log directional accuracy result."""
        row = [
            result.experiment_id, result.symbol,
            result.daily_up_correct, result.daily_up_total,
            result.daily_down_correct, result.daily_down_total, result.daily_accuracy,
            result.weekly_up_correct, result.weekly_up_total,
            result.weekly_down_correct, result.weekly_down_total, result.weekly_accuracy,
            result.monthly_up_correct, result.monthly_up_total,
            result.monthly_down_correct, result.monthly_down_total, result.monthly_accuracy
        ]
        
        with open(self.accuracy_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_combination_result(self, experiment_id: str, combination_type: str,
                                combination_desc: str, num_factors: int,
                                factors_list: List[str], num_stocks: int,
                                n_holdings: int, optimization_method: str,
                                sharpe: float, total_return: float,
                                max_dd: float, win_rate: float,
                                daily_accuracy: float, exec_time: float, status: str):
        """Log combination result for easy comparison."""
        row = [
            experiment_id, combination_type, combination_desc,
            num_factors, '|'.join(factors_list), num_stocks, n_holdings, optimization_method,
            sharpe, total_return, max_dd, win_rate,
            daily_accuracy, exec_time, status
        ]
        
        with open(self.combination_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all logged experiments."""
        try:
            df = pd.read_csv(self.experiment_log_path)
            
            successful = df[df['status'] == 'success']
            
            return {
                'total_experiments': len(df),
                'successful': len(successful),
                'failed': len(df) - len(successful),
                'best_sharpe': successful['sharpe_ratio'].max() if len(successful) > 0 else 0,
                'best_return': successful['total_return'].max() if len(successful) > 0 else 0,
                'best_win_rate': successful['win_rate'].max() if len(successful) > 0 else 0,
                'avg_execution_time': successful['execution_time_seconds'].mean() if len(successful) > 0 else 0,
                'output_dir': str(self.run_dir)
            }
        except:
            return {'error': 'No experiments logged yet'}


class ComprehensiveExperimentRunner:
    """Runs comprehensive experiments across all parameter combinations."""
    
    def __init__(self, output_dir: str = None):
        """Initialize runner."""
        self.logger = ExperimentLogger(output_dir)
        self.experiment_count = 0
        
        # Define all possible configurations
        self.all_factors = ['value', 'momentum', 'quality', 'low_vol', 'sentiment']
        
        self.stock_counts = [5, 10, 20, 30, 50]  # Different universe sizes
        self.holding_counts = [5, 10, 15, 20]    # Different portfolio sizes
        self.optimization_methods = ['equal_weight', 'risk_parity', 'max_sharpe', 'min_volatility']
        
        # Feature categories to track
        self.feature_categories = [
            'trend', 'momentum', 'volatility', 'volume',
            'price_pattern', 'market_relative', 'time_based', 'sentiment'
        ]
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        self.experiment_count += 1
        return f"EXP_{self.logger.run_timestamp}_{self.experiment_count:04d}"
    
    def _calculate_directional_accuracy(
        self,
        prices: pd.DataFrame,
        predictions: Dict[str, float]  # Expected direction based on factor scores
    ) -> Tuple[float, float, float, Dict[str, DirectionalAccuracyResult]]:
        """Calculate directional accuracy at daily, weekly, monthly levels."""
        
        results = {}
        daily_accuracies = []
        weekly_accuracies = []
        monthly_accuracies = []
        
        for symbol in predictions.keys():
            if symbol not in prices.columns:
                continue
            
            price_series = prices[symbol].dropna()
            if len(price_series) < 30:
                continue
            
            # Daily returns and directions
            daily_returns = price_series.pct_change().dropna()
            daily_direction = (daily_returns > 0).astype(int)
            
            # Weekly returns (resample)
            weekly_prices = price_series.resample('W').last().dropna()
            weekly_returns = weekly_prices.pct_change().dropna()
            weekly_direction = (weekly_returns > 0).astype(int)
            
            # Monthly returns
            monthly_prices = price_series.resample('M').last().dropna()
            monthly_returns = monthly_prices.pct_change().dropna()
            monthly_direction = (monthly_returns > 0).astype(int)
            
            # Factor score predicts direction (positive score = expect up)
            predicted_direction = 1 if predictions[symbol] > 0.5 else 0
            
            # Calculate accuracy
            daily_up_correct = ((daily_direction == 1) & (predicted_direction == 1)).sum()
            daily_up_total = (daily_direction == 1).sum()
            daily_down_correct = ((daily_direction == 0) & (predicted_direction == 0)).sum()
            daily_down_total = (daily_direction == 0).sum()
            daily_acc = (daily_up_correct + daily_down_correct) / len(daily_direction) if len(daily_direction) > 0 else 0
            
            weekly_up_correct = ((weekly_direction == 1) & (predicted_direction == 1)).sum()
            weekly_up_total = (weekly_direction == 1).sum()
            weekly_down_correct = ((weekly_direction == 0) & (predicted_direction == 0)).sum()
            weekly_down_total = (weekly_direction == 0).sum()
            weekly_acc = (weekly_up_correct + weekly_down_correct) / len(weekly_direction) if len(weekly_direction) > 0 else 0
            
            monthly_up_correct = ((monthly_direction == 1) & (predicted_direction == 1)).sum()
            monthly_up_total = (monthly_direction == 1).sum()
            monthly_down_correct = ((monthly_direction == 0) & (predicted_direction == 0)).sum()
            monthly_down_total = (monthly_direction == 0).sum()
            monthly_acc = (monthly_up_correct + monthly_down_correct) / len(monthly_direction) if len(monthly_direction) > 0 else 0
            
            results[symbol] = DirectionalAccuracyResult(
                experiment_id='',  # Filled by caller
                symbol=symbol,
                daily_up_correct=int(daily_up_correct),
                daily_up_total=int(daily_up_total),
                daily_down_correct=int(daily_down_correct),
                daily_down_total=int(daily_down_total),
                daily_accuracy=daily_acc,
                weekly_up_correct=int(weekly_up_correct),
                weekly_up_total=int(weekly_up_total),
                weekly_down_correct=int(weekly_down_correct),
                weekly_down_total=int(weekly_down_total),
                weekly_accuracy=weekly_acc,
                monthly_up_correct=int(monthly_up_correct),
                monthly_up_total=int(monthly_up_total),
                monthly_down_correct=int(monthly_down_correct),
                monthly_down_total=int(monthly_down_total),
                monthly_accuracy=monthly_acc
            )
            
            daily_accuracies.append(daily_acc)
            weekly_accuracies.append(weekly_acc)
            monthly_accuracies.append(monthly_acc)
        
        avg_daily = np.mean(daily_accuracies) if daily_accuracies else 0
        avg_weekly = np.mean(weekly_accuracies) if weekly_accuracies else 0
        avg_monthly = np.mean(monthly_accuracies) if monthly_accuracies else 0
        
        return avg_daily, avg_weekly, avg_monthly, results
    
    def _grade_strategy(self, sharpe: float, win_rate: float, max_dd: float) -> str:
        """Assign letter grade based on performance metrics."""
        score = self._score_strategy(sharpe, win_rate, max_dd)
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        elif score >= 40:
            return 'D'
        else:
            return 'F'
    
    def _score_strategy(self, sharpe: float, win_rate: float, max_dd: float) -> float:
        """Calculate overall strategy score (0-100)."""
        sharpe = sharpe or 0
        win_rate = win_rate or 0
        max_dd = abs(max_dd or 0)
        
        # Sharpe component (0-40 points)
        sharpe_score = min(40, max(0, (sharpe + 0.5) / 3.0 * 40))
        
        # Win rate component (0-30 points)
        win_score = min(30, max(0, win_rate * 30))
        
        # Drawdown component (0-30 points, penalize high drawdown)
        dd_score = min(30, max(0, (1 - max_dd) * 30))
        
        return sharpe_score + win_score + dd_score
    
    def run_single_experiment(
        self,
        factors: List[str],
        factor_weights: Dict[str, float],
        num_stocks: int,
        n_holdings: int,
        optimization_method: str,
        start_date: str = '2022-01-01',
        end_date: str = None,
        initial_capital: float = 1000000
    ) -> Tuple[ExperimentConfig, ExperimentResult]:
        """Run a single experiment with given parameters."""
        
        experiment_id = self._generate_experiment_id()
        timestamp = datetime.now().isoformat()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT: {experiment_id}")
        logger.info(f"  Factors: {factors}")
        logger.info(f"  Stocks: {num_stocks}, Holdings: {n_holdings}")
        logger.info(f"  Method: {optimization_method}")
        logger.info(f"{'='*60}")
        
        # Initialize timing
        total_start = time.time()
        times = {
            'data_collection': 0,
            'feature_engineering': 0,
            'factor_analysis': 0,
            'optimization': 0,
            'backtest': 0
        }
        
        try:
            # Get stock symbols - use ALL_STOCKS from config
            all_symbols = config.ALL_STOCKS[:num_stocks]
            
            # ============================================================
            # STEP 1: DATA COLLECTION
            # ============================================================
            step_start = time.time()
            collector = DataCollector(cache_dir=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'raw'
            ))
            
            stock_data, market_data = collector.collect_all(
                symbols=all_symbols,
                start_date=start_date,
                end_date=end_date
            )
            collection_stats = collector.collection_stats  # Get stats from collector
            times['data_collection'] = time.time() - step_start
            
            if not stock_data:
                raise ValueError("No stock data collected")
            
            # Calculate trading days and samples
            sample_df = list(stock_data.values())[0]
            total_trading_days = len(sample_df)
            
            # Split samples (60% train, 20% val, 20% test)
            train_samples = int(total_trading_days * 0.6)
            val_samples = int(total_trading_days * 0.2)
            test_samples = total_trading_days - train_samples - val_samples
            
            # ============================================================
            # STEP 2: FEATURE ENGINEERING
            # ============================================================
            step_start = time.time()
            engineer = FeatureEngineer()
            features = engineer.compute_all_features(stock_data, market_data)
            times['feature_engineering'] = time.time() - step_start
            
            num_features = engineer.feature_count if hasattr(engineer, 'feature_count') else 82
            
            # ============================================================
            # STEP 3: FACTOR ANALYSIS
            # ============================================================
            step_start = time.time()
            analyzer = FactorAnalyzer()
            analyzer.set_factor_weights(factor_weights)  # Set custom weights
            factor_scores = analyzer.compute_factors(stock_data, features)
            times['factor_analysis'] = time.time() - step_start
            
            # ============================================================
            # STEP 4: PORTFOLIO OPTIMIZATION
            # ============================================================
            step_start = time.time()
            optimizer = PortfolioOptimizer()
            portfolio = optimizer.optimize(
                price_data=stock_data,
                factor_scores=factor_scores,
                n_holdings=n_holdings,
                method=optimization_method
            )
            times['optimization'] = time.time() - step_start
            
            # ============================================================
            # STEP 5: BACKTEST
            # ============================================================
            step_start = time.time()
            validator = BacktestValidator()  # No initial_capital in init
            
            # Get portfolio weights
            if portfolio is None:
                raise ValueError("Portfolio optimization failed - no allocation returned")
                
            allocation = portfolio.weights
            benchmark = market_data.get('NIFTY50') if market_data else None
            
            backtest_results = validator.run_backtest(
                price_data=stock_data,
                allocation=allocation,
                initial_capital=initial_capital,
                benchmark=benchmark
            )
            times['backtest'] = time.time() - step_start
            
            # ============================================================
            # CALCULATE DIRECTIONAL ACCURACY
            # ============================================================
            # Build price matrix
            price_df = pd.DataFrame({
                symbol: df['close']
                for symbol, df in stock_data.items()
            })
            
            # Convert factor_scores list to dict for easy access
            factor_scores_dict = {score.symbol: score for score in factor_scores}
            
            # Use factor scores as predictions
            predictions = {
                score.symbol: score.combined_score
                for score in factor_scores
            }
            
            daily_acc, weekly_acc, monthly_acc, accuracy_results = self._calculate_directional_accuracy(
                price_df, predictions
            )
            
            # ============================================================
            # LOG RESULTS
            # ============================================================
            
            # Create config
            exp_config = ExperimentConfig(
                experiment_id=experiment_id,
                timestamp=timestamp,
                factors_used=factors,
                num_factors=len(factors),
                factor_weights=factor_weights,
                num_stocks=len(stock_data),
                stock_symbols=list(stock_data.keys()),
                start_date=start_date,
                end_date=end_date or datetime.now().strftime('%Y-%m-%d'),
                total_trading_days=total_trading_days,
                num_features=num_features,
                feature_categories=self.feature_categories,
                n_holdings=n_holdings,
                optimization_method=optimization_method,
                initial_capital=initial_capital,
                rebalance_frequency='monthly',
                transaction_cost=0.001,
                slippage=0.0005,
                train_pct=0.6,
                val_pct=0.2,
                test_pct=0.2,
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples
            )
            
            # Calculate additional metrics from BacktestResults dataclass
            if backtest_results is None:
                raise ValueError("Backtest failed - no results returned")
            
            # Calculate alpha (excess return over benchmark)
            benchmark_annual = (backtest_results.benchmark_return or 0) / 4  # ~4 years of data
            alpha = (backtest_results.annual_return or 0) - benchmark_annual
            
            # Calculate average trade return (not directly available, estimate from win rate)
            avg_trade_return = 0  # Would need trade history
            
            # Create result
            exp_result = ExperimentResult(
                experiment_id=experiment_id,
                total_return=(backtest_results.total_return or 0) * 100,
                annual_return=(backtest_results.annual_return or 0) * 100,
                benchmark_return=(backtest_results.benchmark_return or 0) * 100,
                excess_return=(backtest_results.excess_return or 0) * 100,
                alpha=alpha * 100,
                volatility=(backtest_results.volatility or 0) * 100,
                max_drawdown=(backtest_results.max_drawdown or 0) * 100,
                max_dd_duration=backtest_results.max_drawdown_duration or 0,
                var_95=(backtest_results.var_95 or 0) * 100,
                cvar_95=(backtest_results.cvar_95 or 0) * 100,
                sharpe_ratio=backtest_results.sharpe_ratio or 0,
                sortino_ratio=backtest_results.sortino_ratio or 0,
                calmar_ratio=backtest_results.calmar_ratio or 0,
                information_ratio=backtest_results.information_ratio or 0,
                total_trades=backtest_results.n_trades or 0,
                num_rebalances=backtest_results.n_rebalances or 0,
                win_rate=(backtest_results.win_rate or 0) * 100,
                profit_factor=backtest_results.profit_factor or 0,
                avg_trade_return=avg_trade_return * 100,
                initial_capital=initial_capital,
                final_value=backtest_results.final_value or initial_capital,
                total_profit=(backtest_results.final_value or initial_capital) - initial_capital,
                daily_direction_accuracy=daily_acc * 100,
                weekly_direction_accuracy=weekly_acc * 100,
                monthly_direction_accuracy=monthly_acc * 100,
                execution_time_seconds=time.time() - total_start,
                data_collection_time=times['data_collection'],
                feature_engineering_time=times['feature_engineering'],
                factor_analysis_time=times['factor_analysis'],
                optimization_time=times['optimization'],
                backtest_time=times['backtest'],
                strategy_grade=self._grade_strategy(backtest_results.sharpe_ratio, backtest_results.win_rate, backtest_results.max_drawdown),
                strategy_score=self._score_strategy(backtest_results.sharpe_ratio, backtest_results.win_rate, backtest_results.max_drawdown),
                status='success'
            )
            
            # Log to CSV
            self.logger.log_experiment(exp_config, exp_result)
            
            # Log factor results
            for factor_name in self.all_factors:
                factor_enabled = factor_name in factors
                factor_weight = factor_weights.get(factor_name, 0)
                
                # Get factor scores for this factor
                factor_scores_list = []
                top_stock = ''
                top_score = 0
                
                for score in factor_scores:
                    factor_val = getattr(score, f'{factor_name}_score', 0)
                    factor_scores_list.append(factor_val)
                    if factor_val > top_score:
                        top_score = factor_val
                        top_stock = score.symbol
                
                avg_score = np.mean(factor_scores_list) if factor_scores_list else 0
                score_std = np.std(factor_scores_list) if factor_scores_list else 0
                
                factor_result = FactorResult(
                    experiment_id=experiment_id,
                    factor_name=factor_name,
                    factor_weight=factor_weight,
                    factor_enabled=factor_enabled,
                    top_stock=top_stock,
                    top_score=top_score,
                    avg_score=avg_score,
                    score_std=score_std,
                    factor_return_contribution=factor_weight * exp_result.total_return / 100,
                    factor_risk_contribution=factor_weight * exp_result.volatility / 100
                )
                self.logger.log_factor_result(factor_result)
            
            # Log model performance for each stock in portfolio
            for symbol, weight in portfolio.weights.items():
                sector = config.STOCK_SECTOR_MAP.get(symbol, 'Other')
                score = factor_scores_dict.get(symbol)
                
                if score:
                    factor_score_dict = {
                        'combined': score.combined_score,
                        'value': score.value_score,
                        'momentum': score.momentum_score,
                        'quality': score.quality_score,
                        'low_vol': score.low_vol_score,
                        'sentiment': score.sentiment_score
                    }
                else:
                    factor_score_dict = {}
                
                # Calculate stock return from data
                if symbol in stock_data:
                    df = stock_data[symbol]
                    stock_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                    stock_vol = df['close'].pct_change().std() * np.sqrt(252) * 100
                else:
                    stock_return = 0
                    stock_vol = 0
                
                self.logger.log_model_performance(
                    experiment_id, symbol, sector, weight,
                    factor_score_dict, stock_return, stock_vol,
                    weight * exp_result.total_return
                )
            
            # Log directional accuracy
            for symbol, acc_result in accuracy_results.items():
                acc_result.experiment_id = experiment_id
                self.logger.log_directional_accuracy(acc_result)
            
            # Log combination result
            self.logger.log_combination_result(
                experiment_id=experiment_id,
                combination_type='factor_portfolio',
                combination_desc=f"{len(factors)}F_{num_stocks}S_{n_holdings}H_{optimization_method}",
                num_factors=len(factors),
                factors_list=factors,
                num_stocks=len(stock_data),
                n_holdings=n_holdings,
                optimization_method=optimization_method,
                sharpe=exp_result.sharpe_ratio,
                total_return=exp_result.total_return,
                max_dd=exp_result.max_drawdown,
                win_rate=exp_result.win_rate,
                daily_accuracy=exp_result.daily_direction_accuracy,
                exec_time=exp_result.execution_time_seconds,
                status='success'
            )
            
            logger.success(f"Experiment {experiment_id} completed successfully")
            logger.info(f"  Return: {exp_result.total_return:.2f}%")
            logger.info(f"  Sharpe: {exp_result.sharpe_ratio:.2f}")
            logger.info(f"  Direction Accuracy: {exp_result.daily_direction_accuracy:.1f}%")
            
            return exp_config, exp_result
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {str(e)}")
            
            # Log failed experiment
            exp_config = ExperimentConfig(
                experiment_id=experiment_id,
                timestamp=timestamp,
                factors_used=factors,
                num_factors=len(factors),
                factor_weights=factor_weights,
                num_stocks=num_stocks,
                stock_symbols=[],
                start_date=start_date,
                end_date=end_date or datetime.now().strftime('%Y-%m-%d'),
                total_trading_days=0,
                num_features=0,
                feature_categories=[],
                n_holdings=n_holdings,
                optimization_method=optimization_method,
                initial_capital=initial_capital,
                rebalance_frequency='monthly',
                transaction_cost=0.001,
                slippage=0.0005
            )
            
            exp_result = ExperimentResult(
                experiment_id=experiment_id,
                total_return=0, annual_return=0, benchmark_return=0, excess_return=0, alpha=0,
                volatility=0, max_drawdown=0, max_dd_duration=0, var_95=0, cvar_95=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, information_ratio=0,
                total_trades=0, num_rebalances=0, win_rate=0, profit_factor=0, avg_trade_return=0,
                initial_capital=initial_capital, final_value=initial_capital, total_profit=0,
                daily_direction_accuracy=0, weekly_direction_accuracy=0, monthly_direction_accuracy=0,
                execution_time_seconds=time.time() - total_start,
                data_collection_time=times['data_collection'],
                feature_engineering_time=times['feature_engineering'],
                factor_analysis_time=times['factor_analysis'],
                optimization_time=times['optimization'],
                backtest_time=times['backtest'],
                strategy_grade='F',
                strategy_score=0,
                status='failed',
                error_message=str(e)
            )
            
            self.logger.log_experiment(exp_config, exp_result)
            
            # Log failed combination
            self.logger.log_combination_result(
                experiment_id=experiment_id,
                combination_type='factor_portfolio',
                combination_desc=f"{len(factors)}F_{num_stocks}S_{n_holdings}H_{optimization_method}",
                num_factors=len(factors),
                factors_list=factors,
                num_stocks=num_stocks,
                n_holdings=n_holdings,
                optimization_method=optimization_method,
                sharpe=0, total_return=0, max_dd=0, win_rate=0,
                daily_accuracy=0,
                exec_time=time.time() - total_start,
                status='failed'
            )
            
            return exp_config, exp_result
    
    def run_all_factor_combinations(
        self,
        num_stocks: int = 30,
        n_holdings: int = 15,
        optimization_method: str = 'risk_parity'
    ):
        """Run experiments for all factor combinations (1 to 5 factors)."""
        
        logger.info("\n" + "="*80)
        logger.info("RUNNING ALL FACTOR COMBINATIONS")
        logger.info("="*80)
        
        # Generate all combinations of factors
        all_combinations = []
        
        for r in range(1, len(self.all_factors) + 1):
            for combo in itertools.combinations(self.all_factors, r):
                all_combinations.append(list(combo))
        
        logger.info(f"Total factor combinations to test: {len(all_combinations)}")
        
        for i, factors in enumerate(all_combinations):
            logger.info(f"\n[{i+1}/{len(all_combinations)}] Testing factors: {factors}")
            
            # Create equal weights for selected factors
            weight = 1.0 / len(factors)
            factor_weights = {f: weight if f in factors else 0.0 for f in self.all_factors}
            
            self.run_single_experiment(
                factors=factors,
                factor_weights=factor_weights,
                num_stocks=num_stocks,
                n_holdings=n_holdings,
                optimization_method=optimization_method
            )
    
    def run_all_portfolio_sizes(
        self,
        factors: List[str] = None,
        optimization_method: str = 'risk_parity'
    ):
        """Run experiments for different stock universe and portfolio sizes."""
        
        if factors is None:
            factors = self.all_factors
        
        logger.info("\n" + "="*80)
        logger.info("RUNNING ALL PORTFOLIO SIZE COMBINATIONS")
        logger.info("="*80)
        
        combinations = []
        for num_stocks in self.stock_counts:
            for n_holdings in self.holding_counts:
                if n_holdings <= num_stocks:
                    combinations.append((num_stocks, n_holdings))
        
        logger.info(f"Total portfolio size combinations to test: {len(combinations)}")
        
        # Equal weights for all factors
        factor_weights = {f: 1.0/len(factors) for f in factors}
        
        for i, (num_stocks, n_holdings) in enumerate(combinations):
            logger.info(f"\n[{i+1}/{len(combinations)}] Testing: {num_stocks} stocks, {n_holdings} holdings")
            
            self.run_single_experiment(
                factors=factors,
                factor_weights=factor_weights,
                num_stocks=num_stocks,
                n_holdings=n_holdings,
                optimization_method=optimization_method
            )
    
    def run_all_optimization_methods(
        self,
        factors: List[str] = None,
        num_stocks: int = 30,
        n_holdings: int = 15
    ):
        """Run experiments for all optimization methods."""
        
        if factors is None:
            factors = self.all_factors
        
        logger.info("\n" + "="*80)
        logger.info("RUNNING ALL OPTIMIZATION METHODS")
        logger.info("="*80)
        
        logger.info(f"Total optimization methods to test: {len(self.optimization_methods)}")
        
        # Equal weights for all factors
        factor_weights = {f: 1.0/len(factors) for f in factors}
        
        for i, method in enumerate(self.optimization_methods):
            logger.info(f"\n[{i+1}/{len(self.optimization_methods)}] Testing method: {method}")
            
            self.run_single_experiment(
                factors=factors,
                factor_weights=factor_weights,
                num_stocks=num_stocks,
                n_holdings=n_holdings,
                optimization_method=method
            )
    
    def run_comprehensive_experiments(self, quick_mode: bool = False):
        """Run comprehensive experiments across all combinations."""
        
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE EXPERIMENT SUITE")
        logger.info("="*80)
        
        if quick_mode:
            # Quick mode: reduced combinations
            stock_counts = [10, 30]
            holding_counts = [5, 15]
            methods = ['risk_parity', 'equal_weight']
            factor_combos = [
                self.all_factors,  # All 5 factors
                ['momentum', 'quality', 'low_vol'],  # 3 factors
                ['value', 'momentum'],  # 2 factors
            ]
        else:
            # Full mode: all combinations
            stock_counts = self.stock_counts
            holding_counts = self.holding_counts
            methods = self.optimization_methods
            factor_combos = []
            for r in range(1, len(self.all_factors) + 1):
                for combo in itertools.combinations(self.all_factors, r):
                    factor_combos.append(list(combo))
        
        # Calculate total experiments
        total = 0
        for factors in factor_combos:
            for num_stocks in stock_counts:
                for n_holdings in holding_counts:
                    if n_holdings <= num_stocks:
                        for method in methods:
                            total += 1
        
        logger.info(f"Total experiments to run: {total}")
        
        current = 0
        for factors in factor_combos:
            factor_weights = {f: 1.0/len(factors) if f in factors else 0.0 for f in self.all_factors}
            
            for num_stocks in stock_counts:
                for n_holdings in holding_counts:
                    if n_holdings > num_stocks:
                        continue
                    
                    for method in methods:
                        current += 1
                        logger.info(f"\n[{current}/{total}] Running experiment...")
                        
                        self.run_single_experiment(
                            factors=factors,
                            factor_weights=factor_weights,
                            num_stocks=num_stocks,
                            n_holdings=n_holdings,
                            optimization_method=method
                        )
        
        # Print summary
        summary = self.logger.get_summary()
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT SUITE COMPLETE")
        logger.info("="*80)
        logger.info(f"Total experiments: {summary.get('total_experiments', 0)}")
        logger.info(f"Successful: {summary.get('successful', 0)}")
        logger.info(f"Failed: {summary.get('failed', 0)}")
        logger.info(f"Best Sharpe Ratio: {summary.get('best_sharpe', 0):.2f}")
        logger.info(f"Best Total Return: {summary.get('best_return', 0):.2f}%")
        logger.info(f"Best Win Rate: {summary.get('best_win_rate', 0):.2f}%")
        logger.info(f"Average Execution Time: {summary.get('avg_execution_time', 0):.1f}s")
        logger.info(f"\nResults saved to: {summary.get('output_dir', 'N/A')}")
        
        return summary


def main():
    """Main function to run experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive trading experiments')
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer combinations')
    parser.add_argument('--factors', action='store_true', help='Run all factor combinations only')
    parser.add_argument('--sizes', action='store_true', help='Run all portfolio size combinations only')
    parser.add_argument('--methods', action='store_true', help='Run all optimization methods only')
    parser.add_argument('--single', action='store_true', help='Run single experiment with defaults')
    
    args = parser.parse_args()
    
    runner = ComprehensiveExperimentRunner()
    
    if args.single:
        # Run single experiment with all 5 factors
        runner.run_single_experiment(
            factors=['value', 'momentum', 'quality', 'low_vol', 'sentiment'],
            factor_weights={'value': 0.2, 'momentum': 0.2, 'quality': 0.2, 'low_vol': 0.2, 'sentiment': 0.2},
            num_stocks=30,
            n_holdings=15,
            optimization_method='risk_parity'
        )
    elif args.factors:
        runner.run_all_factor_combinations()
    elif args.sizes:
        runner.run_all_portfolio_sizes()
    elif args.methods:
        runner.run_all_optimization_methods()
    else:
        runner.run_comprehensive_experiments(quick_mode=args.quick)
    
    # Print final summary
    summary = runner.logger.get_summary()
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Results directory: {summary.get('output_dir', 'N/A')}")
    print("\nCSV Files Generated:")
    print("  - experiment_log.csv          (Master log of all experiments)")
    print("  - factor_analysis_log.csv     (Factor combination results)")
    print("  - feature_importance_log.csv  (Feature contribution analysis)")
    print("  - model_performance_log.csv   (Individual stock/model results)")
    print("  - directional_accuracy_log.csv (Prediction accuracy analysis)")
    print("  - combination_results_log.csv (Easy comparison of all combinations)")


if __name__ == '__main__':
    main()
