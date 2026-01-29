"""
================================================================================
COMPREHENSIVE ML EXPERIMENT RUNNER
================================================================================

This module runs end-to-end ML experiments with:
- Data collection and feature engineering
- ML model training (XGBoost, LightGBM, LSTM, GRU, Ensemble)
- Walk-forward backtesting with ML predictions
- Detailed performance metrics
- Stock-by-stock visualization (plots)
- CSV output for all results
- Comprehensive PDF/HTML report generation

Usage:
    python pipeline/8_ml_experiment_runner.py --symbols 10 --quick
    python pipeline/8_ml_experiment_runner.py --full

Output:
    results/ml_experiments/
    ├── plots/              # Per-stock prediction plots
    ├── metrics/            # Performance metrics CSV
    ├── predictions/        # Raw predictions CSV
    └── report.html         # Comprehensive report

================================================================================
"""

import os
import sys
import csv
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from importlib import import_module

# Import pipeline modules
_step1 = import_module('pipeline.1_data_collection')
_step2 = import_module('pipeline.2_feature_engineering')
_step7 = import_module('pipeline.7_ml_models')

DataCollector = _step1.DataCollector
FeatureEngineer = _step2.FeatureEngineer
MLModelTrainer = _step7.MLModelTrainer


@dataclass
class StockResult:
    """Results for a single stock."""
    symbol: str
    sector: str
    # Data info
    n_samples: int
    n_features: int
    start_date: str
    end_date: str
    # Model metrics
    best_model: str
    rmse: float
    mae: float
    mape: float
    r2: float
    direction_accuracy: float
    # Trading metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    # Individual model metrics
    xgboost_rmse: float = 0
    xgboost_direction: float = 0
    lightgbm_rmse: float = 0
    lightgbm_direction: float = 0
    lstm_rmse: float = 0
    lstm_direction: float = 0
    gru_rmse: float = 0
    gru_direction: float = 0
    ensemble_rmse: float = 0
    ensemble_direction: float = 0


@dataclass
class ExperimentSummary:
    """Summary of entire experiment."""
    experiment_id: str
    timestamp: str
    # Config
    n_stocks: int
    n_features: int
    start_date: str
    end_date: str
    initial_capital: float
    # Aggregate performance
    avg_direction_accuracy: float
    avg_rmse: float
    avg_sharpe: float
    avg_return: float
    best_stock: str
    worst_stock: str
    # Execution
    total_time: float
    status: str


class MLExperimentRunner:
    """
    Comprehensive ML experiment runner with visualization and reporting.
    """

    def __init__(self, output_dir: str = None):
        """Initialize experiment runner."""
        self.output_dir = output_dir or os.path.join(config.RESULTS_DIR, 'ml_experiments')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.output_dir, f'run_{self.timestamp}')

        # Create directories
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.run_dir, 'plots')).mkdir(exist_ok=True)
        Path(os.path.join(self.run_dir, 'metrics')).mkdir(exist_ok=True)
        Path(os.path.join(self.run_dir, 'predictions')).mkdir(exist_ok=True)

        # Storage
        self.stock_results: List[StockResult] = []
        self.all_predictions: Dict[str, pd.DataFrame] = {}
        self.model_metrics: Dict[str, pd.DataFrame] = {}

        logger.info(f"MLExperimentRunner initialized")
        logger.info(f"Output directory: {self.run_dir}")

    def run_experiment(
        self,
        symbols: List[str] = None,
        start_date: str = '2020-01-01',
        end_date: str = None,
        initial_capital: float = 1000000,
        verbose: bool = True
    ) -> ExperimentSummary:
        """
        Run complete ML experiment pipeline.

        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            initial_capital: Initial capital for backtesting
            verbose: Print progress

        Returns:
            ExperimentSummary with all results
        """
        experiment_id = f"MLEXP_{self.timestamp}"
        total_start = time.time()

        if symbols is None:
            symbols = config.ALL_STOCKS

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info("=" * 80)
        logger.info("ML EXPERIMENT RUNNER")
        logger.info("=" * 80)
        logger.info(f"Experiment ID: {experiment_id}")
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Capital: Rs {initial_capital:,.0f}")
        logger.info("=" * 80)

        try:
            # ================================================================
            # STEP 1: DATA COLLECTION
            # ================================================================
            logger.info("\n[STEP 1/5] DATA COLLECTION")
            step_start = time.time()

            collector = DataCollector()
            price_data, market_data = collector.collect_all(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )

            logger.info(f"  Collected {len(price_data)} stocks in {time.time() - step_start:.1f}s")

            if len(price_data) < 2:
                raise ValueError("Insufficient stock data collected")

            # ================================================================
            # STEP 2: FEATURE ENGINEERING
            # ================================================================
            logger.info("\n[STEP 2/5] FEATURE ENGINEERING")
            step_start = time.time()

            engineer = FeatureEngineer()
            features = engineer.compute_all_features(price_data, market_data)

            n_features = engineer.get_feature_count()
            logger.info(f"  Computed {n_features} features in {time.time() - step_start:.1f}s")

            # ================================================================
            # STEP 3: TRAIN MODELS FOR EACH STOCK
            # ================================================================
            logger.info("\n[STEP 3/5] MODEL TRAINING")

            for i, symbol in enumerate(price_data.keys()):
                logger.info(f"\n  [{i+1}/{len(price_data)}] Training models for {symbol}...")

                try:
                    result = self._train_and_evaluate_stock(
                        symbol=symbol,
                        features_df=features.get(symbol),
                        price_df=price_data.get(symbol),
                        initial_capital=initial_capital
                    )

                    if result is not None:
                        self.stock_results.append(result)
                        logger.info(f"    Direction Accuracy: {result.direction_accuracy:.2%}")
                        logger.info(f"    Sharpe Ratio: {result.sharpe_ratio:.2f}")

                except Exception as e:
                    logger.error(f"    Failed for {symbol}: {e}")

            # ================================================================
            # STEP 4: GENERATE PLOTS
            # ================================================================
            logger.info("\n[STEP 4/5] GENERATING PLOTS")
            self._generate_all_plots()

            # ================================================================
            # STEP 5: SAVE RESULTS TO CSV
            # ================================================================
            logger.info("\n[STEP 5/5] SAVING RESULTS")
            self._save_all_results()

            # ================================================================
            # GENERATE SUMMARY
            # ================================================================
            summary = self._generate_summary(
                experiment_id=experiment_id,
                n_stocks=len(price_data),
                n_features=n_features,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                total_time=time.time() - total_start
            )

            # Generate HTML report
            self._generate_html_report(summary)

            logger.info("\n" + "=" * 80)
            logger.info("EXPERIMENT COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Results saved to: {self.run_dir}")

            return summary

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()

            return ExperimentSummary(
                experiment_id=experiment_id,
                timestamp=self.timestamp,
                n_stocks=len(symbols),
                n_features=0,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                avg_direction_accuracy=0,
                avg_rmse=0,
                avg_sharpe=0,
                avg_return=0,
                best_stock='N/A',
                worst_stock='N/A',
                total_time=time.time() - total_start,
                status='failed'
            )

    def _train_and_evaluate_stock(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        price_df: pd.DataFrame,
        initial_capital: float
    ) -> Optional[StockResult]:
        """Train models and evaluate for a single stock."""

        if features_df is None or len(features_df) < 200:
            logger.warning(f"    Insufficient data for {symbol}")
            return None

        if price_df is None or len(price_df) < 200:
            logger.warning(f"    Insufficient price data for {symbol}")
            return None

        # Merge features with price data (features don't have OHLCV)
        features_df = features_df.copy()

        # Align indices
        common_idx = features_df.index.intersection(price_df.index)
        features_df = features_df.loc[common_idx]
        price_df = price_df.loc[common_idx]

        # Add OHLCV columns from price data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in price_df.columns:
                features_df[col] = price_df[col]

        # Prepare target: next day return
        features_df['close_return'] = features_df['close'].pct_change().shift(-1)
        features_df = features_df.dropna()

        if len(features_df) < 150:
            logger.warning(f"    Insufficient samples after target computation: {len(features_df)}")
            return None

        # Prepare ML data
        trainer = MLModelTrainer()
        X, y, feature_names, dates = trainer.prepare_data(features_df, 'close_return')

        # Train all models
        models = trainer.train_all_models(X, y, feature_names, test_size=0.2)

        if not models:
            return None

        # Get metrics
        metrics_df = trainer.get_metrics_df()

        # Find best model
        if len(metrics_df) > 0:
            best_idx = metrics_df['rmse'].idxmin()
            best_model = metrics_df.loc[best_idx, 'model']
            best_rmse = metrics_df.loc[best_idx, 'rmse']
            best_mae = metrics_df.loc[best_idx, 'mae']
            best_mape = metrics_df.loc[best_idx, 'mape']
            best_r2 = metrics_df.loc[best_idx, 'r2']
            best_direction = metrics_df.loc[best_idx, 'direction_accuracy']
        else:
            return None

        # Make predictions for backtesting
        test_size = int(len(X) * 0.2)
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        dates_test = dates[-test_size:]
        prices_test = features_df['close'].iloc[-test_size:].values

        predictions = trainer.predict_ensemble(X_test)

        # Align predictions (sequence models may have fewer)
        pred_len = min(len(predictions), len(y_test))
        predictions = predictions[-pred_len:]
        y_test_aligned = y_test[-pred_len:]
        dates_aligned = dates_test[-pred_len:]
        prices_aligned = prices_test[-pred_len:]

        # Backtest with predictions
        trading_metrics = self._backtest_predictions(
            predictions=predictions,
            actual_returns=y_test_aligned,
            prices=prices_aligned,
            initial_capital=initial_capital
        )

        # Store predictions for plotting
        self.all_predictions[symbol] = pd.DataFrame({
            'date': dates_aligned,
            'actual_return': y_test_aligned,
            'predicted_return': predictions,
            'price': prices_aligned
        })

        # Store model metrics
        self.model_metrics[symbol] = metrics_df

        # Get individual model metrics
        def get_model_metric(name, col):
            row = metrics_df[metrics_df['model'] == name]
            return row[col].values[0] if len(row) > 0 else 0

        return StockResult(
            symbol=symbol,
            sector=config.STOCK_SECTOR_MAP.get(symbol, 'Other'),
            n_samples=len(X),
            n_features=len(feature_names),
            start_date=str(dates[0])[:10],
            end_date=str(dates[-1])[:10],
            best_model=best_model,
            rmse=best_rmse,
            mae=best_mae,
            mape=best_mape,
            r2=best_r2,
            direction_accuracy=best_direction,
            total_return=trading_metrics['total_return'],
            annual_return=trading_metrics['annual_return'],
            sharpe_ratio=trading_metrics['sharpe_ratio'],
            max_drawdown=trading_metrics['max_drawdown'],
            win_rate=trading_metrics['win_rate'],
            profit_factor=trading_metrics['profit_factor'],
            xgboost_rmse=get_model_metric('xgboost', 'rmse'),
            xgboost_direction=get_model_metric('xgboost', 'direction_accuracy'),
            lightgbm_rmse=get_model_metric('lightgbm', 'rmse'),
            lightgbm_direction=get_model_metric('lightgbm', 'direction_accuracy'),
            lstm_rmse=get_model_metric('lstm', 'rmse'),
            lstm_direction=get_model_metric('lstm', 'direction_accuracy'),
            gru_rmse=get_model_metric('gru', 'rmse'),
            gru_direction=get_model_metric('gru', 'direction_accuracy'),
            ensemble_rmse=get_model_metric('ensemble', 'rmse'),
            ensemble_direction=get_model_metric('ensemble', 'direction_accuracy')
        )

    def _backtest_predictions(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        prices: np.ndarray,
        initial_capital: float
    ) -> Dict[str, float]:
        """Backtest trading based on ML predictions."""

        if len(predictions) < 10:
            return {
                'total_return': 0, 'annual_return': 0, 'sharpe_ratio': 0,
                'max_drawdown': 0, 'win_rate': 0.5, 'profit_factor': 1
            }

        # Simple strategy: go long if predicted return > 0
        positions = (predictions > 0).astype(int)
        strategy_returns = positions[:-1] * actual_returns[1:]  # Shift by 1 day

        # Calculate metrics
        total_return = np.sum(strategy_returns)
        n_days = len(strategy_returns)
        annual_return = total_return * (252 / n_days) if n_days > 0 else 0

        # Sharpe ratio
        if strategy_returns.std() > 0:
            sharpe_ratio = (strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0

        # Drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Win rate
        wins = (strategy_returns > 0).sum()
        total_trades = (positions[:-1] != 0).sum()
        win_rate = wins / total_trades if total_trades > 0 else 0.5

        # Profit factor
        gross_profit = strategy_returns[strategy_returns > 0].sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 2.0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

    def _generate_all_plots(self):
        """Generate plots for all stocks."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")
            return

        plots_dir = os.path.join(self.run_dir, 'plots')

        for symbol, pred_df in self.all_predictions.items():
            try:
                self._plot_stock_predictions(symbol, pred_df, plots_dir)
            except Exception as e:
                logger.warning(f"Failed to plot {symbol}: {e}")

        # Generate summary plots
        self._plot_summary_charts(plots_dir)

    def _plot_stock_predictions(self, symbol: str, pred_df: pd.DataFrame, output_dir: str):
        """Generate prediction plot for a single stock."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        dates = pd.to_datetime(pred_df['date'])

        # Plot 1: Actual vs Predicted Returns
        ax1 = axes[0]
        ax1.plot(dates, pred_df['actual_return'] * 100, label='Actual Return', alpha=0.7, linewidth=1)
        ax1.plot(dates, pred_df['predicted_return'] * 100, label='Predicted Return', alpha=0.7, linewidth=1)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(f'{symbol} - Actual vs Predicted Daily Returns', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Return (%)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative Returns
        ax2 = axes[1]
        actual_cumulative = (1 + pred_df['actual_return']).cumprod() - 1
        predicted_cumulative = (1 + pred_df['predicted_return']).cumprod() - 1

        # Strategy: trade based on predictions
        positions = (pred_df['predicted_return'] > 0).astype(int)
        strategy_returns = positions.shift(1).fillna(0) * pred_df['actual_return']
        strategy_cumulative = (1 + strategy_returns).cumprod() - 1

        ax2.plot(dates, actual_cumulative * 100, label='Buy & Hold', linewidth=1.5)
        ax2.plot(dates, strategy_cumulative * 100, label='ML Strategy', linewidth=1.5)
        ax2.set_title(f'{symbol} - Cumulative Returns: Buy & Hold vs ML Strategy', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Direction Accuracy Over Time
        ax3 = axes[2]
        correct_direction = ((pred_df['actual_return'] > 0) == (pred_df['predicted_return'] > 0)).astype(int)
        rolling_accuracy = correct_direction.rolling(window=20, min_periods=1).mean() * 100

        ax3.plot(dates, rolling_accuracy, label='20-day Rolling Accuracy', color='green', linewidth=1.5)
        ax3.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        ax3.fill_between(dates, 50, rolling_accuracy, where=rolling_accuracy >= 50,
                        alpha=0.3, color='green', label='Above Random')
        ax3.fill_between(dates, 50, rolling_accuracy, where=rolling_accuracy < 50,
                        alpha=0.3, color='red', label='Below Random')
        ax3.set_title(f'{symbol} - Direction Prediction Accuracy', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_ylim(30, 80)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{symbol}_predictions.png'), dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"    Saved plot for {symbol}")

    def _plot_summary_charts(self, output_dir: str):
        """Generate summary charts across all stocks."""
        import matplotlib.pyplot as plt

        if not self.stock_results:
            return

        # Convert to DataFrame
        results_df = pd.DataFrame([asdict(r) for r in self.stock_results])

        # Chart 1: Direction Accuracy by Stock
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        sorted_df = results_df.sort_values('direction_accuracy', ascending=True)
        colors = ['green' if x > 0.55 else 'orange' if x > 0.5 else 'red'
                 for x in sorted_df['direction_accuracy']]
        ax1.barh(sorted_df['symbol'], sorted_df['direction_accuracy'] * 100, color=colors)
        ax1.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Random')
        ax1.axvline(x=55, color='green', linestyle='--', alpha=0.7, label='Target')
        ax1.set_xlabel('Direction Accuracy (%)')
        ax1.set_title('Direction Accuracy by Stock', fontsize=12, fontweight='bold')
        ax1.legend()

        # Chart 2: Model Comparison
        ax2 = axes[0, 1]
        model_cols = ['xgboost_direction', 'lightgbm_direction', 'lstm_direction',
                     'gru_direction', 'ensemble_direction']
        model_names = ['XGBoost', 'LightGBM', 'LSTM', 'GRU', 'Ensemble']
        model_avgs = [results_df[col].mean() * 100 for col in model_cols]

        bars = ax2.bar(model_names, model_avgs, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'])
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Average Direction Accuracy (%)')
        ax2.set_title('Model Comparison: Direction Accuracy', fontsize=12, fontweight='bold')
        for bar, val in zip(bars, model_avgs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=10)

        # Chart 3: Sharpe Ratio by Sector
        ax3 = axes[1, 0]
        sector_sharpe = results_df.groupby('sector')['sharpe_ratio'].mean().sort_values()
        colors = ['green' if x > 1 else 'orange' if x > 0 else 'red' for x in sector_sharpe]
        ax3.barh(sector_sharpe.index, sector_sharpe.values, color=colors)
        ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax3.axvline(x=1, color='green', linestyle='--', alpha=0.7, label='Good (>1)')
        ax3.set_xlabel('Average Sharpe Ratio')
        ax3.set_title('Sharpe Ratio by Sector', fontsize=12, fontweight='bold')

        # Chart 4: Return vs Risk Scatter
        ax4 = axes[1, 1]
        scatter = ax4.scatter(
            results_df['max_drawdown'] * 100,
            results_df['annual_return'] * 100,
            c=results_df['sharpe_ratio'],
            cmap='RdYlGn',
            s=100,
            alpha=0.7
        )
        ax4.set_xlabel('Max Drawdown (%)')
        ax4.set_ylabel('Annual Return (%)')
        ax4.set_title('Return vs Risk by Stock', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax4, label='Sharpe Ratio')

        # Add labels
        for _, row in results_df.iterrows():
            ax4.annotate(row['symbol'],
                        (row['max_drawdown'] * 100, row['annual_return'] * 100),
                        fontsize=8, alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'summary_charts.png'), dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("  Saved summary charts")

    def _save_all_results(self):
        """Save all results to CSV files."""
        metrics_dir = os.path.join(self.run_dir, 'metrics')
        predictions_dir = os.path.join(self.run_dir, 'predictions')

        # 1. Stock Results CSV
        if self.stock_results:
            results_df = pd.DataFrame([asdict(r) for r in self.stock_results])
            results_df.to_csv(os.path.join(metrics_dir, 'stock_results.csv'), index=False)
            logger.info(f"  Saved stock_results.csv ({len(results_df)} stocks)")

        # 2. All Predictions CSV
        for symbol, pred_df in self.all_predictions.items():
            pred_df.to_csv(os.path.join(predictions_dir, f'{symbol}_predictions.csv'), index=False)

        logger.info(f"  Saved {len(self.all_predictions)} prediction files")

        # 3. Model Comparison CSV
        all_model_metrics = []
        for symbol, metrics_df in self.model_metrics.items():
            metrics_df = metrics_df.copy()
            metrics_df['symbol'] = symbol
            all_model_metrics.append(metrics_df)

        if all_model_metrics:
            combined_metrics = pd.concat(all_model_metrics, ignore_index=True)
            combined_metrics.to_csv(os.path.join(metrics_dir, 'model_comparison.csv'), index=False)
            logger.info(f"  Saved model_comparison.csv")

        # 4. Sector Summary CSV
        if self.stock_results:
            results_df = pd.DataFrame([asdict(r) for r in self.stock_results])
            sector_summary = results_df.groupby('sector').agg({
                'direction_accuracy': 'mean',
                'sharpe_ratio': 'mean',
                'annual_return': 'mean',
                'max_drawdown': 'mean',
                'win_rate': 'mean',
                'symbol': 'count'
            }).rename(columns={'symbol': 'n_stocks'})
            sector_summary.to_csv(os.path.join(metrics_dir, 'sector_summary.csv'))
            logger.info(f"  Saved sector_summary.csv")

    def _generate_summary(
        self,
        experiment_id: str,
        n_stocks: int,
        n_features: int,
        start_date: str,
        end_date: str,
        initial_capital: float,
        total_time: float
    ) -> ExperimentSummary:
        """Generate experiment summary."""

        if not self.stock_results:
            return ExperimentSummary(
                experiment_id=experiment_id,
                timestamp=self.timestamp,
                n_stocks=n_stocks,
                n_features=n_features,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                avg_direction_accuracy=0,
                avg_rmse=0,
                avg_sharpe=0,
                avg_return=0,
                best_stock='N/A',
                worst_stock='N/A',
                total_time=total_time,
                status='no_results'
            )

        # Calculate averages
        results_df = pd.DataFrame([asdict(r) for r in self.stock_results])

        avg_direction = results_df['direction_accuracy'].mean()
        avg_rmse = results_df['rmse'].mean()
        avg_sharpe = results_df['sharpe_ratio'].mean()
        avg_return = results_df['annual_return'].mean()

        best_idx = results_df['sharpe_ratio'].idxmax()
        worst_idx = results_df['sharpe_ratio'].idxmin()
        best_stock = results_df.loc[best_idx, 'symbol']
        worst_stock = results_df.loc[worst_idx, 'symbol']

        return ExperimentSummary(
            experiment_id=experiment_id,
            timestamp=self.timestamp,
            n_stocks=len(results_df),
            n_features=n_features,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            avg_direction_accuracy=avg_direction,
            avg_rmse=avg_rmse,
            avg_sharpe=avg_sharpe,
            avg_return=avg_return,
            best_stock=best_stock,
            worst_stock=worst_stock,
            total_time=total_time,
            status='success'
        )

    def _generate_html_report(self, summary: ExperimentSummary):
        """Generate comprehensive HTML report."""

        results_df = pd.DataFrame([asdict(r) for r in self.stock_results]) if self.stock_results else pd.DataFrame()

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ML Experiment Report - {summary.experiment_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .summary-card.good {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .summary-card.warning {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .summary-card h3 {{ margin: 0 0 10px 0; font-size: 14px; opacity: 0.9; }}
        .summary-card .value {{ font-size: 28px; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .good {{ color: #27ae60; font-weight: bold; }}
        .warning {{ color: #f39c12; font-weight: bold; }}
        .bad {{ color: #e74c3c; font-weight: bold; }}
        .chart-container {{ text-align: center; margin: 30px 0; }}
        .chart-container img {{ max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ML Stock Prediction Experiment Report</h1>

        <h2>Experiment Summary</h2>
        <div class="summary-grid">
            <div class="summary-card {'good' if summary.avg_direction_accuracy > 0.55 else 'warning' if summary.avg_direction_accuracy > 0.5 else ''}">
                <h3>AVG DIRECTION ACCURACY</h3>
                <div class="value">{summary.avg_direction_accuracy:.1%}</div>
            </div>
            <div class="summary-card {'good' if summary.avg_sharpe > 1 else 'warning' if summary.avg_sharpe > 0 else ''}">
                <h3>AVG SHARPE RATIO</h3>
                <div class="value">{summary.avg_sharpe:.2f}</div>
            </div>
            <div class="summary-card {'good' if summary.avg_return > 0.1 else 'warning' if summary.avg_return > 0 else ''}">
                <h3>AVG ANNUAL RETURN</h3>
                <div class="value">{summary.avg_return:.1%}</div>
            </div>
            <div class="summary-card">
                <h3>STOCKS ANALYZED</h3>
                <div class="value">{summary.n_stocks}</div>
            </div>
        </div>

        <h2>Experiment Configuration</h2>
        <table>
            <tr><td><strong>Experiment ID</strong></td><td>{summary.experiment_id}</td></tr>
            <tr><td><strong>Period</strong></td><td>{summary.start_date} to {summary.end_date}</td></tr>
            <tr><td><strong>Features</strong></td><td>{summary.n_features}</td></tr>
            <tr><td><strong>Initial Capital</strong></td><td>Rs {summary.initial_capital:,.0f}</td></tr>
            <tr><td><strong>Execution Time</strong></td><td>{summary.total_time:.1f} seconds</td></tr>
            <tr><td><strong>Best Stock</strong></td><td>{summary.best_stock}</td></tr>
            <tr><td><strong>Worst Stock</strong></td><td>{summary.worst_stock}</td></tr>
        </table>

        <h2>Summary Charts</h2>
        <div class="chart-container">
            <img src="plots/summary_charts.png" alt="Summary Charts">
        </div>

        <h2>Stock-by-Stock Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Sector</th>
                    <th>Best Model</th>
                    <th>Direction Acc</th>
                    <th>RMSE</th>
                    <th>Sharpe</th>
                    <th>Annual Return</th>
                    <th>Max DD</th>
                    <th>Win Rate</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add rows for each stock
        for _, row in results_df.iterrows() if len(results_df) > 0 else []:
            dir_class = 'good' if row['direction_accuracy'] > 0.55 else 'warning' if row['direction_accuracy'] > 0.5 else 'bad'
            sharpe_class = 'good' if row['sharpe_ratio'] > 1 else 'warning' if row['sharpe_ratio'] > 0 else 'bad'

            html += f"""
                <tr>
                    <td><strong>{row['symbol']}</strong></td>
                    <td>{row['sector']}</td>
                    <td>{row['best_model']}</td>
                    <td class="{dir_class}">{row['direction_accuracy']:.1%}</td>
                    <td>{row['rmse']:.4f}</td>
                    <td class="{sharpe_class}">{row['sharpe_ratio']:.2f}</td>
                    <td>{row['annual_return']:.1%}</td>
                    <td>{row['max_drawdown']:.1%}</td>
                    <td>{row['win_rate']:.1%}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>

        <h2>Model Performance Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Avg Direction Accuracy</th>
                    <th>Avg RMSE</th>
                    <th>Stocks Trained</th>
                </tr>
            </thead>
            <tbody>
"""

        # Model comparison
        if len(results_df) > 0:
            for model_name, col in [('XGBoost', 'xgboost'), ('LightGBM', 'lightgbm'),
                                    ('LSTM', 'lstm'), ('GRU', 'gru'), ('Ensemble', 'ensemble')]:
                dir_col = f'{col}_direction'
                rmse_col = f'{col}_rmse'
                if dir_col in results_df.columns:
                    valid = results_df[results_df[dir_col] > 0]
                    if len(valid) > 0:
                        avg_dir = valid[dir_col].mean()
                        avg_rmse = valid[rmse_col].mean()
                        dir_class = 'good' if avg_dir > 0.55 else 'warning' if avg_dir > 0.5 else 'bad'
                        html += f"""
                <tr>
                    <td><strong>{model_name}</strong></td>
                    <td class="{dir_class}">{avg_dir:.1%}</td>
                    <td>{avg_rmse:.4f}</td>
                    <td>{len(valid)}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>

        <h2>Individual Stock Plots</h2>
        <p>See the <code>plots/</code> directory for detailed prediction charts for each stock.</p>

        <div class="footer">
            <p>Generated by ML Experiment Runner | {timestamp}</p>
            <p>Results Directory: {run_dir}</p>
        </div>
    </div>
</body>
</html>
""".format(timestamp=self.timestamp, run_dir=self.run_dir)

        # Save report
        report_path = os.path.join(self.run_dir, 'report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"  Saved report.html")

        # Also save summary JSON
        summary_path = os.path.join(self.run_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(asdict(summary), f, indent=2, default=str)

        logger.info(f"  Saved summary.json")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run ML Stock Prediction Experiment')
    parser.add_argument('--symbols', type=int, default=None,
                       help='Number of stocks to analyze (default: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with 5 stocks')
    parser.add_argument('--start', default='2020-01-01',
                       help='Start date (default: 2020-01-01)')
    parser.add_argument('--capital', type=float, default=1000000,
                       help='Initial capital (default: 1000000)')

    args = parser.parse_args()

    # Determine symbols
    if args.quick:
        symbols = config.ALL_STOCKS[:5]
    elif args.symbols:
        symbols = config.ALL_STOCKS[:args.symbols]
    else:
        symbols = config.ALL_STOCKS

    # Run experiment
    runner = MLExperimentRunner()
    summary = runner.run_experiment(
        symbols=symbols,
        start_date=args.start,
        initial_capital=args.capital
    )

    # Print final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Status: {summary.status}")
    print(f"Stocks Analyzed: {summary.n_stocks}")
    print(f"Average Direction Accuracy: {summary.avg_direction_accuracy:.1%}")
    print(f"Average Sharpe Ratio: {summary.avg_sharpe:.2f}")
    print(f"Average Annual Return: {summary.avg_return:.1%}")
    print(f"Best Stock: {summary.best_stock}")
    print(f"Execution Time: {summary.total_time:.1f}s")
    print(f"\nResults saved to: {runner.run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
