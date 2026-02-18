"""
================================================================================
UNIFIED ORCHESTRATOR - 8-Step Pipeline
================================================================================
Coordinates the complete trading pipeline with progress callbacks.

Steps:
1. DATA COLLECTION      -> Download OHLCV + market indices
2. FEATURE ENGINEERING   -> ~130 features
3. SENTIMENT ANALYSIS    -> Google News RSS (VADER + TextBlob)
4. FACTOR ANALYSIS       -> Value, Momentum, Quality, Low-Vol, Sentiment
5. ML MODEL TRAINING     -> XGBoost, LightGBM, LSTM, GRU, Ensemble
6. PORTFOLIO OPTIMIZATION -> Risk parity / Max Sharpe / Min Vol / Equal Weight
7. BACKTESTING           -> Per-stock + portfolio-level
8. SIGNAL GENERATION     -> BUY/SELL/HOLD with confidence
================================================================================
"""

import os
import sys
import time
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from engine.data_collector import DataCollector
from engine.feature_engine import AdvancedFeatureEngine, FeatureEngineer
from engine.sentiment import FastSentimentEngine
from engine.factor_analyzer import FactorAnalyzer
from engine.ml_models import ProductionModel, MLModelTrainer, ModelEvaluator
from engine.portfolio_optimizer import PortfolioOptimizer
from engine.backtester import WalkForwardBacktester, BacktestValidator
from engine.signal_generator import SignalGenerator
from engine.broker import create_broker
from engine.csv_logger import ProfessionalCSVLogger  # NEW: CSV Logger

# Phase 1: Strategy Optimization Components
from engine.entry_optimizer import EntryOptimizer
from engine.exit_optimizer import ExitOptimizer
from engine.position_sizer import PositionSizer
from engine.risk_manager import RiskManager


@dataclass
class StepStatus:
    """Status of a single pipeline step."""
    step_number: int
    name: str
    status: str = 'pending'  # pending, running, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0
    details: Dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineStatus:
    """Overall pipeline status."""
    job_id: str
    status: str = 'pending'  # pending, running, completed, failed
    current_step: int = 0
    total_steps: int = 8
    steps: List[StepStatus] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    config: Dict = field(default_factory=dict)


class UnifiedOrchestrator:
    """
    Unified 8-step pipeline orchestrator with progress callbacks.

    Supports both:
    - Production-style: per-stock data -> features -> model -> backtest -> signals
    - Pipeline-style: multi-stock data -> factors -> portfolio optimization -> backtest
    """

    def __init__(
        self,
        symbols: List[str] = None,
        initial_capital: float = 100000,
        paper_trading: bool = True,
        results_dir: str = None,
        progress_callback: Callable[[StepStatus], None] = None
    ):
        self.symbols = symbols or config.ALL_STOCKS[:10]
        self.initial_capital = initial_capital
        self.paper_trading = paper_trading
        self.results_dir = results_dir or os.path.join(config.BASE_DIR, 'results')
        self.progress_callback = progress_callback

        # Components
        self.data_collector = DataCollector()
        self.feature_engine = AdvancedFeatureEngine(
            include_sentiment=True,
            include_market_context=False
        )
        self.feature_engineer = FeatureEngineer()
        self.sentiment_engine = None  # Lazy init
        self.factor_analyzer = FactorAnalyzer()
        self.production_model = ProductionModel()
        self.ml_trainer = MLModelTrainer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.backtester = WalkForwardBacktester(initial_capital=initial_capital)
        self.backtest_validator = BacktestValidator(initial_capital=initial_capital)
        self.signal_generator = SignalGenerator()
        self.broker = None
        self.csv_logger = ProfessionalCSVLogger()  # NEW: Initialize CSV logger
        
        # Phase 1: Strategy Optimization Components
        self.entry_optimizer = EntryOptimizer(min_confidence=0.60, min_quality_score=0.70)
        self.exit_optimizer = ExitOptimizer(atr_multiplier=1.5, max_holding_days=5)
        self.position_sizer = PositionSizer(total_capital=initial_capital, max_position_pct=0.10)
        self.risk_manager = RiskManager(initial_capital=initial_capital, max_risk_per_trade=0.01)

        # State / caches
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')  # Unique run identifier
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.features_cache: Dict[str, pd.DataFrame] = {}
        self.sentiment_cache: Dict[str, Dict] = {}
        self.factor_scores = []
        self.allocation = None
        self.backtest_results: Dict[str, Any] = {}
        self.signals = None

        # Pipeline tracking
        self.pipeline_status = None

        # Setup directories
        os.makedirs(self.results_dir, exist_ok=True)
        for subdir in ['data', 'features', 'models', 'backtest', 'signals', 'reports']:
            os.makedirs(os.path.join(self.results_dir, subdir), exist_ok=True)

        logger.info(f"UnifiedOrchestrator initialized: {len(self.symbols)} symbols, "
                    f"capital={initial_capital}")

    def _update_step(self, step: StepStatus):
        """Notify progress callback."""
        if self.progress_callback:
            self.progress_callback(step)


    def _create_pipeline_summary_csv(self, status: PipelineStatus):
        """Create a comprehensive summary CSV combining all pipeline steps."""
        summary_dir = os.path.join(self.results_dir, 'pipeline_summaries')
        os.makedirs(summary_dir, exist_ok=True)
        
        # Collect all step results
        summary_data = []
        
        for step in status.steps:
            row = {
                'run_id': self.run_id,
                'pipeline_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'step_number': step.step_number,
                'step_name': step.name,
                'step_status': step.status,
                'duration_seconds': step.duration_seconds
            }
            
            # Add step-specific metrics
            if step.details:
                for key, value in step.details.items():
                    # Convert nested dicts to strings
                    if isinstance(value, dict):
                        row[f'{step.name}_{key}'] = str(value)
                    elif isinstance(value, (int, float, str)):
                        row[f'{step.name}_{key}'] = value
            
            summary_data.append(row)
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary CSV
        summary_filename = f"pipeline_summary_{self.run_id}.csv"
        summary_filepath = os.path.join(summary_dir, summary_filename)
        summary_df.to_csv(summary_filepath, index=False)
        
        logger.info(f"✅ Pipeline summary saved -> {summary_filepath}")
        
        # Also create a master CSV that appends to history
        master_filepath = os.path.join(summary_dir, "all_pipeline_runs.csv")
        
        # Append to master file
        if os.path.exists(master_filepath):
            summary_df.to_csv(master_filepath, mode='a', header=False, index=False)
        else:
            summary_df.to_csv(master_filepath, index=False)
        
        logger.info(f"✅ Master pipeline history updated -> {master_filepath}")
        
        return summary_filepath

    def run_pipeline(
        self,
        optimization_method: str = 'risk_parity',
        n_holdings: int = 15,
        start_date: str = '2022-01-01',
        force_download: bool = True,
        models_to_train: List[str] = None
    ) -> PipelineStatus:
        """
        Run the complete 8-step pipeline.

        Args:
            optimization_method: Portfolio optimization method
            n_holdings: Number of stocks to hold
            start_date: Backtest start date
            force_download: Force fresh data download
            models_to_train: List of model names to train

        Returns:
            PipelineStatus with all step results
        """
        job_id = str(uuid.uuid4())[:8]
        self.pipeline_status = PipelineStatus(
            job_id=job_id,
            status='running',
            started_at=datetime.now().isoformat(),
            symbols=self.symbols,
            config={
                'optimization_method': optimization_method,
                'n_holdings': n_holdings,
                'start_date': start_date,
                'initial_capital': self.initial_capital
            }
        )

        # Initialize step statuses
        step_names = [
            'Data Collection', 'Feature Engineering', 'Sentiment Analysis',
            'Factor Analysis', 'ML Model Training', 'Portfolio Optimization',
            'Backtesting', 'Signal Generation'
        ]
        self.pipeline_status.steps = [
            StepStatus(step_number=i+1, name=name)
            for i, name in enumerate(step_names)
        ]

        logger.info("#" * 60)
        logger.info(f"# UNIFIED PIPELINE - Job {job_id}")
        logger.info(f"# Symbols: {len(self.symbols)}")
        logger.info(f"# Method: {optimization_method}")
        logger.info("#" * 60)

        pipeline_start = time.time()

        try:
            # Step 1: Data Collection
            self._run_step(0, self._step_data_collection, force_download, start_date)

            # Step 2: Feature Engineering
            self._run_step(1, self._step_feature_engineering)

            # Step 3: Sentiment Analysis
            self._run_step(2, self._step_sentiment_analysis)

            # Step 4: Factor Analysis
            self._run_step(3, self._step_factor_analysis)

            # Step 5: ML Model Training
            self._run_step(4, self._step_ml_training, models_to_train)

            # Step 6: Portfolio Optimization
            self._run_step(5, self._step_portfolio_optimization,
                          optimization_method, n_holdings)

            # Step 7: Backtesting
            self._run_step(6, self._step_backtesting, start_date)

            # Step 8: Signal Generation
            self._run_step(7, self._step_signal_generation)

            self.pipeline_status.status = 'completed'
            self.pipeline_status.completed_at = datetime.now().isoformat()
            
            # Create comprehensive pipeline summary CSV
            self._create_pipeline_summary_csv(self.pipeline_status)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.pipeline_status.status = 'failed'

        elapsed = time.time() - pipeline_start
        logger.info(f"Pipeline completed in {elapsed/60:.1f} minutes")

        return self.pipeline_status

    def _run_step(self, step_idx: int, func, *args):
        """Run a pipeline step with timing and error handling. Also saves step results to CSV."""
        step = self.pipeline_status.steps[step_idx]
        step.status = 'running'
        step.started_at = datetime.now().isoformat()
        self.pipeline_status.current_step = step_idx + 1
        self._update_step(step)

        logger.info(f"\n{'='*60}")
        logger.info(f"STEP {step_idx+1}/8: {step.name}")
        logger.info(f"{'='*60}")

        start = time.time()
        try:
            details = func(*args)
            step.details = details or {}
            step.status = 'completed'
            # --- Save step results to CSV ---
            job_id = getattr(self.pipeline_status, 'job_id', 'unknown')
            step_dir = os.path.join(self.results_dir, 'pipeline_runs', job_id)
            os.makedirs(step_dir, exist_ok=True)
            # Save details as CSV if possible
            if details:
                # If details is a dict of lists or DataFrames, try to save each as CSV
                for k, v in details.items():
                    if isinstance(v, pd.DataFrame):
                        v.to_csv(os.path.join(step_dir, f'step{step_idx+1}_{step.name.replace(" ", "_").lower()}_{k}.csv'), index=False)
                    elif isinstance(v, list) and v and isinstance(v[0], dict):
                        pd.DataFrame(v).to_csv(os.path.join(step_dir, f'step{step_idx+1}_{step.name.replace(" ", "_").lower()}_{k}.csv'), index=False)
                    elif isinstance(v, dict) and v and all(isinstance(val, (int, float, str)) for val in v.values()):
                        pd.DataFrame([v]).to_csv(os.path.join(step_dir, f'step{step_idx+1}_{step.name.replace(" ", "_").lower()}_{k}.csv'), index=False)
                # If details itself is a DataFrame
                if isinstance(details, pd.DataFrame):
                    details.to_csv(os.path.join(step_dir, f'step{step_idx+1}_{step.name.replace(" ", "_").lower()}.csv'), index=False)
                # If details is a list of dicts
                elif isinstance(details, list) and details and isinstance(details[0], dict):
                    pd.DataFrame(details).to_csv(os.path.join(step_dir, f'step{step_idx+1}_{step.name.replace(" ", "_").lower()}.csv'), index=False)
                # If details is a dict of simple values
                elif isinstance(details, dict) and details and all(isinstance(val, (int, float, str)) for val in details.values()):
                    pd.DataFrame([details]).to_csv(os.path.join(step_dir, f'step{step_idx+1}_{step.name.replace(" ", "_").lower()}.csv'), index=False)
        except Exception as e:
            step.status = 'failed'
            step.error = str(e)
            logger.error(f"Step {step_idx+1} failed: {e}")
            raise

        step.duration_seconds = time.time() - start
        step.completed_at = datetime.now().isoformat()
        self._update_step(step)

    # ==================== STEP IMPLEMENTATIONS ====================

    def _step_data_collection(self, force_download: bool, start_date: str) -> Dict:
        """Step 1: Collect data for all symbols."""
        price_data, market_data = self.data_collector.collect_all(
            symbols=self.symbols,
            start_date=start_date,
            force_download=force_download
        )

        self.data_cache = price_data
        self.market_data = market_data
        
        # Save per-stock collected data to CSV
        data_dir = os.path.join(self.results_dir, 'data_collected')
        os.makedirs(data_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for symbol, df in price_data.items():
            if len(df) > 0:
                df_save = df.copy()
                df_save['collection_timestamp'] = timestamp
                df_save['run_id'] = self.run_id
                
                filename = f"{symbol}_{timestamp}.csv"
                filepath = os.path.join(data_dir, filename)
                df_save.to_csv(filepath, index=False)
                logger.info(f"Saved {symbol} data: {len(df)} rows -> {filepath}")

        return {
            'symbols_collected': len(price_data),
            'market_indices': len(market_data),
            'total_rows': sum(len(df) for df in price_data.values()),
            'csv_saved': len(price_data),
            'date_range': {
                sym: f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
                for sym, df in list(price_data.items())[:3]
                if 'timestamp' in df.columns
            }
        }

    def _step_feature_engineering(self) -> Dict:
        """Step 2: Compute features for all symbols."""
        for symbol, df in self.data_cache.items():
            try:
                feature_set = self.feature_engine.compute_all_features(df, symbol)
                feature_df = self.feature_engine.compute_targets(feature_set.df)
                feature_df = feature_df.dropna()
                self.features_cache[symbol] = feature_df
            except Exception as e:
                logger.error(f"Features failed for {symbol}: {e}")
        
        # Save combined features to CSV
        if self.features_cache:
            features_dir = os.path.join(self.results_dir, 'features_combined')
            os.makedirs(features_dir, exist_ok=True)
            
            # Combine all features with symbol column
            combined_dfs = []
            for symbol, df in self.features_cache.items():
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                df_copy['run_id'] = self.run_id
                df_copy['feature_engineering_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                combined_dfs.append(df_copy)
            
            combined_features = pd.concat(combined_dfs, ignore_index=True)
            
            # Save to CSV
            filename = f"features_combined_{self.run_id}.csv"
            filepath = os.path.join(features_dir, filename)
            combined_features.to_csv(filepath, index=False)
            
            logger.info(f"Saved combined features: {len(combined_features)} rows, {len(combined_features.columns)} columns -> {filepath}")

        return {
            'symbols_processed': len(self.features_cache),
            'avg_features': int(np.mean([len(df.columns) for df in self.features_cache.values()])) if self.features_cache else 0,
            'avg_samples': int(np.mean([len(df) for df in self.features_cache.values()])) if self.features_cache else 0,
            'csv_saved': 1 if self.features_cache else 0
        }

    def _step_sentiment_analysis(self) -> Dict:
        """Step 3: Compute sentiment for all symbols."""
        try:
            self.sentiment_engine = FastSentimentEngine()
        except Exception as e:
            logger.warning(f"Sentiment engine not available: {e}")
            return {'status': 'skipped', 'reason': str(e)}

        sentiments = {}
        for symbol in self.features_cache.keys():
            try:
                result = self.sentiment_engine.get_sentiment_scores(symbol)
                sentiments[symbol] = result
                self.sentiment_cache[symbol] = result
            except Exception as e:
                logger.warning(f"Sentiment failed for {symbol}: {e}")

        return {
            'symbols_analyzed': len(sentiments),
            'avg_score': float(np.mean([s.get('score', 0) for s in sentiments.values()])) if sentiments else 0
        }

    def _step_factor_analysis(self) -> Dict:
        """Step 4: Compute factor scores."""
        # Need price data with DatetimeIndex for factor analysis
        price_data_indexed = {}
        for symbol, df in self.data_cache.items():
            df_copy = df.copy()
            if 'timestamp' in df_copy.columns:
                df_copy = df_copy.set_index('timestamp')
            price_data_indexed[symbol] = df_copy

        # Feature data for pipeline-style analysis
        features_indexed = {}
        for symbol, df in self.features_cache.items():
            if 'timestamp' in df.columns:
                features_indexed[symbol] = df.set_index('timestamp')
            else:
                features_indexed[symbol] = df

        self.factor_scores = self.factor_analyzer.compute_factors(
            price_data_indexed, features_indexed
        )

        top_10 = self.factor_analyzer.get_top_stocks(self.factor_scores, n=10)

        return {
            'stocks_scored': len(self.factor_scores),
            'top_10': [
                {'symbol': s.symbol, 'score': round(s.combined_score, 3)}
                for s in top_10
            ]
        }

    def _step_ml_training(self, models_to_train: List[str] = None) -> Dict:
        """Step 5: Train ML models with proper temporal splits."""
        if not self.features_cache:
            return {'status': 'skipped', 'reason': 'No features available'}

        # CRITICAL FIX: Split each stock FIRST, then combine training portions only
        # This prevents mixing recent data from Stock A with old data from Stock B
        
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        logger.info("Performing per-symbol temporal splits...")
        
        for sym, df in self.features_cache.items():
            df_copy = df.copy()
            df_copy['_symbol'] = sym
            
            # Preserve temporal order with index
            df_copy = df_copy.reset_index(drop=True)
            df_copy['_index'] = range(len(df_copy))
            
            # 70% train, 15% val, 15% test (chronological)
            n = len(df_copy)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            
            train_portion = df_copy.iloc[:train_end]
            val_portion = df_copy.iloc[train_end:val_end]
            test_portion = df_copy.iloc[val_end:]
            
            if len(train_portion) > 0:
                train_dfs.append(train_portion)
            if len(val_portion) > 0:
                val_dfs.append(val_portion)
            if len(test_portion) > 0:
                test_dfs.append(test_portion)
            
            logger.info(f"  {sym}: train={len(train_portion)}, val={len(val_portion)}, test={len(test_portion)}")
        
        # Combine AFTER splitting
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        
        logger.info(f"Combined: train={len(train_df)}, val={len(val_df)}")

        # Get feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                       'date', 'symbol', '_symbol', '_index'] + [c for c in train_df.columns if c.startswith('target_')]
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values
        
        # Prepare targets
        close_return_train = train_df['target_close_return'].values.copy()
        close_return_val = val_df['target_close_return'].values.copy()
        close_return_train = np.nan_to_num(close_return_train, nan=0.0, posinf=0.0, neginf=0.0)
        close_return_val = np.nan_to_num(close_return_val, nan=0.0, posinf=0.0, neginf=0.0)

        close_return_5d_train = None
        close_return_5d_val = None
        if 'target_close_return' in train_df:
            close_return_5d_train = train_df['target_close_return'].rolling(5, min_periods=1).sum().values
            close_return_5d_val = val_df['target_close_return'].rolling(5, min_periods=1).sum().values
            close_return_5d_train = np.nan_to_num(close_return_5d_train, nan=0.0, posinf=0.0, neginf=0.0)
            close_return_5d_val = np.nan_to_num(close_return_5d_val, nan=0.0, posinf=0.0, neginf=0.0)

        y_train = {
            'direction': train_df['target_direction'].values,
            'close_return': close_return_train,
            'close_return_5d': close_return_5d_train
        }
        
        y_val = {
            'direction': val_df['target_direction'].values,
            'close_return': close_return_val,
            'close_return_5d': close_return_5d_val
        }
        
        # Log class balance
        n_up = (y_train['direction'] == 1).sum()
        n_down = (y_train['direction'] == 0).sum()
        logger.info(f"Class balance: UP={n_up} ({n_up/len(y_train['direction'])*100:.1f}%), DOWN={n_down} ({n_down/len(y_train['direction'])*100:.1f}%)")
        
        # ============================================================================
        # CRITICAL: Force fresh model training - delete old models and reset state
        # ============================================================================
        model_dir = os.path.join(self.results_dir, 'models', 'combined')
        
        # Delete all old model files to prevent loading leaked models
        if os.path.exists(model_dir):
            logger.warning(f"Deleting old models from {model_dir} to prevent data leakage...")
            try:
                import shutil
                shutil.rmtree(model_dir)
                logger.success("Old models deleted successfully")
            except Exception as e:
                logger.error(f"Failed to delete old models: {e}")
        
        # Create fresh model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Create FRESH ProductionModel instance with new scalers
        from engine.ml_models import ProductionModel
        self.production_model = ProductionModel()  # Fresh instance, no loaded state
        logger.info("Created fresh ProductionModel instance with new scalers")

        # Train production model from scratch
        metrics = self.production_model.train(X_train, y_train, X_val, y_val,
                                             feature_names=feature_cols)

        # Save model
        self.production_model.save(model_dir)
        
        # Save training metrics to CSV
        metrics_dir = os.path.join(self.results_dir, 'training_metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Create comprehensive metrics DataFrame
        metrics_data = {
            'run_id': [self.run_id],
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'samples_trained': [len(X_train)],
            'samples_validated': [len(X_val)],
            'features_used': [len(feature_cols)],
            'n_up': [n_up],
            'n_down': [n_down],
            'class_balance_pct': [f"{n_up/len(y_train['direction'])*100:.1f}% UP / {n_down/len(y_train['direction'])*100:.1f}% DOWN"]
        }
        
        # Add all model metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_data[key] = [round(value, 4) if isinstance(value, float) else value]
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Save to CSV (append mode to track all runs)
        metrics_filename = f"training_metrics_{self.run_id}.csv"
        metrics_filepath = os.path.join(metrics_dir, metrics_filename)
        metrics_df.to_csv(metrics_filepath, index=False)
        
        logger.info(f"Saved training metrics -> {metrics_filepath}")

        # Always include 'direction_accuracy' as ensemble accuracy for frontend
        metrics_out = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
        if 'direction_accuracy_ensemble' in metrics:
            metrics_out['direction_accuracy'] = metrics['direction_accuracy_ensemble']
        elif 'direction_accuracy_xgb' in metrics:
            metrics_out['direction_accuracy'] = metrics['direction_accuracy_xgb']
        
        return {
            'samples_trained': len(X_train),
            'samples_validated': len(X_val),
            'features_used': len(feature_cols),
            'class_balance': f"{n_up}/{n_down}",
            'metrics': metrics_out
        }

    def _step_portfolio_optimization(self, method: str, n_holdings: int) -> Dict:
        """Step 6: Optimize portfolio allocation."""
        price_data_indexed = {}
        for symbol, df in self.data_cache.items():
            df_copy = df.copy()
            if 'timestamp' in df_copy.columns:
                df_copy = df_copy.set_index('timestamp')
            price_data_indexed[symbol] = df_copy

        try:
            self.allocation = self.portfolio_optimizer.optimize(
                price_data=price_data_indexed,
                factor_scores=self.factor_scores,
                n_holdings=n_holdings,
                method=method
            )

            weights = self.allocation.weights if hasattr(self.allocation, 'weights') else {}
            return {
                'method': method,
                'n_holdings': len(weights),
                'weights': {k: round(v, 4) for k, v in list(weights.items())[:20]},
                'expected_sharpe': round(getattr(self.allocation, 'sharpe_ratio', 0), 2),
                'expected_return': round(getattr(self.allocation, 'expected_return', 0), 4)
            }
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _step_backtesting(self, start_date: str) -> Dict:
        """Step 7: Run backtests."""
        results = {}
        
        # Ensure we use the exact features the model was trained on
        trained_features = self.production_model.feature_names
        if not trained_features:
            logger.warning("No trained features found in model, falling back to heuristic")
        
        # Per-stock backtests
        for symbol in list(self.features_cache.keys()):
            try:
                df = self.features_cache[symbol]
                
                # Use trained features if available, otherwise derive them
                if trained_features:
                    feature_cols = trained_features
                    # Ensure all columns exist, fill missing with 0
                    test_df_full = df.reindex(columns=list(df.columns) + [c for c in feature_cols if c not in df.columns], fill_value=0)
                else:
                    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                                   'date', 'symbol'] + [c for c in df.columns if c.startswith('target_')]
                    feature_cols = [c for c in df.columns if c not in exclude_cols]
                    test_df_full = df

                split_idx = int(len(df) * 0.8)
                test_df = test_df_full.iloc[split_idx:].reset_index(drop=True)
                
                if len(test_df) == 0:
                    results[symbol] = {'error': 'Insufficient data for backtest'}
                    continue

                X_test = test_df[feature_cols].values

                predictions = self.production_model.predict(X_test)
                directions = np.array([p.direction for p in predictions])
                confidences = np.array([p.confidence for p in predictions])
                expected_returns = np.array([p.expected_return for p in predictions])

                try:
                    result = self.backtester.run_backtest(
                        test_df, directions, confidences, expected_returns, min_confidence=0.50
                    )
                except Exception as e:
                    logger.error(f"Backtester run failed for {symbol}: {e}")
                    # Create empty result with safe defaults
                    from engine.backtester import BacktestResult
                    result = BacktestResult(
                        total_return=0.0, annualized_return=0.0, benchmark_return=0.0, excess_return=0.0,
                        volatility=0.0, sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
                        max_drawdown_duration=0, total_trades=0, winning_trades=0, losing_trades=0,
                        win_rate=0.0, avg_win=0.0, avg_loss=0.0, profit_factor=0.0, avg_holding_days=0.0,
                        final_value=self.initial_capital, initial_value=self.initial_capital, trades=[]
                    )

                # Format equity curve
                equity_curve = []
                if result.equity_curve is not None:
                    try:
                        for idx, value in result.equity_curve.items():
                            equity_curve.append({
                                'date': str(idx)[:10],
                                'equity': float(value)
                            })
                    except:
                        pass

                trades_formatted = []
                for t in result.trades:
                    trades_formatted.append({
                        'entry_date': str(t.entry_date)[:10] if t.entry_date else '',
                        'exit_date': str(t.exit_date)[:10] if t.exit_date else '',
                        'direction': t.direction,
                        'entry_price': float(t.entry_price),
                        'exit_price': float(t.exit_price) if t.exit_price else 0,
                        'shares': int(t.quantity),
                        'pnl': float(t.pnl),
                        'return_pct': float(t.pnl_pct),
                        'exit_reason': t.exit_reason
                    })

                # Calculate directional accuracy
                directional_accuracy = 0.0
                try:
                    actual_directions = test_df['target_direction'].values
                    # Handle NaNs in target
                    valid_idx = ~np.isnan(actual_directions)
                    if valid_idx.sum() > 0:
                        directional_accuracy = (directions[valid_idx] == actual_directions[valid_idx]).mean()
                except Exception as e:
                    logger.warning(f"Accuracy calc failed for {symbol}: {e}")
                
                # Calculate per-model predictions
                xgb_accuracy = 0.0
                lgb_accuracy = 0.0
                try:
                    xgb_preds = self.production_model.predict_with_model(X_test, 'xgb')
                    lgb_preds = self.production_model.predict_with_model(X_test, 'lgb')
                    
                    if valid_idx.sum() > 0:
                        xgb_directions = np.array([p.direction for p in xgb_preds])
                        lgb_directions = np.array([p.direction for p in lgb_preds])
                        
                        xgb_accuracy = (xgb_directions[valid_idx] == actual_directions[valid_idx]).mean()
                        lgb_accuracy = (lgb_directions[valid_idx] == actual_directions[valid_idx]).mean()
                except Exception as e:
                    logger.warning(f"Could not compute per-model predictions for {symbol}: {e}")
                    xgb_accuracy = None
                    lgb_accuracy = None
                
                # Get feature importance (top 10)
                try:
                    # Pass feature_cols if needed, but model handles it if stored
                    feature_importance = self.production_model.get_feature_importance(feature_cols)
                    top_features = feature_importance[:10] if len(feature_importance) > 10 else feature_importance
                except Exception as e:
                    logger.warning(f"Could not get feature importance for {symbol}: {e}")
                    top_features = []

                results[symbol] = {
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades,
                    'profit_factor': result.profit_factor,
                    'directional_accuracy': float(directional_accuracy),
                    'model_predictions': {
                        'xgb_accuracy': float(xgb_accuracy) if xgb_accuracy is not None else None,
                        'lgb_accuracy': float(lgb_accuracy) if lgb_accuracy is not None else None,
                        'ensemble_accuracy': float(directional_accuracy)
                    },
                    'feature_importance': top_features,
                    'equity_curve': equity_curve,
                    'trades': trades_formatted
                }

            except Exception as e:
                logger.error(f"Critical backtest failure for {symbol}: {e}")
                results[symbol] = {'error': str(e)}

        self.backtest_results = results

        return {
            'symbols_backtested': len(results),
            'summary': {
                sym: {
                    'return': round(r.get('total_return', 0) * 100, 2),
                    'sharpe': round(r.get('sharpe_ratio', 0), 2),
                    'win_rate': round(r.get('win_rate', 0) * 100, 1),
                    'trades': r.get('total_trades', 0)
                }
                for sym, r in results.items()
                # Include even if error to show "Failed" in logs, but orchestrator return usually expects dict
                if 'error' not in r
            }
        }

    def _step_signal_generation(self) -> Dict:
        """Step 8: Generate trading signals with Phase 1 optimization."""
        signals = {}
        filtered_count = 0
        
        # Get feature names from model
        trained_features = self.production_model.feature_names

        for symbol in self.features_cache.keys():
            df = self.features_cache[symbol]
            if len(df) < 2:
                continue

            if trained_features:
                feature_cols = trained_features
                # Ensure all columns exist, fill missing with 0
                latest = df.iloc[-1].reindex(feature_cols, fill_value=0)
            else:
                exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                               'date', 'symbol'] + [c for c in df.columns if c.startswith('target_')]
                feature_cols = [c for c in df.columns if c not in exclude_cols]
                latest = df.iloc[-1]

            X = latest[feature_cols].values.reshape(1, -1)

            try:
                pred = self.production_model.predict_single(X)

                if pred.direction_probability > 0.52:
                    direction = 1  # BUY
                    action = "BUY"
                    confidence = min((pred.direction_probability - 0.5) * 2 + 0.5, 0.95)
                elif pred.direction_probability < 0.48:
                    direction = -1  # SELL
                    action = "SELL"
                    confidence = min((0.5 - pred.direction_probability) * 2 + 0.5, 0.95)
                else:
                    direction = 0  # HOLD
                    action = "HOLD"
                    confidence = 0.5

                # NEW: Calculate Phase 1 metrics for ALL signals (even HOLD)
                quality_score = 0.5
                entry_reasons = ["Model neutral (HOLD)"]
                position_size_usd = 0
                risk_pct = 0
                approved = False  # HOLD signals are not approved for entry
                
                current_price = float(latest.get('close', 0))
                atr = latest.get('atr_pct', 0.015)
                
                # Always evaluate quality to show why it's HOLD or BUY/SELL
                if direction == 0:
                    # HOLD signal - still calculate what quality would be for BUY
                    # This helps user see if they should wait or if stock is just neutral
                    test_direction = 1  # Test as if BUY
                    entry_signal = self.entry_optimizer.evaluate(
                        prediction={'direction': test_direction, 'confidence': 0.5},
                        features=latest,
                        market_data=None
                    )
                    quality_score = entry_signal.quality_score
                    entry_reasons = ["HOLD: " + r for r in entry_signal.reasons[:2]]
                else:
                    # BUY/SELL signal - run full optimization
                    # Evaluate entry quality
                    entry_signal = self.entry_optimizer.evaluate(
                        prediction={'direction': direction, 'confidence': confidence},
                        features=latest,
                        market_data=None  # Could add NIFTY data here
                    )
                    
                    quality_score = entry_signal.quality_score
                    entry_reasons = entry_signal.reasons[:3]  # Top 3 reasons
                    
                    if not entry_signal.should_enter:
                        # Filter out low-quality signals
                        action = "HOLD"
                        filtered_count += 1
                        approved = False
                    else:
                        # Calculate stop loss
                        stop_loss = self.exit_optimizer.calculate_initial_stop_loss(
                            current_price, direction, atr
                        )
                        
                        # Calculate position size
                        size_result = self.position_sizer.calculate_position_size(
                            symbol=symbol,
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            confidence=confidence,
                            volatility=atr,
                            current_positions=None  # Could track open positions
                        )
                        
                        position_size_usd = size_result['position_value']
                        risk_pct = size_result['risk_pct']
                        
                        # NEW: Risk manager approval
                        risk_check = self.risk_manager.approve_trade(
                            symbol=symbol,
                            direction=direction,
                            entry_price=current_price,
                            position_size=position_size_usd,
                            stop_loss=stop_loss
                        )
                        
                        if not risk_check.approved:
                            action = "HOLD"
                            filtered_count += 1
                            approved = False
                            entry_reasons.append(f"Risk rejected: {risk_check.reason}")
                        else:
                            approved = True

                signals[symbol] = {
                    'action': action,
                    'confidence': round(float(confidence), 3),
                    'direction_probability': round(float(pred.direction_probability), 3),
                    'expected_return': round(float(pred.expected_return), 6),
                    'current_price': round(float(latest.get('close', 0)), 2),
                    # NEW: Phase 1 metrics
                    'quality_score': round(float(quality_score), 3),
                    'entry_approved': approved,
                    'position_size_usd': round(float(position_size_usd), 2),
                    'risk_pct': round(float(risk_pct), 4),
                    'entry_reasons': ', '.join(entry_reasons[:2])  # Top 2 reasons
                }
            except Exception as e:
                logger.error(f"Signal generation failed for {symbol}: {e}")

        self.signals = signals

        # Save signals
        signals_path = os.path.join(
            self.results_dir, 'signals',
            f'signals_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
        )
        with open(signals_path, 'w') as f:
            json.dump(signals, f, indent=2)

        buy_signals = [s for s in signals.values() if s['action'] == 'BUY']
        sell_signals = [s for s in signals.values() if s['action'] == 'SELL']

        logger.info(f"Signal generation complete: {len(buy_signals)} BUY, {len(sell_signals)} SELL, "
                   f"{filtered_count} filtered by optimizers")

        return {
            'total_signals': len(signals),
            'buy_count': len(buy_signals),
            'sell_count': len(sell_signals),
            'hold_count': len(signals) - len(buy_signals) - len(sell_signals),
            'filtered_count': filtered_count,  # NEW
            'signals': signals
        }

    # ==================== CONVENIENCE METHODS ====================

    def get_status(self) -> Dict:
        """Get current pipeline status."""
        if self.pipeline_status:
            return {
                'job_id': self.pipeline_status.job_id,
                'status': self.pipeline_status.status,
                'current_step': self.pipeline_status.current_step,
                'total_steps': self.pipeline_status.total_steps,
                'steps': [
                    {
                        'step': s.step_number,
                        'name': s.name,
                        'status': s.status,
                        'duration': round(s.duration_seconds, 1),
                        'details': s.details
                    }
                    for s in self.pipeline_status.steps
                ]
            }
        return {'status': 'not_started'}

    def get_backtest_results(self) -> Dict:
        return self.backtest_results

    def get_signals(self) -> Dict:
        return self.signals or {}

    def get_allocation(self) -> Dict:
        if self.allocation and hasattr(self.allocation, 'weights'):
            return {
                'weights': self.allocation.weights,
                'method': getattr(self.allocation, 'method', 'unknown'),
                'sharpe_ratio': getattr(self.allocation, 'sharpe_ratio', 0),
                'expected_return': getattr(self.allocation, 'expected_return', 0)
            }
        return {}
