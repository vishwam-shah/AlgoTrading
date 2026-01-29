"""
================================================================================
PROFESSIONAL CSV LOGGING SYSTEM
================================================================================
Comprehensive data persistence with organized folder structure and naming
convention: stock_date_timestamp

Structure:
results/
├── pipeline_runs/
│   └── {SYMBOL}_{DATE}_{TIMESTAMP}/    # e.g., SBIN_20260129_141530
│       ├── run_summary.csv              # Configuration & results overview
│       ├── backtest_results.csv         # Per-stock performance
│       ├── model_metrics.csv            # ML model accuracy
│       ├── features_used.csv            # ✅ ALL 60+ features listed here
│       ├── feature_importance.csv       # Top features ranked
│       ├── signals.csv                  # Trading signals generated
│       ├── portfolio_allocation.csv     # Final weights
│       └── trades.csv                   # Individual trade log
├── master_logs/                         # Aggregated logs across all runs
│   ├── all_pipeline_runs.csv
│   ├── all_backtest_results.csv
│   └── all_model_metrics.csv
└── exports/                             # User-requested exports

================================================================================
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger


class ProfessionalCSVLogger:
    """
    Professional CSV logging system with organized folder structure.
    Each pipeline run creates a dedicated folder with all details.
    """
    
    def __init__(self, base_dir: str = 'results'):
        self.base_dir = Path(base_dir)
        self.pipeline_runs_dir = self.base_dir / 'pipeline_runs'
        self.master_logs_dir = self.base_dir / 'master_logs'
        self.exports_dir = self.base_dir / 'exports'
        
        # Create directories
        for dir_path in [self.pipeline_runs_dir, self.master_logs_dir, self.exports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_run_folder(self, symbols: List[str], job_id: str) -> Path:
        """
        Create folder for this pipeline run with naming convention:
        {PRIMARY_SYMBOL}_{DATE}_{TIMESTAMP}
        
        Example: SBIN_20260129_141530
        """
        timestamp = datetime.now()
        primary_symbol = symbols[0] if symbols else 'MULTI'
        date_str = timestamp.strftime('%Y%m%d')
        time_str = timestamp.strftime('%H%M%S')
        
        folder_name = f"{primary_symbol}_{date_str}_{time_str}"
        run_folder = self.pipeline_runs_dir / folder_name
        run_folder.mkdir(exist_ok=True)
        
        logger.info(f"Created run folder: {run_folder}")
        return run_folder
    
    def log_run_summary(self, run_folder: Path, config: Dict, status: Dict, duration: float):
        """
        1. RUN_SUMMARY.CSV
        Professional summary of the pipeline run with all configuration details.
        """
        summary = {
            # Identification
            'job_id': config.get('job_id'),
            'run_timestamp': datetime.now().isoformat(),
            'run_folder': str(run_folder.name),
            
            # Configuration
            'symbols': '|'.join(config.get('symbols', [])),
            'num_symbols': len(config.get('symbols', [])),
            'sectors': '|'.join(config.get('sectors', [])) if config.get('sectors') else 'N/A',
            'optimization_method': config.get('optimization_method', 'risk_parity'),
            'n_holdings': config.get('n_holdings', 15),
            'start_date': config.get('start_date'),
            'initial_capital': config.get('capital', 100000),
            
            # Models
            'models_trained': '|'.join(config.get('models_to_train', ['xgboost', 'lightgbm'])),
            
            # Results
            'status': status.get('status'),
            'duration_seconds': duration,
            'duration_formatted': f"{int(duration // 60)}m {int(duration % 60)}s",
            'steps_completed': status.get('steps_completed', 0),
            'total_steps': status.get('total_steps', 8),
            
            # Performance Summary
            'avg_direction_accuracy': status.get('avg_direction_accuracy', 0),
            'avg_sharpe_ratio': status.get('avg_sharpe_ratio', 0),
            'total_features_used': status.get('total_features', 0),
        }
        
        df = pd.DataFrame([summary])
        csv_path = run_folder / 'run_summary.csv'
        df.to_csv(csv_path, index=False)
        logger.success(f"Saved run summary: {csv_path}")
        
        # Also append to master log
        master_path = self.master_logs_dir / 'all_pipeline_runs.csv'
        df.to_csv(master_path, mode='a', header=not master_path.exists(), index=False)
    
    def log_backtest_results(self, run_folder: Path, job_id: str, results: Dict[str, Any]):
        """
        2. BACKTEST_RESULTS.CSV
        Detailed backtest results for each symbol.
        """
        rows = []
        for symbol, result in results.items():
            if 'error' not in result:
                row = {
                    'job_id': job_id,
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'sector': result.get('sector', 'Unknown'),
                    
                    # Returns
                    'total_return': result.get('total_return', 0),
                    'annual_return': result.get('annual_return', 0),
                    'monthly_return': result.get('monthly_return', 0),
                    
                    # Risk Metrics
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'sortino_ratio': result.get('sortino_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'max_drawdown_duration': result.get('max_dd_duration', 0),
                    'volatility': result.get('volatility', 0),
                    
                    # Trading Metrics
                    'total_trades': result.get('total_trades', 0),
                    'win_rate': result.get('win_rate', 0),
                    'profit_factor': result.get('profit_factor', 0),
                    'avg_win': result.get('avg_win', 0),
                    'avg_loss': result.get('avg_loss', 0),
                    'largest_win': result.get('largest_win', 0),
                    'largest_loss': result.get('largest_loss', 0),
                    
                    # Alpha/Beta
                    'alpha': result.get('alpha', 0),
                    'beta': result.get('beta', 0),
                    
                    # Portfolio Weight
                    'portfolio_weight': result.get('weight', 0),
                    'position_value': result.get('position_value', 0),
                }
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_path = run_folder / 'backtest_results.csv'
            df.to_csv(csv_path, index=False)
            logger.success(f"Saved backtest results: {csv_path}")
            
            # Append to master
            master_path = self.master_logs_dir / 'all_backtest_results.csv'
            df.to_csv(master_path, mode='a', header=not master_path.exists(), index=False)
    
    def log_model_metrics(self, run_folder: Path, job_id: str, metrics: Dict[str, Any]):
        """
        3. MODEL_METRICS.CSV
        Model performance metrics for each symbol and model type.
        """
        rows = []
        for symbol, symbol_metrics in metrics.items():
            for model_type, model_data in symbol_metrics.items():
                row = {
                    'job_id': job_id,
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'model_type': model_type,
                    
                    # Classification Metrics
                    'direction_accuracy': model_data.get('direction_accuracy', 0),
                    'direction_accuracy_high_conf': model_data.get('high_conf_accuracy', 0),
                    'precision': model_data.get('precision', 0),
                    'recall': model_data.get('recall', 0),
                    'f1_score': model_data.get('f1_score', 0),
                    
                    # Regression Metrics
                    'rmse': model_data.get('rmse', 0),
                    'mae': model_data.get('mae', 0),
                    'mape': model_data.get('mape', 0),
                    'r2_score': model_data.get('r2_score', 0),
                    
                    # Training Info
                    'training_samples': model_data.get('training_samples', 0),
                    'validation_samples': model_data.get('validation_samples', 0),
                    'features_used': model_data.get('features_used', 0),
                    'training_time_seconds': model_data.get('training_time', 0),
                    
                    # Hyperparameters
                    'learning_rate': model_data.get('learning_rate', 0),
                    'n_estimators': model_data.get('n_estimators', 0),
                    'max_depth': model_data.get('max_depth', 0),
                }
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_path = run_folder / 'model_metrics.csv'
            df.to_csv(csv_path, index=False)
            logger.success(f"Saved model metrics: {csv_path}")
            
            # Append to master
            master_path = self.master_logs_dir / 'all_model_metrics.csv'
            df.to_csv(master_path, mode='a', header=not master_path.exists(), index=False)
    
    def log_features_used(self, run_folder: Path, job_id: str, features_list: List[str], 
                          feature_descriptions: Optional[Dict[str, str]] = None):
        """
        4. FEATURES_USED.CSV
        **COMPLETE LIST OF ALL FEATURES USED IN THIS RUN**
        
        This is critical - shows exactly which features were used for training.
        """
        rows = []
        for idx, feature in enumerate(features_list, 1):
            row = {
                'job_id': job_id,
                'feature_number': idx,
                'feature_name': feature,
                'category': self._categorize_feature(feature),
                'description': feature_descriptions.get(feature, '') if feature_descriptions else '',
                'data_type': 'numeric',  # Most features are numeric
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = run_folder / 'features_used.csv'
        df.to_csv(csv_path, index=False)
        logger.success(f"Saved features list ({len(features_list)} features): {csv_path}")
    
    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize feature by name pattern"""
        if any(x in feature_name.lower() for x in ['rsi', 'macd', 'stoch', 'cci', 'williams']):
            return 'Momentum'
        elif any(x in feature_name.lower() for x in ['vwap', 'obv', 'volume', 'mfi']):
            return 'Volume'
        elif any(x in feature_name.lower() for x in ['bb', 'atr', 'volatility', 'keltner']):
            return 'Volatility'
        elif any(x in feature_name.lower() for x in ['sma', 'ema', 'adx', 'ichimoku', 'supertrend']):
            return 'Trend'
        elif any(x in feature_name.lower() for x in ['sentiment', 'news']):
            return 'Sentiment'
        elif any(x in feature_name.lower() for x in ['regime', 'market']):
            return 'Market Regime'
        else:
            return 'Other'
    
    def log_feature_importance(self, run_folder: Path, job_id: str, 
                               importance_data: Dict[str, Dict[str, float]]):
        """
        5. FEATURE_IMPORTANCE.CSV
        Feature importance scores from models (helps identify best features).
        """
        rows = []
        for symbol, importances in importance_data.items():
            for feature, importance in importances.items():
                row = {
                    'job_id': job_id,
                    'symbol': symbol,
                    'feature_name': feature,
                    'importance_score': importance,
                    'category': self._categorize_feature(feature),
                }
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            # Sort by importance
            df = df.sort_values('importance_score', ascending=False)
            df['rank'] = df.groupby('symbol').cumcount() + 1
            
            csv_path = run_folder / 'feature_importance.csv'
            df.to_csv(csv_path, index=False)
            logger.success(f"Saved feature importance: {csv_path}")
    
    def log_signals(self, run_folder: Path, job_id: str, signals: Dict[str, Any]):
        """
        6. SIGNALS.CSV
        All trading signals generated with detailed information.
        """
        rows = []
        for symbol, signal in signals.items():
            row = {
                'job_id': job_id,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'sector': signal.get('sector', 'Unknown'),
                
                # Signal
                'action': signal.get('action'),  # BUY/SELL/HOLD
                'strength': signal.get('strength'),  # STRONG/MODERATE/WEAK
                'confidence': signal.get('confidence', 0),
                'direction_probability': signal.get('direction_probability', 0),
                
                # Prices
                'current_price': signal.get('current_price', 0),
                'target_price': signal.get('target_price', 0),
                'stop_loss': signal.get('stop_loss', 0),
                
                # Risk/Reward
                'expected_return_pct': signal.get('expected_return', 0) * 100,
                'risk_pct': signal.get('risk_pct', 0),
                'risk_reward_ratio': signal.get('risk_reward_ratio', 0),
                
                # Position Sizing
                'suggested_position_pct': signal.get('suggested_position_pct', 0),
                'suggested_quantity': signal.get('suggested_quantity', 0),
                'position_value': signal.get('position_value', 0),
                
                # Technical Summary
                'market_regime': signal.get('market_regime', ''),
                'technical_summary': signal.get('technical_summary', ''),
            }
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_path = run_folder / 'signals.csv'
            df.to_csv(csv_path, index=False)
            logger.success(f"Saved signals: {csv_path}")
    
    def log_portfolio_allocation(self, run_folder: Path, job_id: str, 
                                 allocation: Dict[str, float], allocation_metadata: Dict):
        """
        7. PORTFOLIO_ALLOCATION.CSV
        Final portfolio weights and allocation details.
        """
        rows = []
        for symbol, weight in allocation.items():
            row = {
                'job_id': job_id,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'weight_pct': weight * 100,
                'sector': allocation_metadata.get('sectors', {}).get(symbol, 'Unknown'),
                'factor_score': allocation_metadata.get('factor_scores', {}).get(symbol, 0),
                'expected_return': allocation_metadata.get('expected_returns', {}).get(symbol, 0),
                'volatility': allocation_metadata.get('volatilities', {}).get(symbol, 0),
            }
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values('weight_pct', ascending=False)
            
            csv_path = run_folder / 'portfolio_allocation.csv'
            df.to_csv(csv_path, index=False)
            logger.success(f"Saved portfolio allocation: {csv_path}")
    
    def log_trades(self, run_folder: Path, job_id: str, trades_data: List[Dict]):
        """
        8. TRADES.CSV
        All individual trades executed during backtest.
        """
        if not trades_data:
            return
        
        rows = []
        for trade in trades_data:
            row = {
                'job_id': job_id,
                'trade_id': trade.get('trade_id'),
                'symbol': trade.get('symbol'),
                'entry_date': trade.get('entry_date'),
                'exit_date': trade.get('exit_date'),
                'duration_days': trade.get('duration_days', 0),
                
                # Direction
                'direction': trade.get('direction'),  # LONG/SHORT
                
                # Prices
                'entry_price': trade.get('entry_price', 0),
                'exit_price': trade.get('exit_price', 0),
                'shares': trade.get('shares', 0),
                
                # P&L
                'pnl': trade.get('pnl', 0),
                'pnl_pct': trade.get('return_pct', 0) * 100,
                
                # Reasons
                'entry_reason': trade.get('entry_reason', ''),
                'exit_reason': trade.get('exit_reason', ''),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = run_folder / 'trades.csv'
        df.to_csv(csv_path, index=False)
        logger.success(f"Saved {len(trades_data)} trades: {csv_path}")


# Usage Example:
"""
logger = ProfessionalCSVLogger()

# Create run folder
run_folder = logger.create_run_folder(symbols=['SBIN', 'HDFCBANK'], job_id='pipeline_20260129_141530')

# Log run summary
logger.log_run_summary(run_folder, config, status, duration=245.3)

# Log backtest results
logger.log_backtest_results(run_folder, job_id, backtest_results)

# Log model metrics
logger.log_model_metrics(run_folder, job_id, model_metrics)

# **IMPORTANT: Log ALL features used**
logger.log_features_used(run_folder, job_id, feature_names_list, feature_descriptions_dict)

# Log feature importance
logger.log_feature_importance(run_folder, job_id, feature_importance_dict)

# Log signals
logger.log_signals(run_folder, job_id, signals_dict)

# Log portfolio allocation
logger.log_portfolio_allocation(run_folder, job_id, weights, metadata)

# Log trades
logger.log_trades(run_folder, job_id, trades_list)
"""
