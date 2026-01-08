"""
================================================================================
PIPELINE LOGGER
================================================================================
Logs all pipeline runs, data collection, training, and predictions to CSV files.
Each run is timestamped for tracking and reproducibility.

Usage:
    from pipeline.utils.pipeline_logger import PipelineLogger

    logger = PipelineLogger()
    logger.log_data_collection('HDFCBANK', rows=2700, start='2015-01-01', end='2025-12-05')
    logger.log_training('HDFCBANK', 'xgboost', metrics={'rmse': 0.015, 'mae': 0.012})
    logger.log_prediction('HDFCBANK', 'xgboost', predictions_df)
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class PipelineLogger:
    """Centralized logging for all pipeline operations."""

    def __init__(self):
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_timestamp = datetime.now()

        # Initialize log files if they don't exist
        self._init_log_files()

    def _init_log_files(self):
        """Initialize CSV log files with headers if they don't exist."""
        # Data collection log
        if not os.path.exists(config.DATA_COLLECTION_LOG):
            df = pd.DataFrame(columns=[
                'run_id', 'timestamp', 'symbol', 'sector', 'source',
                'rows', 'start_date', 'end_date', 'years',
                'train_rows', 'val_rows', 'test_rows',
                'status', 'error_message'
            ])
            df.to_csv(config.DATA_COLLECTION_LOG, index=False)

        # Training log
        if not os.path.exists(config.TRAINING_LOG):
            df = pd.DataFrame(columns=[
                'run_id', 'timestamp', 'symbol', 'sector', 'model_type',
                'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end',
                'train_rows', 'val_rows', 'test_rows', 'n_features',
                'train_rmse', 'train_mae', 'train_mape', 'train_r2', 'train_direction_acc',
                'val_rmse', 'val_mae', 'val_mape', 'val_r2', 'val_direction_acc',
                'test_rmse', 'test_mae', 'test_mape', 'test_r2', 'test_direction_acc',
                'training_time_sec', 'model_path', 'status', 'error_message'
            ])
            df.to_csv(config.TRAINING_LOG, index=False)

        # Prediction log
        if not os.path.exists(config.PREDICTION_LOG):
            df = pd.DataFrame(columns=[
                'run_id', 'timestamp', 'symbol', 'model_type',
                'prediction_date', 'actual_close', 'predicted_close',
                'actual_return', 'predicted_return',
                'actual_direction', 'predicted_direction',
                'error_pct', 'correct_direction'
            ])
            df.to_csv(config.PREDICTION_LOG, index=False)

        # Pipeline runs log
        if not os.path.exists(config.PIPELINE_LOG_FILE):
            df = pd.DataFrame(columns=[
                'run_id', 'start_time', 'end_time', 'duration_sec',
                'step', 'symbols_processed', 'success_count', 'error_count',
                'train_period', 'val_period', 'test_period',
                'status', 'notes'
            ])
            df.to_csv(config.PIPELINE_LOG_FILE, index=False)

    def log_data_collection(self, symbol: str, rows: int, start_date: str, end_date: str,
                            train_rows: int = 0, val_rows: int = 0, test_rows: int = 0,
                            status: str = 'success', error_message: str = None):
        """Log data collection for a stock."""
        sector = config.STOCK_SECTOR_MAP.get(symbol, 'Unknown')
        years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365

        record = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'sector': sector,
            'source': config.DATA_SOURCE,
            'rows': rows,
            'start_date': start_date,
            'end_date': end_date,
            'years': round(years, 2),
            'train_rows': train_rows,
            'val_rows': val_rows,
            'test_rows': test_rows,
            'status': status,
            'error_message': error_message
        }

        df = pd.DataFrame([record])
        df.to_csv(config.DATA_COLLECTION_LOG, mode='a', header=False, index=False)

        logger.info(f"Logged data collection: {symbol} - {rows} rows")

    def log_training(self, symbol: str, model_type: str,
                     train_metrics: dict, val_metrics: dict, test_metrics: dict,
                     data_info: dict, model_path: str,
                     training_time: float, status: str = 'success', error_message: str = None):
        """Log model training results."""
        sector = config.STOCK_SECTOR_MAP.get(symbol, 'Unknown')

        record = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'sector': sector,
            'model_type': model_type,
            'train_start': data_info.get('train_start'),
            'train_end': data_info.get('train_end'),
            'val_start': data_info.get('val_start'),
            'val_end': data_info.get('val_end'),
            'test_start': data_info.get('test_start'),
            'test_end': data_info.get('test_end'),
            'train_rows': data_info.get('train_rows', 0),
            'val_rows': data_info.get('val_rows', 0),
            'test_rows': data_info.get('test_rows', 0),
            'n_features': data_info.get('n_features', 0),
            'train_rmse': train_metrics.get('rmse'),
            'train_mae': train_metrics.get('mae'),
            'train_mape': train_metrics.get('mape'),
            'train_r2': train_metrics.get('r2'),
            'train_direction_acc': train_metrics.get('direction_accuracy'),
            'val_rmse': val_metrics.get('rmse'),
            'val_mae': val_metrics.get('mae'),
            'val_mape': val_metrics.get('mape'),
            'val_r2': val_metrics.get('r2'),
            'val_direction_acc': val_metrics.get('direction_accuracy'),
            'test_rmse': test_metrics.get('rmse'),
            'test_mae': test_metrics.get('mae'),
            'test_mape': test_metrics.get('mape'),
            'test_r2': test_metrics.get('r2'),
            'test_direction_acc': test_metrics.get('direction_accuracy'),
            'training_time_sec': round(training_time, 2),
            'model_path': model_path,
            'status': status,
            'error_message': error_message
        }

        df = pd.DataFrame([record])
        df.to_csv(config.TRAINING_LOG, mode='a', header=False, index=False)

        logger.info(f"Logged training: {symbol} - {model_type} - Test RMSE: {test_metrics.get('rmse', 'N/A')}")

    def log_predictions(self, symbol: str, model_type: str, predictions_df: pd.DataFrame):
        """Log individual predictions."""
        records = []

        for _, row in predictions_df.iterrows():
            record = {
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'model_type': model_type,
                'prediction_date': row.get('date') or row.get('timestamp'),
                'actual_close': row.get('actual_close'),
                'predicted_close': row.get('predicted_close'),
                'actual_return': row.get('actual_return'),
                'predicted_return': row.get('predicted_return'),
                'actual_direction': row.get('actual_direction'),
                'predicted_direction': row.get('predicted_direction'),
                'error_pct': row.get('error_pct'),
                'correct_direction': row.get('correct_direction')
            }
            records.append(record)

        if records:
            df = pd.DataFrame(records)
            df.to_csv(config.PREDICTION_LOG, mode='a', header=False, index=False)
            logger.info(f"Logged {len(records)} predictions: {symbol} - {model_type}")

    def log_pipeline_run(self, step: str, symbols: list, success_count: int, error_count: int,
                         start_time: datetime, end_time: datetime = None, notes: str = None):
        """Log a pipeline run."""
        if end_time is None:
            end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        record = {
            'run_id': self.run_id,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_sec': round(duration, 2),
            'step': step,
            'symbols_processed': len(symbols),
            'success_count': success_count,
            'error_count': error_count,
            'train_period': f"{config.TRAIN_START_DATE} to {config.TRAIN_END_DATE}",
            'val_period': f"{config.VAL_START_DATE} to {config.VAL_END_DATE}",
            'test_period': f"{config.TEST_START_DATE} to {config.END_DATE}",
            'status': 'success' if error_count == 0 else 'partial',
            'notes': notes
        }

        df = pd.DataFrame([record])
        df.to_csv(config.PIPELINE_LOG_FILE, mode='a', header=False, index=False)

        logger.info(f"Logged pipeline run: {step} - {success_count}/{len(symbols)} successful")

    def get_latest_runs(self, n: int = 10) -> pd.DataFrame:
        """Get the latest n pipeline runs."""
        if os.path.exists(config.PIPELINE_LOG_FILE):
            df = pd.read_csv(config.PIPELINE_LOG_FILE)
            return df.tail(n)
        return pd.DataFrame()

    def get_training_history(self, symbol: str = None, model_type: str = None) -> pd.DataFrame:
        """Get training history, optionally filtered by symbol or model."""
        if os.path.exists(config.TRAINING_LOG):
            df = pd.read_csv(config.TRAINING_LOG)
            if symbol:
                df = df[df['symbol'] == symbol]
            if model_type:
                df = df[df['model_type'] == model_type]
            return df
        return pd.DataFrame()

    def get_data_collection_summary(self) -> pd.DataFrame:
        """Get summary of data collection runs."""
        if os.path.exists(config.DATA_COLLECTION_LOG):
            df = pd.read_csv(config.DATA_COLLECTION_LOG)
            # Get latest entry per symbol
            df = df.sort_values('timestamp').groupby('symbol').last().reset_index()
            return df
        return pd.DataFrame()


def get_run_id() -> str:
    """Generate a unique run ID based on current timestamp."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


if __name__ == '__main__':
    # Test logging
    logger_instance = PipelineLogger()
    print(f"Run ID: {logger_instance.run_id}")

    # Example usage
    logger_instance.log_data_collection(
        symbol='HDFCBANK',
        rows=2700,
        start_date='2015-01-01',
        end_date='2025-12-05',
        train_rows=1728,
        val_rows=493,
        test_rows=479
    )

    print("\nData collection log created!")
    print(f"Check: {config.DATA_COLLECTION_LOG}")
