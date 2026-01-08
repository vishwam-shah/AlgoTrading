"""
================================================================================
STEP 1: DATA COLLECTION
================================================================================
Downloads and validates stock price data from NSE India.

Features:
- Downloads OHLCV data for specified stocks
- Validates data quality (missing values, duplicates, outliers)
- Saves to data/raw/{STOCK}.csv
- Logs collection details to logs/data_collection_log.csv

Usage:
    # Single stock
    python pipeline/01_data_collection.py --symbol RELIANCE
    
    # Multiple stocks
    python pipeline/01_data_collection.py --symbols RELIANCE TCS INFY
    
    # All stocks from config
    python pipeline/01_data_collection.py --all

Output:
    - data/raw/{STOCK}.csv - Raw OHLCV data with timestamps
    - logs/data_collection_log.csv - Collection metadata
================================================================================
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import yfinance as yf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipeline.utils.pipeline_logger import PipelineLogger


def download_stock_data(symbol: str, start_date: str = '2015-01-01', end_date: str = None) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (default: today)
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Add .NS suffix for NSE stocks
    ticker = f"{symbol}.NS"
    
    logger.info(f"Downloading {symbol} from {start_date} to {end_date}")
    
    try:
        # Download data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            logger.error(f"No data found for {symbol}")
            return None
        
        # Reset index to get date as column
        df = df.reset_index()
        
        # Handle multi-index columns from newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Rename columns to lowercase
        df.columns = [col.lower() if isinstance(col, str) else str(col).lower() for col in df.columns]
        
        # Rename 'date' to 'timestamp' if exists
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        
        # Ensure timestamp is timezone-naive
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Select required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        logger.success(f"Downloaded {len(df)} rows for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading {symbol}: {e}")
        return None


def validate_data(df: pd.DataFrame, symbol: str) -> dict:
    """
    Validate data quality and return quality metrics.
    
    Args:
        df: DataFrame with stock data
        symbol: Stock symbol
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        'symbol': symbol,
        'total_rows': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_dates': df['timestamp'].duplicated().sum(),
        'date_gaps': 0,
        'zero_volume_days': (df['volume'] == 0).sum(),
        'negative_prices': ((df['close'] < 0) | (df['open'] < 0) | (df['high'] < 0) | (df['low'] < 0)).sum(),
        'price_anomalies': 0,
        'status': 'PASS'
    }
    
    # Check for date gaps (weekends excluded)
    dates = pd.to_datetime(df['timestamp'])
    expected_dates = pd.date_range(dates.min(), dates.max(), freq='D')
    # Filter out weekends
    expected_dates = expected_dates[expected_dates.dayofweek < 5]
    actual_dates = set(dates.dt.date)
    expected_dates_set = set(expected_dates.date)
    validation['date_gaps'] = len(expected_dates_set - actual_dates)
    
    # Check for price anomalies (OHLC relationships)
    anomalies = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ).sum()
    validation['price_anomalies'] = anomalies
    
    # Set status
    if validation['missing_values'] > 0 or validation['negative_prices'] > 0 or validation['price_anomalies'] > 0:
        validation['status'] = 'FAIL'
    elif validation['date_gaps'] > 100 or validation['zero_volume_days'] > 50:
        validation['status'] = 'WARNING'
    
    return validation


def save_stock_data(df: pd.DataFrame, symbol: str, output_dir: str = None) -> str:
    """
    Save stock data to CSV file.
    
    Args:
        df: DataFrame with stock data
        symbol: Stock symbol
        output_dir: Output directory (default: config.RAW_DATA_DIR)
    
    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = config.RAW_DATA_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f"{symbol}.csv")
    df.to_csv(filepath, index=False)
    
    logger.success(f"Saved {symbol} data to {filepath}")
    return filepath


def collect_stock(symbol: str, start_date: str = '2015-01-01', pipeline_logger: PipelineLogger = None) -> bool:
    """
    Collect data for a single stock.
    
    Args:
        symbol: Stock symbol
        start_date: Start date for data collection
        pipeline_logger: PipelineLogger instance for logging
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"COLLECTING DATA: {symbol}")
    logger.info(f"{'='*80}")
    
    try:
        # Download data
        df = download_stock_data(symbol, start_date=start_date)
        
        if df is None or df.empty:
            if pipeline_logger:
                pipeline_logger.log_data_collection(
                    symbol=symbol,
                    rows=0,
                    start_date='',
                    end_date='',
                    status='FAIL',
                    error_message='No data downloaded'
                )
            return False
        
        # Validate data
        validation = validate_data(df, symbol)
        
        # Log validation results
        logger.info(f"Validation Results:")
        logger.info(f"  Total Rows: {validation['total_rows']}")
        logger.info(f"  Missing Values: {validation['missing_values']}")
        logger.info(f"  Duplicate Dates: {validation['duplicate_dates']}")
        logger.info(f"  Date Gaps: {validation['date_gaps']}")
        logger.info(f"  Zero Volume Days: {validation['zero_volume_days']}")
        logger.info(f"  Negative Prices: {validation['negative_prices']}")
        logger.info(f"  Price Anomalies: {validation['price_anomalies']}")
        logger.info(f"  Status: {validation['status']}")
        
        # Save data
        filepath = save_stock_data(df, symbol)
        
        # Log to pipeline logger
        if pipeline_logger:
            pipeline_logger.log_data_collection(
                symbol=symbol,
                rows=len(df),
                start_date=df['timestamp'].min().strftime('%Y-%m-%d'),
                end_date=df['timestamp'].max().strftime('%Y-%m-%d'),
                status=validation['status'],
                error_message='' if validation['status'] == 'PASS' else 'Data quality issues detected'
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Error collecting {symbol}: {e}")
        if pipeline_logger:
            pipeline_logger.log_data_collection(
                symbol=symbol,
                rows=0,
                start_date='',
                end_date='',
                status='FAIL',
                error_message=str(e)
            )
        return False


def main():
    """Main entry point for data collection."""
    parser = argparse.ArgumentParser(description='Step 1: Collect stock price data')
    parser.add_argument('--symbol', type=str, help='Single stock symbol to collect')
    parser.add_argument('--symbols', nargs='+', help='Multiple stock symbols to collect')
    parser.add_argument('--all', action='store_true', help='Collect all stocks from config')
    parser.add_argument('--start-date', type=str, default='2015-01-01', help='Start date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Determine which stocks to collect
    if args.symbol:
        stocks = [args.symbol]
    elif args.symbols:
        stocks = args.symbols
    elif args.all:
        stocks = config.TOP_100_STOCKS if hasattr(config, 'TOP_100_STOCKS') else []
    else:
        parser.print_help()
        return
    
    # Initialize logger
    pipeline_logger = PipelineLogger()
    
    # Collect data for each stock
    logger.info(f"\nCollecting data for {len(stocks)} stocks...")
    success_count = 0
    fail_count = 0
    
    for i, symbol in enumerate(stocks, 1):
        logger.info(f"\nProgress: {i}/{len(stocks)}")
        
        if collect_stock(symbol, start_date=args.start_date, pipeline_logger=pipeline_logger):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"DATA COLLECTION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Success: {success_count}/{len(stocks)}")
    logger.info(f"Failed: {fail_count}/{len(stocks)}")
    logger.info(f"Data saved to: {config.RAW_DATA_DIR}")
    logger.info(f"Logs saved to: {config.DATA_COLLECTION_LOG}")


if __name__ == '__main__':
    main()
