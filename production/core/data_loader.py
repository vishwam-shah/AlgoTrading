"""
================================================================================
DATA LOADER - Unified Data Collection
================================================================================
Handles all data collection from yfinance with caching and validation.

Features:
- Download OHLCV data from yfinance
- Smart caching to avoid redundant downloads
- Data validation and quality checks
- Handles yfinance API changes
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class DataLoader:
    """
    Unified data loader with caching and validation.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize DataLoader.
        
        Args:
            cache_dir: Directory for caching data
        """
        self.cache_dir = cache_dir or os.path.join(config.BASE_DIR, 'data', 'raw')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check yfinance version
        try:
            self.yf_version = yf.__version__
            logger.info(f"yfinance version: {self.yf_version}")
        except:
            self.yf_version = "unknown"
    
    def download_stock(
        self, 
        symbol: str, 
        days: int = 500,
        force_download: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Download stock data from yfinance.
        
        Args:
            symbol: Stock symbol (e.g., 'HDFCBANK')
            days: Number of historical days
            force_download: Force fresh download even if cached
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        cache_path = os.path.join(self.cache_dir, f'{symbol}.csv')
        
        # Check cache first
        if not force_download and os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    last_date = df['timestamp'].max()
                    
                    # Use cache if data is recent (within 1 day)
                    if (datetime.now() - last_date).days <= 1:
                        logger.info(f"  {symbol}: Using cached data ({len(df)} rows)")
                        return df
            except Exception as e:
                logger.warning(f"  {symbol}: Cache read failed - {e}")
        
        # Download from yfinance
        ticker_symbol = f"{symbol}.NS"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for attempt in range(3):
            try:
                logger.info(f"  Downloading {ticker_symbol} from yfinance...")
                ticker = yf.Ticker(ticker_symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if df is not None and len(df) > 0:
                    df = self._process_dataframe(df, symbol)
                    
                    # Save to cache
                    df.to_csv(cache_path, index=False)
                    logger.success(f"  {symbol}: Downloaded {len(df)} rows (latest: {df['timestamp'].iloc[-1]})")
                    return df
                else:
                    logger.warning(f"  {symbol}: Attempt {attempt+1} - no data returned")
                    
            except Exception as e:
                logger.warning(f"  {symbol}: Attempt {attempt+1} failed - {e}")
            
            import time
            time.sleep(1)
        
        # Fallback to cache
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path)
                logger.warning(f"  {symbol}: Using cached data as fallback ({len(df)} rows)")
                return df
            except:
                pass
        
        logger.error(f"  {symbol}: All download attempts failed")
        return None
    
    def _process_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process raw yfinance DataFrame into standard format."""
        # Handle MultiIndex columns (yfinance 1.0+)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Reset index to get date as column
        df = df.reset_index()
        
        # Standardize column names
        column_map = {
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        df = df.rename(columns=column_map)
        
        # Ensure required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                if col == 'timestamp' and 'date' in df.columns:
                    df['timestamp'] = df['date']
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # Add symbol
        df['symbol'] = symbol
        
        # Select and order columns
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        if 'adj_close' in df.columns:
            cols.append('adj_close')
        
        return df[cols]
    
    def download_multiple(
        self, 
        symbols: List[str], 
        days: int = 500,
        force_download: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            days: Number of historical days
            force_download: Force fresh download
            
        Returns:
            Dict of symbol -> DataFrame
        """
        data = {}
        
        for symbol in symbols:
            df = self.download_stock(symbol, days, force_download)
            if df is not None and len(df) >= 50:
                data[symbol] = df
            else:
                logger.warning(f"  {symbol}: Insufficient data, skipping")
        
        logger.success(f"Data collection complete: {len(data)}/{len(symbols)} symbols")
        return data
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data quality.
        
        Returns:
            (is_valid, list of issues)
        """
        issues = []
        
        # Check for missing values
        missing = df[['open', 'high', 'low', 'close', 'volume']].isnull().sum()
        if missing.any():
            issues.append(f"Missing values: {missing[missing > 0].to_dict()}")
        
        # Check for zeros
        zeros = (df[['open', 'high', 'low', 'close']] == 0).sum()
        if zeros.any():
            issues.append(f"Zero values: {zeros[zeros > 0].to_dict()}")
        
        # Check OHLC consistency
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()
        if invalid_ohlc > 0:
            issues.append(f"Invalid OHLC: {invalid_ohlc} rows")
        
        # Check for sufficient data
        if len(df) < 100:
            issues.append(f"Insufficient data: {len(df)} rows (need 100+)")
        
        return len(issues) == 0, issues


# For backwards compatibility
def download_stock_data(symbol: str, days: int = 500, force_download: bool = False):
    """Legacy function for backwards compatibility."""
    loader = DataLoader()
    return loader.download_stock(symbol, days, force_download)
