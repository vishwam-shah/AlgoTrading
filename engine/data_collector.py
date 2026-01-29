"""
================================================================================
UNIFIED DATA COLLECTOR
================================================================================
Merged data collection from pipeline/ and production/ systems.

Features:
- yfinance download with .NS suffix for NSE stocks
- MultiIndex column handling (yfinance 1.0+)
- Retry logic with exponential backoff
- 12-hour file caching
- Data validation: missing values, zero volumes, extreme outliers
- Market data download (NIFTY50, BANKNIFTY, VIX, USD/INR)
- Quality reporting
================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DataCollector:
    """
    Unified data collector merging pipeline and production data logic.

    Handles:
    - yfinance download with retry and MultiIndex handling
    - File-based caching (12-hour TTL)
    - Data validation and cleaning
    - Market context data (indices, VIX, forex)
    """

    def __init__(self, cache_hours: int = 12):
        self.cache_hours = cache_hours
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.quality_stats: Dict[str, Dict] = {}
        logger.info("DataCollector initialized")

    def collect_all(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None,
        force_download: bool = False
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Collect price data for all symbols + market data.

        Args:
            symbols: List of stock symbols
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            force_download: Bypass cache

        Returns:
            Tuple of (price_data dict, market_data dict)
        """
        logger.info("=" * 60)
        logger.info("DATA COLLECTION")
        logger.info("=" * 60)
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Period: {start_date or 'default'} to {end_date or 'today'}")

        price_data = {}

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Collecting {symbol}...")

            df = self._collect_symbol(
                symbol, start_date, end_date, force_download
            )

            if df is not None and len(df) > 0:
                df = self._validate_and_clean(df, symbol)
                price_data[symbol] = df
                self.data_cache[symbol] = df
                logger.info(f"  {symbol}: {len(df)} rows")
            else:
                logger.warning(f"  {symbol}: No data collected")

        # Collect market data
        market_data = self._collect_market_data(start_date, end_date, force_download)

        logger.success(f"Data collection complete: {len(price_data)} symbols, "
                      f"{len(market_data)} market indices")

        return price_data, market_data

    def collect_single(
        self,
        symbol: str,
        days: int = 1000,
        force_download: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Collect data for a single symbol (production-style with days param).

        Args:
            symbol: Stock symbol
            days: Number of calendar days to fetch
            force_download: Bypass cache

        Returns:
            DataFrame with OHLCV data or None
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = self._collect_symbol(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            force_download
        )

        if df is not None and len(df) > 0:
            df = self._validate_and_clean(df, symbol)
            df['symbol'] = symbol
            self.data_cache[symbol] = df

        return df

    def _collect_symbol(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        force_download: bool = False
    ) -> Optional[pd.DataFrame]:
        """Download data for a single symbol with retry logic."""
        import yfinance as yf

        # Check file cache first
        if not force_download:
            cached = self._load_from_cache(symbol)
            if cached is not None:
                return cached

        # Default dates
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        ticker = f"{symbol}.NS"

        # Retry with backoff
        for attempt in range(3):
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )

                if len(df) > 0:
                    df = df.reset_index()

                    # Handle yfinance 1.0+ MultiIndex columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] if isinstance(col, tuple) else col
                                     for col in df.columns]

                    # Standardize column names
                    df.columns = [c.lower() for c in df.columns]

                    # Rename date column
                    if 'date' in df.columns:
                        df = df.rename(columns={'date': 'timestamp'})

                    # Ensure timestamp is datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])

                    # Add symbol column
                    df['symbol'] = symbol

                    # Save to cache
                    self._save_to_cache(symbol, df)

                    return df
                else:
                    logger.warning(f"  {symbol}: Attempt {attempt+1} - no data returned")
                    time.sleep(1 + attempt)

            except Exception as e:
                logger.warning(f"  {symbol}: Attempt {attempt+1} failed - {e}")
                time.sleep(1 + attempt)

        # Fallback: try loading from raw data dir
        raw_path = os.path.join(config.RAW_DATA_DIR, f'{symbol}.csv')
        if os.path.exists(raw_path):
            try:
                df = pd.read_csv(raw_path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.warning(f"  {symbol}: Using cached file ({len(df)} rows)")
                return df
            except Exception as e:
                logger.error(f"  {symbol}: Failed to read cached data - {e}")

        return None

    def _collect_market_data(
        self,
        start_date: str = None,
        end_date: str = None,
        force_download: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Collect market context data (indices, VIX, forex)."""
        import yfinance as yf

        market_data = {}

        for name, ticker in config.MARKET_SYMBOLS.items():
            try:
                df = yf.download(
                    ticker,
                    start=start_date or '2015-01-01',
                    end=end_date or datetime.now().strftime('%Y-%m-%d'),
                    progress=False,
                    auto_adjust=True
                )

                if len(df) > 0:
                    df = df.reset_index()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] if isinstance(col, tuple) else col
                                     for col in df.columns]
                    df.columns = [c.lower() for c in df.columns]

                    if 'date' in df.columns:
                        df = df.rename(columns={'date': 'timestamp'})
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp')

                    market_data[name] = df
                    logger.info(f"  Market {name}: {len(df)} rows")

            except Exception as e:
                logger.warning(f"  Market {name}: Failed - {e}")

        return market_data

    def _validate_and_clean(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean price data."""
        initial_rows = len(df)
        issues = []

        # Check for required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                logger.error(f"  {symbol}: Missing required column: {col}")
                return df

        # Remove rows with missing prices
        price_cols = ['open', 'high', 'low', 'close']
        missing_mask = df[price_cols].isnull().any(axis=1)
        if missing_mask.sum() > 0:
            issues.append(f"Removed {missing_mask.sum()} rows with missing prices")
            df = df[~missing_mask]

        # Remove zero volume rows (market holidays incorrectly included)
        zero_vol = df['volume'] == 0
        if zero_vol.sum() > 0:
            issues.append(f"Removed {zero_vol.sum()} zero-volume rows")
            df = df[~zero_vol]

        # Check for extreme outliers (>50% daily moves)
        if len(df) > 1:
            daily_returns = df['close'].pct_change().abs()
            extreme_mask = daily_returns > 0.50
            if extreme_mask.sum() > 0:
                issues.append(f"Found {extreme_mask.sum()} extreme daily moves (>50%)")

        # Sort by date
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()

        # Store quality stats
        self.quality_stats[symbol] = {
            'initial_rows': initial_rows,
            'final_rows': len(df),
            'issues': issues,
            'date_range': f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}" if 'timestamp' in df.columns else 'N/A'
        }

        if issues:
            for issue in issues:
                logger.debug(f"  {symbol}: {issue}")

        return df

    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from file cache if fresh enough."""
        cache_path = os.path.join(config.RAW_DATA_DIR, f'{symbol}.csv')

        if os.path.exists(cache_path):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
            if file_age < timedelta(hours=self.cache_hours):
                try:
                    df = pd.read_csv(cache_path)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    logger.debug(f"  {symbol}: Loaded from cache ({len(df)} rows)")
                    return df
                except Exception:
                    pass

        return None

    def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        """Save data to file cache."""
        cache_path = os.path.join(config.RAW_DATA_DIR, f'{symbol}.csv')
        try:
            df.to_csv(cache_path, index=False)
        except Exception as e:
            logger.warning(f"  {symbol}: Failed to save cache - {e}")

    def get_quality_report(self) -> Dict:
        """Get data quality report for all collected symbols."""
        return {
            'symbols_collected': len(self.data_cache),
            'quality_stats': self.quality_stats,
            'total_rows': sum(len(df) for df in self.data_cache.values())
        }
