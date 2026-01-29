"""
================================================================================
STEP 1: DATA COLLECTION
================================================================================
Robust data collection with validation, caching, and quality checks.

Features:
- Multi-source data fetching (yfinance, backup sources)
- Data validation and cleaning
- Quality metrics and reporting
- Caching for efficiency
- Market data (indices, VIX, FX rates)

Usage:
    from pipeline.step_1_data_collection import DataCollector
    collector = DataCollector()
    price_data, market_data = collector.collect_all(symbols, start_date, end_date)

Or run directly:
    python pipeline/step_1_data_collection.py
================================================================================
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DataCollector:
    """
    Robust data collection with validation and quality checks.
    
    Features:
    - Downloads OHLCV data from yfinance
    - Validates data quality (missing values, outliers, gaps)
    - Caches data for efficiency
    - Downloads market context data (NIFTY, VIX, etc.)
    - Generates quality reports
    """
    
    def __init__(self, cache_dir: str = None):
        """Initialize DataCollector."""
        self.cache_dir = Path(cache_dir or config.RAW_DATA_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.market_cache_dir = Path(config.MARKET_DATA_DIR)
        self.market_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.quality_report = {}
        self.collection_stats = {
            'total_symbols': 0,
            'successful': 0,
            'failed': 0,
            'cached': 0,
            'downloaded': 0
        }
        
        logger.info("DataCollector initialized")
        logger.info(f"  Cache directory: {self.cache_dir}")
        
    def collect_all(
        self,
        symbols: List[str],
        start_date: str = '2022-01-01',
        end_date: str = None,
        use_cache: bool = True,
        cache_expiry_hours: int = 12
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Collect price and market data for all symbols.
        
        Args:
            symbols: List of stock symbols (e.g., ['HDFCBANK', 'TCS'])
            start_date: Start date for data collection
            end_date: End date (defaults to today)
            use_cache: Whether to use cached data
            cache_expiry_hours: Cache expiry time
            
        Returns:
            Tuple of (price_data dict, market_data dict)
        """
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        logger.info("=" * 60)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("=" * 60)
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Period: {start_date} to {end_date}")
        
        self.collection_stats['total_symbols'] = len(symbols)
        
        # Collect price data
        price_data = {}
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Collecting {symbol}...")
            
            df = self._collect_symbol(
                symbol, start_date, end_date, 
                use_cache, cache_expiry_hours
            )
            
            if df is not None and len(df) > 0:
                price_data[symbol] = df
                self.collection_stats['successful'] += 1
            else:
                self.collection_stats['failed'] += 1
                logger.warning(f"  Failed to collect {symbol}")
        
        # Collect market data
        market_data = self._collect_market_data(start_date, end_date, use_cache)
        
        # Generate quality report
        self._generate_quality_report(price_data)
        
        # Log summary
        logger.success(f"Data collection complete:")
        logger.info(f"  Successful: {self.collection_stats['successful']}/{len(symbols)}")
        logger.info(f"  Failed: {self.collection_stats['failed']}")
        logger.info(f"  Cached: {self.collection_stats['cached']}")
        logger.info(f"  Downloaded: {self.collection_stats['downloaded']}")
        
        return price_data, market_data
    
    def _collect_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool,
        cache_expiry_hours: int
    ) -> Optional[pd.DataFrame]:
        """Collect data for a single symbol."""
        
        # Check cache
        cache_file = self.cache_dir / f"{symbol}_raw.csv"
        
        if use_cache and cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.total_seconds() < cache_expiry_hours * 3600:
                try:
                    df = pd.read_csv(cache_file, parse_dates=['date'], index_col='date')
                    # Filter to requested date range
                    df = df.loc[start_date:end_date]
                    if len(df) > 100:
                        self.collection_stats['cached'] += 1
                        return df
                except Exception as e:
                    logger.warning(f"  Cache read failed: {e}")
        
        # Download from yfinance
        try:
            ticker = f"{symbol}.NS"
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                return None
            
            # Handle multi-level columns from newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten multi-level columns (take first level)
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            # Standardize columns to lowercase
            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]
            
            # Rename columns if needed
            if 'adj close' in df.columns:
                df = df.drop('adj close', axis=1, errors='ignore')
            
            # Handle 'price' column from some yfinance versions
            if 'price' in df.columns and 'close' not in df.columns:
                df['close'] = df['price']
            
            # Ensure required columns
            required = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in df.columns:
                    logger.warning(f"  Missing column: {col}")
                    logger.warning(f"  Available columns: {df.columns.tolist()}")
                    return None
            
            df = df.set_index('date')
            
            # Validate data
            df = self._validate_and_clean(df, symbol)
            
            if df is not None and len(df) > 0:
                # Save to cache
                df.to_csv(cache_file)
                self.collection_stats['downloaded'] += 1
            
            return df
            
        except Exception as e:
            logger.error(f"  Download error for {symbol}: {e}")
            return None
    
    def _validate_and_clean(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """Validate and clean data."""
        if df is None or len(df) == 0:
            return None
        
        original_len = len(df)
        
        # Remove rows with missing critical values
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Remove rows with zero/negative prices
        df = df[(df['close'] > 0) & (df['open'] > 0) & 
                (df['high'] > 0) & (df['low'] > 0)]
        
        # Remove rows with zero volume
        df = df[df['volume'] > 0]
        
        # Fix OHLC consistency (high >= low, high >= close, etc.)
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Remove extreme outliers (>50% daily change)
        returns = df['close'].pct_change().abs()
        df = df[returns < 0.5]
        
        # Sort by date
        df = df.sort_index()
        
        # Store quality metrics
        self.quality_report[symbol] = {
            'original_rows': original_len,
            'clean_rows': len(df),
            'missing_pct': (original_len - len(df)) / original_len * 100,
            'date_range': f"{df.index.min()} to {df.index.max()}" if len(df) > 0 else "N/A",
            'trading_days': len(df)
        }
        
        return df
    
    def _collect_market_data(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool
    ) -> Dict[str, pd.DataFrame]:
        """Collect market context data."""
        logger.info("Collecting market data...")
        
        market_data = {}
        
        # Market symbols to collect
        market_symbols = {
            'NIFTY50': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'INDIA_VIX': '^INDIAVIX',
            'USD_INR': 'INR=X'
        }
        
        for name, ticker in market_symbols.items():
            cache_file = self.market_cache_dir / f"{name}.csv"
            
            # Check cache
            if use_cache and cache_file.exists():
                try:
                    df = pd.read_csv(cache_file, parse_dates=['date'], index_col='date')
                    df = df.loc[start_date:end_date]
                    if len(df) > 100:
                        market_data[name] = df
                        continue
                except:
                    pass
            
            # Download
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    # Handle multi-level columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                    
                    df = df.reset_index()
                    df.columns = [str(c).lower() for c in df.columns]
                    df = df.set_index('date')
                    df.to_csv(cache_file)
                    market_data[name] = df
                    logger.info(f"  {name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"  Failed to collect {name}: {e}")
        
        return market_data
    
    def _generate_quality_report(self, price_data: Dict[str, pd.DataFrame]):
        """Generate data quality report."""
        if not price_data:
            return
        
        # Aggregate quality metrics
        total_rows = sum(len(df) for df in price_data.values())
        avg_rows = total_rows / len(price_data)
        
        # Find date coverage
        all_dates = set()
        for df in price_data.values():
            all_dates.update(df.index.tolist())
        
        self.quality_report['summary'] = {
            'total_symbols': len(price_data),
            'total_rows': total_rows,
            'avg_rows_per_symbol': avg_rows,
            'unique_dates': len(all_dates),
            'date_range': f"{min(all_dates)} to {max(all_dates)}" if all_dates else "N/A"
        }
    
    def get_quality_report(self) -> Dict:
        """Get data quality report."""
        return self.quality_report
    
    def save_quality_report(self, filepath: str = None):
        """Save quality report to file."""
        filepath = filepath or str(self.cache_dir / 'quality_report.json')
        with open(filepath, 'w') as f:
            json.dump(self.quality_report, f, indent=2, default=str)
        logger.info(f"Quality report saved to {filepath}")


def test_data_collection():
    """Test data collection with sample stocks."""
    print("\n" + "=" * 80)
    print("TESTING STEP 1: DATA COLLECTION")
    print("=" * 80)
    
    # Test with 10 stocks
    test_symbols = [
        'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK',
        'TCS', 'INFY', 'RELIANCE', 'TATASTEEL', 'HINDUNILVR'
    ]
    
    collector = DataCollector()
    price_data, market_data = collector.collect_all(
        symbols=test_symbols,
        start_date='2022-01-01',
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    print("\n" + "=" * 80)
    print("DATA COLLECTION RESULTS")
    print("=" * 80)
    
    print(f"\n✓ Price Data: {len(price_data)} symbols collected")
    for symbol, df in price_data.items():
        print(f"  {symbol}: {len(df)} rows, "
              f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    
    print(f"\n✓ Market Data: {len(market_data)} indices collected")
    for name, df in market_data.items():
        print(f"  {name}: {len(df)} rows")
    
    # Quality report
    report = collector.get_quality_report()
    if 'summary' in report:
        print(f"\n✓ Quality Summary:")
        print(f"  Total Rows: {report['summary']['total_rows']:,}")
        print(f"  Avg Rows/Symbol: {report['summary']['avg_rows_per_symbol']:.0f}")
        print(f"  Date Range: {report['summary']['date_range']}")
    
    # Validation tests
    print("\n" + "-" * 40)
    print("VALIDATION TESTS")
    print("-" * 40)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Data completeness
    tests_total += 1
    if len(price_data) >= 8:
        print("✓ Test 1: Data completeness - PASSED")
        tests_passed += 1
    else:
        print("✗ Test 1: Data completeness - FAILED")
    
    # Test 2: Sufficient history
    tests_total += 1
    min_rows = min(len(df) for df in price_data.values()) if price_data else 0
    if min_rows >= 200:
        print(f"✓ Test 2: Sufficient history ({min_rows} rows min) - PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 2: Sufficient history ({min_rows} rows min) - FAILED")
    
    # Test 3: No missing values in critical columns
    tests_total += 1
    has_missing = False
    for symbol, df in price_data.items():
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
            has_missing = True
            break
    if not has_missing:
        print("✓ Test 3: No missing values - PASSED")
        tests_passed += 1
    else:
        print("✗ Test 3: No missing values - FAILED")
    
    # Test 4: Market data available
    tests_total += 1
    if 'NIFTY50' in market_data:
        print("✓ Test 4: Market data available - PASSED")
        tests_passed += 1
    else:
        print("✗ Test 4: Market data available - FAILED")
    
    print(f"\n{'=' * 40}")
    print(f"TESTS: {tests_passed}/{tests_total} passed")
    print("=" * 40)
    
    return price_data, market_data


if __name__ == "__main__":
    test_data_collection()
