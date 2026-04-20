"""
Download historical OHLCV data from yfinance for NSE stocks.
Caches to parquet for reuse.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, List

# Load verified ticker map
try:
    _CONFIG_DIR = Path(__file__).resolve().parent.parent / "00_config"
    sys.path.insert(0, str(_CONFIG_DIR))
    from tickers import to_yf  # type: ignore
except Exception:
    def to_yf(s: str) -> str: return f"{s}.NS"  # type: ignore


class DataDownloader:
    """Download and cache stock data from yfinance."""

    def __init__(self, data_dir: Path, delay: float = 1.0):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay  # seconds between downloads

    def download(
        self,
        symbol: str,
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Download stock data. Use cache if available and use_cache=True.

        Args:
            symbol: Stock ticker (e.g., "SBIN" — ".NS" suffix added automatically)
            start_date: Start date for download (YYYY-MM-DD)
            end_date: End date (defaults to today)
            use_cache: Use cached parquet if available

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        cache_file = self.data_dir / f"{symbol}.parquet"

        # Check cache
        if use_cache and cache_file.exists():
            return self._load_cached(symbol, cache_file, start_date, end_date)

        # Download fresh
        ticker = to_yf(symbol)
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date or datetime.now().strftime("%Y-%m-%d"),
                progress=False,
            )

            if data.empty:
                print(f"⚠ No data for {symbol}")
                return None

            # Normalize columns
            data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            data.columns = ["open", "high", "low", "close", "volume"]
            data = data.reset_index()
            data.columns = ["timestamp", "open", "high", "low", "close", "volume"]

            # Clean
            data["volume"] = data["volume"].astype(int)
            data = data.dropna()

            # Cache
            data.to_parquet(cache_file, compression="snappy", index=False)
            time.sleep(self.delay)  # Rate limit

            print(f"✓ Downloaded {symbol}: {len(data)} rows")
            return data

        except Exception as e:
            print(f"✗ Error downloading {symbol}: {e}")
            return None

    def _load_cached(
        self,
        symbol: str,
        cache_file: Path,
        start_date: str,
        end_date: Optional[str],
    ) -> pd.DataFrame:
        """Load cached data and filter by date range."""
        df = pd.read_parquet(cache_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if start_date:
            df = df[df["timestamp"] >= start_date]
        if end_date:
            df = df[df["timestamp"] <= end_date]

        return df if len(df) > 0 else None

    def download_incremental(
        self,
        symbol: str,
        data_start_date: str = "2018-01-01",
    ) -> Optional[pd.DataFrame]:
        """
        Download stock data incrementally — only fetch new rows, don't re-download everything.

        If cache exists: loads existing parquet, checks last date, fetches only newer data.
        If cache missing: downloads full history from data_start_date.

        This is MUCH faster for daily updates than re-downloading the entire history.

        Args:
            symbol: Stock ticker (e.g., "SBIN")
            data_start_date: If no cache, fetch from this date

        Returns:
            DataFrame with all available data (cached + new), or None on error
        """
        cache_file = self.data_dir / f"{symbol}.parquet"

        try:
            # If cache exists, only fetch new rows
            if cache_file.exists():
                existing = pd.read_parquet(cache_file)
                existing["timestamp"] = pd.to_datetime(existing["timestamp"])
                last_date = existing["timestamp"].max()

                # Fetch from 4 days before last_date (to fill any gaps from holidays)
                fetch_start = (last_date - pd.Timedelta(days=4)).strftime("%Y-%m-%d")

                # If we're up-to-date, return existing
                if pd.Timestamp(fetch_start) >= pd.Timestamp.today():
                    return existing

                # Fetch new data
                ticker = f"{symbol}.NS"
                new_data = yf.download(
                    to_yf(symbol),
                    start=fetch_start,
                    end=datetime.now().strftime("%Y-%m-%d"),
                    progress=False,
                )

                if new_data.empty:
                    return existing  # No new data, return what we have

                # Normalize new data
                new_data = new_data[["Open", "High", "Low", "Close", "Volume"]].copy()
                new_data.columns = ["open", "high", "low", "close", "volume"]
                new_data = new_data.reset_index()
                new_data.columns = ["timestamp", "open", "high", "low", "close", "volume"]
                new_data["volume"] = new_data["volume"].astype(int)
                new_data = new_data.dropna()

                # Combine: concat + drop duplicates by timestamp + sort
                combined = pd.concat([existing, new_data], ignore_index=True)
                combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
                combined = combined.sort_values("timestamp").reset_index(drop=True)

                # Save back
                combined.to_parquet(cache_file, compression="snappy", index=False)
                time.sleep(self.delay)

                return combined

            else:
                # No cache: download full history
                return self.download(symbol, start_date=data_start_date, use_cache=False)

        except Exception as e:
            print(f"✗ Error downloading {symbol} incrementally: {e}")
            return None

    def download_batch(
        self,
        symbols: List[str],
        start_date: str = "2019-01-01",
        use_cache: bool = True,
    ) -> dict:
        """Download multiple stocks. Returns {symbol: DataFrame}."""
        results = {}
        for symbol in symbols:
            df = self.download(symbol, start_date=start_date, use_cache=use_cache)
            if df is not None:
                results[symbol] = df
        return results

    def download_batch_incremental(
        self,
        symbols: List[str],
        data_start_date: str = "2018-01-01",
    ) -> dict:
        """Download multiple stocks incrementally. Returns {symbol: DataFrame}."""
        results = {}
        for symbol in symbols:
            df = self.download_incremental(symbol, data_start_date=data_start_date)
            if df is not None:
                results[symbol] = df
        return results
