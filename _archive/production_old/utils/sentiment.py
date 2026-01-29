"""
================================================================================
HISTORICAL SENTIMENT SCRAPER
================================================================================
Fetches historical news sentiment for stocks using Google News.
Stores daily sentiment scores in cache to avoid re-fetching.

Key Features:
- Free (no API key required)
- Historical data via date filtering
- Caches sentiment to data/sentiment/{SYMBOL}_sentiment.csv
- Fetches 3-7 days worth of news per date for robustness

Usage:
    from pipeline.utils.sentiment_scraper import get_historical_sentiment
    
    # Get sentiment for specific date
    sentiment = get_historical_sentiment('HDFCBANK', '2024-11-28')
    
    # Bulk update for all dates in dataframe
    df = update_sentiment_cache('HDFCBANK', df)
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from gnews import GNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Sentiment cache directory
SENTIMENT_DIR = os.path.join(config.DATA_DIR, 'sentiment')
os.makedirs(SENTIMENT_DIR, exist_ok=True)


def get_news_for_date(symbol: str, target_date: str, lookback_days: int = 7) -> float:
    """
    Fetch news sentiment for a specific date.
    Uses Google News with date filtering.
    
    Args:
        symbol: Stock symbol
        target_date: Date in 'YYYY-MM-DD' format
        lookback_days: Number of days before target_date to search
        
    Returns:
        Sentiment score (-1 to +1)
    """
    try:
        target_dt = pd.to_datetime(target_date)
        start_date = target_dt - timedelta(days=lookback_days)
        
        # Initialize Google News for specific date range
        google_news = GNews(
            language='en',
            country='IN',
            period=f'{lookback_days}d',
            start_date=start_date.to_pydatetime(),
            end_date=target_dt.to_pydatetime(),
            max_results=10
        )
        
        analyzer = SentimentIntensityAnalyzer()
        
        # Search for news
        news = google_news.get_news(symbol)
        
        if not news:
            return 0.0
        
        # Analyze sentiment
        sentiments = []
        for article in news[:10]:
            title = article.get('title', '')
            if title:
                sentiment = analyzer.polarity_scores(title)
                sentiments.append(sentiment['compound'])
        
        if sentiments:
            return float(np.mean(sentiments))
        
    except Exception as e:
        logger.debug(f"{symbol} {target_date}: Sentiment fetch failed - {str(e)}")
    
    return 0.0


def load_sentiment_cache(symbol: str) -> pd.DataFrame:
    """
    Load existing sentiment cache for a symbol.
    
    Returns:
        DataFrame with columns: date, sentiment
    """
    cache_path = os.path.join(SENTIMENT_DIR, f"{symbol}_sentiment.csv")
    
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.warning(f"{symbol}: Could not load sentiment cache - {str(e)}")
    
    return pd.DataFrame(columns=['date', 'sentiment'])


def save_sentiment_cache(symbol: str, cache_df: pd.DataFrame):
    """Save sentiment cache to CSV."""
    cache_path = os.path.join(SENTIMENT_DIR, f"{symbol}_sentiment.csv")
    cache_df = cache_df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    cache_df.to_csv(cache_path, index=False)


def update_sentiment_cache(symbol: str, price_df: pd.DataFrame, 
                           batch_size: int = 50,
                           delay: float = 1.0,
                           max_fetch: int = None,
                           lookback_days: int = None) -> pd.DataFrame:
    """
    Update sentiment cache for all dates in price dataframe.
    Only fetches sentiment for dates not already in cache.
    
    Args:
        symbol: Stock symbol
        price_df: DataFrame with 'timestamp' column
        batch_size: Fetch sentiment for this many dates per batch
        delay: Delay between batches (seconds) to avoid rate limits
        max_fetch: Maximum number of new dates to fetch (None = use config default)
        lookback_days: Only fetch sentiment for last N days (None = use config default)
        
    Returns:
        Updated cache DataFrame
    """
    import config
    
    # Use config defaults if not specified
    if max_fetch is None:
        max_fetch = config.SENTIMENT_MAX_FETCH
    if lookback_days is None:
        lookback_days = getattr(config, 'SENTIMENT_LOOKBACK_DAYS', 365)
    
    cache_df = load_sentiment_cache(symbol)
    
    # Get dates that need sentiment
    price_dates = pd.to_datetime(price_df['timestamp']).dt.date
    
    # Filter to only recent dates if lookback_days is specified
    if lookback_days:
        cutoff_date = pd.Timestamp.now().date() - pd.Timedelta(days=lookback_days)
        price_dates = [d for d in price_dates if d >= cutoff_date]
        logger.info(f"{symbol}: Limiting sentiment to last {lookback_days} days ({len(price_dates)} dates)")
    
    cached_dates = set(cache_df['date'].dt.date if len(cache_df) > 0 else [])
    missing_dates = sorted(set(price_dates) - cached_dates)
    
    if not missing_dates:
        logger.info(f"{symbol}: Sentiment cache up to date")
        return cache_df
    
    # Limit fetching to max_fetch most recent dates
    if max_fetch and len(missing_dates) > max_fetch:
        logger.warning(f"{symbol}: {len(missing_dates)} missing dates, fetching only {max_fetch} most recent")
        missing_dates = missing_dates[-max_fetch:]
    
    logger.info(f"{symbol}: Fetching sentiment for {len(missing_dates)} dates...")
    
    # Fetch in batches to avoid overwhelming API
    new_rows = []
    for i in range(0, len(missing_dates), batch_size):
        batch = missing_dates[i:i+batch_size]
        
        for date in batch:
            try:
                sentiment = get_news_for_date(symbol, str(date))
                new_rows.append({'date': date, 'sentiment': sentiment})
            except Exception as e:
                logger.debug(f"{symbol} {date}: Sentiment failed - {str(e)}")
                # Use neutral sentiment on error
                new_rows.append({'date': date, 'sentiment': 0.0})
        
        logger.info(f"{symbol}: Fetched {min(i+batch_size, len(missing_dates))}/{len(missing_dates)}")
        
        # Rate limiting
        if i + batch_size < len(missing_dates):
            time.sleep(delay)
    
    # Merge with cache
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        new_df['date'] = pd.to_datetime(new_df['date'])
        cache_df = pd.concat([cache_df, new_df], ignore_index=True)
        save_sentiment_cache(symbol, cache_df)
        logger.success(f"{symbol}: Added {len(new_rows)} sentiment records to cache")
    
    return cache_df


def get_historical_sentiment(symbol: str, target_date: str) -> float:
    """
    Get sentiment for a specific date (from cache or fetch).
    
    Args:
        symbol: Stock symbol
        target_date: Date in 'YYYY-MM-DD' format
        
    Returns:
        Sentiment score (-1 to +1)
    """
    cache_df = load_sentiment_cache(symbol)
    target_dt = pd.to_datetime(target_date).date()
    
    # Check cache first
    if len(cache_df) > 0:
        match = cache_df[cache_df['date'].dt.date == target_dt]
        if len(match) > 0:
            return float(match.iloc[0]['sentiment'])
    
    # Fetch if not in cache
    sentiment = get_news_for_date(symbol, target_date)
    
    # Add to cache
    new_row = pd.DataFrame([{'date': target_dt, 'sentiment': sentiment}])
    cache_df = pd.concat([cache_df, new_row], ignore_index=True)
    save_sentiment_cache(symbol, cache_df)
    
    return sentiment


def add_sentiment_to_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Add sentiment column to dataframe using historical sentiment cache.
    Missing dates get neutral sentiment (0.0).
    
    Args:
        df: DataFrame with 'timestamp' column
        symbol: Stock symbol
        
    Returns:
        DataFrame with 'news_sentiment' column added
    """
    df = df.copy()
    
    # Ensure cache is up to date (only recent dates)
    cache_df = update_sentiment_cache(symbol, df)
    
    # Merge sentiment with price data
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    cache_df['date'] = cache_df['date'].dt.date
    
    df = df.merge(cache_df[['date', 'sentiment']], on='date', how='left')
    df.rename(columns={'sentiment': 'news_sentiment'}, inplace=True)
    
    # Fill missing sentiment with 0.0 (neutral) for dates we didn't fetch
    df['news_sentiment'] = df['news_sentiment'].fillna(0.0)
    
    # Count how many have sentiment
    sentiment_count = (df['news_sentiment'] != 0.0).sum()
    logger.info(f"{symbol}: {sentiment_count}/{len(df)} rows have sentiment data")
    
    # Add derived features
    df['news_sentiment_bullish'] = (df['news_sentiment'] > 0.2).astype(int)
    df['news_sentiment_bearish'] = (df['news_sentiment'] < -0.2).astype(int)
    df['news_sentiment_neutral'] = ((df['news_sentiment'] >= -0.2) & 
                                    (df['news_sentiment'] <= 0.2)).astype(int)
    
    # Rolling sentiment features
    df['sentiment_ma_7d'] = df['news_sentiment'].rolling(7, min_periods=1).mean()
    df['sentiment_ma_30d'] = df['news_sentiment'].rolling(30, min_periods=1).mean()
    df['sentiment_trend'] = df['news_sentiment'] - df['sentiment_ma_7d']
    
    df = df.drop(columns=['date'])
    
    return df


if __name__ == '__main__':
    # Test the sentiment scraper
    symbol = 'HDFCBANK'
    
    # Test single date
    sentiment = get_historical_sentiment(symbol, '2024-11-28')
    print(f"{symbol} sentiment on 2024-11-28: {sentiment:.3f}")
    
    # Test with date range
    dates = pd.date_range('2024-11-01', '2024-11-30', freq='D')
    df = pd.DataFrame({'timestamp': dates})
    df = add_sentiment_to_dataframe(df, symbol)
    
    print(f"\nSentiment summary for {symbol} in Nov 2024:")
    print(df[['timestamp', 'news_sentiment', 'sentiment_ma_7d']].tail(10))
