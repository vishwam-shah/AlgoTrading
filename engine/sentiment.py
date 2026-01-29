"""
================================================================================
FAST SENTIMENT ENGINE - Google News RSS
================================================================================
Ultra-fast sentiment analysis using Google News RSS feeds.

Key Features:
- No API keys required
- No rate limits (RSS is free and fast)
- Batch fetching (all news in one request)
- Caching to avoid redundant fetches
- VADER + TextBlob ensemble sentiment

Comparison to old implementation:
- Old: 1 HTTP request per date (very slow, 10+ minutes for 100 dates)
- New: 1 RSS request per symbol (< 1 second total)

Usage:
    from engine.sentiment import FastSentimentEngine
    
    engine = FastSentimentEngine()
    df = engine.add_sentiment_features(df, 'HDFCBANK')
================================================================================
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Cache directory
SENTIMENT_CACHE_DIR = os.path.join(config.DATA_DIR, 'sentiment')
os.makedirs(SENTIMENT_CACHE_DIR, exist_ok=True)


class FastSentimentEngine:
    """
    Fast sentiment analysis using Google News RSS.
    
    Fetches recent news via RSS (no rate limits), caches results,
    and computes sentiment using VADER + TextBlob ensemble.
    """
    
    # Stock name mappings for better news search
    STOCK_NAMES = {
        'HDFCBANK': 'HDFC Bank',
        'ICICIBANK': 'ICICI Bank',
        'SBIN': 'State Bank of India SBI',
        'AXISBANK': 'Axis Bank',
        'KOTAKBANK': 'Kotak Mahindra Bank',
        'TCS': 'Tata Consultancy Services TCS',
        'INFY': 'Infosys',
        'WIPRO': 'Wipro',
        'HCLTECH': 'HCL Technologies',
        'TECHM': 'Tech Mahindra',
        'RELIANCE': 'Reliance Industries',
        'TATAMOTORS': 'Tata Motors',
        'TATASTEEL': 'Tata Steel',
        'ITC': 'ITC Limited',
        'LT': 'Larsen Toubro L&T',
        'BHARTIARTL': 'Bharti Airtel',
        'HINDUNILVR': 'Hindustan Unilever HUL',
        'MARUTI': 'Maruti Suzuki',
        'BAJFINANCE': 'Bajaj Finance',
        'ADANIENT': 'Adani Enterprises',
        'ADANIPORTS': 'Adani Ports',
        'ASIANPAINT': 'Asian Paints',
        'SUNPHARMA': 'Sun Pharma',
        'DRREDDY': 'Dr Reddys',
        'DIVISLAB': 'Divis Laboratories',
        'CIPLA': 'Cipla',
        'COALINDIA': 'Coal India',
        'NTPC': 'NTPC',
        'POWERGRID': 'Power Grid',
        'ONGC': 'ONGC',
        'BPCL': 'BPCL',
        'IOC': 'Indian Oil IOC',
        'GAIL': 'GAIL India',
        'ULTRACEMCO': 'UltraTech Cement',
        'GRASIM': 'Grasim Industries',
        'JSWSTEEL': 'JSW Steel',
        'HINDALCO': 'Hindalco',
        'M&M': 'Mahindra Mahindra M&M',
        'EICHERMOTOR': 'Eicher Motors',
        'HEROMOTOCO': 'Hero MotoCorp',
        'BAJAJ-AUTO': 'Bajaj Auto',
        'APOLLOHOSP': 'Apollo Hospitals',
        'NESTLEIND': 'Nestle India',
        'BRITANNIA': 'Britannia',
        'TITAN': 'Titan Company',
        'INDUSINDBK': 'IndusInd Bank',
        'SBILIFE': 'SBI Life Insurance',
        'HDFCLIFE': 'HDFC Life Insurance',
    }
    
    def __init__(self, cache_hours: int = 6):
        """
        Initialize sentiment engine.
        
        Args:
            cache_hours: Hours before re-fetching news (default 6)
        """
        self.cache_hours = cache_hours
        self.vader = SentimentIntensityAnalyzer()
        self._news_cache: Dict[str, Tuple[datetime, List[Dict]]] = {}
        
    def _get_search_term(self, symbol: str) -> str:
        """Get human-readable search term for a symbol."""
        return self.STOCK_NAMES.get(symbol, symbol)
    
    def _fetch_google_news_rss(self, symbol: str, max_results: int = 50) -> List[Dict]:
        """
        Fetch news from Google News RSS feed.
        
        Args:
            symbol: Stock symbol
            max_results: Maximum news items to fetch
            
        Returns:
            List of news items with title, date, source
        """
        search_term = self._get_search_term(symbol)
        
        # Build Google News RSS URL
        query = quote(f'{search_term} stock NSE')
        rss_url = f'https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en'
        
        try:
            feed = feedparser.parse(rss_url)
            
            news_items = []
            for entry in feed.entries[:max_results]:
                # Parse publication date
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6])
                else:
                    pub_date = datetime.now()
                
                # Clean title (remove source suffix)
                title = entry.title
                if ' - ' in title:
                    title = title.rsplit(' - ', 1)[0]
                
                news_items.append({
                    'title': title,
                    'date': pub_date,
                    'source': entry.get('source', {}).get('title', 'Unknown'),
                    'link': entry.link
                })
            
            logger.debug(f"{symbol}: Fetched {len(news_items)} news items from RSS")
            return news_items
            
        except Exception as e:
            logger.warning(f"{symbol}: RSS fetch failed - {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment using VADER + TextBlob ensemble.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1 to +1)
        """
        if not text:
            return 0.0
        
        try:
            # VADER sentiment
            vader_score = self.vader.polarity_scores(text)['compound']
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_score = blob.sentiment.polarity
            
            # Ensemble (weighted average - VADER is better for headlines)
            ensemble_score = vader_score * 0.7 + textblob_score * 0.3
            
            return float(np.clip(ensemble_score, -1, 1))
            
        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}")
            return 0.0
    
    def _get_cached_news(self, symbol: str) -> Optional[List[Dict]]:
        """Get news from cache if still valid."""
        if symbol in self._news_cache:
            cache_time, news = self._news_cache[symbol]
            if datetime.now() - cache_time < timedelta(hours=self.cache_hours):
                return news
        return None
    
    def get_sentiment_scores(self, symbol: str) -> Dict[str, float]:
        """
        Get aggregated sentiment scores for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with sentiment metrics
        """
        # Check cache
        news = self._get_cached_news(symbol)
        
        if news is None:
            news = self._fetch_google_news_rss(symbol)
            self._news_cache[symbol] = (datetime.now(), news)
        
        if not news:
            return {
                'current': 0.0,
                'avg_7d': 0.0,
                'bullish_ratio': 0.0,
                'bearish_ratio': 0.0,
                'news_count': 0
            }
        
        # Analyze each headline
        sentiments = []
        recent_sentiments = []  # Last 7 days
        now = datetime.now()
        
        for item in news:
            score = self._analyze_sentiment(item['title'])
            sentiments.append({
                'score': score,
                'date': item['date'],
                'title': item['title']
            })
            
            if now - item['date'] <= timedelta(days=7):
                recent_sentiments.append(score)
        
        # Aggregate
        all_scores = [s['score'] for s in sentiments]
        
        return {
            'current': float(np.mean(all_scores)) if all_scores else 0.0,
            'avg_7d': float(np.mean(recent_sentiments)) if recent_sentiments else 0.0,
            'bullish_ratio': float(sum(1 for s in all_scores if s > 0.1) / len(all_scores)) if all_scores else 0.0,
            'bearish_ratio': float(sum(1 for s in all_scores if s < -0.1) / len(all_scores)) if all_scores else 0.0,
            'news_count': len(news)
        }
    
    def add_sentiment_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add sentiment features to dataframe (fast - single RSS fetch).
        
        Args:
            df: DataFrame with price data
            symbol: Stock symbol
            
        Returns:
            DataFrame with sentiment columns added
        """
        df = df.copy()
        
        # Fetch news once (all recent news)
        news = self._get_cached_news(symbol)
        if news is None:
            news = self._fetch_google_news_rss(symbol, max_results=100)
            self._news_cache[symbol] = (datetime.now(), news)
        
        if not news:
            logger.warning(f"{symbol}: No news found, using neutral sentiment")
            df['news_sentiment'] = 0.0
            df['news_sentiment_bullish'] = 0
            df['news_sentiment_bearish'] = 0
            df['news_sentiment_neutral'] = 1
            df['sentiment_ma_7d'] = 0.0
            df['sentiment_ma_30d'] = 0.0
            df['sentiment_trend'] = 0.0
            df['news_volume'] = 0
            return df
        
        # Create date -> sentiment mapping
        date_sentiment = {}
        date_news_count = {}
        
        for item in news:
            news_date = item['date'].date()
            score = self._analyze_sentiment(item['title'])
            
            if news_date not in date_sentiment:
                date_sentiment[news_date] = []
                date_news_count[news_date] = 0
            
            date_sentiment[news_date].append(score)
            date_news_count[news_date] += 1
        
        # Compute daily average sentiment
        daily_sentiment = {d: np.mean(scores) for d, scores in date_sentiment.items()}
        
        # Map to dataframe
        if 'timestamp' in df.columns:
            df['_date'] = pd.to_datetime(df['timestamp']).dt.date
        else:
            df['_date'] = pd.to_datetime(df.index).date
        
        df['news_sentiment'] = df['_date'].map(daily_sentiment).fillna(0.0)
        df['news_volume'] = df['_date'].map(date_news_count).fillna(0).astype(int)
        
        # Forward fill sentiment for dates without news (carry last known sentiment)
        df['news_sentiment'] = df['news_sentiment'].replace(0, np.nan).ffill().fillna(0)
        
        # Add derived features
        df['news_sentiment_bullish'] = (df['news_sentiment'] > 0.15).astype(int)
        df['news_sentiment_bearish'] = (df['news_sentiment'] < -0.15).astype(int)
        df['news_sentiment_neutral'] = ((df['news_sentiment'] >= -0.15) & 
                                        (df['news_sentiment'] <= 0.15)).astype(int)
        
        # Rolling features
        df['sentiment_ma_7d'] = df['news_sentiment'].rolling(7, min_periods=1).mean()
        df['sentiment_ma_30d'] = df['news_sentiment'].rolling(30, min_periods=1).mean()
        df['sentiment_trend'] = df['news_sentiment'] - df['sentiment_ma_7d']
        
        df = df.drop(columns=['_date'])
        
        sentiment_coverage = (df['news_sentiment'] != 0).sum() / len(df) * 100
        logger.success(f"{symbol}: Added sentiment features (coverage: {sentiment_coverage:.1f}%, {len(news)} articles)")
        
        return df
    
    def save_cache(self, symbol: str):
        """Save sentiment cache to disk."""
        if symbol not in self._news_cache:
            return
        
        cache_time, news = self._news_cache[symbol]
        
        cache_data = []
        for item in news:
            score = self._analyze_sentiment(item['title'])
            cache_data.append({
                'date': item['date'].strftime('%Y-%m-%d'),
                'sentiment': score,
                'title': item['title'],
                'source': item['source']
            })
        
        cache_path = os.path.join(SENTIMENT_CACHE_DIR, f"{symbol}_sentiment_cache.csv")
        pd.DataFrame(cache_data).to_csv(cache_path, index=False)
        logger.info(f"{symbol}: Saved {len(cache_data)} sentiment records to cache")


# Convenience function for feature_engine.py
def add_sentiment_to_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Drop-in replacement for old sentiment function (much faster)."""
    engine = FastSentimentEngine()
    return engine.add_sentiment_features(df, symbol)


if __name__ == '__main__':
    # Test the fast sentiment engine
    import time
    
    engine = FastSentimentEngine()
    
    symbols = ['SBIN', 'HDFCBANK', 'TCS', 'RELIANCE']
    
    for symbol in symbols:
        start = time.time()
        scores = engine.get_sentiment_scores(symbol)
        elapsed = time.time() - start
        
        print(f"\n{symbol} (fetched in {elapsed:.2f}s):")
        print(f"  Current sentiment: {scores['current']:.3f}")
        print(f"  7-day average: {scores['avg_7d']:.3f}")
        print(f"  Bullish ratio: {scores['bullish_ratio']:.1%}")
        print(f"  Bearish ratio: {scores['bearish_ratio']:.1%}")
        print(f"  News count: {scores['news_count']}")
    
    # Test dataframe integration
    print("\n" + "="*60)
    print("Testing DataFrame integration:")
    dates = pd.date_range('2024-11-01', '2024-12-31', freq='D')
    df = pd.DataFrame({'timestamp': dates, 'close': np.random.uniform(100, 150, len(dates))})
    
    start = time.time()
    df = engine.add_sentiment_features(df, 'SBIN')
    elapsed = time.time() - start
    
    print(f"Added sentiment to {len(df)} rows in {elapsed:.2f}s")
    print(df[['timestamp', 'news_sentiment', 'sentiment_ma_7d', 'news_volume']].tail(10))
