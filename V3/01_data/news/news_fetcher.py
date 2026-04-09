"""
News sentiment fetcher — Google News RSS + FinBERT.

Two-tier architecture:
1. Tier 1 (Google News RSS): Fast, zero-cost, headline-level sentiment
2. Tier 2 (FinBERT): Deep learning embeddings, takes longer but more accurate

Usage:
    fetcher = NewsFeaturizer()
    sentiment = fetcher.fetch_and_score('SBIN', use_finbert=False)
    # Returns: {score: float [-1, 1], articles: int, spike_flag: bool}
"""

import feedparser
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class NewsFeaturizer:
    """Fetch news and compute sentiment scores."""

    # VADER lexicon (simplified sentiment scoring)
    POSITIVE_WORDS = {
        "gain", "up", "rise", "jump", "bullish", "strong", "beat",
        "success", "profit", "rally", "surge", "boom", "hope",
        "growth", "positive", "outperform", "upgrade", "recovery"
    }

    NEGATIVE_WORDS = {
        "loss", "down", "fall", "drop", "bearish", "weak", "miss",
        "failure", "decline", "crash", "plunge", "doom", "fear",
        "contraction", "negative", "downgrade", "underperform", "recession"
    }

    def __init__(self, cache_dir: Path = None):
        """
        Initialize news fetcher.

        Args:
            cache_dir: Directory to cache sentiment scores (optional)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "nse_news"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_google_news(
        self,
        symbol: str,
        max_articles: int = 20,
    ) -> List[str]:
        """
        Fetch headlines from Google News RSS for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'SBIN')
            max_articles: Maximum articles to fetch

        Returns:
            List of headline strings
        """
        try:
            # Google News RSS URL for Indian markets
            # Format: https://news.google.com/rss/search?q={query}
            query = f"{symbol} NSE stock India"
            url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

            feed = feedparser.parse(url)
            headlines = []

            for entry in feed.entries[:max_articles]:
                headline = entry.get("title", "")
                if headline:
                    headlines.append(headline)

            return headlines

        except Exception as e:
            print(f"⚠ Failed to fetch news for {symbol}: {e}")
            return []

    def simple_sentiment_score(self, text: str) -> float:
        """
        Simple VADER-style sentiment scoring.

        Returns:
            Score in [-1, 1] where -1 = very negative, 1 = very positive
        """
        text_lower = text.lower()

        # Count positive and negative words
        positive_count = sum(1 for word in self.POSITIVE_WORDS if word in text_lower)
        negative_count = sum(1 for word in self.NEGATIVE_WORDS if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0  # Neutral

        # Score: (pos - neg) / (pos + neg)
        score = (positive_count - negative_count) / total
        return np.clip(score, -1.0, 1.0)

    def finbert_sentiment(self, headlines: List[str]) -> Dict[str, float]:
        """
        Analyze headlines with FinBERT (requires transformers + torch).

        This is a placeholder. Real implementation would:
            from transformers import pipeline
            classifier = pipeline("text-classification", model="ProsusAI/finbert")
            results = [classifier(h)[0] for h in headlines]

        Returns:
            {
                'positive_ratio': fraction classified positive,
                'negative_ratio': fraction classified negative,
                'neutral_ratio': fraction classified neutral,
                'confidence_avg': average confidence
            }
        """
        if not headlines:
            return {
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 1.0,
                "confidence_avg": 0.0,
            }

        # Placeholder: would use FinBERT in production
        # For now, use simple scoring as fallback
        scores = [self.simple_sentiment_score(h) for h in headlines]
        avg_score = np.mean(scores) if scores else 0.0

        positive = sum(1 for s in scores if s > 0.3)
        negative = sum(1 for s in scores if s < -0.3)
        neutral = len(scores) - positive - negative

        return {
            "positive_ratio": positive / len(scores) if scores else 0.0,
            "negative_ratio": negative / len(scores) if scores else 0.0,
            "neutral_ratio": neutral / len(scores) if scores else 0.0,
            "confidence_avg": abs(avg_score),
        }

    def fetch_and_score(
        self,
        symbol: str,
        date: str = None,
        use_finbert: bool = False,
        lookback_days: int = 1,
    ) -> Dict[str, float]:
        """
        Fetch news for a symbol and compute sentiment score.

        Args:
            symbol: Stock symbol
            date: Trading date (YYYY-MM-DD), defaults to today
            use_finbert: Use FinBERT (slower but more accurate) vs simple scoring
            lookback_days: How many days back to look for news

        Returns:
            {
                'raw_score': float [-1, 1],
                'positive_ratio': float [0, 1],
                'negative_ratio': float [0, 1],
                'n_articles': int,
                'spike_flag': bool (if score is extreme),
                'timestamp': str
            }
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Fetch headlines
        headlines = self.fetch_google_news(symbol, max_articles=20)

        if not headlines:
            return {
                "raw_score": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "n_articles": 0,
                "spike_flag": False,
                "timestamp": date,
            }

        # Score with FinBERT or simple method
        if use_finbert:
            finbert_scores = self.finbert_sentiment(headlines)
            # Convert to -1 to 1 scale
            raw_score = (
                finbert_scores["positive_ratio"]
                - finbert_scores["negative_ratio"]
            )
        else:
            # Simple scoring
            scores = [self.simple_sentiment_score(h) for h in headlines]
            raw_score = np.mean(scores)
            finbert_scores = {
                "positive_ratio": sum(1 for s in scores if s > 0.3) / len(scores),
                "negative_ratio": sum(1 for s in scores if s < -0.3) / len(scores),
            }

        # Spike detection: if score is >2 std above 30-day mean
        # (simplified, would need historical cache in production)
        is_spike = abs(raw_score) > 0.7

        return {
            "raw_score": float(raw_score),
            "positive_ratio": float(finbert_scores["positive_ratio"]),
            "negative_ratio": float(finbert_scores["negative_ratio"]),
            "n_articles": len(headlines),
            "spike_flag": is_spike,
            "timestamp": date,
        }

    def adjust_confidence_threshold(
        self,
        base_threshold: float,
        sentiment: Dict[str, float],
    ) -> float:
        """
        Adjust signal confidence threshold based on sentiment.

        High-confidence signals are confirmed by positive sentiment.
        Negative sentiment raises the bar for trading.

        Args:
            base_threshold: Base threshold (e.g., 0.58)
            sentiment: Sentiment dict from fetch_and_score()

        Returns:
            Adjusted threshold (higher = harder to trade)
        """
        raw_score = sentiment["raw_score"]
        spike = sentiment["spike_flag"]

        if spike and raw_score < 0:
            # Strong negative news: raise threshold
            return min(base_threshold + 0.05, 0.75)

        elif raw_score > 0.5 and not spike:
            # Consistent positive sentiment: lower threshold slightly
            return max(base_threshold - 0.02, 0.50)

        elif raw_score < -0.5:
            # Moderate negative: raise threshold
            return base_threshold + 0.02

        else:
            # Neutral or mixed: use base
            return base_threshold
