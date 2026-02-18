"""
Alternative Data Module for V3 Pipeline

Provides unique data sources beyond technical indicators:
- Google Trends: Retail interest signals
- Twitter Sentiment: Social media sentiment (TODO)
- Options Data: Smart money positioning (TODO)
"""

from .google_trends import GoogleTrendsCollector

__all__ = ['GoogleTrendsCollector']
