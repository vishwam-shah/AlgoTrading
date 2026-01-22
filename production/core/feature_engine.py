"""
================================================================================
ADVANCED FEATURE ENGINE
================================================================================
Production-grade feature engineering combining:
- Technical indicators (50+ features)
- Market regime detection (10 features)
- Sentiment analysis (8 features)
- Macro market context (15 features)
- Statistical features (20 features)
- Unique alpha signals (10 features)

Total: ~113 carefully selected features
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class FeatureSet:
    """Container for computed features"""
    df: pd.DataFrame
    feature_names: List[str]
    n_features: int
    timestamp: datetime


class AdvancedFeatureEngine:
    """
    Production-grade feature engineering.

    Unique aspects that differentiate from competition:
    1. Adaptive indicators that adjust to market regime
    2. Cross-asset signals (Nifty, VIX, sector indices)
    3. Sentiment integration from news
    4. Statistical regime detection
    5. Volume-price divergence signals
    """

    def __init__(self, include_sentiment: bool = True, include_market_context: bool = True):
        self.include_sentiment = include_sentiment
        self.include_market_context = include_market_context
        self._feature_names = []

    def compute_all_features(self, df: pd.DataFrame, symbol: str = None) -> FeatureSet:
        """
        Compute all features for a stock.

        Args:
            df: DataFrame with OHLCV columns
            symbol: Stock symbol (for sector-specific features)

        Returns:
            FeatureSet with all computed features
        """
        logger.info(f"Computing features for {symbol or 'stock'}...")

        df = df.copy()

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # 1. Core Technical Features (50 features)
        df = self._compute_technical_features(df)

        # 2. Volatility & Risk Features (15 features)
        df = self._compute_volatility_features(df)

        # 3. Volume Analysis Features (12 features)
        df = self._compute_volume_features(df)

        # 4. Momentum & Trend Features (15 features)
        df = self._compute_momentum_features(df)

        # 5. Statistical Features (10 features)
        df = self._compute_statistical_features(df)

        # 6. Market Regime Detection (8 features)
        df = self._compute_regime_features(df)

        # 7. Alpha Signals - Unique to our system (10 features)
        df = self._compute_alpha_signals(df)

        # 8. Market Context (if enabled)
        if self.include_market_context:
            df = self._add_market_context(df, symbol)

        # 9. Sentiment (if enabled)
        if self.include_sentiment:
            df = self._add_sentiment_features(df, symbol)

        # Handle infinities and NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Get feature names (exclude OHLCV and timestamp)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'date', 'symbol']
        self._feature_names = [c for c in df.columns if c not in exclude_cols
                               and not c.startswith('target_')]

        logger.info(f"Computed {len(self._feature_names)} features")

        return FeatureSet(
            df=df,
            feature_names=self._feature_names,
            n_features=len(self._feature_names),
            timestamp=datetime.now()
        )

    def _compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Core technical indicators."""

        # === PRICE RETURNS (7) ===
        for period in [1, 2, 3, 5, 10, 20, 60]:
            df[f'return_{period}d'] = df['close'].pct_change(period)

        # === MOVING AVERAGES & RATIOS (12) ===
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']

        # EMA
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # === RSI (3) ===
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # === MACD (4) ===
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_divergence'] = df['macd'].diff(5)

        # === BOLLINGER BANDS (5) ===
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # === STOCHASTIC (4) ===
        for period in [14, 21]:
            low_n = df['low'].rolling(period).min()
            high_n = df['high'].rolling(period).max()
            df[f'stoch_k_{period}'] = 100 * (df['close'] - low_n) / (high_n - low_n + 1e-10)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()

        # === ATR (3) ===
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)

        for period in [7, 14, 21]:
            df[f'atr_{period}'] = tr.rolling(period).mean()

        df['atr_pct'] = df['atr_14'] / df['close'] * 100

        # === ADX (3) ===
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr_14 = tr.rolling(14).sum()
        plus_di = 100 * plus_dm.rolling(14).sum() / (tr_14 + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).sum() / (tr_14 + 1e-10)

        df['adx_plus_di'] = plus_di
        df['adx_minus_di'] = minus_di
        df['adx'] = 100 * abs(plus_di - minus_di).rolling(14).mean() / (plus_di + minus_di + 1e-10)

        # === CANDLESTICK PATTERNS (6) ===
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['candle_lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']
        df['candle_body_ratio'] = df['candle_body'] / (df['candle_range'] + 1e-10)
        df['is_doji'] = (df['candle_body'] < df['candle_range'] * 0.1).astype(int)

        return df

    def _compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility and risk measures."""

        # Historical volatility (annualized)
        for period in [5, 10, 20, 60]:
            df[f'volatility_{period}d'] = df['return_1d'].rolling(period).std() * np.sqrt(252)

        # Parkinson volatility (using high-low)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) *
            ((np.log(df['high'] / df['low']) ** 2).rolling(20).mean())
        ) * np.sqrt(252)

        # Garman-Klass volatility
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        df['gk_volatility'] = np.sqrt(
            (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(20).mean()
        ) * np.sqrt(252)

        # Volatility change
        df['vol_change_5d'] = df['volatility_20d'].pct_change(5)
        df['vol_change_10d'] = df['volatility_20d'].pct_change(10)

        # Volatility ratio (short-term vs long-term)
        df['vol_ratio'] = df['volatility_10d'] / (df['volatility_60d'] + 1e-10)

        # High-low range as volatility proxy
        df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['range_ma_20'] = df['range_pct'].rolling(20).mean()

        # Volatility regime
        df['high_vol_regime'] = (df['volatility_20d'] > df['volatility_60d']).astype(int)

        return df

    def _compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume analysis features."""

        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()

        # Volume ratio
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)

        # OBV (On Balance Volume)
        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv'] = obv
        df['obv_slope_5'] = obv.diff(5) / 5
        df['obv_slope_20'] = obv.diff(20) / 20

        # Volume-Price Trend
        vpt = (df['close'].pct_change() * df['volume']).cumsum()
        df['vpt'] = vpt
        df['vpt_slope'] = vpt.diff(10)

        # Accumulation/Distribution
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        df['ad_line'] = (clv * df['volume']).cumsum()

        # Volume-Price divergence (unique alpha signal)
        price_trend = df['close'].pct_change(10)
        vol_trend = df['volume'].pct_change(10)
        df['vol_price_divergence'] = price_trend - vol_trend

        return df

    def _compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum and trend features."""

        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period) * 100

        # Momentum (absolute)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)

        # Price acceleration
        df['acceleration'] = df['momentum_10'].diff(5)

        # Trend strength
        df['trend_5'] = (df['close'] > df['sma_5']).astype(int)
        df['trend_20'] = (df['close'] > df['sma_20']).astype(int)
        df['trend_50'] = (df['close'] > df['sma_50']).astype(int)
        df['trend_strength'] = df['trend_5'] + df['trend_20'] + df['trend_50']

        # Higher highs / higher lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['uptrend_score'] = (df['higher_high'].rolling(5).sum() +
                               df['higher_low'].rolling(5).sum())

        # Distance from 52-week high/low
        df['dist_52w_high'] = df['close'] / df['high'].rolling(252).max() - 1
        df['dist_52w_low'] = df['close'] / df['low'].rolling(252).min() - 1

        return df

    def _compute_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features for pattern detection."""

        returns = df['return_1d']

        # Rolling statistics
        df['return_mean_20'] = returns.rolling(20).mean()
        df['return_std_20'] = returns.rolling(20).std()
        df['return_skew_20'] = returns.rolling(20).skew()
        df['return_kurt_20'] = returns.rolling(20).kurt()

        # Z-score
        df['price_zscore'] = (df['close'] - df['close'].rolling(20).mean()) / (
            df['close'].rolling(20).std() + 1e-10)

        # Return percentile
        df['return_percentile'] = returns.rolling(60).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
        )

        # Autocorrelation
        df['autocorr_1'] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )

        # Hurst exponent approximation (trending vs mean-reverting)
        df['hurst_proxy'] = df['return_std_20'] / (df['return_std_20'].rolling(5).mean() + 1e-10)

        # Consecutive up/down days
        df['consec_up'] = (returns > 0).astype(int)
        df['consec_up'] = df['consec_up'].groupby(
            (df['consec_up'] != df['consec_up'].shift()).cumsum()
        ).cumsum() * df['consec_up']

        return df

    def _compute_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market regime detection."""

        # Trend regime (based on moving averages)
        df['ma_trend_regime'] = np.where(
            (df['sma_20'] > df['sma_50']) & (df['sma_50'] > df['sma_200']), 2,  # Strong uptrend
            np.where(df['sma_20'] > df['sma_50'], 1,  # Mild uptrend
            np.where((df['sma_20'] < df['sma_50']) & (df['sma_50'] < df['sma_200']), -2,  # Strong downtrend
            np.where(df['sma_20'] < df['sma_50'], -1, 0)))  # Mild downtrend / sideways
        )

        # Volatility regime
        vol_median = df['volatility_20d'].rolling(60).median()
        df['vol_regime'] = np.where(
            df['volatility_20d'] > vol_median * 1.5, 2,  # High vol
            np.where(df['volatility_20d'] > vol_median, 1,  # Above average
            np.where(df['volatility_20d'] < vol_median * 0.5, -1, 0))  # Low vol
        )

        # Momentum regime
        df['momentum_regime'] = np.where(
            (df['rsi_14'] > 70) & (df['adx'] > 25), 2,  # Strong overbought
            np.where(df['rsi_14'] > 60, 1,  # Overbought
            np.where((df['rsi_14'] < 30) & (df['adx'] > 25), -2,  # Strong oversold
            np.where(df['rsi_14'] < 40, -1, 0)))  # Oversold
        )

        # Combined regime score
        df['regime_score'] = df['ma_trend_regime'] + df['vol_regime'] + df['momentum_regime']

        # Regime changes
        df['regime_change'] = df['ma_trend_regime'].diff().abs()

        # Days since regime change
        regime_changes = df['ma_trend_regime'].diff().abs() > 0
        df['days_in_regime'] = (~regime_changes).astype(int).groupby(
            regime_changes.cumsum()
        ).cumsum()

        return df

    def _compute_alpha_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Unique alpha signals - our competitive edge.
        These are proprietary signals not commonly found in other systems.
        """

        # 1. Smart Money Flow - combines price and volume in a unique way
        price_change = df['close'].pct_change()
        vol_change = df['volume'].pct_change()
        df['smart_money_flow'] = np.where(
            (price_change > 0) & (vol_change > 0.5), 2,  # Strong buying
            np.where((price_change > 0) & (vol_change < -0.3), 1,  # Stealth buying
            np.where((price_change < 0) & (vol_change > 0.5), -2,  # Strong selling
            np.where((price_change < 0) & (vol_change < -0.3), -1, 0)))  # Stealth selling
        )
        df['smart_money_flow_ma'] = df['smart_money_flow'].rolling(5).mean()

        # 2. Gap Analysis - overnight sentiment
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
        df['gap_filled'] = ((df['gap_pct'] > 0) & (df['low'] < df['close'].shift(1)) |
                           (df['gap_pct'] < 0) & (df['high'] > df['close'].shift(1))).astype(int)

        # 3. Intraday Strength - where did price close within the day's range
        df['intraday_strength'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['intraday_strength_ma'] = df['intraday_strength'].rolling(5).mean()

        # 4. Price Efficiency Ratio - trend strength vs noise
        net_change = abs(df['close'] - df['close'].shift(10))
        total_change = df['close'].diff().abs().rolling(10).sum()
        df['efficiency_ratio'] = net_change / (total_change + 1e-10)

        # 5. Relative Volume Breakout - unusual activity
        df['rel_volume_breakout'] = (df['volume'] > df['volume_sma_20'] * 2).astype(int)

        # 6. Mean Reversion Signal
        df['mean_reversion_signal'] = -df['price_zscore'] * (1 - df['efficiency_ratio'])

        return df

    def _add_market_context(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Add market-wide context features."""
        import yfinance as yf

        try:
            # Get market data timeframe
            start_date = df['timestamp'].min() if 'timestamp' in df.columns else df.index.min()
            end_date = df['timestamp'].max() if 'timestamp' in df.columns else df.index.max()

            # Download Nifty 50
            nifty = yf.download('^NSEI', start=start_date, end=end_date, progress=False)
            if len(nifty) > 0:
                nifty = nifty.reset_index()
                if isinstance(nifty.columns, pd.MultiIndex):
                    nifty.columns = nifty.columns.get_level_values(0)
                nifty.columns = [c.lower() for c in nifty.columns]

                # Nifty features
                nifty['nifty_return'] = nifty['close'].pct_change()
                nifty['nifty_volatility'] = nifty['nifty_return'].rolling(20).std() * np.sqrt(252)
                nifty['nifty_trend'] = (nifty['close'] > nifty['close'].rolling(20).mean()).astype(int)

                # Merge with main dataframe
                merge_col = 'timestamp' if 'timestamp' in df.columns else 'date'
                nifty_merge_col = 'date' if 'date' in nifty.columns else nifty.columns[0]

                if merge_col in df.columns:
                    df[merge_col] = pd.to_datetime(df[merge_col]).dt.date
                    nifty[nifty_merge_col] = pd.to_datetime(nifty[nifty_merge_col]).dt.date

                    df = df.merge(
                        nifty[[nifty_merge_col, 'nifty_return', 'nifty_volatility', 'nifty_trend']],
                        left_on=merge_col, right_on=nifty_merge_col, how='left'
                    )

                    # Stock beta (relative to market)
                    if 'return_1d' in df.columns:
                        df['stock_beta'] = df['return_1d'].rolling(60).cov(df['nifty_return']) / (
                            df['nifty_return'].rolling(60).var() + 1e-10
                        )
                        df['stock_alpha'] = df['return_1d'] - df['stock_beta'] * df['nifty_return']

                    logger.info("Added market context features")

        except Exception as e:
            logger.warning(f"Could not add market context: {e}")
            # Add placeholder columns
            df['nifty_return'] = 0
            df['nifty_volatility'] = 0
            df['nifty_trend'] = 0
            df['stock_beta'] = 1
            df['stock_alpha'] = 0

        return df

    def _add_sentiment_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add news sentiment features."""

        try:
            # Try to load from cache
            from pipeline.utils.sentiment_scraper import add_sentiment_to_dataframe
            df = add_sentiment_to_dataframe(df, symbol)
            logger.info(f"Added sentiment features for {symbol}")
        except Exception as e:
            logger.warning(f"Could not add sentiment for {symbol}: {e}")
            # Add placeholder columns
            df['news_sentiment'] = 0
            df['news_sentiment_bullish'] = 0
            df['news_sentiment_bearish'] = 0
            df['news_sentiment_neutral'] = 1
            df['sentiment_ma_7d'] = 0
            df['sentiment_ma_30d'] = 0
            df['sentiment_trend'] = 0

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of computed feature names."""
        return self._feature_names

    @staticmethod
    def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
        """Compute prediction targets."""
        df = df.copy()

        # Next day close return
        df['target_close_return'] = df['close'].pct_change(-1)  # Negative shift for next day

        # Next day high return (from today's close)
        df['target_high_return'] = (df['high'].shift(-1) - df['close']) / df['close']

        # Next day low return (from today's close)
        df['target_low_return'] = (df['low'].shift(-1) - df['close']) / df['close']

        # Direction (binary: up=1, down=0)
        df['target_direction'] = (df['target_close_return'] > 0).astype(int)

        # 5-class direction
        thresholds = config.DIRECTION_THRESHOLDS
        df['target_direction_5class'] = pd.cut(
            df['target_close_return'],
            bins=[-np.inf, thresholds['strong_bear'], thresholds['weak_bear'],
                  thresholds['neutral'], thresholds['weak_bull'], np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

        # Actual next day prices (for evaluation)
        df['target_close'] = df['close'].shift(-1)
        df['target_high'] = df['high'].shift(-1)
        df['target_low'] = df['low'].shift(-1)

        return df
