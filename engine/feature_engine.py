"""
================================================================================
UNIFIED FEATURE ENGINE
================================================================================
Merged feature engineering from production/ and pipeline/ systems.

Production AdvancedFeatureEngine: ~113 features (single-stock, per-symbol)
Pipeline FeatureEngineer: ~68 features (multi-stock, dict-based)

Unified: ~130 deduplicated features across 10 categories:
1. Technical (50): SMA, EMA, RSI, MACD, Bollinger, Stochastic, ATR, ADX
2. Volatility (15): Historical, Parkinson, Garman-Klass, regime
3. Volume (12): OBV, VPT, AD line, MFI, VWAP
4. Momentum (15): ROC, momentum, acceleration, trend strength
5. Statistical (10): Skewness, kurtosis, z-scores, autocorrelation
6. Regime (8): Trend, volatility, momentum regime detection
7. Alpha (6): Smart money flow, gaps, efficiency ratio, mean reversion
8. Candlestick (6): Body ratio, shadows, doji detection
9. Market Relative (5): Beta, relative strength, correlation
10. Time-based (5): Day of week, month, quarter, month boundaries
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import enhanced indicators module
try:
    from engine.enhanced_indicators import add_all_enhanced_indicators
    ENHANCED_INDICATORS_AVAILABLE = True
except ImportError:
    ENHANCED_INDICATORS_AVAILABLE = False
    logger.warning("Enhanced indicators module not available. Using base indicators only.")


@dataclass
class FeatureSet:
    """Container for computed features."""
    df: pd.DataFrame
    feature_names: List[str]
    n_features: int
    timestamp: datetime


class AdvancedFeatureEngine:
    """
    Production-grade feature engineering (single-stock interface).

    Takes a single DataFrame with OHLCV columns and returns a FeatureSet.
    Used by the production orchestrator for per-symbol processing.
    """

    def __init__(self, include_sentiment: bool = True, include_market_context: bool = True):
        self.include_sentiment = include_sentiment
        self.include_market_context = include_market_context
        self._feature_names = []

    def compute_all_features(self, df: pd.DataFrame, symbol: str = None) -> FeatureSet:
        """Compute all features for a single stock DataFrame."""
        logger.info(f"Computing features for {symbol or 'stock'}...")

        df = df.copy()

        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Core feature categories
        df = _compute_technical_features(df)
        df = _compute_volatility_features(df)
        df = _compute_volume_features(df)
        df = _compute_momentum_features(df)
        df = _compute_statistical_features(df)
        df = _compute_regime_features(df)
        df = _compute_alpha_signals(df)

        # NEW: Enhanced indicators (60+ additional features)
        if ENHANCED_INDICATORS_AVAILABLE:
            logger.info("Adding 60+ enhanced indicators...")
            df = add_all_enhanced_indicators(df)
        
        # NEW: Lagged features for better accuracy (40+ features, no data leakage)
        try:
            from engine.lagged_features import add_lagged_features
            logger.info("Adding lagged features for improved accuracy...")
            df = add_lagged_features(df)
        except Exception as e:
            logger.warning(f"Could not add lagged features: {e}")

        # Market context (optional)
        if self.include_market_context:
            df = _add_market_context(df, symbol)

        # Sentiment (optional)
        if self.include_sentiment:
            df = _add_sentiment_features(df, symbol)

        # Handle infinities and NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Get feature names
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

    def get_feature_names(self) -> List[str]:
        return self._feature_names

    @staticmethod
    def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
        """Compute prediction targets.
        
        CRITICAL: Targets must represent FUTURE values we want to predict
        from CURRENT features. We use shift(-1) to move future values forward.
        
        Example: Row i contains:
        - Features: computed from data up to and including day i
        - Targets: what we want to predict for day i+1
        """
        df = df.copy()

        # Target: tomorrow's return (what we're trying to predict)
        # We compute pct_change() which gives us today's return from yesterday,
        # then shift(-1) to get tomorrow's return into today's row
        df['target_close_return'] = df['close'].pct_change().shift(-1)
        
        # Target: tomorrow's high/low relative to today's close
        df['target_high_return'] = ((df['high'].shift(-1) - df['close']) / df['close'])
        df['target_low_return'] = ((df['low'].shift(-1) - df['close']) / df['close'])
        
        # Binary direction: will price go up tomorrow?
        df['target_direction'] = (df['target_close_return'] > 0).astype(int)

        # Multi-class direction (5 classes: strong bear, weak bear, neutral, weak bull, strong bull)
        thresholds = config.DIRECTION_THRESHOLDS
        df['target_direction_5class'] = pd.cut(
            df['target_close_return'],
            bins=[-np.inf, thresholds['strong_bear'], thresholds['weak_bear'],
                  thresholds['neutral'], thresholds['weak_bull'], np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

        # Raw future prices (for regression models)
        df['target_close'] = df['close'].shift(-1)
        df['target_high'] = df['high'].shift(-1)
        df['target_low'] = df['low'].shift(-1)

        return df


class FeatureEngineer:
    """
    Multi-stock feature engineering (pipeline-style interface).

    Takes a dict of symbol -> OHLCV DataFrames and returns a dict of
    symbol -> feature DataFrames. Used by the pipeline orchestrator.
    """

    def __init__(self):
        self.feature_count = 0
        self.feature_names = []
        self.computation_stats = {}
        logger.info("FeatureEngineer initialized")

    def compute_all_features(
        self,
        price_data: Dict[str, pd.DataFrame],
        market_data: Dict[str, pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """Compute all features for all symbols."""
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING")
        logger.info("=" * 60)
        logger.info(f"Symbols: {len(price_data)}")

        features = {}

        for i, (symbol, df) in enumerate(price_data.items(), 1):
            logger.info(f"[{i}/{len(price_data)}] Computing features for {symbol}...")

            try:
                feature_df = self._compute_symbol_features(df, market_data, symbol)
                if feature_df is not None:
                    features[symbol] = feature_df
                    self.computation_stats[symbol] = {
                        'rows': len(feature_df),
                        'features': len(feature_df.columns),
                        'success': True
                    }
            except Exception as e:
                logger.error(f"  Error computing features for {symbol}: {e}")
                self.computation_stats[symbol] = {'success': False, 'error': str(e)}

        if features:
            self.feature_names = list(list(features.values())[0].columns)
            self.feature_count = len(self.feature_names)

        logger.success(f"Feature engineering complete: {len(features)} symbols, {self.feature_count} features")
        return features

    def _compute_symbol_features(
        self,
        df: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
        symbol: str
    ) -> pd.DataFrame:
        """Compute all features for a single symbol (pipeline interface)."""
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        volume = df['volume']

        # === TREND ===
        for period in [5, 10, 20, 50, 200]:
            features[f'sma_{period}'] = close.rolling(period).mean()
            features[f'sma_{period}_slope'] = features[f'sma_{period}'].diff(5) / 5

        for period in [12, 26, 50]:
            features[f'ema_{period}'] = close.ewm(span=period).mean()

        features['price_vs_sma20'] = close / features['sma_20'] - 1
        features['price_vs_sma50'] = close / features['sma_50'] - 1
        features['price_vs_sma200'] = close / features['sma_200'] - 1
        features['sma_20_50_cross'] = (features['sma_20'] > features['sma_50']).astype(int)
        features['sma_50_200_cross'] = (features['sma_50'] > features['sma_200']).astype(int)

        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        # === MOMENTUM ===
        for period in [1, 2, 3, 5, 10, 20, 63, 126, 252]:
            features[f'return_{period}d'] = close.pct_change(period)

        for period in [7, 14, 21]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        for period in [14, 21]:
            low_n = low.rolling(period).min()
            high_n = high.rolling(period).max()
            features[f'stoch_k_{period}'] = 100 * (close - low_n) / (high_n - low_n + 1e-10)
            features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()

        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        features['williams_r'] = (high_14 - close) / (high_14 - low_14 + 1e-10) * -100

        for period in [5, 10, 20]:
            features[f'roc_{period}'] = close.pct_change(period) * 100

        features['momentum_10'] = close - close.shift(10)
        features['momentum_20'] = close - close.shift(20)
        features['acceleration'] = features['momentum_10'].diff(5)

        features['trend_5'] = (close > features['sma_5']).astype(int)
        features['trend_20'] = (close > features['sma_20']).astype(int)
        features['trend_50'] = (close > features['sma_50']).astype(int)
        features['trend_strength'] = features['trend_5'] + features['trend_20'] + features['trend_50']

        features['higher_high'] = (high > high.shift(1)).astype(int)
        features['higher_low'] = (low > low.shift(1)).astype(int)
        features['lower_low'] = (low < low.shift(1)).astype(int)
        features['uptrend_score'] = (features['higher_high'].rolling(5).sum() +
                                     features['higher_low'].rolling(5).sum())

        features['dist_52w_high'] = close / high.rolling(252).max() - 1
        features['dist_52w_low'] = close / low.rolling(252).min() - 1

        # === VOLATILITY ===
        for period in [5, 10, 20, 60]:
            features[f'volatility_{period}d'] = close.pct_change().rolling(period).std() * np.sqrt(252)

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        for period in [7, 14, 21]:
            features[f'atr_{period}'] = tr.rolling(period).mean()
        features['atr_pct'] = features['atr_14'] / close * 100

        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['bb_upper'] = sma_20 + 2 * std_20
        features['bb_lower'] = sma_20 - 2 * std_20
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)

        features['keltner_upper'] = features['ema_26'] + 2 * features['atr_14']
        features['keltner_lower'] = features['ema_26'] - 2 * features['atr_14']

        features['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) *
            (np.log(high / low) ** 2).rolling(20).mean()
        ) * np.sqrt(252)

        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_price) ** 2
        features['gk_volatility'] = np.sqrt(
            (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(20).mean()
        ) * np.sqrt(252)

        features['vol_change_5d'] = features['volatility_20d'].pct_change(5)
        features['vol_ratio'] = features['volatility_10d'] / (features['volatility_60d'] + 1e-10)
        features['range_pct'] = (high - low) / close * 100
        features['high_vol_regime'] = (features['volatility_20d'] > features['volatility_60d']).astype(int)

        # === VOLUME ===
        for period in [5, 10, 20]:
            features[f'volume_sma_{period}'] = volume.rolling(period).mean()
        features['volume_ratio'] = volume / (features['volume_sma_20'] + 1e-10)

        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_slope_5'] = obv.diff(5) / 5
        features['obv_slope_20'] = obv.diff(20) / 20

        features['vpt'] = (close.pct_change() * volume).cumsum()
        features['vpt_slope'] = features['vpt'].diff(10)

        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        features['ad_line'] = (clv * volume).cumsum()
        features['vol_price_divergence'] = close.pct_change(10) - volume.pct_change(10)

        typical_price = (high + low + close) / 3
        raw_mf = typical_price * volume
        mf_positive = raw_mf.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        mf_negative = raw_mf.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        features['mfi'] = 100 - (100 / (1 + mf_positive / (mf_negative + 1e-10)))

        features['vwap'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        features['price_vs_vwap'] = close / features['vwap'] - 1

        features['cmf_20'] = (clv * volume).rolling(20).sum() / volume.rolling(20).sum()

        # === ADX ===
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr_14 = tr.rolling(14).sum()
        plus_di = 100 * plus_dm.rolling(14).sum() / (tr_14 + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).sum() / (tr_14 + 1e-10)
        features['adx_plus_di'] = plus_di
        features['adx_minus_di'] = minus_di
        features['adx'] = 100 * abs(plus_di - minus_di).rolling(14).mean() / (plus_di + minus_di + 1e-10)

        # === STATISTICAL ===
        returns = close.pct_change()
        features['return_mean_20'] = returns.rolling(20).mean()
        features['return_std_20'] = returns.rolling(20).std()
        features['return_skew_20'] = returns.rolling(20).skew()
        features['return_kurt_20'] = returns.rolling(20).kurt()
        features['price_zscore'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-10)
        features['return_percentile'] = returns.rolling(60).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
        )
        features['autocorr_1'] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )
        features['hurst_proxy'] = features['return_std_20'] / (features['return_std_20'].rolling(5).mean() + 1e-10)

        consec_up = (returns > 0).astype(int)
        features['consec_up'] = consec_up.groupby(
            (consec_up != consec_up.shift()).cumsum()
        ).cumsum() * consec_up

        # === REGIME ===
        features['ma_trend_regime'] = np.where(
            (features['sma_20'] > features['sma_50']) & (features['sma_50'] > features['sma_200']), 2,
            np.where(features['sma_20'] > features['sma_50'], 1,
            np.where((features['sma_20'] < features['sma_50']) & (features['sma_50'] < features['sma_200']), -2,
            np.where(features['sma_20'] < features['sma_50'], -1, 0)))
        )

        vol_median = features['volatility_20d'].rolling(60).median()
        features['vol_regime'] = np.where(
            features['volatility_20d'] > vol_median * 1.5, 2,
            np.where(features['volatility_20d'] > vol_median, 1,
            np.where(features['volatility_20d'] < vol_median * 0.5, -1, 0))
        )

        features['momentum_regime'] = np.where(
            (features['rsi_14'] > 70) & (features['adx'] > 25), 2,
            np.where(features['rsi_14'] > 60, 1,
            np.where((features['rsi_14'] < 30) & (features['adx'] > 25), -2,
            np.where(features['rsi_14'] < 40, -1, 0)))
        )

        features['regime_score'] = features['ma_trend_regime'] + features['vol_regime'] + features['momentum_regime']
        features['regime_change'] = features['ma_trend_regime'].diff().abs()
        regime_changes = features['ma_trend_regime'].diff().abs() > 0
        features['days_in_regime'] = (~regime_changes).astype(int).groupby(
            regime_changes.cumsum()
        ).cumsum()

        # === ALPHA SIGNALS ===
        price_change = close.pct_change()
        vol_change = volume.pct_change()
        features['smart_money_flow'] = np.where(
            (price_change > 0) & (vol_change > 0.5), 2,
            np.where((price_change > 0) & (vol_change < -0.3), 1,
            np.where((price_change < 0) & (vol_change > 0.5), -2,
            np.where((price_change < 0) & (vol_change < -0.3), -1, 0)))
        )
        features['smart_money_flow_ma'] = pd.Series(features['smart_money_flow'], index=df.index).rolling(5).mean()

        features['gap_pct'] = (open_price - close.shift(1)) / close.shift(1) * 100
        features['gap_up'] = (open_price > close.shift(1)).astype(int)
        features['gap_down'] = (open_price < close.shift(1)).astype(int)

        features['intraday_strength'] = (close - low) / (high - low + 1e-10)
        features['intraday_strength_ma'] = features['intraday_strength'].rolling(5).mean()

        net_change = abs(close - close.shift(10))
        total_change = close.diff().abs().rolling(10).sum()
        features['efficiency_ratio'] = net_change / (total_change + 1e-10)

        features['rel_volume_breakout'] = (volume > features['volume_sma_20'] * 2).astype(int)
        features['mean_reversion_signal'] = -features['price_zscore'] * (1 - features['efficiency_ratio'])

        # === CANDLESTICK ===
        body = abs(close - open_price)
        candle_range = high - low
        features['candle_body'] = body
        features['candle_upper_shadow'] = high - pd.concat([close, open_price], axis=1).max(axis=1)
        features['candle_lower_shadow'] = pd.concat([close, open_price], axis=1).min(axis=1) - low
        features['candle_range'] = candle_range
        features['candle_body_ratio'] = body / (candle_range + 1e-10)
        features['is_doji'] = (body < candle_range * 0.1).astype(int)
        features['body_pct'] = (close - open_price) / open_price

        # N-day high/low
        for period in [20, 52]:
            features[f'is_{period}d_high'] = (close >= close.rolling(period).max()).astype(int)
            features[f'is_{period}d_low'] = (close <= close.rolling(period).min()).astype(int)

        # === MARKET RELATIVE ===
        if market_data and 'NIFTY50' in market_data:
            nifty = market_data['NIFTY50']['close'].reindex(df.index, method='ffill')
            stock_ret = close.pct_change()
            nifty_ret = nifty.pct_change()
            cov = stock_ret.rolling(60).cov(nifty_ret)
            var = nifty_ret.rolling(60).var()
            features['beta_60d'] = cov / (var + 1e-10)
            for period in [20, 60]:
                features[f'rs_vs_nifty_{period}d'] = close.pct_change(period) - nifty.pct_change(period)
            features['corr_nifty_60d'] = stock_ret.rolling(60).corr(nifty_ret)

        # === TIME-BASED ===
        if isinstance(df.index, pd.DatetimeIndex):
            features['day_of_week'] = df.index.dayofweek
            features['month'] = df.index.month
            features['quarter'] = df.index.quarter
            features['is_month_end'] = df.index.is_month_end.astype(int)
            features['is_month_start'] = df.index.is_month_start.astype(int)

        # === SENTIMENT PROXY ===
        features['fear_greed_proxy'] = (
            (features['rsi_14'] / 100) * 0.25 +
            features['bb_position'].clip(0, 1) * 0.25 +
            (features['stoch_k_14'] / 100) * 0.25 +
            (features['mfi'] / 100) * 0.25
        )

        # Clean
        features = features.ffill().bfill()
        features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        return features

    def get_feature_count(self) -> int:
        return self.feature_count

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def get_summary(self) -> Dict:
        return {
            'total_features': self.feature_count,
            'symbols_processed': len(self.computation_stats),
            'computation_stats': self.computation_stats
        }


# ============================================================================
# SHARED FEATURE COMPUTATION FUNCTIONS (used by AdvancedFeatureEngine)
# ============================================================================

def _compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Core technical indicators."""
    for period in [1, 2, 3, 5, 10, 20, 60]:
        df[f'return_{period}d'] = df['close'].pct_change(period)

    for period in [5, 10, 20, 50, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']

    for period in [12, 26]:
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_divergence'] = df['macd'].diff(5)

    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    for period in [14, 21]:
        low_n = df['low'].rolling(period).min()
        high_n = df['high'].rolling(period).max()
        df[f'stoch_k_{period}'] = 100 * (df['close'] - low_n) / (high_n - low_n + 1e-10)
        df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()

    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)

    for period in [7, 14, 21]:
        df[f'atr_{period}'] = tr.rolling(period).mean()
    df['atr_pct'] = df['atr_14'] / df['close'] * 100

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

    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['candle_lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['candle_range'] = df['high'] - df['low']
    df['candle_body_ratio'] = df['candle_body'] / (df['candle_range'] + 1e-10)
    df['is_doji'] = (df['candle_body'] < df['candle_range'] * 0.1).astype(int)

    return df


def _compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility and risk measures."""
    for period in [5, 10, 20, 60]:
        df[f'volatility_{period}d'] = df['return_1d'].rolling(period).std() * np.sqrt(252)

    df['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) *
        ((np.log(df['high'] / df['low']) ** 2).rolling(20).mean())
    ) * np.sqrt(252)

    log_hl = np.log(df['high'] / df['low']) ** 2
    log_co = np.log(df['close'] / df['open']) ** 2
    df['gk_volatility'] = np.sqrt(
        (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(20).mean()
    ) * np.sqrt(252)

    df['vol_change_5d'] = df['volatility_20d'].pct_change(5)
    df['vol_change_10d'] = df['volatility_20d'].pct_change(10)
    df['vol_ratio'] = df['volatility_10d'] / (df['volatility_60d'] + 1e-10)
    df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100
    df['range_ma_20'] = df['range_pct'].rolling(20).mean()
    df['high_vol_regime'] = (df['volatility_20d'] > df['volatility_60d']).astype(int)

    return df


def _compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume analysis features."""
    for period in [5, 10, 20]:
        df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()

    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)

    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv'] = obv
    df['obv_slope_5'] = obv.diff(5) / 5
    df['obv_slope_20'] = obv.diff(20) / 20

    vpt = (df['close'].pct_change() * df['volume']).cumsum()
    df['vpt'] = vpt
    df['vpt_slope'] = vpt.diff(10)

    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    df['ad_line'] = (clv * df['volume']).cumsum()

    price_trend = df['close'].pct_change(10)
    vol_trend = df['volume'].pct_change(10)
    df['vol_price_divergence'] = price_trend - vol_trend

    return df


def _compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum and trend features."""
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = df['close'].pct_change(period) * 100

    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)
    df['acceleration'] = df['momentum_10'].diff(5)

    df['trend_5'] = (df['close'] > df['sma_5']).astype(int)
    df['trend_20'] = (df['close'] > df['sma_20']).astype(int)
    df['trend_50'] = (df['close'] > df['sma_50']).astype(int)
    df['trend_strength'] = df['trend_5'] + df['trend_20'] + df['trend_50']

    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
    df['uptrend_score'] = (df['higher_high'].rolling(5).sum() + df['higher_low'].rolling(5).sum())

    df['dist_52w_high'] = df['close'] / df['high'].rolling(252).max() - 1
    df['dist_52w_low'] = df['close'] / df['low'].rolling(252).min() - 1

    return df


def _compute_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Statistical features."""
    returns = df['return_1d']

    df['return_mean_20'] = returns.rolling(20).mean()
    df['return_std_20'] = returns.rolling(20).std()
    df['return_skew_20'] = returns.rolling(20).skew()
    df['return_kurt_20'] = returns.rolling(20).kurt()

    df['price_zscore'] = (df['close'] - df['close'].rolling(20).mean()) / (
        df['close'].rolling(20).std() + 1e-10)

    df['return_percentile'] = returns.rolling(60).apply(
        lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
    )
    df['autocorr_1'] = returns.rolling(20).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    df['hurst_proxy'] = df['return_std_20'] / (df['return_std_20'].rolling(5).mean() + 1e-10)

    consec_up = (returns > 0).astype(int)
    df['consec_up'] = consec_up.groupby(
        (consec_up != consec_up.shift()).cumsum()
    ).cumsum() * consec_up

    return df


def _compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Market regime detection."""
    df['ma_trend_regime'] = np.where(
        (df['sma_20'] > df['sma_50']) & (df['sma_50'] > df['sma_200']), 2,
        np.where(df['sma_20'] > df['sma_50'], 1,
        np.where((df['sma_20'] < df['sma_50']) & (df['sma_50'] < df['sma_200']), -2,
        np.where(df['sma_20'] < df['sma_50'], -1, 0)))
    )

    vol_median = df['volatility_20d'].rolling(60).median()
    df['vol_regime'] = np.where(
        df['volatility_20d'] > vol_median * 1.5, 2,
        np.where(df['volatility_20d'] > vol_median, 1,
        np.where(df['volatility_20d'] < vol_median * 0.5, -1, 0))
    )

    df['momentum_regime'] = np.where(
        (df['rsi_14'] > 70) & (df['adx'] > 25), 2,
        np.where(df['rsi_14'] > 60, 1,
        np.where((df['rsi_14'] < 30) & (df['adx'] > 25), -2,
        np.where(df['rsi_14'] < 40, -1, 0)))
    )

    df['regime_score'] = df['ma_trend_regime'] + df['vol_regime'] + df['momentum_regime']
    df['regime_change'] = df['ma_trend_regime'].diff().abs()

    regime_changes = df['ma_trend_regime'].diff().abs() > 0
    df['days_in_regime'] = (~regime_changes).astype(int).groupby(
        regime_changes.cumsum()
    ).cumsum()

    return df


def _compute_alpha_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha signals."""
    price_change = df['close'].pct_change()
    vol_change = df['volume'].pct_change()
    df['smart_money_flow'] = np.where(
        (price_change > 0) & (vol_change > 0.5), 2,
        np.where((price_change > 0) & (vol_change < -0.3), 1,
        np.where((price_change < 0) & (vol_change > 0.5), -2,
        np.where((price_change < 0) & (vol_change < -0.3), -1, 0)))
    )
    df['smart_money_flow_ma'] = df['smart_money_flow'].rolling(5).mean()

    df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    df['gap_filled'] = ((df['gap_pct'] > 0) & (df['low'] < df['close'].shift(1)) |
                       (df['gap_pct'] < 0) & (df['high'] > df['close'].shift(1))).astype(int)

    df['intraday_strength'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['intraday_strength_ma'] = df['intraday_strength'].rolling(5).mean()

    net_change = abs(df['close'] - df['close'].shift(10))
    total_change = df['close'].diff().abs().rolling(10).sum()
    df['efficiency_ratio'] = net_change / (total_change + 1e-10)

    df['rel_volume_breakout'] = (df['volume'] > df['volume_sma_20'] * 2).astype(int)
    df['mean_reversion_signal'] = -df['price_zscore'] * (1 - df['efficiency_ratio'])

    return df


def _add_market_context(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """Add market-wide context features."""
    import yfinance as yf

    try:
        start_date = df['timestamp'].min() if 'timestamp' in df.columns else df.index.min()
        end_date = df['timestamp'].max() if 'timestamp' in df.columns else df.index.max()

        nifty = yf.download('^NSEI', start=start_date, end=end_date, progress=False)
        if len(nifty) > 0:
            nifty = nifty.reset_index()
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = [col[0] if isinstance(col, tuple) else col for col in nifty.columns]
            nifty.columns = [c.lower() for c in nifty.columns]

            nifty['nifty_return'] = nifty['close'].pct_change()
            nifty['nifty_volatility'] = nifty['nifty_return'].rolling(20).std() * np.sqrt(252)
            nifty['nifty_trend'] = (nifty['close'] > nifty['close'].rolling(20).mean()).astype(int)

            merge_col = 'timestamp' if 'timestamp' in df.columns else 'date'
            nifty_merge_col = 'date' if 'date' in nifty.columns else nifty.columns[0]

            if merge_col in df.columns:
                df[merge_col] = pd.to_datetime(df[merge_col]).dt.date
                nifty[nifty_merge_col] = pd.to_datetime(nifty[nifty_merge_col]).dt.date

                df = df.merge(
                    nifty[[nifty_merge_col, 'nifty_return', 'nifty_volatility', 'nifty_trend']],
                    left_on=merge_col, right_on=nifty_merge_col, how='left'
                )

                if 'return_1d' in df.columns:
                    df['stock_beta'] = df['return_1d'].rolling(60).cov(df['nifty_return']) / (
                        df['nifty_return'].rolling(60).var() + 1e-10
                    )
                    df['stock_alpha'] = df['return_1d'] - df['stock_beta'] * df['nifty_return']

    except Exception as e:
        logger.warning(f"Could not add market context: {e}")
        df['nifty_return'] = 0
        df['nifty_volatility'] = 0
        df['nifty_trend'] = 0
        df['stock_beta'] = 1
        df['stock_alpha'] = 0

    return df


def _add_sentiment_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add news sentiment features using fast RSS-based engine."""
    try:
        from engine.sentiment import FastSentimentEngine
        engine = FastSentimentEngine()
        df = engine.add_sentiment_features(df, symbol)
        logger.info(f"Added fast sentiment features for {symbol}")
    except Exception as e:
        logger.warning(f"Could not add sentiment for {symbol}: {e}")
        df['news_sentiment'] = 0
        df['news_sentiment_bullish'] = 0
        df['news_sentiment_bearish'] = 0
        df['news_sentiment_neutral'] = 1
        df['sentiment_ma_7d'] = 0
        df['sentiment_ma_30d'] = 0
        df['sentiment_trend'] = 0
        df['news_volume'] = 0

    return df
