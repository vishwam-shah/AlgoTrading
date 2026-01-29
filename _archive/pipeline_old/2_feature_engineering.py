"""
================================================================================
STEP 2: FEATURE ENGINEERING
================================================================================
Comprehensive feature engineering for stock analysis.

Features computed:
1. Technical Indicators (RSI, MACD, Bollinger, ATR, etc.)
2. Price Patterns (candlestick patterns, support/resistance)
3. Volume Analysis (OBV, VWAP, volume trends)
4. Momentum Indicators (momentum, rate of change)
5. Volatility Measures (historical vol, Parkinson's vol)
6. Market Relative Features (vs NIFTY, sector indices)
7. Time-based Features (day of week, month, seasonality)

Usage:
    from pipeline.step_2_feature_engineering import FeatureEngineer
    engineer = FeatureEngineer()
    features = engineer.compute_all_features(price_data, market_data)

Or run directly:
    python pipeline/step_2_feature_engineering.py
================================================================================
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from loguru import logger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FeatureEngineer:
    """
    Comprehensive feature engineering for stock data.
    
    Computes 50+ technical and fundamental features organized into categories:
    - Trend indicators
    - Momentum indicators
    - Volatility indicators
    - Volume indicators
    - Price patterns
    - Market relative features
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.feature_count = 0
        self.feature_names = []
        self.computation_stats = {}
        
        logger.info("FeatureEngineer initialized")
    
    def compute_all_features(
        self,
        price_data: Dict[str, pd.DataFrame],
        market_data: Dict[str, pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute all features for all symbols.
        
        Args:
            price_data: Dict of symbol -> OHLCV DataFrame
            market_data: Dict of market index -> DataFrame
            
        Returns:
            Dict of symbol -> features DataFrame
        """
        logger.info("=" * 60)
        logger.info("STEP 2: FEATURE ENGINEERING")
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
        
        # Update feature count
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
        """Compute all features for a single symbol."""
        
        features = pd.DataFrame(index=df.index)
        
        # Use pandas Series for all calculations (not numpy arrays)
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        volume = df['volume']
        
        # ======================================================================
        # 1. TREND INDICATORS
        # ======================================================================
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'sma_{period}_slope'] = features[f'sma_{period}'].diff(5) / 5
        
        # Exponential Moving Averages
        for period in [12, 26, 50]:
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Price vs MAs
        features['price_vs_sma20'] = df['close'] / features['sma_20'] - 1
        features['price_vs_sma50'] = df['close'] / features['sma_50'] - 1
        features['price_vs_sma200'] = df['close'] / features['sma_200'] - 1
        
        # MA Crossovers
        features['sma_20_50_cross'] = (features['sma_20'] > features['sma_50']).astype(int)
        features['sma_50_200_cross'] = (features['sma_50'] > features['sma_200']).astype(int)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # ======================================================================
        # 2. MOMENTUM INDICATORS
        # ======================================================================
        
        # Returns
        for period in [1, 5, 10, 20, 63, 126, 252]:
            features[f'return_{period}d'] = df['close'].pct_change(period)
        
        # RSI
        for period in [14, 28]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        features['stoch_k'] = (df['close'] - low_14) / (high_14 - low_14 + 1e-10) * 100
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        # Williams %R
        features['williams_r'] = (high_14 - df['close']) / (high_14 - low_14 + 1e-10) * -100
        
        # Rate of Change
        for period in [10, 20]:
            features[f'roc_{period}'] = df['close'].pct_change(period) * 100
        
        # Momentum
        features['momentum_10'] = df['close'] - df['close'].shift(10)
        features['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # ======================================================================
        # 3. VOLATILITY INDICATORS
        # ======================================================================
        
        # Historical Volatility
        for period in [10, 20, 60]:
            features[f'volatility_{period}d'] = df['close'].pct_change().rolling(period).std() * np.sqrt(252)
        
        # Average True Range (ATR)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()
        features['atr_pct'] = features['atr_14'] / df['close'] * 100
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_upper'] = sma_20 + 2 * std_20
        features['bb_lower'] = sma_20 - 2 * std_20
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
        
        # Keltner Channels
        features['keltner_upper'] = features['ema_26'] + 2 * features['atr_14']
        features['keltner_lower'] = features['ema_26'] - 2 * features['atr_14']
        
        # Parkinson's Volatility (more efficient estimator)
        features['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(df['high'] / df['low']) ** 2).rolling(20).mean()
        ) * np.sqrt(252)
        
        # ======================================================================
        # 4. VOLUME INDICATORS
        # ======================================================================
        
        # Volume Moving Averages
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / (features['volume_sma_20'] + 1)
        
        # On-Balance Volume (OBV) - using pandas operations
        obv_direction = np.sign(close.diff())
        features['obv'] = (obv_direction * volume).fillna(0).cumsum()
        features['obv_sma'] = features['obv'].rolling(20).mean()
        
        # Money Flow Index (MFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_mf = typical_price * df['volume']
        
        mf_positive = raw_mf.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        mf_negative = raw_mf.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        features['mfi'] = 100 - (100 / (1 + mf_positive / (mf_negative + 1e-10)))
        
        # Volume-Price Trend (VPT)
        features['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
        
        # VWAP (intraday proxy using daily data)
        features['vwap'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        features['price_vs_vwap'] = df['close'] / features['vwap'] - 1
        
        # ======================================================================
        # 5. PRICE PATTERNS
        # ======================================================================
        
        # Candlestick patterns (simplified)
        body = close - open_price
        body_pct = body / open_price
        range_hl = high - low
        
        features['body_pct'] = body_pct
        features['upper_shadow'] = (high - pd.concat([close, open_price], axis=1).max(axis=1)) / (range_hl + 1e-10)
        features['lower_shadow'] = (pd.concat([close, open_price], axis=1).min(axis=1) - low) / (range_hl + 1e-10)
        
        # Doji (small body)
        features['is_doji'] = (np.abs(body_pct) < 0.001).astype(int)
        
        # Gap analysis (using pandas shift to maintain index alignment)
        features['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        features['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        features['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Higher highs / Lower lows
        features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # N-day high/low
        for period in [20, 52]:
            features[f'is_{period}d_high'] = (df['close'] >= df['close'].rolling(period).max()).astype(int)
            features[f'is_{period}d_low'] = (df['close'] <= df['close'].rolling(period).min()).astype(int)
        
        # ======================================================================
        # 6. MARKET RELATIVE FEATURES
        # ======================================================================
        
        if market_data and 'NIFTY50' in market_data:
            nifty = market_data['NIFTY50']['close'].reindex(df.index, method='ffill')
            
            # Beta (60-day rolling)
            stock_ret = df['close'].pct_change()
            nifty_ret = nifty.pct_change()
            
            cov = stock_ret.rolling(60).cov(nifty_ret)
            var = nifty_ret.rolling(60).var()
            features['beta_60d'] = cov / (var + 1e-10)
            
            # Relative Strength vs NIFTY
            for period in [20, 60]:
                stock_perf = df['close'].pct_change(period)
                nifty_perf = nifty.pct_change(period)
                features[f'rs_vs_nifty_{period}d'] = stock_perf - nifty_perf
            
            # Correlation with NIFTY
            features['corr_nifty_60d'] = stock_ret.rolling(60).corr(nifty_ret)
        
        # ======================================================================
        # 7. TIME-BASED FEATURES
        # ======================================================================
        
        features['day_of_week'] = df.index.dayofweek
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        features['is_month_end'] = df.index.is_month_end.astype(int)
        features['is_month_start'] = df.index.is_month_start.astype(int)
        
        # ======================================================================
        # 8. SENTIMENT PROXY (from price patterns)
        # ======================================================================
        
        # Accumulation/Distribution
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        features['ad_line'] = (mfm * volume).cumsum()
        
        # Chaikin Money Flow
        features['cmf_20'] = (mfm * volume).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Fear/Greed proxy (combining multiple indicators)
        features['fear_greed_proxy'] = (
            (features['rsi_14'] / 100) * 0.25 +
            features['bb_position'].clip(0, 1) * 0.25 +
            (features['stoch_k'] / 100) * 0.25 +
            ((features['mfi'] / 100) if 'mfi' in features else 0.5) * 0.25
        )
        
        # ======================================================================
        # CLEAN AND FINALIZE
        # ======================================================================
        
        # Forward fill then backfill any remaining NaN
        features = features.ffill().bfill()
        
        # Replace inf with NaN then fill
        features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        return features
    
    def get_feature_count(self) -> int:
        """Get total number of features computed."""
        return self.feature_count
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
    
    def get_summary(self) -> Dict:
        """Get feature engineering summary."""
        return {
            'total_features': self.feature_count,
            'feature_categories': {
                'trend': 15,
                'momentum': 12,
                'volatility': 10,
                'volume': 8,
                'price_patterns': 10,
                'market_relative': 5,
                'time_based': 5,
                'sentiment_proxy': 3
            },
            'symbols_processed': len(self.computation_stats),
            'computation_stats': self.computation_stats
        }


def test_feature_engineering():
    """Test feature engineering with sample data."""
    print("\n" + "=" * 80)
    print("TESTING STEP 2: FEATURE ENGINEERING")
    print("=" * 80)
    
    # First run data collection
    from step_1_data_collection import DataCollector
    
    test_symbols = ['HDFCBANK', 'ICICIBANK', 'SBIN', 'TCS', 'RELIANCE']
    
    collector = DataCollector()
    price_data, market_data = collector.collect_all(
        symbols=test_symbols,
        start_date='2022-01-01'
    )
    
    # Now test feature engineering
    engineer = FeatureEngineer()
    features = engineer.compute_all_features(price_data, market_data)
    
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING RESULTS")
    print("=" * 80)
    
    print(f"\n✓ Features computed: {engineer.get_feature_count()} features")
    print(f"\n✓ Symbols processed: {len(features)}")
    
    # Show sample features for first symbol
    if features:
        first_symbol = list(features.keys())[0]
        first_df = features[first_symbol]
        
        print(f"\n✓ Sample features for {first_symbol}:")
        print(f"  Rows: {len(first_df)}")
        print(f"  Columns: {len(first_df.columns)}")
        print(f"\n  Feature categories:")
        
        # Count features by category
        trend_features = [c for c in first_df.columns if 'sma' in c or 'ema' in c or 'macd' in c]
        momentum_features = [c for c in first_df.columns if 'rsi' in c or 'stoch' in c or 'momentum' in c or 'return' in c]
        volatility_features = [c for c in first_df.columns if 'vol' in c or 'atr' in c or 'bb_' in c]
        volume_features = [c for c in first_df.columns if 'volume' in c or 'obv' in c or 'mfi' in c]
        
        print(f"    Trend: {len(trend_features)}")
        print(f"    Momentum: {len(momentum_features)}")
        print(f"    Volatility: {len(volatility_features)}")
        print(f"    Volume: {len(volume_features)}")
        
        # Show latest values
        print(f"\n  Latest feature values ({first_df.index[-1].strftime('%Y-%m-%d')}):")
        for col in ['rsi_14', 'macd', 'bb_position', 'volatility_20d', 'fear_greed_proxy']:
            if col in first_df.columns:
                print(f"    {col}: {first_df[col].iloc[-1]:.4f}")
    
    # Validation tests
    print("\n" + "-" * 40)
    print("VALIDATION TESTS")
    print("-" * 40)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Feature count
    tests_total += 1
    if engineer.get_feature_count() >= 50:
        print(f"✓ Test 1: Sufficient features ({engineer.get_feature_count()} >= 50) - PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 1: Sufficient features ({engineer.get_feature_count()} < 50) - FAILED")
    
    # Test 2: No NaN in critical features
    tests_total += 1
    has_nan = False
    for symbol, df in features.items():
        # Skip first 252 rows (need lookback period)
        if df.iloc[252:].isnull().any().any():
            has_nan = True
            break
    if not has_nan:
        print("✓ Test 2: No NaN values (after warmup) - PASSED")
        tests_passed += 1
    else:
        print("✗ Test 2: No NaN values (after warmup) - FAILED")
    
    # Test 3: RSI in valid range
    tests_total += 1
    rsi_valid = True
    for symbol, df in features.items():
        rsi = df['rsi_14'].iloc[252:]
        if (rsi < 0).any() or (rsi > 100).any():
            rsi_valid = False
            break
    if rsi_valid:
        print("✓ Test 3: RSI in valid range [0, 100] - PASSED")
        tests_passed += 1
    else:
        print("✗ Test 3: RSI in valid range [0, 100] - FAILED")
    
    # Test 4: Beta reasonable
    tests_total += 1
    beta_valid = True
    for symbol, df in features.items():
        if 'beta_60d' in df.columns:
            beta = df['beta_60d'].iloc[252:].dropna()
            if len(beta) > 0 and (beta.abs() > 5).any():
                beta_valid = False
                break
    if beta_valid:
        print("✓ Test 4: Beta in reasonable range - PASSED")
        tests_passed += 1
    else:
        print("✗ Test 4: Beta in reasonable range - FAILED")
    
    print(f"\n{'=' * 40}")
    print(f"TESTS: {tests_passed}/{tests_total} passed")
    print("=" * 40)
    
    return features


if __name__ == "__main__":
    test_feature_engineering()
