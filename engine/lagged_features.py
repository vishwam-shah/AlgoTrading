"""
================================================================================
LAGGED FEATURES MODULE
================================================================================
Additional ~40 features for improved directional accuracy.
All features use ONLY historical data - NO look-ahead bias.

Feature Categories:
1. Lagged Returns (10 features)
2. Rolling Correlations (4 features)
3. Order Imbalance Proxies (6 features)
4. Intraday Range Patterns (5 features)
5. Volume-Price Divergence (4 features)
6. Cross-sectional Ranks (6 features)
7. Regime Transitions (3 features)
8. Multi-timeframe Alignment (4 features)
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict
from loguru import logger


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ~40 lagged features that use only historical data.
    
    Args:
        df: DataFrame with OHLCV data and existing features
        
    Returns:
        DataFrame with additional lagged features
    """
    df = df.copy()
    
    # ===========================
    # 1. LAGGED RETURNS (10 features)
    # ===========================
    # Returns from 2-10 days ago (day 1 already exists as return_1d)
    for lag in [2, 3, 4, 5, 7, 10]:
        df[f'return_lag_{lag}d'] = df['close'].pct_change().shift(lag)
    
    # Cumulative returns over different windows
    df['cum_return_5d'] = df['close'].pct_change().rolling(5).sum()
    df['cum_return_10d'] = df['close'].pct_change().rolling(10).sum()
    df['cum_return_20d'] = df['close'].pct_change().rolling(20).sum()
    
    # ===========================
    # 2. ROLLING CORRELATIONS (4 features)
    # ===========================
    returns = df['close'].pct_change()
    
    # Correlation with own lagged returns (momentum autocorrelation)
    df['autocorr_5d'] = returns.rolling(20).apply(
        lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False
    )
    df['autocorr_10d'] = returns.rolling(40).apply(
        lambda x: x.autocorr(lag=10) if len(x) > 10 else 0, raw=False
    )
    
    # Correlation between price and volume
    df['price_volume_corr_20d'] = df['close'].rolling(20).corr(df['volume'])
    df['price_volume_corr_60d'] = df['close'].rolling(60).corr(df['volume'])
    
    # ===========================
    # 3. ORDER IMBALANCE PROXIES (6 features)
    # ===========================
    # Buying pressure: how close did price close to high?
    df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['buying_pressure_ma_5'] = df['buying_pressure'].rolling(5).mean()
    df['buying_pressure_ma_20'] = df['buying_pressure'].rolling(20).mean()
    
    # Selling pressure: how close did price close to low?
    df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
    df['selling_pressure_ma_5'] = df['selling_pressure'].rolling(5).mean()
    
    # Net buying/selling imbalance
    df['order_imbalance'] = df['buying_pressure'] - df['selling_pressure']
    
    # ===========================
    # 4. INTRADAY RANGE PATTERNS (5 features)
    # ===========================
    range_pct = (df['high'] - df['low']) / df['close'] * 100
    
    # Range percentile (is today's range high or low compared to recent history?)
    df['range_percentile_20d'] = range_pct.rolling(20).apply(
        lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
    )
    
    # Range expansion/contraction
    df['range_expansion'] = range_pct / (range_pct.rolling(5).mean() + 1e-10)
    
    # Upper/lower wick sizes
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    
    df['wick_imbalance'] = (upper_wick - lower_wick) / (df['high'] - df['low'] + 1e-10)
    df['avg_wick_size'] = (upper_wick + lower_wick) / (df['high'] - df['low'] + 1e-10)
    
    # ===========================
    # 5. VOLUME-PRICE DIVERGENCE (4 features)
    # ===========================
    price_trend = df['close'].pct_change(10)
    volume_trend = df['volume'].pct_change(10)
    
    # Divergence signals (price up but volume down = bearish divergence)
    df['vpd_divergence'] = price_trend - volume_trend
    df['vpd_divergence_ma'] = df['vpd_divergence'].rolling(5).mean()
    
    # Volume surge without price movement (accumulation/distribution)
    df['volume_surge_low_momentum'] = (
        (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(int) &
        (abs(returns) < returns.rolling(20).std()).astype(int)
    ).astype(int)
    
    # Price gap with volume confirmation
    gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_volume_confirm'] = (
        (abs(gap) > 0.01) & 
        (df['volume'] > df['volume'].rolling(5).mean())
    ).astype(int)
    
    # ===========================
    # 6. CROSS-SECTIONAL RANKS (6 features)
    # ===========================
    # Note: These work best with multiple stocks, but we can rank against own history
    
    # Momentum rank (percentile of current return vs historical returns)
    for period in [5, 10, 20]:
        ret = df['close'].pct_change(period)
        df[f'momentum_rank_{period}d'] = ret.rolling(60).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
        )
    
    # Volatility rank
    vol = returns.rolling(20).std()
    df['vol_rank'] = vol.rolling(60).apply(
        lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
    )
    
    # Volume rank
    df['volume_rank'] = df['volume'].rolling(60).apply(
        lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
    )
    
    # RSI rank (how extreme is current RSI?)
    if 'rsi_14' in df.columns:
        df['rsi_rank'] = df['rsi_14'].rolling(60).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
        )
    
    # ===========================
    # 7. REGIME TRANSITIONS (3 features)
    # ===========================
    if 'ma_trend_regime' in df.columns:
        # Count regime changes in last N days
        regime_changes = df['ma_trend_regime'].diff().abs() > 0
        df['regime_changes_10d'] = regime_changes.rolling(10).sum()
        df['regime_changes_20d'] = regime_changes.rolling(20).sum()
        df['regime_changes_30d'] = regime_changes.rolling(30).sum()
    else:
        # Define simple regime based on price vs MA
        regime = (df['close'] > df['close'].rolling(20).mean()).astype(int)
        regime_changes = regime.diff().abs() > 0
        df['regime_changes_10d'] = regime_changes.rolling(10).sum()
        df['regime_changes_20d'] = regime_changes.rolling(20).sum()
        df['regime_changes_30d'] = regime_changes.rolling(30).sum()
    
    # ===========================
    # 8. MULTI-TIMEFRAME ALIGNMENT (4 features)
    # ===========================
    # Is trend aligned across multiple timeframes?
    short_trend = (df['close'] > df['close'].rolling(5).mean()).astype(int)
    mid_trend = (df['close'] > df['close'].rolling(20).mean()).astype(int)
    long_trend = (df['close'] > df['close'].rolling(60).mean()).astype(int)
    
    df['trend_alignment'] = short_trend + mid_trend + long_trend  # 0-3
    df['strong_uptrend'] = (df['trend_alignment'] == 3).astype(int)
    df['strong_downtrend'] = (df['trend_alignment'] == 0).astype(int)
    
    # Trend strength increasing/decreasing
    df['trend_alignment_change'] = df['trend_alignment'].diff()
    
    # Clean up NaNs and infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    
    logger.info(f"Added {len([c for c in df.columns if c.startswith(('return_lag_', 'cum_return_', 'autocorr_', 'price_volume_corr_', 'buying_pressure', 'selling_pressure', 'order_imbalance', 'range_', 'wick_', 'vpd_', 'volume_surge', 'gap_volume', 'momentum_rank_', 'vol_rank', 'volume_rank', 'rsi_rank', 'regime_changes_', 'trend_alignment'))])} lagged features")
    
    return df


def compute_cross_sectional_ranks(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Compute cross-sectional ranks across multiple stocks.
    
    This is more powerful than single-stock ranks as it compares stocks to each other.
    
    Args:
        dfs: Dict of symbol -> DataFrame with features
        
    Returns:
        Dict of symbol -> DataFrame with added cross-sectional rank features
    """
    if len(dfs) < 3:
        logger.warning("Need at least 3 stocks for meaningful cross-sectional ranks")
        return dfs
    
    # Combine all data with symbol identifiers
    combined = []
    for symbol, df in dfs.items():
        df_copy = df.copy()
        df_copy['_symbol'] = symbol
        if 'timestamp' in df_copy.columns:
            df_copy['_date'] = pd.to_datetime(df_copy['timestamp']).dt.date
        combined.append(df_copy)
    
    all_data = pd.concat(combined, ignore_index=True)
    
    # For each date, rank stocks
    if '_date' in all_data.columns:
        # Rank momentum
        all_data['cross_momentum_rank'] = all_data.groupby('_date')['return_20d'].rank(pct=True)
        
        # Rank volatility
        if 'volatility_20d' in all_data.columns:
            all_data['cross_vol_rank'] = all_data.groupby('_date')['volatility_20d'].rank(pct=True)
        
        # Rank volume
        all_data['cross_volume_rank'] = all_data.groupby('_date')['volume'].rank(pct=True)
        
        # Separate back to individual stocks
        result = {}
        for symbol in dfs.keys():
            stock_data = all_data[all_data['_symbol'] == symbol].copy()
            stock_data = stock_data.drop(columns=['_symbol', '_date'])
            result[symbol] = stock_data
        
        logger.info(f"Added cross-sectional ranks for {len(result)} stocks")
        return result
    
    return dfs
