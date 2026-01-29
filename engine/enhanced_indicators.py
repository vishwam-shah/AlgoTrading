"""
================================================================================
ENHANCED TECHNICAL INDICATORS (60+ NEW INDICATORS)
================================================================================
Advanced indicators to improve directional accuracy from 53% to target 65-70%.

Categories:
1. Advanced Momentum (12): MFI variations, TSI, Ultimate Oscillator, etc.
2. Advanced Volume (10): Chaikin Money Flow, Ease of Movement, Volume Oscillator, etc.
3. Advanced Volatility (8): Ulcer Index, Chandelier Exit, Keltner Width, etc.
4. Advanced Trend (12): Ichimoku, SuperTrend, Parabolic SAR, Donchian, etc.
5. Market Strength (6): Aroon, Know Sure Thing, Trix, etc.
6. Candlestick Patterns (12): Engulfing, Hammer, Doji Star, etc.
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Tuple


def add_enhanced_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 12 advanced momentum indicators.
    
    Indicators:
    - Money Flow Index (MFI) - multi-period
    - True Strength Index (TSI)
    - Ultimate Oscillator
    - Commodity Channel Index (CCI) - multi-period
    - Percentage Price Oscillator (PPO)
    - Detrended Price Oscillator (DPO)
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Money Flow Index (MFI) - already in base, add variations
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    for period in [10, 14, 20]:
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        mfi_ratio = positive_flow / (negative_flow + 1e-10)
        df[f'mfi_{period}'] = 100 - (100 / (1 + mfi_ratio))
    
    # True Strength Index (TSI)
    price_change = close.diff()
    double_smoothed_pc = price_change.ewm(span=25).mean().ewm(span=13).mean()
    double_smoothed_abs_pc = abs(price_change).ewm(span=25).mean().ewm(span=13).mean()
    df['tsi'] = 100 * (double_smoothed_pc / (double_smoothed_abs_pc + 1e-10))
    df['tsi_signal'] = df['tsi'].ewm(span=7).mean()
    df['tsi_histogram'] = df['tsi'] - df['tsi_signal']
    
    # Ultimate Oscillator (combines 7, 14, 28 periods)
    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    
    avg7 = bp.rolling(7).sum() / (tr.rolling(7).sum() + 1e-10)
    avg14 = bp.rolling(14).sum() / (tr.rolling(14).sum() + 1e-10)
    avg28 = bp.rolling(28).sum() / (tr.rolling(28).sum() + 1e-10)
    df['ultimate_oscillator'] = 100 * ((4 * avg7 + 2 * avg14 + avg28) / 7)
    
    # Commodity Channel Index (CCI) - multi-period
    for period in [14, 20, 30]:
        tp = typical_price
        sma_tp = tp.rolling(period).mean()
        mean_dev = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mean_dev + 1e-10)
    
    # Percentage Price Oscillator (PPO)
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    df['ppo'] = ((ema_12 - ema_26) / (ema_26 + 1e-10)) * 100
    df['ppo_signal'] = df['ppo'].ewm(span=9).mean()
    df['ppo_histogram'] = df['ppo'] - df['ppo_signal']
    
    # Detrended Price Oscillator (DPO) - removes trend, shows cycles
    period = 20
    df['dpo'] = close.shift(int(period/2 + 1)) - close.rolling(period).mean()
    
    return df


def add_enhanced_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 10 advanced volume indicators.
    
    Indicators:
    - Chaikin Money Flow (CMF) - multi-period
    - Ease of Movement (EMV)
    - Volume Price Trend (VPT) variations
    - Force Index
    - Accumulation/Distribution Index variations
    - Volume Oscillator
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Chaikin Money Flow (CMF) - already have CMF_20, add variations
    clv = ((close - low) - (high - close)) / (high - low + 1e-10)
    for period in [10, 20, 30]:
        df[f'cmf_{period}'] = (clv * volume).rolling(period).sum() / (volume.rolling(period).sum() + 1e-10)
    
    # Ease of Movement (EMV)
    distance_moved = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    box_ratio = (volume / 1000000) / (high - low + 1e-10)
    emv = distance_moved / (box_ratio + 1e-10)
    df['emv_14'] = emv.rolling(14).mean()
    df['emv_trend'] = df['emv_14'].diff(5)
    
    # Force Index - combines price change and volume
    force_index = close.diff() * volume
    df['force_index'] = force_index
    df['force_index_13'] = force_index.ewm(span=13).mean()
    df['force_index_signal'] = (df['force_index_13'] > 0).astype(int)
    
    # Volume Oscillator
    volume_short = volume.rolling(5).mean()
    volume_long = volume.rolling(10).mean()
    df['volume_oscillator'] = ((volume_short - volume_long) / (volume_long + 1e-10)) * 100
    
    # Klinger Oscillator (advanced volume indicator)
    typical_price = (high + low + close) / 3
    dm = high - low
    trend = (typical_price > typical_price.shift(1)).astype(int) * 2 - 1  # 1 or -1
    cm = trend * dm * volume
    df['klinger_vf'] = cm.ewm(span=34).mean() - cm.ewm(span=55).mean()
    df['klinger_signal'] = df['klinger_vf'].ewm(span=13).mean()
    
    # Negative Volume Index (NVI) - tracks what smart money is doing
    nvi = pd.Series(index=df.index, dtype=float)
    nvi.iloc[0] = 1000
    for i in range(1, len(df)):
        if volume.iloc[i] < volume.iloc[i-1]:
            nvi.iloc[i] = nvi.iloc[i-1] + ((close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]) * nvi.iloc[i-1]
        else:
            nvi.iloc[i] = nvi.iloc[i-1]
    df['nvi'] = nvi
    df['nvi_sma_255'] = nvi.rolling(255).mean()
    
    return df


def add_enhanced_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 8 advanced volatility indicators.
    
    Indicators:
    - Ulcer Index
    - Chandelier Exit
    - Keltner Channel Width
    - Mass Index
    - Historical Volatility (multiple periods)
    - Volatility Ratio
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Ulcer Index - measures downside volatility
    for period in [14, 28]:
        highest_close = close.rolling(period).max()
        pct_drawdown = ((close - highest_close) / (highest_close + 1e-10)) * 100
        df[f'ulcer_index_{period}'] = np.sqrt((pct_drawdown ** 2).rolling(period).mean())
    
    # Chandelier Exit (volatility-based trailing stop)
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    atr_22 = tr.rolling(22).mean()
    
    df['chandelier_exit_long'] = high.rolling(22).max() - (atr_22 * 3)
    df['chandelier_exit_short'] = low.rolling(22).min() + (atr_22 * 3)
    df['chandelier_position'] = np.where(close > df['chandelier_exit_long'], 1,
                                         np.where(close < df['chandelier_exit_short'], -1, 0))
    
    # Keltner Channel Width
    ema_20 = close.ewm(span=20).mean()
    atr_10 = tr.rolling(10).mean()
    keltner_upper = ema_20 + (2 * atr_10)
    keltner_lower = ema_20 - (2 * atr_10)
    df['keltner_width'] = (keltner_upper - keltner_lower) / ema_20
    df['keltner_pct'] = (close - keltner_lower) / (keltner_upper - keltner_lower + 1e-10)
    
    # Mass Index - detects trend reversals based on range expansion
    hl_range = high - low
    ema_9 = hl_range.ewm(span=9).mean()
    ema_9_9 = ema_9.ewm(span=9).mean()
    ema_ratio = ema_9 / (ema_9_9 + 1e-10)
    df['mass_index'] = ema_ratio.rolling(25).sum()
    
    # Historical Volatility - multiple timeframes
    returns = close.pct_change()
    for period in [10, 30, 60, 90]:
        df[f'hist_vol_{period}d'] = returns.rolling(period).std() * np.sqrt(252) * 100
    
    # Volatility Ratio
    df['volatility_ratio_short_long'] = df['hist_vol_10d'] / (df['hist_vol_60d'] + 1e-10)
    
    return df


def add_enhanced_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 12 advanced trend indicators.
    
    Indicators:
    - Ichimoku Cloud (5 components)
    - SuperTrend
    - Parabolic SAR
    - Donchian Channels
    - Aroon Indicator
    - Linear Regression Slope
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Ichimoku Cloud
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = high.rolling(9).max()
    period9_low = low.rolling(9).min()
    df['ichimoku_tenkan'] = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = high.rolling(26).max()
    period26_low = low.rolling(26).min()
    df['ichimoku_kijun'] = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted +26
    df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, shifted +26
    period52_high = high.rolling(52).max()
    period52_low = low.rolling(52).min()
    df['ichimoku_senkou_b'] = ((period52_high + period52_low) / 2).shift(26)
    
    # Chikou Span (Lagging Span): Close shifted -26
    df['ichimoku_chikou'] = close.shift(-26)
    
    # Ichimoku signals
    df['ichimoku_cloud_green'] = (df['ichimoku_senkou_a'] > df['ichimoku_senkou_b']).astype(int)
    df['ichimoku_price_above_cloud'] = (close > df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].max(axis=1)).astype(int)
    df['ichimoku_tk_cross'] = (df['ichimoku_tenkan'] > df['ichimoku_kijun']).astype(int)
    
    # SuperTrend Indicator
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(10).mean()
    
    hl_avg = (high + low) / 2
    multiplier = 3
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = 1
        else:
            # Update bands
            if close.iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
            
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
    
    df['supertrend'] = supertrend
    df['supertrend_direction'] = direction
    df['supertrend_distance'] = (close - supertrend) / close * 100
    
    # Parabolic SAR
    # Simplified version
    df['sar'] = close.rolling(5).mean()  # Placeholder - full SAR is complex
    df['sar_signal'] = (close > df['sar']).astype(int)
    
    # Donchian Channels
    for period in [20, 50]:
        df[f'donchian_high_{period}'] = high.rolling(period).max()
        df[f'donchian_low_{period}'] = low.rolling(period).min()
        df[f'donchian_mid_{period}'] = (df[f'donchian_high_{period}'] + df[f'donchian_low_{period}']) / 2
        df[f'donchian_width_{period}'] = (df[f'donchian_high_{period}'] - df[f'donchian_low_{period}']) / df[f'donchian_mid_{period}']
    
    # Linear Regression Slope (trend strength)
    for period in [10, 20, 50]:
        def lin_reg_slope(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        df[f'lr_slope_{period}'] = close.rolling(period).apply(lin_reg_slope, raw=True)
        df[f'lr_r2_{period}'] = close.rolling(period).apply(
            lambda y: np.corrcoef(np.arange(len(y)), y)[0,1]**2 if len(y) > 1 else 0, raw=True
        )
    
    return df


def add_market_strength_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 6 market strength indicators.
    
    Indicators:
    - Aroon Up/Down
    - Know Sure Thing (KST)
    - Trix
    - Vortex Indicator
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Aroon Indicator
    period = 25
    aroon_up = high.rolling(period + 1).apply(lambda x: (period - (len(x) - 1 - x.argmax())) / period * 100, raw=True)
    aroon_down = low.rolling(period + 1).apply(lambda x: (period - (len(x) - 1 - x.argmin())) / period * 100, raw=True)
    df['aroon_up'] = aroon_up
    df['aroon_down'] = aroon_down
    df['aroon_oscillator'] = aroon_up - aroon_down
    
    # Know Sure Thing (KST) - momentum oscillator
    roc1 = close.pct_change(10) * 100
    roc2 = close.pct_change(15) * 100
    roc3 = close.pct_change(20) * 100
    roc4 = close.pct_change(30) * 100
    
    kst = (roc1.rolling(10).mean() * 1 +
           roc2.rolling(10).mean() * 2 +
           roc3.rolling(10).mean() * 3 +
           roc4.rolling(15).mean() * 4)
    df['kst'] = kst
    df['kst_signal'] = kst.rolling(9).mean()
    df['kst_histogram'] = kst - df['kst_signal']
    
    # Trix - triple exponential average
    ema1 = close.ewm(span=15).mean()
    ema2 = ema1.ewm(span=15).mean()
    ema3 = ema2.ewm(span=15).mean()
    df['trix'] = ema3.pct_change() * 100
    df['trix_signal'] = df['trix'].ewm(span=9).mean()
    
    # Vortex Indicator
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    
    vm_plus = abs(high - low.shift(1))
    vm_minus = abs(low - high.shift(1))
    
    period = 14
    vi_plus = vm_plus.rolling(period).sum() / (tr.rolling(period).sum() + 1e-10)
    vi_minus = vm_minus.rolling(period).sum() / (tr.rolling(period).sum() + 1e-10)
    
    df['vortex_plus'] = vi_plus
    df['vortex_minus'] = vi_minus
    df['vortex_diff'] = vi_plus - vi_minus
    
    return df


def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 12 candlestick pattern detections.
    
    Patterns:
    - Bullish/Bearish Engulfing
    - Hammer / Inverted Hammer
    - Shooting Star / Hanging Man
    - Morning Star / Evening Star
    - Doji variations
    """
    open_price = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    
    body = abs(close - open_price)
    candle_range = high - low
    upper_shadow = high - pd.concat([open_price, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_price, close], axis=1).min(axis=1) - low
    
    # Bullish Engulfing
    prev_body = abs(close.shift(1) - open_price.shift(1))
    bullish_engulfing = (
        (close.shift(1) < open_price.shift(1)) &  # Previous candle bearish
        (close > open_price) &  # Current candle bullish
        (open_price < close.shift(1)) &  # Opens below previous close
        (close > open_price.shift(1))  # Closes above previous open
    ).astype(int)
    df['pattern_bullish_engulfing'] = bullish_engulfing
    
    # Bearish Engulfing
    bearish_engulfing = (
        (close.shift(1) > open_price.shift(1)) &  # Previous candle bullish
        (close < open_price) &  # Current candle bearish
        (open_price > close.shift(1)) &  # Opens above previous close
        (close < open_price.shift(1))  # Closes below previous open
    ).astype(int)
    df['pattern_bearish_engulfing'] = bearish_engulfing
    
    # Hammer (bullish reversal at bottom)
    hammer = (
        (lower_shadow > body * 2) &  # Long lower shadow
        (upper_shadow < body * 0.3) &  # Small upper shadow
        (close > open_price)  # Bullish body
    ).astype(int)
    df['pattern_hammer'] = hammer
    
    # Shooting Star (bearish reversal at top)
    shooting_star = (
        (upper_shadow > body * 2) &  # Long upper shadow
        (lower_shadow < body * 0.3) &  # Small lower shadow
        (close < open_price)  # Bearish body
    ).astype(int)
    df['pattern_shooting_star'] = shooting_star
    
    # Doji (indecision)
    doji = (body < candle_range * 0.1).astype(int)
    df['pattern_doji'] = doji
    
    # Dragonfly Doji (bullish)
    dragonfly_doji = (
        doji &
        (lower_shadow > candle_range * 0.7) &
        (upper_shadow < candle_range * 0.1)
    ).astype(int)
    df['pattern_dragonfly_doji'] = dragonfly_doji
    
    # Gravestone Doji (bearish)
    gravestone_doji = (
        doji &
        (upper_shadow > candle_range * 0.7) &
        (lower_shadow < candle_range * 0.1)
    ).astype(int)
    df['pattern_gravestone_doji'] = gravestone_doji
    
    # Spinning Top (indecision with small body)
    spinning_top = (
        (body < candle_range * 0.3) &
        (upper_shadow > body * 0.5) &
        (lower_shadow > body * 0.5)
    ).astype(int)
    df['pattern_spinning_top'] = spinning_top
    
    # Marubozu (strong trend - no shadows)
    bullish_marubozu = (
        (close > open_price) &
        (upper_shadow < body * 0.01) &
        (lower_shadow < body * 0.01)
    ).astype(int)
    bearish_marubozu = (
        (close < open_price) &
        (upper_shadow < body * 0.01) &
        (lower_shadow < body * 0.01)
    ).astype(int)
    df['pattern_bullish_marubozu'] = bullish_marubozu
    df['pattern_bearish_marubozu'] = bearish_marubozu
    
    # Three White Soldiers (strong bullish)
    three_white_soldiers = (
        (close > open_price) &
        (close.shift(1) > open_price.shift(1)) &
        (close.shift(2) > open_price.shift(2)) &
        (close > close.shift(1)) &
        (close.shift(1) > close.shift(2)) &
        (open_price > open_price.shift(1)) &
        (open_price.shift(1) > open_price.shift(2))
    ).astype(int)
    df['pattern_three_white_soldiers'] = three_white_soldiers
    
    # Three Black Crows (strong bearish)
    three_black_crows = (
        (close < open_price) &
        (close.shift(1) < open_price.shift(1)) &
        (close.shift(2) < open_price.shift(2)) &
        (close < close.shift(1)) &
        (close.shift(1) < close.shift(2)) &
        (open_price < open_price.shift(1)) &
        (open_price.shift(1) < open_price.shift(2))
    ).astype(int)
    df['pattern_three_black_crows'] = three_black_crows
    
    return df


def add_all_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all 60+ enhanced indicators to the dataframe.
    
    Returns:
        DataFrame with all original and enhanced indicators.
    """
    df = df.copy()
    
    # Add each category
    df = add_enhanced_momentum_indicators(df)
    df = add_enhanced_volume_indicators(df)
    df = add_enhanced_volatility_indicators(df)
    df = add_enhanced_trend_indicators(df)
    df = add_market_strength_indicators(df)
    df = add_candlestick_patterns(df)
    
    # Clean infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    
    return df
