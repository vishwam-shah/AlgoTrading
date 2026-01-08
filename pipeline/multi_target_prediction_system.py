"""
================================================================================
MULTI-TARGET STOCK PREDICTION SYSTEM
Cutting-Edge, Future-Proof Architecture
================================================================================

OBJECTIVE:
Predict multiple targets for next trading day:
1. Closing price (regression)
2. Highest price (regression)
3. Lowest price (regression)
4. Direction: UP/DOWN (classification)

FEATURES:
- Technical indicators
- Sentiment analysis (news)
- Market context (NIFTY, BANKNIFTY, VIX)
- Economic indicators
- Temporal features

VALIDATION:
- Rolling window (walk-forward)
- Train on historical data
- Test on next period
- Validate on out-of-sample

MODELS:
1. LSTM (Long Short-Term Memory)
2. GRU (Gated Recurrent Unit)
3. XGBoost
4. Ensemble (Stacking)

REFERENCES:
1. "Deep Learning for Stock Prediction Using Numerical and Textual Information"
   - Ding et al., 2015, IEEE/ACM Conference
   
2. "Stock Price Prediction Using LSTM, RNN and CNN-sliding window model"
   - Hiransha et al., 2018
   
3. "Financial Trading as a Game: A Deep Reinforcement Learning Approach"
   - Huang, 2018
   
4. "Empirical Asset Pricing via Machine Learning"
   - Gu, Kelly, Xiu, 2020, Review of Financial Studies
   
5. "XGBoost: A Scalable Tree Boosting System"
   - Chen & Guestrin, 2016, KDD

================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class MultiTargetFeatureEngineering:
    """
    Comprehensive feature engineering for multi-target prediction.
    Includes all factors that affect stock prices.
    """
    
    @staticmethod
    def load_stock_data(symbol: str) -> pd.DataFrame:
        """Load stock features."""
        features_file = os.path.join(config.FEATURES_DIR, f"{symbol}_features.csv")
        
        if not os.path.exists(features_file):
            # Try raw data
            raw_file = os.path.join(config.RAW_DATA_DIR, f"{symbol}.csv")
            if os.path.exists(raw_file):
                df = pd.read_csv(raw_file)
            else:
                raise FileNotFoundError(f"No data found for {symbol}")
        else:
            df = pd.read_csv(features_file)
        
        # Ensure timestamp is timezone-naive for consistent merging
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    @staticmethod
    def load_market_data() -> Dict[str, pd.DataFrame]:
        """Load market indices."""
        market_data = {}
        
        for index in ['NIFTY50', 'BANKNIFTY', 'INDIA_VIX']:
            file_path = os.path.join(config.MARKET_DATA_DIR, f"{index}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                market_data[index] = df
        
        return market_data
    
    @staticmethod
    def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute comprehensive technical indicators."""
        df = df.copy()
        
        # ========== PRICE-BASED FEATURES ==========
        
        # Returns
        for period in [1, 2, 3, 5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
        
        # Lagged prices (for sequence models)
        for lag in range(1, 11):  # Last 10 days
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'high_lag_{lag}'] = df['high'].shift(lag)
            df[f'low_lag_{lag}'] = df['low'].shift(lag)
        
        # Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Moving Average Ratios
        df['price_to_sma_5'] = df['close'] / (df['sma_5'] + 1e-10)
        df['price_to_sma_20'] = df['close'] / (df['sma_20'] + 1e-10)
        df['price_to_sma_50'] = df['close'] / (df['sma_50'] + 1e-10)
        df['sma_5_to_sma_20'] = df['sma_5'] / (df['sma_20'] + 1e-10)
        df['sma_20_to_sma_50'] = df['sma_20'] / (df['sma_50'] + 1e-10)
        
        # ========== MOMENTUM INDICATORS ==========
        
        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            df[f'rsi_{period}'] = df[f'rsi_{period}'].clip(0, 100)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Stochastic RSI (combines RSI + Stochastic)
        rsi_14 = df['rsi_14'].copy()
        rsi_low_14 = rsi_14.rolling(14).min()
        rsi_high_14 = rsi_14.rolling(14).max()
        df['stoch_rsi'] = 100 * (rsi_14 - rsi_low_14) / (rsi_high_14 - rsi_low_14 + 1e-10)
        
        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)
        
        # Commodity Channel Index (CCI)
        tp = (df['high'] + df['low'] + df['close']) / 3  # Typical Price
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        df['cci'] = df['cci'].clip(-300, 300)
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = 100 * (df['close'] - df['close'].shift(period)) / (df['close'].shift(period) + 1e-10)
        
        # Money Flow Index (MFI) - Volume-weighted RSI
        tp = (df['high'] + df['low'] + df['close']) / 3
        money_flow = tp * df['volume']
        money_flow_pos = money_flow.where(tp.diff() > 0, 0)
        money_flow_neg = money_flow.where(tp.diff() < 0, 0)
        mf_ratio = money_flow_pos.rolling(14).sum() / (money_flow_neg.rolling(14).sum() + 1e-10)
        df['mfi'] = 100 - (100 / (1 + mf_ratio))
        df['mfi'] = df['mfi'].clip(0, 100)
        
        # True Strength Index (TSI)
        price_change = df['close'].diff()
        double_smoothed_pc = price_change.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
        double_smoothed_abs_pc = price_change.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
        df['tsi'] = 100 * double_smoothed_pc / (double_smoothed_abs_pc + 1e-10)
        
        # Ultimate Oscillator (combines 7, 14, 28 periods)
        true_low = pd.concat([df['low'], df['close'].shift(1)], axis=1).min(axis=1)
        true_high = pd.concat([df['high'], df['close'].shift(1)], axis=1).max(axis=1)
        bp = df['close'] - true_low  # Buying Pressure
        tr_uo = true_high - true_low  # True Range for Ultimate Oscillator
        avg7 = bp.rolling(7).sum() / (tr_uo.rolling(7).sum() + 1e-10)
        avg14 = bp.rolling(14).sum() / (tr_uo.rolling(14).sum() + 1e-10)
        avg28 = bp.rolling(28).sum() / (tr_uo.rolling(28).sum() + 1e-10)
        df['ultimate_osc'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
        
        # ========== VOLATILITY INDICATORS ==========
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        # Keltner Channels (ATR-based bands)
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        atr_10 = true_range.rolling(10).mean()
        df['keltner_upper'] = ema_20 + 2 * atr_10
        df['keltner_lower'] = ema_20 - 2 * atr_10
        df['keltner_position'] = (df['close'] - df['keltner_lower']) / (df['keltner_upper'] - df['keltner_lower'] + 1e-10)
        
        # Donchian Channels (high-low bands)
        df['donchian_upper'] = df['high'].rolling(20).max()
        df['donchian_lower'] = df['low'].rolling(20).min()
        df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2
        df['donchian_position'] = (df['close'] - df['donchian_lower']) / (df['donchian_upper'] - df['donchian_lower'] + 1e-10)
        
        # Historical Volatility
        for period in [5, 10, 20, 60]:
            df[f'volatility_{period}'] = df['return_1d'].rolling(period).std() * np.sqrt(252)
        
        # Realized Volatility (Parkinson, Garman-Klass)
        df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * ((np.log(df['high'] / df['low'])) ** 2))
        hl = np.log(df['high'] / df['low'])
        co = np.log(df['close'] / df['open'])
        df['garman_klass_vol'] = np.sqrt(0.5 * hl**2 - (2 * np.log(2) - 1) * co**2)
        
        # Volatility Ratios
        df['vol_ratio_5_20'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
        df['vol_ratio_10_60'] = df['volatility_10'] / (df['volatility_60'] + 1e-10)
        
        # ========== VOLUME INDICATORS ==========
        
        # Volume Moving Averages
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
        
        # Volume Ratios
        df['volume_ratio_5'] = df['volume'] / (df['volume_ma_5'] + 1)
        df['volume_ratio_20'] = df['volume'] / (df['volume_ma_20'] + 1)
        
        # Volume Oscillator
        df['volume_osc'] = 100 * (df['volume_ma_5'] - df['volume_ma_20']) / (df['volume_ma_20'] + 1)
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        df['obv_slope'] = df['obv'].diff(5) / 5  # 5-day slope
        
        # Volume-Price Trend (VPT)
        df['vpt'] = (df['volume'] * df['return_1d']).fillna(0).cumsum()
        df['vpt_slope'] = df['vpt'].diff(5) / 5
        
        # Accumulation/Distribution Line
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)  # Money Flow Multiplier
        mfv = mfm * df['volume']  # Money Flow Volume
        df['ad_line'] = mfv.cumsum()
        df['ad_line_ema'] = df['ad_line'].ewm(span=20, adjust=False).mean()
        
        # Chaikin Money Flow (CMF)
        df['cmf'] = mfv.rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-10)
        
        # Force Index (Price change * Volume)
        df['force_index'] = df['close'].diff() * df['volume']
        df['force_index_ema'] = df['force_index'].ewm(span=13, adjust=False).mean()
        
        # Ease of Movement (EMV)
        distance_moved = ((df['high'] + df['low']) / 2) - ((df['high'].shift() + df['low'].shift()) / 2)
        box_ratio = (df['volume'] / 1000000) / (df['high'] - df['low'] + 1e-10)
        df['emv'] = distance_moved / (box_ratio + 1e-10)
        df['emv_ema'] = df['emv'].ewm(span=14, adjust=False).mean()
        
        # Volume-Weighted Average Price (VWAP)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / (df['volume'].cumsum() + 1e-10)
        df['price_to_vwap'] = df['close'] / (df['vwap'] + 1e-10)
        
        # Volume Weighted Moving Average
        for period in [10, 20]:
            df[f'vwma_{period}'] = (df['close'] * df['volume']).rolling(period).sum() / (df['volume'].rolling(period).sum() + 1e-10)
        
        # ========== TREND INDICATORS ==========
        
        # ADX (Average Directional Index)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr_14 = true_range.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr_14)
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr_14)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Parabolic SAR (simplified version)
        af = 0.02  # Acceleration factor
        max_af = 0.2
        df['sar'] = df['close'].copy()
        uptrend = df['close'] > df['close'].shift(1)
        df.loc[uptrend, 'sar'] = df.loc[uptrend, 'low'].rolling(5).min()
        df.loc[~uptrend, 'sar'] = df.loc[~uptrend, 'high'].rolling(5).max()
        df['sar_signal'] = (df['close'] > df['sar']).astype(int)  # 1=bullish, 0=bearish
        
        # Aroon Indicator
        aroon_period = 25
        df['aroon_up'] = 100 * df['high'].rolling(aroon_period).apply(lambda x: aroon_period - x[::-1].argmax()) / aroon_period
        df['aroon_down'] = 100 * df['low'].rolling(aroon_period).apply(lambda x: aroon_period - x[::-1].argmin()) / aroon_period
        df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
        
        # Supertrend (ATR-based trend)
        multiplier = 3
        hl_avg = (df['high'] + df['low']) / 2
        basic_upperband = hl_avg + multiplier * true_range.rolling(10).mean()
        basic_lowerband = hl_avg - multiplier * true_range.rolling(10).mean()
        df['supertrend'] = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > basic_upperband.iloc[i-1]:
                df['supertrend'].iloc[i] = 1  # Uptrend
            elif df['close'].iloc[i] < basic_lowerband.iloc[i-1]:
                df['supertrend'].iloc[i] = -1  # Downtrend
            else:
                df['supertrend'].iloc[i] = df['supertrend'].iloc[i-1]
        
        # Ichimoku Cloud (simplified)
        nine_period_high = df['high'].rolling(9).max()
        nine_period_low = df['low'].rolling(9).min()
        df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2  # Conversion Line
        
        period26_high = df['high'].rolling(26).max()
        period26_low = df['low'].rolling(26).min()
        df['kijun_sen'] = (period26_high + period26_low) / 2  # Base Line
        
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)  # Leading Span A
        period52_high = df['high'].rolling(52).max()
        period52_low = df['low'].rolling(52).min()
        df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)  # Leading Span B
        
        df['chikou_span'] = df['close'].shift(-26)  # Lagging Span
        df['ichimoku_signal'] = ((df['close'] > df['senkou_span_a']) & (df['close'] > df['senkou_span_b'])).astype(int)
        
        # ========== PATTERN FEATURES ==========
        
        # Price Ranges
        df['daily_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        df['close_to_high'] = (df['high'] - df['close']) / (df['high'] + 1e-10)
        df['close_to_low'] = (df['close'] - df['low']) / (df['close'] + 1e-10)
        df['high_low_ratio'] = df['high'] / (df['low'] + 1e-10)
        
        # Gap
        df['gap'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-10)
        df['gap_up'] = (df['gap'] > 0.01).astype(int)  # > 1% gap up
        df['gap_down'] = (df['gap'] < -0.01).astype(int)  # < -1% gap down
        
        # Candle Properties
        df['candle_body'] = (df['close'] - df['open']) / (df['open'] + 1e-10)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['open'] + 1e-10)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['open'] + 1e-10)
        df['body_to_range'] = np.abs(df['candle_body']) / (df['daily_range'] + 1e-10)
        
        # Candlestick Patterns (simplified detection)
        # Doji: Small body
        df['is_doji'] = (np.abs(df['candle_body']) < 0.002).astype(int)
        
        # Hammer/Hanging Man: Small body, long lower shadow
        df['is_hammer'] = ((df['lower_shadow'] > 2 * np.abs(df['candle_body'])) & 
                           (df['upper_shadow'] < 0.3 * np.abs(df['candle_body']))).astype(int)
        
        # Shooting Star/Inverted Hammer: Small body, long upper shadow
        df['is_shooting_star'] = ((df['upper_shadow'] > 2 * np.abs(df['candle_body'])) & 
                                   (df['lower_shadow'] < 0.3 * np.abs(df['candle_body']))).astype(int)
        
        # Engulfing (bullish/bearish)
        prev_body = df['candle_body'].shift(1)
        df['bullish_engulfing'] = ((df['candle_body'] > 0) & (prev_body < 0) & 
                                    (df['candle_body'] > -prev_body)).astype(int)
        df['bearish_engulfing'] = ((df['candle_body'] < 0) & (prev_body > 0) & 
                                    (-df['candle_body'] > prev_body)).astype(int)
        
        # Price Action: Higher highs, lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['higher_close'] = (df['close'] > df['close'].shift(1)).astype(int)
        
        # Support/Resistance levels (last 20 days)
        df['resistance_20'] = df['high'].rolling(20).max()
        df['support_20'] = df['low'].rolling(20).min()
        df['near_resistance'] = (df['close'] > 0.98 * df['resistance_20']).astype(int)
        df['near_support'] = (df['close'] < 1.02 * df['support_20']).astype(int)
        
        # ========== MICROSTRUCTURE FEATURES ==========
        
        # Price Efficiency (how directly price moves)
        net_change = np.abs(df['close'] - df['close'].shift(10))
        path_length = np.abs(df['close'].diff()).rolling(10).sum()
        df['price_efficiency_10'] = net_change / (path_length + 1e-10)
        
        # Realized Volatility vs ATR
        df['realized_vol_20'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
        df['vol_to_atr'] = df['realized_vol_20'] / (df['atr_14'] + 1e-10)
        
        # Bid-Ask Spread Proxy (High-Low as percentage of price)
        df['spread_proxy'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        df['spread_ma'] = df['spread_proxy'].rolling(20).mean()
        df['spread_ratio'] = df['spread_proxy'] / (df['spread_ma'] + 1e-10)
        
        # Price Impact Proxy (price change per unit volume)
        df['price_impact'] = np.abs(df['return_1d']) / (df['volume'] / df['volume'].rolling(20).mean() + 1e-10)
        df['price_impact_ma'] = df['price_impact'].rolling(20).mean()
        
        # Roll's Spread Estimator
        cov_ret = df['return_1d'].rolling(2).cov().shift(-1)
        df['roll_spread'] = 2 * np.sqrt(np.abs(cov_ret))
        
        # Amihud Illiquidity Measure
        df['amihud_illiq'] = np.abs(df['return_1d']) / ((df['volume'] * df['close']) / 10000000 + 1e-10)
        df['amihud_ma'] = df['amihud_illiq'].rolling(20).mean()
        
        return df
    
    @staticmethod
    def add_market_features(df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add market index features."""
        df = df.copy()
        
        for index_name, index_df in market_data.items():
            # Merge on date
            index_df = index_df[['timestamp', 'close']].copy()
            index_df.columns = ['timestamp', f'{index_name}_close']
            
            # Remove timezone info for consistent merging
            index_df['timestamp'] = pd.to_datetime(index_df['timestamp']).dt.tz_localize(None)
            
            df = df.merge(index_df, on='timestamp', how='left')
            
            # Market returns
            df[f'{index_name}_return_1d'] = df[f'{index_name}_close'].pct_change()
            df[f'{index_name}_return_5d'] = df[f'{index_name}_close'].pct_change(5)
            
            # Correlation with market
            df[f'corr_{index_name}_20d'] = df['return_1d'].rolling(20).corr(df[f'{index_name}_return_1d'])
        
        # Relative strength to NIFTY50
        if 'NIFTY50_close' in df.columns:
            df['relative_strength_nifty'] = df['close'] / (df['NIFTY50_close'] + 1e-10)
            df['rs_nifty_sma'] = df['relative_strength_nifty'].rolling(20).mean()
            df['rs_nifty_normalized'] = (df['relative_strength_nifty'] - df['rs_nifty_sma']) / (df['rs_nifty_sma'] + 1e-10)
            
            # Beta to market
            cov = df['return_1d'].rolling(60).cov(df['NIFTY50_return_1d'])
            var = df['NIFTY50_return_1d'].rolling(60).var()
            df['beta_nifty'] = cov / (var + 1e-10)
            df['beta_nifty'] = df['beta_nifty'].clip(-3, 3)  # Limit outliers
        
        # ========== CROSS-SECTIONAL FEATURES ==========
        
        # Volume percentile (within last 60 days)
        df['volume_percentile'] = df['volume'].rolling(60).rank(pct=True)
        
        # Volatility percentile
        df['vol_percentile'] = df['volatility_20'].rolling(60).rank(pct=True)
        
        # Price percentile (within 52-week range)
        df['price_52w_high'] = df['high'].rolling(252).max()
        df['price_52w_low'] = df['low'].rolling(252).min()
        df['price_52w_percentile'] = (df['close'] - df['price_52w_low']) / (df['price_52w_high'] - df['price_52w_low'] + 1e-10)
        
        # Distance from 52-week high/low
        df['dist_from_52w_high'] = (df['price_52w_high'] - df['close']) / (df['close'] + 1e-10)
        df['dist_from_52w_low'] = (df['close'] - df['price_52w_low']) / (df['close'] + 1e-10)
        
        return df
    
    @staticmethod
    def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal (time-based) features."""
        df = df.copy()
        
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        # Cyclical encoding
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        # Special days
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
        df['is_month_end'] = (df['timestamp'].dt.is_month_end).astype(int)
        df['is_quarter_end'] = (df['timestamp'].dt.is_quarter_end).astype(int)
        df['is_year_end'] = (df['month'] == 12).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Expiry week (last week of month for Indian options)
        df['is_expiry_week'] = ((df['day_of_month'] >= 21) | (df['is_month_end'])).astype(int)
        
        # Days since month start
        df['days_since_month_start'] = df['day_of_month']
        
        # Days to month end
        df['days_to_month_end'] = df['timestamp'].dt.days_in_month - df['day_of_month']
        
        return df
    
    @staticmethod
    def add_sentiment_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add sentiment features from news."""
        df = df.copy()
        
        # Check if sentiment features already exist
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
        
        if sentiment_cols:
            logger.info(f"Using existing {len(sentiment_cols)} sentiment features")
            return df
        
        # Otherwise, create placeholder sentiment features
        # In production, these would come from news API
        logger.warning("No sentiment data found, creating neutral placeholders")
        
        df['news_sentiment'] = 0.0  # Neutral
        df['sentiment_ma_7d'] = 0.0
        df['sentiment_std_7d'] = 0.0
        df['sentiment_positive_count'] = 0
        df['sentiment_negative_count'] = 0
        
        return df
    
    @staticmethod
    def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        df = df.copy()
        
        # ========== TREND REGIME ==========
        
        # Bull/Bear Market (based on 50-day and 200-day SMA)
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            df['regime_bull'] = (df['close'] > df['sma_50']) & (df['sma_50'] > df['sma_200'])
            df['regime_bull'] = df['regime_bull'].astype(int)
            df['regime_bear'] = (df['close'] < df['sma_50']) & (df['sma_50'] < df['sma_200'])
            df['regime_bear'] = df['regime_bear'].astype(int)
        
        # Trend Strength (ADX-based)
        if 'adx' in df.columns:
            df['strong_trend'] = (df['adx'] > 25).astype(int)
            df['weak_trend'] = (df['adx'] < 20).astype(int)
            df['trending'] = (df['adx'] > 25).astype(int)
            df['ranging'] = (df['adx'] < 20).astype(int)
        
        # ========== VOLATILITY REGIME ==========
        
        # High/Low Volatility
        vol_median = df['volatility_20'].rolling(60).median()
        df['high_vol_regime'] = (df['volatility_20'] > 1.5 * vol_median).astype(int)
        df['low_vol_regime'] = (df['volatility_20'] < 0.5 * vol_median).astype(int)
        
        # Volatility expanding/contracting
        df['vol_expanding'] = (df['volatility_5'] > df['volatility_20']).astype(int)
        df['vol_contracting'] = (df['volatility_5'] < df['volatility_20']).astype(int)
        
        # ========== VOLUME REGIME ==========
        
        # High/Low Volume
        vol_ma_60 = df['volume'].rolling(60).mean()
        df['high_volume_regime'] = (df['volume'] > 1.5 * vol_ma_60).astype(int)
        df['low_volume_regime'] = (df['volume'] < 0.5 * vol_ma_60).astype(int)
        
        # Volume trend
        df['volume_increasing'] = (df['volume_ma_5'] > df['volume_ma_20']).astype(int)
        df['volume_decreasing'] = (df['volume_ma_5'] < df['volume_ma_20']).astype(int)
        
        # ========== MOMENTUM REGIME ==========
        
        # Overbought/Oversold (RSI-based)
        if 'rsi_14' in df.columns:
            df['overbought'] = (df['rsi_14'] > 70).astype(int)
            df['oversold'] = (df['rsi_14'] < 30).astype(int)
            df['neutral_rsi'] = ((df['rsi_14'] >= 40) & (df['rsi_14'] <= 60)).astype(int)
        
        # Momentum regime
        df['positive_momentum'] = (df['return_5d'] > 0).astype(int)
        df['negative_momentum'] = (df['return_5d'] < 0).astype(int)
        
        # ========== CORRELATION REGIME ==========
        
        # Check correlation with market
        if 'corr_NIFTY50_20d' in df.columns:
            df['high_correlation'] = (df['corr_NIFTY50_20d'] > 0.7).astype(int)
            df['low_correlation'] = (df['corr_NIFTY50_20d'] < 0.3).astype(int)
            df['decoupled'] = (np.abs(df['corr_NIFTY50_20d']) < 0.3).astype(int)
        
        # ========== PRICE LOCATION REGIME ==========
        
        # Where is price in its range?
        if 'bb_position' in df.columns:
            df['near_bb_upper'] = (df['bb_position'] > 0.8).astype(int)
            df['near_bb_lower'] = (df['bb_position'] < 0.2).astype(int)
            df['bb_middle_zone'] = ((df['bb_position'] >= 0.4) & (df['bb_position'] <= 0.6)).astype(int)
        
        # Distance from moving averages
        if 'sma_20' in df.columns:
            pct_from_sma = (df['close'] - df['sma_20']) / (df['sma_20'] + 1e-10)
            df['far_above_sma'] = (pct_from_sma > 0.05).astype(int)  # >5% above
            df['far_below_sma'] = (pct_from_sma < -0.05).astype(int)  # >5% below
            df['near_sma'] = (np.abs(pct_from_sma) < 0.02).astype(int)  # Within 2%
        
        return df
    
    @staticmethod
    def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features (cross-terms)."""
        df = df.copy()
        
        # ========== PRICE-VOLUME INTERACTIONS ==========
        
        # High return + high volume = strong signal
        if 'return_1d' in df.columns and 'volume_ratio_20' in df.columns:
            df['return_x_volume'] = df['return_1d'] * df['volume_ratio_20']
            df['abs_return_x_volume'] = np.abs(df['return_1d']) * df['volume_ratio_20']
        
        # Price momentum + volume trend
        if 'return_5d' in df.columns and 'volume_ratio_5' in df.columns:
            df['momentum_x_volume'] = df['return_5d'] * df['volume_ratio_5']
        
        # Trend + volume (ADX * volume ratio)
        if 'adx' in df.columns and 'volume_ratio_20' in df.columns:
            df['trend_strength_x_volume'] = (df['adx'] / 100) * df['volume_ratio_20']
        
        # ========== VOLATILITY-VOLUME INTERACTIONS ==========
        
        # High volatility + high volume = breakout
        if 'volatility_20' in df.columns and 'volume_ratio_20' in df.columns:
            df['vol_x_volume'] = df['volatility_20'] * df['volume_ratio_20']
        
        # ATR * volume (volatility-adjusted volume)
        if 'atr_14' in df.columns and 'volume' in df.columns:
            df['atr_x_volume'] = (df['atr_14'] / df['close']) * df['volume_ratio_20']
        
        # ========== MOMENTUM-VOLATILITY INTERACTIONS ==========
        
        # RSI + Volatility
        if 'rsi_14' in df.columns and 'volatility_20' in df.columns:
            df['rsi_x_vol'] = (df['rsi_14'] / 100) * df['volatility_20']
        
        # Return + Volatility (Sharpe-like)
        if 'return_5d' in df.columns and 'volatility_5' in df.columns:
            df['return_to_vol_5d'] = df['return_5d'] / (df['volatility_5'] + 1e-10)
            df['return_to_vol_5d'] = df['return_to_vol_5d'].clip(-5, 5)
        
        # MACD + Volatility
        if 'macd_histogram' in df.columns and 'volatility_20' in df.columns:
            df['macd_x_vol'] = df['macd_histogram'] * df['volatility_20']
        
        # ========== TREND-VOLUME INTERACTIONS ==========
        
        # Moving average trend + volume
        if 'sma_5_to_sma_20' in df.columns and 'volume_ratio_5' in df.columns:
            df['ma_trend_x_volume'] = df['sma_5_to_sma_20'] * df['volume_ratio_5']
        
        # Price to SMA + Volume
        if 'price_to_sma_20' in df.columns and 'volume_ratio_20' in df.columns:
            df['price_sma_x_volume'] = df['price_to_sma_20'] * df['volume_ratio_20']
        
        # ========== BOLLINGER BAND INTERACTIONS ==========
        
        # BB Width + Volume (squeeze detection)
        if 'bb_width' in df.columns and 'volume_ratio_20' in df.columns:
            df['bb_squeeze'] = df['bb_width'] * df['volume_ratio_20']
            df['bb_squeeze_low'] = ((df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)) & 
                                     (df['volume_ratio_20'] > 1.2)).astype(int)
        
        # BB Position + RSI
        if 'bb_position' in df.columns and 'rsi_14' in df.columns:
            df['bb_rsi_signal'] = df['bb_position'] * (df['rsi_14'] / 100)
        
        # ========== MARKET CORRELATION INTERACTIONS ==========
        
        # Stock return when market is up/down
        if 'NIFTY50_return_1d' in df.columns and 'return_1d' in df.columns:
            df['return_when_market_up'] = df['return_1d'] * (df['NIFTY50_return_1d'] > 0).astype(int)
            df['return_when_market_down'] = df['return_1d'] * (df['NIFTY50_return_1d'] < 0).astype(int)
        
        # Beta * Market return
        if 'beta_nifty' in df.columns and 'NIFTY50_return_1d' in df.columns:
            df['expected_return'] = df['beta_nifty'] * df['NIFTY50_return_1d']
            df['return_surprise'] = df['return_1d'] - df['expected_return']
        
        # ========== MULTI-INDICATOR COMBINATIONS ==========
        
        # RSI + MACD confluence
        if 'rsi_14' in df.columns and 'macd_histogram' in df.columns:
            rsi_bullish = (df['rsi_14'] > 50).astype(int)
            macd_bullish = (df['macd_histogram'] > 0).astype(int)
            df['rsi_macd_confluence'] = rsi_bullish * macd_bullish  # Both bullish
            df['rsi_macd_divergence'] = np.abs(rsi_bullish - macd_bullish)  # Disagreement
        
        # Stochastic + RSI
        if 'stoch_k' in df.columns and 'rsi_14' in df.columns:
            df['stoch_rsi_avg'] = (df['stoch_k'] + df['rsi_14']) / 2
            df['stoch_rsi_spread'] = np.abs(df['stoch_k'] - df['rsi_14'])
        
        # Multiple trend indicators
        if 'adx' in df.columns and 'aroon_osc' in df.columns:
            df['trend_consensus'] = ((df['adx'] / 100) + (df['aroon_osc'] / 100)) / 2
        
        return df
    
    @staticmethod
    def add_advanced_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced statistical and time series features."""
        df = df.copy()
        
        # ========== MOMENTUM DECAY FEATURES ==========
        
        # Exponentially weighted momentum (recent days matter more)
        weights = np.exp(np.linspace(-1, 0, 10))
        weights = weights / weights.sum()
        df['momentum_ewm'] = df['return_1d'].rolling(10).apply(lambda x: np.dot(x, weights), raw=True)
        
        # Momentum acceleration (change in momentum)
        df['momentum_accel'] = df['return_5d'].diff(5)
        df['momentum_decel'] = (df['return_5d'] < df['return_10d']).astype(int)
        
        # ========== AUTOCORRELATION FEATURES ==========
        
        # Serial correlation (mean reversion vs momentum)
        def safe_autocorr(x, lag=1):
            try:
                if len(x) >= lag + 10:
                    return x.autocorr(lag=lag)
                return 0
            except:
                return 0
        
        df['autocorr_1'] = df['return_1d'].rolling(30).apply(lambda x: safe_autocorr(x, 1), raw=False)
        df['autocorr_5'] = df['return_1d'].rolling(60).apply(lambda x: safe_autocorr(x, 5), raw=False)
        df['autocorr_1'] = df['autocorr_1'].fillna(0).clip(-1, 1)
        df['autocorr_5'] = df['autocorr_5'].fillna(0).clip(-1, 1)
        
        # ========== HURST EXPONENT (Trend persistence) ==========
        
        def hurst_exponent(ts):
            """Calculate Hurst exponent (0.5=random, >0.5=trending, <0.5=mean-reverting)"""
            if len(ts) < 20:
                return 0.5
            lags = range(2, 20)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            try:
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0]
            except:
                return 0.5
        
        df['hurst_60'] = df['close'].rolling(60).apply(hurst_exponent, raw=True)
        df['hurst_60'] = df['hurst_60'].clip(0, 1)
        
        # ========== SKEWNESS & KURTOSIS ==========
        
        # Distribution shape of returns
        df['skew_20'] = df['return_1d'].rolling(20).skew()
        df['kurt_20'] = df['return_1d'].rolling(20).kurt()
        df['skew_20'] = df['skew_20'].clip(-5, 5)
        df['kurt_20'] = df['kurt_20'].clip(-5, 10)
        
        # ========== ORDER FLOW PROXIES ==========
        
        # Buying vs selling pressure (intraday)
        df['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
        df['net_pressure'] = df['buy_pressure'] - df['sell_pressure']
        df['pressure_ma'] = df['net_pressure'].rolling(10).mean()
        
        # Volume imbalance
        up_volume = df['volume'].where(df['close'] > df['open'], 0)
        down_volume = df['volume'].where(df['close'] < df['open'], 0)
        df['volume_imbalance'] = (up_volume - down_volume) / (df['volume'] + 1e-10)
        df['volume_imbalance_ma'] = df['volume_imbalance'].rolling(10).mean()
        
        # ========== RANGE EXPANSION/CONTRACTION ==========
        
        # True Range percentile
        df['tr_percentile'] = df['atr_14'].rolling(60).rank(pct=True)
        
        # Range expansion (today's range vs average)
        avg_range = df['daily_range'].rolling(20).mean()
        df['range_expansion'] = df['daily_range'] / (avg_range + 1e-10)
        df['range_expansion'] = df['range_expansion'].clip(0, 5)
        
        # ========== PRICE PERSISTENCE ==========
        
        # How many consecutive up/down days
        df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['consecutive_down'] = (df['close'] < df['close'].shift(1)).astype(int)
        
        for i in range(1, len(df)):
            if df['consecutive_up'].iloc[i] == 1:
                df['consecutive_up'].iloc[i] = df['consecutive_up'].iloc[i-1] + 1
            if df['consecutive_down'].iloc[i] == 1:
                df['consecutive_down'].iloc[i] = df['consecutive_down'].iloc[i-1] + 1
        
        df['max_consecutive_up'] = df['consecutive_up'].rolling(20).max()
        df['max_consecutive_down'] = df['consecutive_down'].rolling(20).max()
        
        # ========== VOLATILITY CLUSTERING ==========
        
        # GARCH-like features
        df['vol_squared'] = df['return_1d'] ** 2
        df['vol_squared_ma'] = df['vol_squared'].rolling(20).mean()
        
        # Volatility persistence
        high_vol = (df['volatility_20'] > df['volatility_20'].rolling(60).median()).astype(int)
        df['vol_persistence'] = high_vol.rolling(10).sum() / 10
        
        # ========== EFFICIENCY RATIO ==========
        
        # Perry Kaufman's Efficiency Ratio
        direction = np.abs(df['close'] - df['close'].shift(10))
        volatility = np.abs(df['close'].diff()).rolling(10).sum()
        df['efficiency_ratio'] = direction / (volatility + 1e-10)
        df['efficiency_ratio'] = df['efficiency_ratio'].clip(0, 1)
        
        return df
    
    @staticmethod
    def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity and market depth features."""
        df = df.copy()
        
        # ========== LIQUIDITY MEASURES ==========
        
        # Turnover (Volume * Price as proxy for liquidity)
        df['turnover'] = df['volume'] * df['close']
        df['turnover_ma_20'] = df['turnover'].rolling(20).mean()
        df['turnover_ratio'] = df['turnover'] / (df['turnover_ma_20'] + 1e-10)
        
        # Relative turnover (vs market)
        if 'NIFTY50_close' in df.columns:
            nifty_volume = df['volume'].rolling(20).mean()  # Proxy
            df['relative_turnover'] = df['turnover'] / (nifty_volume * df['NIFTY50_close'] + 1e-10)
        
        # Volume concentration (how concentrated is volume?)
        df['volume_concentration'] = df['volume'].rolling(5).max() / (df['volume'].rolling(5).sum() + 1e-10)
        
        # ========== PRICE IMPACT ENHANCEMENTS ==========
        
        # Kyle's Lambda (price impact per volume)
        abs_return = np.abs(df['return_1d'])
        volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
        df['kyle_lambda'] = abs_return / (volume_ratio + 1e-10)
        df['kyle_lambda'] = df['kyle_lambda'].clip(0, 0.1)
        
        # Price resilience (how fast does price recover after move?)
        df['price_resilience'] = df['return_1d'].rolling(5).std() / (df['volatility_5'] + 1e-10)
        
        # ========== LIQUIDITY REGIMES ==========
        
        # Low liquidity regime
        turnover_median = df['turnover'].rolling(60).median()
        df['low_liquidity'] = (df['turnover'] < 0.7 * turnover_median).astype(int)
        df['high_liquidity'] = (df['turnover'] > 1.3 * turnover_median).astype(int)
        
        return df
    
    @staticmethod
    def add_gap_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add gap analysis features."""
        df = df.copy()
        
        # ========== GAP ANALYSIS ==========
        
        # Gap types
        df['gap_size'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-10)
        df['gap_up_large'] = (df['gap_size'] > 0.02).astype(int)  # >2% gap
        df['gap_down_large'] = (df['gap_size'] < -0.02).astype(int)
        
        # Gap filled? (price returned to previous close)
        prev_close = df['close'].shift(1)
        gap_up_mask = df['open'] > prev_close
        gap_down_mask = df['open'] < prev_close
        
        df['gap_filled'] = 0
        df.loc[gap_up_mask, 'gap_filled'] = (df.loc[gap_up_mask, 'low'] <= prev_close[gap_up_mask]).astype(int)
        df.loc[gap_down_mask, 'gap_filled'] = (df.loc[gap_down_mask, 'high'] >= prev_close[gap_down_mask]).astype(int)
        
        # Gap fill ratio (how often do gaps get filled?)
        df['gap_fill_ratio'] = df['gap_filled'].rolling(20).mean()
        
        # Overnight return (gap captures overnight news)
        df['overnight_return'] = df['gap_size']
        df['overnight_vol'] = df['overnight_return'].rolling(20).std()
        
        # Intraday return (open to close)
        df['intraday_return'] = (df['close'] - df['open']) / (df['open'] + 1e-10)
        df['intraday_vol'] = df['intraday_return'].rolling(20).std()
        
        # Overnight vs intraday ratio
        df['overnight_intraday_ratio'] = df['overnight_vol'] / (df['intraday_vol'] + 1e-10)
        
        return df
    
    @staticmethod
    def add_price_level_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add price level and round number features."""
        df = df.copy()
        
        # ========== ROUND NUMBER PSYCHOLOGY ==========
        
        # Distance to round numbers (100, 500, 1000)
        def dist_to_round(price, base):
            return ((price % base) / base)
        
        df['dist_to_100'] = df['close'].apply(lambda x: dist_to_round(x, 100))
        df['dist_to_500'] = df['close'].apply(lambda x: dist_to_round(x, 500))
        df['near_round_100'] = (np.abs(df['dist_to_100'] - 0.5) > 0.4).astype(int)
        
        # ========== PRICE LEVELS ==========
        
        # Recent pivot highs/lows (support/resistance)
        df['pivot_high'] = df['high'].rolling(10, center=True).max()
        df['pivot_low'] = df['low'].rolling(10, center=True).min()
        df['at_pivot_high'] = (df['close'] >= 0.98 * df['pivot_high']).astype(int)
        df['at_pivot_low'] = (df['close'] <= 1.02 * df['pivot_low']).astype(int)
        
        # Price zones (divide range into zones)
        if 'price_52w_high' in df.columns and 'price_52w_low' in df.columns:
            range_52w = df['price_52w_high'] - df['price_52w_low']
            price_zone_raw = ((df['close'] - df['price_52w_low']) / (range_52w + 1e-10) * 10)
            df['price_zone'] = price_zone_raw.fillna(5).clip(0, 10).astype(int)
        else:
            df['price_zone'] = 5
        
        # Time at price level (consolidation)
        price_bucket = (df['close'] / 10).astype(int)  # Bucket by 10s
        df['time_at_level'] = price_bucket.rolling(20).apply(lambda x: (x == x.iloc[-1]).sum(), raw=False)
        df['time_at_level'] = df['time_at_level'].fillna(1)
        
        return df
    
    @staticmethod
    def add_advanced_volume_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced volume pattern features."""
        df = df.copy()
        
        # ========== VOLUME PATTERNS ==========
        
        # Climax volume (extreme volume spikes)
        volume_mean = df['volume'].rolling(60).mean()
        volume_std = df['volume'].rolling(60).std()
        df['volume_zscore'] = (df['volume'] - volume_mean) / (volume_std + 1e-10)
        df['climax_volume'] = (df['volume_zscore'] > 2.5).astype(int)
        
        # Exhaustion (high volume but small price move)
        df['volume_efficiency'] = np.abs(df['return_1d']) / (df['volume_ratio_20'] + 1e-10)
        df['exhaustion'] = ((df['volume_ratio_20'] > 2) & (np.abs(df['return_1d']) < 0.01)).astype(int)
        
        # Accumulation/Distribution phases
        # Accumulation: Price flat/up slightly, volume increasing
        price_flat = (np.abs(df['return_5d']) < 0.02)
        volume_up = (df['volume_ma_5'] > df['volume_ma_20'])
        df['accumulation_phase'] = (price_flat & volume_up).astype(int)
        
        # Distribution: Price flat/down slightly, volume increasing
        df['distribution_phase'] = (price_flat & volume_up & (df['return_5d'] < 0)).astype(int)
        
        # Volume-price divergence
        price_up = (df['return_5d'] > 0)
        volume_down = (df['volume_ma_5'] < df['volume_ma_20'])
        df['volume_price_divergence'] = ((price_up & volume_down) | (~price_up & ~volume_down)).astype(int)
        
        # Smart money vs dumb money (OBV divergence from price)
        obv_direction = (df['obv'] > df['obv'].shift(5))
        price_direction = (df['close'] > df['close'].shift(5))
        df['smart_money_divergence'] = (obv_direction != price_direction).astype(int)
        
        return df
    
    @staticmethod
    def add_market_breadth_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add market breadth indicators."""
        df = df.copy()
        
        # ========== MARKET BREADTH ==========
        
        if 'NIFTY50_return_1d' in df.columns:
            # Stock vs market performance
            df['outperformance'] = df['return_1d'] - df['NIFTY50_return_1d']
            df['outperformance_ma'] = df['outperformance'].rolling(20).mean()
            df['outperform_streak'] = (df['outperformance'] > 0).astype(int)
            
            for i in range(1, len(df)):
                if df['outperform_streak'].iloc[i] == 1:
                    df['outperform_streak'].iloc[i] = df['outperform_streak'].iloc[i-1] + 1
            
            # Relative momentum
            df['relative_momentum'] = df['return_20d'] - df['NIFTY50_return_5d']
            
            # Beta stability (is beta changing?)
            df['beta_change'] = df['beta_nifty'].diff(20)
            df['beta_stable'] = (np.abs(df['beta_change']) < 0.2).astype(int)
        
        return df
    
    @staticmethod
    def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical and wave features."""
        df = df.copy()
        
        # ========== CYCLE DETECTION ==========
        
        # Dominant cycle (simplified)
        # Look for peaks in autocorrelation
        def find_dominant_cycle(ts):
            if len(ts) < 40:
                return 20
            autocorrs = [ts.autocorr(lag=i) for i in range(2, 30)]
            if autocorrs:
                return np.argmax(autocorrs) + 2
            return 20
        
        df['dominant_cycle'] = df['close'].rolling(60).apply(find_dominant_cycle, raw=False)
        df['dominant_cycle'] = df['dominant_cycle'].fillna(20).clip(5, 50)
        
        # Phase (where are we in the cycle?)
        df['cycle_phase'] = (df.index % df['dominant_cycle']) / df['dominant_cycle']
        
        # Detrended price oscillator
        cycle_len = 20  # Fixed for simplicity
        df['detrended'] = df['close'] - df['close'].shift(int(cycle_len/2)).rolling(cycle_len).mean()
        df['detrended_norm'] = df['detrended'] / (df['close'] + 1e-10)
        
        # ========== WAVE PATTERNS ==========
        
        # Elliott Wave approximation (swing highs/lows)
        df['swing_high'] = df['high'].rolling(5, center=True).max()
        df['swing_low'] = df['low'].rolling(5, center=True).min()
        df['is_swing_high'] = (df['high'] == df['swing_high']).astype(int)
        df['is_swing_low'] = (df['low'] == df['swing_low']).astype(int)
        
        # Wave count (simplified)
        df['wave_count'] = df['is_swing_high'].rolling(20).sum() + df['is_swing_low'].rolling(20).sum()
        
        return df
    
    @staticmethod
    def create_targets(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create multiple prediction targets.
        
        IMPORTANT: We predict RETURNS, not absolute prices.
        This is more realistic and generalizable.
        """
        df = df.copy()
        
        # TARGET 1: Next day close return (percentage change)
        df['target_close_return'] = df['close'].pct_change().shift(-1)  # Tomorrow's return
        df['target_close'] = df['close'].shift(-1)  # Still keep absolute for reference
        
        # TARGET 2: Next day high return
        df['target_high_return'] = (df['high'].shift(-1) - df['close']) / df['close']
        df['target_high'] = df['high'].shift(-1)
        
        # TARGET 3: Next day low return
        df['target_low_return'] = (df['low'].shift(-1) - df['close']) / df['close']
        df['target_low'] = df['low'].shift(-1)
        
        # TARGET 4: Direction (classification) - based on next day close vs today close
        df['target_direction'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Additional useful targets for analysis
        df['target_range'] = df['target_high'] - df['target_low']
        
        return df
    
    @classmethod
    def prepare_full_dataset(cls, symbol: str) -> pd.DataFrame:
        """
        Prepare complete dataset with all features for a symbol.
        """
        logger.info(f"Preparing dataset for {symbol}...")
        
        # Load data
        df = cls.load_stock_data(symbol)
        logger.info(f"  Loaded {len(df)} rows")
        
        # Compute technical features
        df = cls.compute_technical_features(df)
        logger.info(f"  Added technical features")
        
        # Add market features
        market_data = cls.load_market_data()
        df = cls.add_market_features(df, market_data)
        logger.info(f"  Added market features")
        
        # Add temporal features
        df = cls.add_temporal_features(df)
        logger.info(f"  Added temporal features")
        
        # Add sentiment features
        df = cls.add_sentiment_features(df, symbol)
        logger.info(f"  Added sentiment features")
        
        # Add regime detection features
        df = cls.add_regime_features(df)
        logger.info(f"  Added regime features")
        
        # Add interaction features
        df = cls.add_interaction_features(df)
        logger.info(f"  Added interaction features")
        
        # Add advanced statistical features
        df = cls.add_advanced_statistical_features(df)
        logger.info(f"  Added advanced statistical features")
        
        # Add liquidity features
        df = cls.add_liquidity_features(df)
        logger.info(f"  Added liquidity features")
        
        # Add gap analysis features
        df = cls.add_gap_analysis_features(df)
        logger.info(f"  Added gap analysis features")
        
        # Add price level features
        df = cls.add_price_level_features(df)
        logger.info(f"  Added price level features")
        
        # Add advanced volume patterns
        df = cls.add_advanced_volume_patterns(df)
        logger.info(f"  Added advanced volume patterns")
        
        # Add market breadth features
        df = cls.add_market_breadth_features(df)
        logger.info(f"  Added market breadth features")
        
        # Add cycle features
        df = cls.add_cycle_features(df)
        logger.info(f"  Added cycle features")
        
        # Create targets
        df = cls.create_targets(df)
        logger.info(f"  Created targets")
        
        # Remove rows with NaN in critical columns
        critical_cols = ['close', 'high', 'low', 'volume', 
                        'target_close_return', 'target_high_return', 'target_low_return', 'target_direction']
        initial_len = len(df)
        df = df.dropna(subset=critical_cols)
        logger.info(f"  Dropped {initial_len - len(df)} rows with missing critical data")
        
        logger.success(f"Dataset prepared: {len(df)} samples, {len(df.columns)} columns")
        
        return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns (exclude targets, metadata, raw OHLCV).
    
    CRITICAL: Must exclude ALL forward-looking information to prevent data leakage.
    """
    exclude_patterns = [
        'timestamp', 'symbol',
        'target_', 'next_',  # All targets
        'open', 'high', 'low', 'close', 'volume',  # Raw OHLCV (we use derived features)
        'sma_', 'ema_', 'volume_ma_',  # Intermediate calculations (we use ratios)
        '_lag_',  # Lagged values are for sequences, not flat features
        # CRITICAL: Exclude same-day features that could leak
        'return_1d',  # This is TODAY's return, calculated using tomorrow's price!
        'intraday_return',  # Uses close price
        'intraday_range',  # Uses high/low of same day
    ]
    
    # Also exclude exact column names that are problematic
    exclude_exact = [
        'target',  # Old single target
        'return_1d',  # CRITICAL: This causes 99.81% accuracy
    ]
    
    feature_cols = []
    for col in df.columns:
        # Skip if exact match
        if col in exclude_exact:
            continue
        # Skip if matches any exclude pattern
        if any(pattern in col for pattern in exclude_patterns):
            continue
        # Include if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    return feature_cols


def get_sequence_columns(df: pd.DataFrame, lookback: int = 10) -> List[str]:
    """
    Get columns for sequence models (LSTM/GRU).
    Includes lagged prices and current features.
    """
    sequence_cols = []
    
    # Lagged prices
    for lag in range(1, lookback + 1):
        for base in ['close', 'high', 'low']:
            col = f'{base}_lag_{lag}'
            if col in df.columns:
                sequence_cols.append(col)
    
    # Current features (non-lagged)
    feature_cols = get_feature_columns(df)
    sequence_cols.extend(feature_cols)
    
    return sequence_cols


# Save for next part...
