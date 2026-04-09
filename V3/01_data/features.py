"""
Feature engineering for stock price prediction.
Computes ~100 technical and statistical features from OHLCV data.
"""

import numpy as np
import pandas as pd
from ta import momentum, trend, volatility, volume
from typing import List


class FeatureEngineer:
    """Compute technical indicators and statistical features."""

    def __init__(self):
        self.feature_names: List[str] = []

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for a stock.

        Input DataFrame must have: timestamp, open, high, low, close, volume
        Output includes all input columns + computed features.
        """
        df = df.copy().reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Price-based features
        df = self._price_features(df)

        # Technical indicators
        df = self._technical_indicators(df)

        # Volume features
        df = self._volume_features(df)

        # Volatility features
        df = self._volatility_features(df)

        # Momentum features
        df = self._momentum_features(df)

        # Trend features
        df = self._trend_features(df)

        # Statistical features
        df = self._statistical_features(df)

        # Drop NaN from rolling windows
        df = df.dropna(subset=self.feature_names)

        # Replace inf with NaN then drop
        for col in self.feature_names:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        df = df.dropna(subset=self.feature_names)
        return df.reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns, gaps, price ratios."""
        names = []

        # Log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        names.append("log_return")

        # Close-to-open return
        df["co_return"] = np.log(df["close"] / df["open"])
        names.append("co_return")

        # High-Low range (%)
        df["hl_range"] = (df["high"] - df["low"]) / df["open"]
        names.append("hl_range")

        # Gap (prev close to open)
        df["gap"] = np.log(df["open"] / df["close"].shift(1))
        names.append("gap")

        # Moving average ratios
        for period in [5, 10, 20, 50]:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
            df[f"close_to_sma_{period}"] = df["close"] / df[f"sma_{period}"] - 1
            names.extend([f"sma_{period}", f"close_to_sma_{period}"])

        self.feature_names.extend(names)
        return df

    def _technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, MACD, Bollinger Bands, Stochastic."""
        names = []

        # RSI (14)
        rsi = momentum.RSIIndicator(df["close"], window=14).rsi()
        df["rsi_14"] = rsi
        names.append("rsi_14")

        # MACD
        macd = trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
        names.extend(["macd", "macd_signal", "macd_diff"])

        # Bollinger Bands
        bb = volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_wband"] = bb.bollinger_wband()
        names.extend(["bb_high", "bb_low", "bb_mid", "bb_wband"])

        # ADX (trend strength)
        adx = trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
        df["adx"] = adx.adx()
        names.append("adx")

        self.feature_names.extend(names)
        return df

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV, VWAP, volume momentum."""
        names = []

        # OBV
        obv = volume.OnBalanceVolumeIndicator(df["close"], df["volume"])
        df["obv"] = obv.on_balance_volume()
        names.append("obv")

        # Volume SMA ratio
        vol_sma = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / vol_sma
        names.append("vol_ratio")

        # Volume momentum
        df["vol_mom"] = df["volume"].pct_change(5)
        names.append("vol_mom")

        self.feature_names.extend(names)
        return df

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Historical volatility, ATR."""
        names = []

        # Historical volatility (20-day)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        df["hvol_20"] = log_ret.rolling(20).std()
        names.append("hvol_20")

        # ATR (14)
        atr = volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=14
        )
        df["atr_14"] = atr.average_true_range()
        names.append("atr_14")

        self.feature_names.extend(names)
        return df

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ROC, Rate of Change."""
        names = []

        # ROC
        for period in [5, 10, 20]:
            roc = momentum.ROCIndicator(df["close"], window=period)
            df[f"roc_{period}"] = roc.roc()
            names.append(f"roc_{period}")

        self.feature_names.extend(names)
        return df

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend direction, support/resistance."""
        names = []

        # SMA trend (short > long)
        df["sma_5"] = df["close"].rolling(5).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["trend_up"] = (df["sma_5"] > df["sma_20"]).astype(int)
        names.append("trend_up")

        self.feature_names.extend(names)
        return df

    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Skewness, kurtosis, returns stats."""
        names = []

        # Returns stats
        log_ret = np.log(df["close"] / df["close"].shift(1))

        df["ret_mean_5"] = log_ret.rolling(5).mean()
        df["ret_std_5"] = log_ret.rolling(5).std()
        names.extend(["ret_mean_5", "ret_std_5"])

        # Skewness (5-window)
        df["skew_5"] = log_ret.rolling(5).skew()
        names.append("skew_5")

        self.feature_names.extend(names)
        return df
