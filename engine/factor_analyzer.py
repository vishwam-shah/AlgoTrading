"""
================================================================================
FACTOR ANALYSIS - Five-Factor Model
================================================================================
Systematic stock selection using a five-factor model.

Factors:
1. VALUE      - Price relative to intrinsic worth
2. MOMENTUM   - Price trend strength and persistence
3. QUALITY    - Business quality and stability
4. LOW_VOL    - Risk-adjusted returns
5. SENTIMENT  - Market sentiment from price patterns

Research Basis:
- Fama & French (1993): Value and Size factors
- Carhart (1997): Momentum factor
- Frazzini & Pedersen (2014): Low volatility anomaly
- Baker & Wurgler (2006): Investor Sentiment
================================================================================
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class FactorScore:
    """Factor scores for a single stock."""
    symbol: str
    date: datetime

    # Individual factor scores (0-1, higher = better)
    value_score: float
    momentum_score: float
    quality_score: float
    low_vol_score: float
    sentiment_score: float

    # Combined score
    combined_score: float

    # Ranks (1 = best)
    value_rank: int = 0
    momentum_rank: int = 0
    quality_rank: int = 0
    low_vol_rank: int = 0
    sentiment_rank: int = 0
    combined_rank: int = 0

    # Raw metrics for analysis
    metrics: Dict = field(default_factory=dict)


class FactorAnalyzer:
    """
    Five-factor model for stock ranking and selection.

    Computes factor scores for each stock and ranks them within the universe.
    Factors are designed to be uncorrelated for diversification.
    """

    def __init__(
        self,
        momentum_lookback: int = 252,
        volatility_lookback: int = 60,
        min_history: int = 200
    ):
        self.momentum_lookback = momentum_lookback
        self.volatility_lookback = volatility_lookback
        self.min_history = min_history

        self.factor_weights = {
            'value': 0.20,
            'momentum': 0.20,
            'quality': 0.20,
            'low_vol': 0.20,
            'sentiment': 0.20
        }

        self.factor_scores = []

        logger.info("FactorAnalyzer initialized with 5 factors")
        logger.info(f"  Weights: {self.factor_weights}")

    def set_factor_weights(self, weights: Dict[str, float]):
        """Set custom factor weights."""
        self.factor_weights = weights
        logger.info(f"Factor weights updated: {weights}")

    def compute_factors(
        self,
        price_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame] = None,
        benchmark_data: pd.DataFrame = None
    ) -> List[FactorScore]:
        """
        Compute factor scores for all stocks.

        Args:
            price_data: Dict of symbol -> OHLCV DataFrame
            features: Dict of symbol -> features DataFrame (optional)
            benchmark_data: Benchmark index data for relative calculations

        Returns:
            List of FactorScore objects
        """
        logger.info("=" * 60)
        logger.info("FACTOR ANALYSIS (5-Factor Model)")
        logger.info("=" * 60)
        logger.info(f"Analyzing {len(price_data)} stocks")

        self.factor_scores = []

        for symbol, df in price_data.items():
            if len(df) < self.min_history:
                logger.warning(f"  Skipping {symbol}: insufficient history ({len(df)} < {self.min_history})")
                continue

            try:
                feature_df = features.get(symbol) if features else None
                score = self._compute_symbol_factors(symbol, df, feature_df, benchmark_data)
                if score:
                    self.factor_scores.append(score)
            except Exception as e:
                logger.error(f"  Error computing factors for {symbol}: {e}")

        self._compute_ranks()

        logger.success(f"Factor analysis complete: {len(self.factor_scores)} stocks scored")
        self._log_factor_leaders()

        return self.factor_scores

    def _compute_symbol_factors(
        self,
        symbol: str,
        df: pd.DataFrame,
        features: pd.DataFrame,
        benchmark: pd.DataFrame
    ) -> Optional[FactorScore]:
        """Compute all factor scores for a single symbol."""

        close = df['close'].values
        metrics = {}

        value_score = self._compute_value_factor(df, metrics)
        momentum_score = self._compute_momentum_factor(df, metrics)
        quality_score = self._compute_quality_factor(df, metrics)
        low_vol_score = self._compute_low_vol_factor(df, benchmark, metrics)
        sentiment_score = self._compute_sentiment_factor(df, features, metrics)

        combined_score = (
            self.factor_weights['value'] * value_score +
            self.factor_weights['momentum'] * momentum_score +
            self.factor_weights['quality'] * quality_score +
            self.factor_weights['low_vol'] * low_vol_score +
            self.factor_weights['sentiment'] * sentiment_score
        )

        return FactorScore(
            symbol=symbol,
            date=df.index[-1],
            value_score=value_score,
            momentum_score=momentum_score,
            quality_score=quality_score,
            low_vol_score=low_vol_score,
            sentiment_score=sentiment_score,
            combined_score=combined_score,
            metrics=metrics
        )

    def _compute_value_factor(self, df: pd.DataFrame, metrics: Dict) -> float:
        """VALUE FACTOR: Stocks trading below intrinsic value."""
        close = df['close'].values

        high_52w = df['close'].rolling(252).max().iloc[-1]
        current = close[-1]
        discount_52w = 1 - (current / high_52w)
        metrics['discount_52w'] = float(discount_52w)

        sma_200 = df['close'].rolling(200).mean().iloc[-1]
        price_vs_sma200 = current / sma_200 - 1
        metrics['price_vs_sma200'] = float(price_vs_sma200)

        if len(close) >= 756:
            avg_3y = np.mean(close[-756:])
            price_vs_3y = current / avg_3y - 1
        else:
            avg_1y = np.mean(close[-252:])
            price_vs_3y = current / avg_1y - 1
        metrics['price_vs_avg'] = float(price_vs_3y)

        value_signals = [
            np.clip(discount_52w * 2, 0, 1),
            np.clip(-price_vs_sma200 + 0.5, 0, 1),
            np.clip(-price_vs_3y + 0.5, 0, 1)
        ]

        value_score = np.mean(value_signals)
        return float(np.clip(value_score, 0, 1))

    def _compute_momentum_factor(self, df: pd.DataFrame, metrics: Dict) -> float:
        """MOMENTUM FACTOR: Stocks with strong upward trends."""
        close = df['close'].values

        if len(close) >= 252:
            ret_12m = (close[-21] / close[-252]) - 1
        else:
            ret_12m = (close[-1] / close[0]) - 1
        metrics['return_12m'] = float(ret_12m)

        if len(close) >= 126:
            ret_6m = (close[-1] / close[-126]) - 1
        else:
            ret_6m = (close[-1] / close[0]) - 1
        metrics['return_6m'] = float(ret_6m)

        if len(close) >= 63:
            ret_3m = (close[-1] / close[-63]) - 1
        else:
            ret_3m = (close[-1] / close[0]) - 1
        metrics['return_3m'] = float(ret_3m)

        sma_50 = df['close'].rolling(50).mean()
        pct_above_sma = (df['close'] > sma_50).iloc[-63:].mean()
        metrics['pct_above_sma50'] = float(pct_above_sma)

        mom_signals = [
            np.clip(ret_12m / 0.5 + 0.5, 0, 1),
            np.clip(ret_6m / 0.4 + 0.5, 0, 1),
            np.clip(ret_3m / 0.3 + 0.5, 0, 1),
            pct_above_sma
        ]

        momentum_score = np.mean(mom_signals)
        return float(np.clip(momentum_score, 0, 1))

    def _compute_quality_factor(self, df: pd.DataFrame, metrics: Dict) -> float:
        """QUALITY FACTOR: Stocks with stable, consistent performance."""
        close = df['close'].values
        returns = pd.Series(close).pct_change().dropna()

        monthly_returns = df['close'].resample('M').last().pct_change().dropna()
        positive_months = (monthly_returns > 0).sum()
        consistency = positive_months / len(monthly_returns) if len(monthly_returns) > 0 else 0.5
        metrics['monthly_consistency'] = float(consistency)

        rolling_max = df['close'].cummax()
        drawdown = (df['close'] - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())
        metrics['max_drawdown'] = float(max_dd)

        ret_std = returns.std() * np.sqrt(252)
        metrics['return_std'] = float(ret_std)

        skewness = returns.skew()
        metrics['return_skewness'] = float(skewness)

        n = min(252, len(close))
        x = np.arange(n)
        y = close[-n:]
        coeffs = np.polyfit(x, y, 1)
        trend = np.polyval(coeffs, x)
        ss_res = np.sum((y - trend) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        metrics['trend_r_squared'] = float(r_squared)

        quality_signals = [
            consistency,
            np.clip(1 - max_dd * 2, 0, 1),
            np.clip(1 - ret_std, 0, 1),
            np.clip((skewness + 1) / 2, 0, 1),
            np.clip(r_squared, 0, 1)
        ]

        quality_score = np.mean(quality_signals)
        return float(np.clip(quality_score, 0, 1))

    def _compute_low_vol_factor(
        self,
        df: pd.DataFrame,
        benchmark: pd.DataFrame,
        metrics: Dict
    ) -> float:
        """LOW VOLATILITY FACTOR: Lower risk, higher risk-adjusted returns."""
        close = df['close'].values
        returns = pd.Series(close).pct_change().dropna()

        vol_60d = returns.iloc[-60:].std() * np.sqrt(252)
        vol_252d = returns.iloc[-252:].std() * np.sqrt(252)
        metrics['volatility_60d'] = float(vol_60d)
        metrics['volatility_252d'] = float(vol_252d)

        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        atr_pct = tr.rolling(14).mean().iloc[-1] / close[-1]
        metrics['atr_pct'] = float(atr_pct)

        neg_returns = returns[returns < 0]
        downside_vol = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else vol_252d
        metrics['downside_vol'] = float(downside_vol)

        if benchmark is not None and len(benchmark) > 0:
            bench_ret = benchmark['close'].pct_change().dropna()
            aligned = pd.DataFrame({'stock': returns, 'bench': bench_ret}).dropna()
            if len(aligned) >= 60:
                cov = aligned['stock'].cov(aligned['bench'])
                var = aligned['bench'].var()
                beta = cov / var if var > 0 else 1.0
                metrics['beta'] = float(beta)
            else:
                beta = 1.0
                metrics['beta'] = 1.0
        else:
            beta = 1.0
            metrics['beta'] = 1.0

        low_vol_signals = [
            np.clip(1 - vol_60d, 0, 1),
            np.clip(1 - vol_252d, 0, 1),
            np.clip(1 - atr_pct * 10, 0, 1),
            np.clip(1 - downside_vol, 0, 1),
            np.clip(1 - (beta - 0.5), 0, 1)
        ]

        low_vol_score = np.mean(low_vol_signals)
        return float(np.clip(low_vol_score, 0, 1))

    def _compute_sentiment_factor(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        metrics: Dict
    ) -> float:
        """SENTIMENT FACTOR: Market psychology from price patterns."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_price = df['open'].values

        ret_5d = (close[-1] / close[-5]) - 1 if len(close) >= 5 else 0
        ret_10d = (close[-1] / close[-10]) - 1 if len(close) >= 10 else 0
        metrics['momentum_5d'] = float(ret_5d)
        metrics['momentum_10d'] = float(ret_10d)

        if len(close) >= 20:
            returns = np.diff(close[-20:]) / close[-20:-1]
            vol_20d = volume[-19:]
            up_volume = np.sum(vol_20d[returns > 0])
            down_volume = np.sum(vol_20d[returns < 0])
            volume_ratio = up_volume / (down_volume + 1) if down_volume > 0 else 2.0
            metrics['volume_sentiment_ratio'] = float(volume_ratio)
        else:
            volume_ratio = 1.0
            metrics['volume_sentiment_ratio'] = 1.0

        if len(close) >= 21:
            mom_5d_prev = (close[-6] / close[-11]) - 1 if len(close) >= 11 else 0
            mom_5d_now = (close[-1] / close[-6]) - 1
            acceleration = mom_5d_now - mom_5d_prev
            metrics['price_acceleration'] = float(acceleration)
        else:
            acceleration = 0
            metrics['price_acceleration'] = 0

        if len(close) >= 20:
            gaps = (open_price[-19:] - close[-20:-1]) / close[-20:-1]
            avg_gap = np.mean(gaps)
            metrics['avg_gap_20d'] = float(avg_gap)
        else:
            avg_gap = 0
            metrics['avg_gap_20d'] = 0

        if len(close) >= 15:
            returns_14 = np.diff(close[-15:]) / close[-15:-1]
            gains = returns_14[returns_14 > 0]
            losses = -returns_14[returns_14 < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            metrics['rsi_proxy'] = float(rsi)
        else:
            rsi = 50
            metrics['rsi_proxy'] = 50

        sma_20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        price_vs_sma20 = (close[-1] / sma_20) - 1
        metrics['price_vs_sma20'] = float(price_vs_sma20)

        sentiment_signals = [
            np.clip(0.5 + (ret_5d + ret_10d) / 0.20, 0, 1),
            np.clip(volume_ratio / 2, 0, 1),
            np.clip(0.5 + acceleration / 0.10, 0, 1),
            np.clip(0.5 + avg_gap / 0.02, 0, 1),
            rsi / 100,
            np.clip(0.5 + price_vs_sma20 / 0.10, 0, 1)
        ]

        sentiment_score = np.mean(sentiment_signals)
        return float(np.clip(sentiment_score, 0, 1))

    def _compute_ranks(self):
        """Compute percentile ranks across the universe."""
        if not self.factor_scores:
            return

        def get_ranks(scores):
            sorted_idx = np.argsort(scores)[::-1]
            ranks = np.empty_like(sorted_idx)
            ranks[sorted_idx] = np.arange(1, len(scores) + 1)
            return ranks.tolist()

        value_ranks = get_ranks([s.value_score for s in self.factor_scores])
        momentum_ranks = get_ranks([s.momentum_score for s in self.factor_scores])
        quality_ranks = get_ranks([s.quality_score for s in self.factor_scores])
        low_vol_ranks = get_ranks([s.low_vol_score for s in self.factor_scores])
        sentiment_ranks = get_ranks([s.sentiment_score for s in self.factor_scores])
        combined_ranks = get_ranks([s.combined_score for s in self.factor_scores])

        for i, score in enumerate(self.factor_scores):
            score.value_rank = value_ranks[i]
            score.momentum_rank = momentum_ranks[i]
            score.quality_rank = quality_ranks[i]
            score.low_vol_rank = low_vol_ranks[i]
            score.sentiment_rank = sentiment_ranks[i]
            score.combined_rank = combined_ranks[i]

    def _log_factor_leaders(self):
        """Log top stocks by each factor."""
        if not self.factor_scores:
            return

        logger.info("\n" + "-" * 50)
        logger.info("FACTOR LEADERS (Top 5 each)")
        logger.info("-" * 50)

        factors = ['value', 'momentum', 'quality', 'low_vol', 'sentiment', 'combined']

        for factor in factors:
            top_5 = self.get_top_stocks(self.factor_scores, n=5, factor=factor)
            symbols = [s.symbol for s in top_5]
            scores = [getattr(s, f'{factor}_score') for s in top_5]

            logger.info(f"\n{factor.upper()}:")
            for sym, score in zip(symbols, scores):
                logger.info(f"  {sym:12} {score:.3f}")

    def get_top_stocks(
        self,
        scores: List[FactorScore] = None,
        n: int = 10,
        factor: str = 'combined'
    ) -> List[FactorScore]:
        """Get top N stocks by specified factor."""
        scores = scores or self.factor_scores

        if factor == 'combined':
            sorted_scores = sorted(scores, key=lambda x: x.combined_score, reverse=True)
        elif factor == 'value':
            sorted_scores = sorted(scores, key=lambda x: x.value_score, reverse=True)
        elif factor == 'momentum':
            sorted_scores = sorted(scores, key=lambda x: x.momentum_score, reverse=True)
        elif factor == 'quality':
            sorted_scores = sorted(scores, key=lambda x: x.quality_score, reverse=True)
        elif factor == 'low_vol':
            sorted_scores = sorted(scores, key=lambda x: x.low_vol_score, reverse=True)
        elif factor == 'sentiment':
            sorted_scores = sorted(scores, key=lambda x: x.sentiment_score, reverse=True)
        else:
            raise ValueError(f"Unknown factor: {factor}")

        return sorted_scores[:n]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert factor scores to DataFrame."""
        data = []
        for s in self.factor_scores:
            row = {
                'symbol': s.symbol,
                'date': s.date,
                'value_score': s.value_score,
                'momentum_score': s.momentum_score,
                'quality_score': s.quality_score,
                'low_vol_score': s.low_vol_score,
                'sentiment_score': s.sentiment_score,
                'combined_score': s.combined_score,
                'value_rank': s.value_rank,
                'momentum_rank': s.momentum_rank,
                'quality_rank': s.quality_rank,
                'low_vol_rank': s.low_vol_rank,
                'sentiment_rank': s.sentiment_rank,
                'combined_rank': s.combined_rank,
            }
            for key, val in s.metrics.items():
                row[f'metric_{key}'] = val
            data.append(row)

        return pd.DataFrame(data)
