"""
================================================================================
STEP 3: FACTOR ANALYSIS
================================================================================
Five-factor model for systematic stock selection.

Factors:
1. VALUE      - Price relative to intrinsic worth (earnings yield, book value)
2. MOMENTUM   - Price trend strength and persistence  
3. QUALITY    - Business quality and stability
4. LOW_VOL    - Risk-adjusted returns (lower volatility often = higher risk-adjusted)
5. SENTIMENT  - Market sentiment from price patterns

Research Basis:
- Fama & French (1993): Value and Size factors
- Carhart (1997): Momentum factor
- Frazzini & Pedersen (2014): Low volatility anomaly
- Baker & Wurgler (2006): Investor Sentiment

Usage:
    from pipeline.step_3_factor_analysis import FactorAnalyzer
    analyzer = FactorAnalyzer()
    factor_scores = analyzer.compute_factors(price_data, features)

Or run directly:
    python pipeline/step_3_factor_analysis.py
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

# Add parent directory to path
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
        """Initialize FactorAnalyzer."""
        self.momentum_lookback = momentum_lookback
        self.volatility_lookback = volatility_lookback
        self.min_history = min_history
        
        # Factor weights (equal by default - can be customized)
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
        logger.info("STEP 3: FACTOR ANALYSIS (5-Factor Model)")
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
        
        # Compute ranks across universe
        self._compute_ranks()
        
        logger.success(f"Factor analysis complete: {len(self.factor_scores)} stocks scored")
        
        # Log top stocks
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
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_price = df['open'].values
        
        metrics = {}
        
        # ======================================================================
        # FACTOR 1: VALUE (price relative to intrinsic worth)
        # ======================================================================
        value_score = self._compute_value_factor(df, metrics)
        
        # ======================================================================
        # FACTOR 2: MOMENTUM (trend strength)
        # ======================================================================
        momentum_score = self._compute_momentum_factor(df, metrics)
        
        # ======================================================================
        # FACTOR 3: QUALITY (stability and consistency)
        # ======================================================================
        quality_score = self._compute_quality_factor(df, metrics)
        
        # ======================================================================
        # FACTOR 4: LOW VOLATILITY (risk-adjusted returns)
        # ======================================================================
        low_vol_score = self._compute_low_vol_factor(df, benchmark, metrics)
        
        # ======================================================================
        # FACTOR 5: SENTIMENT (market psychology from price patterns)
        # ======================================================================
        sentiment_score = self._compute_sentiment_factor(df, features, metrics)
        
        # ======================================================================
        # COMBINED SCORE
        # ======================================================================
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
        """
        VALUE FACTOR: Stocks trading below intrinsic value.
        
        Uses price-based proxies since we don't have fundamental data:
        - Distance from 52-week high (contrarian value)
        - Price to moving average ratio
        - Mean reversion potential
        """
        close = df['close'].values
        
        # Distance from 52-week high (more discount = more value)
        high_52w = df['close'].rolling(252).max().iloc[-1]
        current = close[-1]
        discount_52w = 1 - (current / high_52w)
        metrics['discount_52w'] = float(discount_52w)
        
        # Price vs 200-day MA (below MA = potential value)
        sma_200 = df['close'].rolling(200).mean().iloc[-1]
        price_vs_sma200 = current / sma_200 - 1
        metrics['price_vs_sma200'] = float(price_vs_sma200)
        
        # Mean reversion: how far from 3-year average
        if len(close) >= 756:  # 3 years
            avg_3y = np.mean(close[-756:])
            price_vs_3y = current / avg_3y - 1
        else:
            avg_1y = np.mean(close[-252:])
            price_vs_3y = current / avg_1y - 1
        metrics['price_vs_avg'] = float(price_vs_3y)
        
        # Combine value signals (lower = better value)
        # Normalize to 0-1 where 1 = best value
        value_signals = [
            np.clip(discount_52w * 2, 0, 1),  # 50% discount = max score
            np.clip(-price_vs_sma200 + 0.5, 0, 1),  # Below SMA = higher score
            np.clip(-price_vs_3y + 0.5, 0, 1)  # Below avg = higher score
        ]
        
        value_score = np.mean(value_signals)
        return float(np.clip(value_score, 0, 1))
    
    def _compute_momentum_factor(self, df: pd.DataFrame, metrics: Dict) -> float:
        """
        MOMENTUM FACTOR: Stocks with strong upward trends.
        
        Research shows momentum persists in short-to-medium term.
        Uses multiple timeframes for robustness.
        """
        close = df['close'].values
        
        # 12-month return (skip last month to avoid reversal)
        if len(close) >= 252:
            ret_12m = (close[-21] / close[-252]) - 1
        else:
            ret_12m = (close[-1] / close[0]) - 1
        metrics['return_12m'] = float(ret_12m)
        
        # 6-month return
        if len(close) >= 126:
            ret_6m = (close[-1] / close[-126]) - 1
        else:
            ret_6m = (close[-1] / close[0]) - 1
        metrics['return_6m'] = float(ret_6m)
        
        # 3-month return
        if len(close) >= 63:
            ret_3m = (close[-1] / close[-63]) - 1
        else:
            ret_3m = (close[-1] / close[0]) - 1
        metrics['return_3m'] = float(ret_3m)
        
        # Trend strength: % of days above 50-day MA
        sma_50 = df['close'].rolling(50).mean()
        pct_above_sma = (df['close'] > sma_50).iloc[-63:].mean()
        metrics['pct_above_sma50'] = float(pct_above_sma)
        
        # Combine momentum signals
        # Normalize returns to 0-1 scale (assuming ±50% range)
        mom_signals = [
            np.clip(ret_12m / 0.5 + 0.5, 0, 1),
            np.clip(ret_6m / 0.4 + 0.5, 0, 1),
            np.clip(ret_3m / 0.3 + 0.5, 0, 1),
            pct_above_sma
        ]
        
        momentum_score = np.mean(mom_signals)
        return float(np.clip(momentum_score, 0, 1))
    
    def _compute_quality_factor(self, df: pd.DataFrame, metrics: Dict) -> float:
        """
        QUALITY FACTOR: Stocks with stable, consistent performance.
        
        Uses price stability as proxy for business quality.
        Consistent performers tend to have sustainable returns.
        """
        close = df['close'].values
        returns = pd.Series(close).pct_change().dropna()
        
        # Return consistency: positive months / total months
        monthly_returns = df['close'].resample('M').last().pct_change().dropna()
        positive_months = (monthly_returns > 0).sum()
        consistency = positive_months / len(monthly_returns) if len(monthly_returns) > 0 else 0.5
        metrics['monthly_consistency'] = float(consistency)
        
        # Drawdown recovery: smaller max drawdown = higher quality
        rolling_max = df['close'].cummax()
        drawdown = (df['close'] - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())
        metrics['max_drawdown'] = float(max_dd)
        
        # Return stability: lower standard deviation of returns
        ret_std = returns.std() * np.sqrt(252)  # Annualized
        metrics['return_std'] = float(ret_std)
        
        # Skewness: positive skew is better (more upside surprises)
        skewness = returns.skew()
        metrics['return_skewness'] = float(skewness)
        
        # Earnings stability proxy: how smooth is the price trend?
        # Use R-squared of linear regression
        n = min(252, len(close))
        x = np.arange(n)
        y = close[-n:]
        
        # Fit linear trend
        coeffs = np.polyfit(x, y, 1)
        trend = np.polyval(coeffs, x)
        ss_res = np.sum((y - trend) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        metrics['trend_r_squared'] = float(r_squared)
        
        # Combine quality signals
        quality_signals = [
            consistency,
            np.clip(1 - max_dd * 2, 0, 1),  # 50% drawdown = 0 score
            np.clip(1 - ret_std, 0, 1),  # Lower vol = higher quality
            np.clip((skewness + 1) / 2, 0, 1),  # Normalize skewness
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
        """
        LOW VOLATILITY FACTOR: Lower risk, higher risk-adjusted returns.
        
        The low volatility anomaly shows that less volatile stocks
        often outperform on a risk-adjusted basis.
        """
        close = df['close'].values
        returns = pd.Series(close).pct_change().dropna()
        
        # Realized volatility (annualized)
        vol_60d = returns.iloc[-60:].std() * np.sqrt(252)
        vol_252d = returns.iloc[-252:].std() * np.sqrt(252)
        metrics['volatility_60d'] = float(vol_60d)
        metrics['volatility_252d'] = float(vol_252d)
        
        # Average True Range as % of price
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        atr_pct = tr.rolling(14).mean().iloc[-1] / close[-1]
        metrics['atr_pct'] = float(atr_pct)
        
        # Downside volatility (semi-deviation)
        neg_returns = returns[returns < 0]
        downside_vol = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else vol_252d
        metrics['downside_vol'] = float(downside_vol)
        
        # Beta (if benchmark available)
        if benchmark is not None and len(benchmark) > 0:
            bench_ret = benchmark['close'].pct_change().dropna()
            # Align returns
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
        
        # Combine low-vol signals (lower = better)
        low_vol_signals = [
            np.clip(1 - vol_60d, 0, 1),  # 0% vol = 1, 100% vol = 0
            np.clip(1 - vol_252d, 0, 1),
            np.clip(1 - atr_pct * 10, 0, 1),  # 10% ATR = 0 score
            np.clip(1 - downside_vol, 0, 1),
            np.clip(1 - (beta - 0.5), 0, 1)  # Beta < 0.5 = max, > 1.5 = min
        ]
        
        low_vol_score = np.mean(low_vol_signals)
        return float(np.clip(low_vol_score, 0, 1))
    
    def _compute_sentiment_factor(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        metrics: Dict
    ) -> float:
        """
        SENTIMENT FACTOR: Market psychology from price patterns.
        
        Uses price-derived indicators that correlate with sentiment:
        - Short-term momentum (recent buying/selling pressure)
        - Volume patterns (conviction behind moves)
        - Price acceleration (enthusiasm/fear)
        - Gap patterns (overnight sentiment)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_price = df['open'].values
        
        # Short-term momentum (5-10 days)
        ret_5d = (close[-1] / close[-5]) - 1 if len(close) >= 5 else 0
        ret_10d = (close[-1] / close[-10]) - 1 if len(close) >= 10 else 0
        metrics['momentum_5d'] = float(ret_5d)
        metrics['momentum_10d'] = float(ret_10d)
        
        # Volume-price relationship (bullish vs bearish volume)
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
        
        # Price acceleration (momentum of momentum)
        if len(close) >= 21:
            mom_5d_prev = (close[-6] / close[-11]) - 1 if len(close) >= 11 else 0
            mom_5d_now = (close[-1] / close[-6]) - 1
            acceleration = mom_5d_now - mom_5d_prev
            metrics['price_acceleration'] = float(acceleration)
        else:
            acceleration = 0
            metrics['price_acceleration'] = 0
        
        # Gap analysis (overnight sentiment)
        if len(close) >= 20:
            gaps = (open_price[-19:] - close[-20:-1]) / close[-20:-1]
            avg_gap = np.mean(gaps)
            metrics['avg_gap_20d'] = float(avg_gap)
        else:
            avg_gap = 0
            metrics['avg_gap_20d'] = 0
        
        # RSI proxy (momentum indicator)
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
        
        # Price vs 20-day MA
        sma_20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        price_vs_sma20 = (close[-1] / sma_20) - 1
        metrics['price_vs_sma20'] = float(price_vs_sma20)
        
        # Combine sentiment signals
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
        
        n = len(self.factor_scores)
        
        # Get arrays for each factor
        value_scores = [s.value_score for s in self.factor_scores]
        momentum_scores = [s.momentum_score for s in self.factor_scores]
        quality_scores = [s.quality_score for s in self.factor_scores]
        low_vol_scores = [s.low_vol_score for s in self.factor_scores]
        sentiment_scores = [s.sentiment_score for s in self.factor_scores]
        combined_scores = [s.combined_score for s in self.factor_scores]
        
        def get_ranks(scores):
            """Get ranks (1 = best, highest score)."""
            sorted_idx = np.argsort(scores)[::-1]
            ranks = np.empty_like(sorted_idx)
            ranks[sorted_idx] = np.arange(1, len(scores) + 1)
            return ranks.tolist()
        
        value_ranks = get_ranks(value_scores)
        momentum_ranks = get_ranks(momentum_scores)
        quality_ranks = get_ranks(quality_scores)
        low_vol_ranks = get_ranks(low_vol_scores)
        sentiment_ranks = get_ranks(sentiment_scores)
        combined_ranks = get_ranks(combined_scores)
        
        # Update factor scores with ranks
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
            # Add metrics
            for key, val in s.metrics.items():
                row[f'metric_{key}'] = val
            data.append(row)
        
        return pd.DataFrame(data)


def test_factor_analysis():
    """Test factor analysis with sample data."""
    print("\n" + "=" * 80)
    print("TESTING STEP 3: FACTOR ANALYSIS")
    print("=" * 80)
    
    # Run previous steps
    from step_1_data_collection import DataCollector
    from step_2_feature_engineering import FeatureEngineer
    
    test_symbols = [
        'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK',
        'TCS', 'INFY', 'RELIANCE', 'TATASTEEL', 'HINDUNILVR'
    ]
    
    # Step 1: Collect data
    collector = DataCollector()
    price_data, market_data = collector.collect_all(
        symbols=test_symbols,
        start_date='2022-01-01'
    )
    
    # Step 2: Compute features
    engineer = FeatureEngineer()
    features = engineer.compute_all_features(price_data, market_data)
    
    # Step 3: Factor analysis
    analyzer = FactorAnalyzer()
    
    # Get benchmark for beta calculation
    benchmark = market_data.get('NIFTY50')
    
    factor_scores = analyzer.compute_factors(price_data, features, benchmark)
    
    print("\n" + "=" * 80)
    print("FACTOR ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"\n✓ Stocks analyzed: {len(factor_scores)}")
    print(f"\n✓ Factor weights: {analyzer.factor_weights}")
    
    # Show all factor scores
    print("\n" + "-" * 80)
    print("COMPLETE FACTOR SCORES")
    print("-" * 80)
    print(f"{'Symbol':<12} {'Value':>7} {'Momentum':>8} {'Quality':>8} {'LowVol':>7} {'Sentiment':>9} {'Combined':>8} {'Rank':>5}")
    print("-" * 80)
    
    sorted_scores = sorted(factor_scores, key=lambda x: x.combined_score, reverse=True)
    for s in sorted_scores:
        print(f"{s.symbol:<12} {s.value_score:>7.3f} {s.momentum_score:>8.3f} "
              f"{s.quality_score:>8.3f} {s.low_vol_score:>7.3f} {s.sentiment_score:>9.3f} "
              f"{s.combined_score:>8.3f} {s.combined_rank:>5}")
    
    # Factor correlation analysis
    print("\n" + "-" * 50)
    print("FACTOR CORRELATION ANALYSIS")
    print("-" * 50)
    
    df = analyzer.to_dataframe()
    factor_cols = ['value_score', 'momentum_score', 'quality_score', 'low_vol_score', 'sentiment_score']
    corr = df[factor_cols].corr()
    
    print("\nFactor correlations (low = good diversification):")
    for i, f1 in enumerate(factor_cols):
        for j, f2 in enumerate(factor_cols):
            if i < j:
                print(f"  {f1[:-6]:12} vs {f2[:-6]:12}: {corr.loc[f1, f2]:>6.3f}")
    
    # Validation tests
    print("\n" + "-" * 40)
    print("VALIDATION TESTS")
    print("-" * 40)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: All stocks scored
    tests_total += 1
    if len(factor_scores) >= 8:
        print(f"✓ Test 1: All stocks scored ({len(factor_scores)}/10) - PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 1: All stocks scored ({len(factor_scores)}/10) - FAILED")
    
    # Test 2: Scores in valid range
    tests_total += 1
    scores_valid = True
    for s in factor_scores:
        for attr in ['value_score', 'momentum_score', 'quality_score', 'low_vol_score', 'sentiment_score', 'combined_score']:
            score = getattr(s, attr)
            if score < 0 or score > 1:
                scores_valid = False
                break
    if scores_valid:
        print("✓ Test 2: All scores in [0, 1] range - PASSED")
        tests_passed += 1
    else:
        print("✗ Test 2: All scores in [0, 1] range - FAILED")
    
    # Test 3: Ranks unique
    tests_total += 1
    combined_ranks = [s.combined_rank for s in factor_scores]
    if len(set(combined_ranks)) == len(combined_ranks):
        print("✓ Test 3: Ranks are unique - PASSED")
        tests_passed += 1
    else:
        print("✗ Test 3: Ranks are unique - FAILED")
    
    # Test 4: Factor diversification (correlations < 0.7)
    tests_total += 1
    max_corr = 0
    for i, f1 in enumerate(factor_cols):
        for j, f2 in enumerate(factor_cols):
            if i < j:
                max_corr = max(max_corr, abs(corr.loc[f1, f2]))
    if max_corr < 0.7:
        print(f"✓ Test 4: Factors diversified (max corr: {max_corr:.3f} < 0.7) - PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 4: Factors diversified (max corr: {max_corr:.3f} >= 0.7) - FAILED")
    
    print(f"\n{'=' * 40}")
    print(f"TESTS: {tests_passed}/{tests_total} passed")
    print("=" * 40)
    
    return factor_scores


if __name__ == "__main__":
    test_factor_analysis()
