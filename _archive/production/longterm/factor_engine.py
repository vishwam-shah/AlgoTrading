"""
================================================================================
FACTOR ENGINE - Long-Term Equity Portfolio
================================================================================
Computes research-backed factors for stock ranking:
- Value: Earnings yield, Book yield, Dividend yield
- Momentum: 12-1 month momentum, risk-adjusted momentum
- Quality: ROE, profit margins, financial stability
- Low Volatility: Historical vol, beta, drawdown

Based on: Fama-French, Jegadeesh-Titman, Novy-Marx, Ang et al.
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FactorScores:
    """Factor scores for a single stock."""
    symbol: str
    date: datetime

    # Individual factors (5 factors now including sentiment)
    value_score: float
    momentum_score: float
    quality_score: float
    low_vol_score: float

    # Combined score
    combined_score: float

    # Raw metrics (for analysis)
    metrics: Dict[str, float]

    # Sentiment factor (with default since it's new)
    sentiment_score: float = 0.5  # Price-derived sentiment factor

    # Ranks (1 = best) - all have defaults
    value_rank: int = 0
    momentum_rank: int = 0
    quality_rank: int = 0
    low_vol_rank: int = 0
    sentiment_rank: int = 0
    combined_rank: int = 0


class FactorEngine:
    """
    Compute factor scores for stock universe.

    Factors are computed using ONLY price data and basic fundamentals.
    For NSE stocks, we derive what we can from price action.
    """

    def __init__(
        self,
        momentum_lookback: int = 252,  # 12 months
        momentum_skip: int = 21,        # Skip last month (reversal)
        volatility_lookback: int = 252,
        min_history: int = 252          # Minimum trading days needed
    ):
        self.momentum_lookback = momentum_lookback
        self.momentum_skip = momentum_skip
        self.volatility_lookback = volatility_lookback
        self.min_history = min_history

        # Factor weights (5 factors - equal by default)
        # Sentiment added as 5th factor based on price-derived indicators
        self.factor_weights = {
            'value': 0.20,
            'momentum': 0.20,
            'quality': 0.20,
            'low_vol': 0.20,
            'sentiment': 0.20  # NEW: Price-derived market sentiment
        }

        logger.info("FactorEngine initialized with 5 factors (including Sentiment)")

    def set_factor_weights(self, weights: Dict[str, float]):
        """Set custom factor weights."""
        assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to 1"
        self.factor_weights = weights
        logger.info(f"Factor weights updated: {weights}")

    def compute_all_factors(
        self,
        price_data: Dict[str, pd.DataFrame],
        fundamental_data: Optional[Dict[str, Dict]] = None,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> List[FactorScores]:
        """
        Compute factor scores for all stocks.

        Args:
            price_data: Dict of {symbol: DataFrame with OHLCV}
            fundamental_data: Optional dict of {symbol: {pe, pb, roe, etc.}}
            benchmark_data: Optional DataFrame for NIFTY50 (for beta calculation)

        Returns:
            List of FactorScores for all stocks
        """
        logger.info(f"Computing factors for {len(price_data)} stocks...")

        all_scores = []

        for symbol, df in price_data.items():
            try:
                if len(df) < self.min_history:
                    logger.warning(f"{symbol}: Insufficient history ({len(df)} < {self.min_history})")
                    continue

                # Get fundamental data if available
                fundamentals = fundamental_data.get(symbol, {}) if fundamental_data else {}

                # Compute individual factor scores (5 factors)
                value = self._compute_value_factor(df, fundamentals)
                momentum = self._compute_momentum_factor(df)
                quality = self._compute_quality_factor(df, fundamentals)
                low_vol = self._compute_low_vol_factor(df, benchmark_data)
                sentiment = self._compute_sentiment_factor(df)  # NEW: Price-derived sentiment

                # Store raw metrics from all 5 factors
                metrics = {**value['metrics'], **momentum['metrics'],
                          **quality['metrics'], **low_vol['metrics'],
                          **sentiment['metrics']}

                # Combined score (5 factors now)
                combined = (
                    self.factor_weights['value'] * value['score'] +
                    self.factor_weights['momentum'] * momentum['score'] +
                    self.factor_weights['quality'] * quality['score'] +
                    self.factor_weights['low_vol'] * low_vol['score'] +
                    self.factor_weights['sentiment'] * sentiment['score']
                )

                # Get latest date
                if 'timestamp' in df.columns:
                    latest_date = pd.to_datetime(df['timestamp'].iloc[-1])
                else:
                    latest_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()

                score = FactorScores(
                    symbol=symbol,
                    date=latest_date,
                    value_score=value['score'],
                    momentum_score=momentum['score'],
                    quality_score=quality['score'],
                    low_vol_score=low_vol['score'],
                    sentiment_score=sentiment['score'],  # NEW
                    combined_score=combined,
                    metrics=metrics
                )

                all_scores.append(score)

            except Exception as e:
                logger.error(f"{symbol}: Factor computation failed - {e}")
                continue

        # Compute ranks across universe
        all_scores = self._compute_ranks(all_scores)

        logger.success(f"Computed factors for {len(all_scores)} stocks")
        return all_scores

    def _compute_value_factor(
        self,
        df: pd.DataFrame,
        fundamentals: Dict
    ) -> Dict:
        """
        Compute value factor.

        If fundamentals available: Use P/E, P/B, dividend yield
        If not: Use price-based proxies (mean reversion signals)
        """
        metrics = {}

        close = df['close'].values

        # --- Fundamental-based (if available) ---
        if fundamentals:
            pe = fundamentals.get('pe_ratio', None)
            pb = fundamentals.get('pb_ratio', None)
            div_yield = fundamentals.get('dividend_yield', None)

            if pe and pe > 0:
                metrics['earnings_yield'] = 1.0 / pe
            if pb and pb > 0:
                metrics['book_yield'] = 1.0 / pb
            if div_yield:
                metrics['dividend_yield'] = div_yield

        # --- Price-based proxies (always computed) ---
        # 52-week high ratio (lower = more "value")
        high_52w = df['high'].rolling(252).max().iloc[-1]
        metrics['pct_from_52w_high'] = (close[-1] - high_52w) / high_52w

        # Price to 200-day SMA ratio (lower = more "value")
        sma_200 = df['close'].rolling(200).mean().iloc[-1]
        metrics['price_to_sma200'] = close[-1] / sma_200 if sma_200 > 0 else 1.0

        # Long-term mean reversion (5-year avg if available)
        if len(df) > 1260:  # 5 years
            avg_5y = df['close'].iloc[-1260:].mean()
            metrics['price_to_5y_avg'] = close[-1] / avg_5y

        # Compute score (normalize to 0-1 range, higher = more value)
        # For value, LOWER price ratios are better
        score_components = []

        if 'earnings_yield' in metrics:
            score_components.append(metrics['earnings_yield'] * 10)  # Scale up
        if 'book_yield' in metrics:
            score_components.append(metrics['book_yield'] * 5)
        if 'dividend_yield' in metrics:
            score_components.append(metrics['dividend_yield'] * 20)

        # Price-based (inverted - lower price = higher score)
        score_components.append(-metrics['pct_from_52w_high'])  # Negative of drawdown
        score_components.append(2 - metrics['price_to_sma200'])  # Inverted

        score = np.mean(score_components) if score_components else 0.5

        return {'score': float(score), 'metrics': metrics}

    def _compute_momentum_factor(self, df: pd.DataFrame) -> Dict:
        """
        Compute momentum factor.

        Primary: 12-1 month momentum (skip last month to avoid reversal)
        Secondary: Risk-adjusted momentum (Sharpe of returns)
        """
        metrics = {}

        close = df['close'].values
        returns = df['close'].pct_change().values

        # 12-month return
        if len(close) >= 252:
            ret_12m = (close[-1] / close[-252]) - 1
            metrics['return_12m'] = ret_12m
        else:
            ret_12m = (close[-1] / close[0]) - 1
            metrics['return_12m'] = ret_12m

        # 1-month return (to skip)
        if len(close) >= 21:
            ret_1m = (close[-1] / close[-21]) - 1
            metrics['return_1m'] = ret_1m
        else:
            ret_1m = 0
            metrics['return_1m'] = ret_1m

        # 12-1 momentum (classic Jegadeesh-Titman)
        momentum_12_1 = ret_12m - ret_1m
        metrics['momentum_12_1'] = momentum_12_1

        # 6-month momentum
        if len(close) >= 126:
            ret_6m = (close[-1] / close[-126]) - 1
            metrics['return_6m'] = ret_6m

        # 3-month momentum
        if len(close) >= 63:
            ret_3m = (close[-1] / close[-63]) - 1
            metrics['return_3m'] = ret_3m

        # Risk-adjusted momentum (Sharpe-like)
        if len(returns) >= 252:
            recent_returns = returns[-252:]
            vol = np.nanstd(recent_returns) * np.sqrt(252)
            sharpe_12m = ret_12m / vol if vol > 0 else 0
            metrics['sharpe_12m'] = sharpe_12m
        else:
            metrics['sharpe_12m'] = 0

        # Momentum consistency (% of positive months)
        if len(df) >= 252:
            monthly_rets = df['close'].resample('M').last().pct_change().dropna()
            if len(monthly_rets) >= 6:
                metrics['momentum_consistency'] = (monthly_rets > 0).mean()

        # Score: Higher momentum = higher score
        score = (
            0.5 * (momentum_12_1 + 1) / 2 +  # Normalize to 0-1
            0.3 * (metrics.get('sharpe_12m', 0) + 2) / 4 +  # Sharpe typically -2 to 2
            0.2 * metrics.get('momentum_consistency', 0.5)
        )

        score = np.clip(score, 0, 1)

        return {'score': float(score), 'metrics': metrics}

    def _compute_quality_factor(
        self,
        df: pd.DataFrame,
        fundamentals: Dict
    ) -> Dict:
        """
        Compute quality factor.

        Fundamental: ROE, profit margins, debt ratios
        Price-based: Earnings stability proxy, trend strength
        """
        metrics = {}

        # --- Fundamental-based (if available) ---
        if fundamentals:
            roe = fundamentals.get('roe', None)
            roa = fundamentals.get('roa', None)
            debt_equity = fundamentals.get('debt_to_equity', None)
            gross_margin = fundamentals.get('gross_margin', None)

            if roe is not None:
                metrics['roe'] = roe
            if roa is not None:
                metrics['roa'] = roa
            if debt_equity is not None:
                metrics['debt_to_equity'] = debt_equity
            if gross_margin is not None:
                metrics['gross_margin'] = gross_margin

        # --- Price-based quality proxies ---
        close = df['close'].values

        # Earnings stability proxy: lower volatility of returns = more stable
        returns = df['close'].pct_change().dropna()
        if len(returns) >= 252:
            # Use volatility of 3-month returns as stability proxy
            quarterly_returns = df['close'].pct_change(63).dropna()
            if len(quarterly_returns) >= 4:
                earnings_stability = 1 / (quarterly_returns.std() + 0.01)
                metrics['earnings_stability_proxy'] = float(earnings_stability)

        # Trend strength (R-squared of price vs time)
        if len(close) >= 252:
            x = np.arange(252)
            y = close[-252:]
            correlation = np.corrcoef(x, y)[0, 1]
            r_squared = correlation ** 2
            metrics['trend_strength'] = r_squared

        # Price stability (lower max drawdown = higher quality)
        cummax = df['close'].cummax()
        drawdown = (df['close'] - cummax) / cummax
        max_dd = drawdown.min()
        metrics['max_drawdown_1y'] = float(max_dd)

        # Score computation
        score_components = []

        if 'roe' in metrics and metrics['roe'] > 0:
            score_components.append(min(metrics['roe'] / 0.30, 1.0))  # Cap at 30% ROE
        if 'gross_margin' in metrics:
            score_components.append(metrics['gross_margin'])
        if 'debt_to_equity' in metrics:
            score_components.append(1 - min(metrics['debt_to_equity'] / 2, 1.0))  # Lower is better

        # Price-based components
        if 'earnings_stability_proxy' in metrics:
            score_components.append(min(metrics['earnings_stability_proxy'] / 10, 1.0))
        if 'trend_strength' in metrics:
            score_components.append(metrics['trend_strength'])

        # Max drawdown (inverted)
        score_components.append(1 + metrics['max_drawdown_1y'])  # DD is negative

        score = np.mean(score_components) if score_components else 0.5
        score = np.clip(score, 0, 1)

        return {'score': float(score), 'metrics': metrics}

    def _compute_low_vol_factor(
        self,
        df: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Compute low volatility factor.

        Lower volatility = HIGHER score (anomaly: low vol outperforms)
        """
        metrics = {}

        returns = df['close'].pct_change().dropna()

        # Historical volatility (annualized)
        if len(returns) >= 252:
            vol_1y = returns.iloc[-252:].std() * np.sqrt(252)
        else:
            vol_1y = returns.std() * np.sqrt(252)
        metrics['volatility_1y'] = float(vol_1y)

        # Beta (if benchmark provided)
        if benchmark_data is not None and len(benchmark_data) >= 252:
            try:
                bench_returns = benchmark_data['close'].pct_change().dropna()

                # Align dates
                common_dates = returns.index.intersection(bench_returns.index)
                if len(common_dates) >= 60:
                    stock_ret = returns.loc[common_dates]
                    bench_ret = bench_returns.loc[common_dates]

                    covariance = np.cov(stock_ret, bench_ret)[0, 1]
                    benchmark_var = np.var(bench_ret)

                    beta = covariance / benchmark_var if benchmark_var > 0 else 1.0
                    metrics['beta'] = float(beta)
            except:
                metrics['beta'] = 1.0
        else:
            metrics['beta'] = 1.0

        # Downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 20:
            downside_dev = negative_returns.std() * np.sqrt(252)
            metrics['downside_deviation'] = float(downside_dev)

        # Max drawdown
        cummax = df['close'].cummax()
        drawdown = (df['close'] - cummax) / cummax
        metrics['max_drawdown'] = float(drawdown.min())

        # Score: LOWER volatility = HIGHER score (inverted)
        # Typical vol range: 15% to 50%
        vol_score = 1 - (vol_1y - 0.10) / 0.50  # Normalize: 10% vol = 1.0, 60% vol = 0.0
        beta_score = 1 - (metrics['beta'] - 0.5) / 1.0  # Beta 0.5 = 1.0, Beta 1.5 = 0.0
        dd_score = 1 + metrics['max_drawdown']  # DD is negative

        score = 0.4 * vol_score + 0.3 * beta_score + 0.3 * dd_score
        score = np.clip(score, 0, 1)

        return {'score': float(score), 'metrics': metrics}

    def _compute_sentiment_factor(self, df: pd.DataFrame) -> Dict:
        """
        Compute SENTIMENT factor from price-derived indicators.

        Since we can't get historical news sentiment, we use price patterns
        that historically correlate with market sentiment:

        1. Short-term momentum (1-2 weeks) - recent price action reflects sentiment
        2. Volume patterns - high volume on up days = bullish sentiment
        3. Price acceleration - 2nd derivative shows enthusiasm/fear
        4. Gap patterns - opening gaps indicate overnight sentiment shifts
        5. Relative strength - outperforming = positive sentiment

        Research basis:
        - Baker & Wurgler (2006): "Investor Sentiment and the Cross-Section of Stock Returns"
        - Lee, Shleifer & Thaler (1991): "Investor Sentiment and Closed-End Fund Discount"
        """
        metrics = {}

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_price = df['open'].values

        # --- 1. SHORT-TERM MOMENTUM (recent sentiment) ---
        # 5-day momentum (1 week)
        if len(close) >= 5:
            ret_5d = (close[-1] / close[-5]) - 1
            metrics['momentum_5d'] = float(ret_5d)
        else:
            ret_5d = 0
            metrics['momentum_5d'] = 0

        # 10-day momentum (2 weeks)
        if len(close) >= 10:
            ret_10d = (close[-1] / close[-10]) - 1
            metrics['momentum_10d'] = float(ret_10d)
        else:
            ret_10d = 0
            metrics['momentum_10d'] = 0

        # --- 2. VOLUME-PRICE RELATIONSHIP (bullish/bearish volume) ---
        if len(close) >= 20:
            returns = np.diff(close) / close[:-1]
            vol_20d = volume[-20:]
            ret_20d = returns[-19:]

            # Up-volume vs down-volume ratio
            up_volume = np.sum(vol_20d[1:][ret_20d > 0])
            down_volume = np.sum(vol_20d[1:][ret_20d < 0])

            volume_ratio = up_volume / (down_volume + 1)  # Avoid div by zero
            metrics['volume_sentiment_ratio'] = float(volume_ratio)

            # Volume trend (is volume increasing?)
            vol_ma_short = np.mean(vol_20d[-5:])
            vol_ma_long = np.mean(vol_20d)
            metrics['volume_trend'] = float(vol_ma_short / vol_ma_long) if vol_ma_long > 0 else 1.0
        else:
            metrics['volume_sentiment_ratio'] = 1.0
            metrics['volume_trend'] = 1.0

        # --- 3. PRICE ACCELERATION (2nd derivative = enthusiasm) ---
        if len(close) >= 21:
            # 1st derivative: momentum
            mom_5d_prev = (close[-6] / close[-11]) - 1 if len(close) >= 11 else 0
            mom_5d_now = (close[-1] / close[-6]) - 1

            # 2nd derivative: acceleration
            acceleration = mom_5d_now - mom_5d_prev
            metrics['price_acceleration'] = float(acceleration)
        else:
            metrics['price_acceleration'] = 0

        # --- 4. GAP ANALYSIS (overnight sentiment) ---
        if len(close) >= 20:
            gaps = (open_price[1:] - close[:-1]) / close[:-1]
            recent_gaps = gaps[-20:]

            # Average gap direction (positive = bullish overnight sentiment)
            avg_gap = np.mean(recent_gaps)
            metrics['avg_gap_20d'] = float(avg_gap)

            # Gap fill tendency (unfilled gaps = strong sentiment)
            gap_unfilled_count = 0
            for i in range(-20, 0):
                if i < -1:
                    gap = open_price[i+1] - close[i]
                    if gap > 0 and low[i+1] > close[i]:  # Bullish gap unfilled
                        gap_unfilled_count += 1
                    elif gap < 0 and high[i+1] < close[i]:  # Bearish gap unfilled
                        gap_unfilled_count -= 1
            metrics['unfilled_gaps_score'] = gap_unfilled_count / 20
        else:
            metrics['avg_gap_20d'] = 0
            metrics['unfilled_gaps_score'] = 0

        # --- 5. RELATIVE STRENGTH (vs own history) ---
        if len(close) >= 63:
            # Price vs 20-day MA (above = bullish)
            sma_20 = np.mean(close[-20:])
            price_vs_sma20 = (close[-1] / sma_20) - 1
            metrics['price_vs_sma20'] = float(price_vs_sma20)

            # RSI-like indicator (without ta library)
            returns_63 = np.diff(close[-64:]) / close[-64:-1]
            gains = returns_63[returns_63 > 0]
            losses = -returns_63[returns_63 < 0]

            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.001

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            metrics['rsi_proxy'] = float(rsi)
        else:
            metrics['price_vs_sma20'] = 0
            metrics['rsi_proxy'] = 50

        # --- 6. SENTIMENT REGIME (trending vs mean-reverting) ---
        if len(close) >= 63:
            # Hurst exponent proxy (autocorrelation)
            returns = np.diff(close[-64:]) / close[-64:-1]
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            metrics['return_autocorrelation'] = float(autocorr)

            # Positive autocorrelation = trending (momentum sentiment)
            # Negative = mean-reverting (contrarian sentiment)
        else:
            metrics['return_autocorrelation'] = 0

        # === COMPUTE FINAL SENTIMENT SCORE ===
        # Normalize each component to 0-1 range and combine

        score_components = []

        # Short-term momentum (higher = more bullish)
        mom_score = 0.5 + (ret_5d + ret_10d) / 0.20  # ±10% range
        score_components.append(np.clip(mom_score, 0, 1))

        # Volume ratio (>1 = bullish)
        vol_score = np.clip(metrics['volume_sentiment_ratio'] / 2, 0, 1)
        score_components.append(vol_score)

        # Acceleration (positive = bullish momentum building)
        accel_score = 0.5 + metrics['price_acceleration'] / 0.10
        score_components.append(np.clip(accel_score, 0, 1))

        # Gap sentiment
        gap_score = 0.5 + metrics['avg_gap_20d'] / 0.02
        score_components.append(np.clip(gap_score, 0, 1))

        # RSI-based (50 = neutral, >70 = overbought but bullish, <30 = oversold)
        rsi_score = metrics['rsi_proxy'] / 100
        score_components.append(rsi_score)

        # Price vs SMA (above = bullish)
        sma_score = 0.5 + metrics['price_vs_sma20'] / 0.10
        score_components.append(np.clip(sma_score, 0, 1))

        # Final sentiment score
        score = np.mean(score_components)
        score = np.clip(score, 0, 1)

        return {'score': float(score), 'metrics': metrics}

    def _compute_ranks(self, scores: List[FactorScores]) -> List[FactorScores]:
        """Compute percentile ranks across universe."""
        if not scores:
            return scores

        n = len(scores)

        # Get arrays for ranking (5 factors)
        value_scores = [s.value_score for s in scores]
        momentum_scores = [s.momentum_score for s in scores]
        quality_scores = [s.quality_score for s in scores]
        low_vol_scores = [s.low_vol_score for s in scores]
        sentiment_scores = [s.sentiment_score for s in scores]  # NEW
        combined_scores = [s.combined_score for s in scores]

        # Compute ranks (higher score = lower rank number = better)
        def get_ranks(values):
            sorted_indices = np.argsort(values)[::-1]  # Descending
            ranks = np.empty(len(values), dtype=int)
            ranks[sorted_indices] = np.arange(1, len(values) + 1)
            return ranks

        value_ranks = get_ranks(value_scores)
        momentum_ranks = get_ranks(momentum_scores)
        quality_ranks = get_ranks(quality_scores)
        low_vol_ranks = get_ranks(low_vol_scores)
        sentiment_ranks = get_ranks(sentiment_scores)  # NEW
        combined_ranks = get_ranks(combined_scores)

        # Update scores with ranks
        for i, score in enumerate(scores):
            score.value_rank = value_ranks[i]
            score.momentum_rank = momentum_ranks[i]
            score.quality_rank = quality_ranks[i]
            score.low_vol_rank = low_vol_ranks[i]
            score.sentiment_rank = sentiment_ranks[i]  # NEW
            score.combined_rank = combined_ranks[i]

        return scores

    def get_top_stocks(
        self,
        scores: List[FactorScores],
        n: int = 20,
        factor: str = 'combined'
    ) -> List[FactorScores]:
        """Get top N stocks by specified factor."""

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
        elif factor == 'sentiment':  # NEW
            sorted_scores = sorted(scores, key=lambda x: x.sentiment_score, reverse=True)
        else:
            raise ValueError(f"Unknown factor: {factor}")

        return sorted_scores[:n]

    def to_dataframe(self, scores: List[FactorScores]) -> pd.DataFrame:
        """Convert factor scores to DataFrame."""
        data = []
        for s in scores:
            row = {
                'symbol': s.symbol,
                'date': s.date,
                'value_score': s.value_score,
                'momentum_score': s.momentum_score,
                'quality_score': s.quality_score,
                'low_vol_score': s.low_vol_score,
                'sentiment_score': s.sentiment_score,  # NEW
                'combined_score': s.combined_score,
                'value_rank': s.value_rank,
                'momentum_rank': s.momentum_rank,
                'quality_rank': s.quality_rank,
                'low_vol_rank': s.low_vol_rank,
                'sentiment_rank': s.sentiment_rank,  # NEW
                'combined_rank': s.combined_rank,
            }
            # Add raw metrics
            for key, val in s.metrics.items():
                row[f'metric_{key}'] = val
            data.append(row)

        return pd.DataFrame(data)

    def generate_report(self, scores: List[FactorScores]) -> str:
        """Generate factor analysis report."""
        lines = []
        lines.append("=" * 80)
        lines.append("FACTOR ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append(f"Universe: {len(scores)} stocks")
        lines.append(f"Factor weights: {self.factor_weights}")
        lines.append("")

        # Top stocks by combined score
        lines.append("-" * 40)
        lines.append("TOP 10 STOCKS (Combined Factor)")
        lines.append("-" * 40)
        top_10 = self.get_top_stocks(scores, n=10, factor='combined')
        for i, s in enumerate(top_10, 1):
            lines.append(f"{i:2}. {s.symbol:12} Score: {s.combined_score:.3f} "
                        f"(V:{s.value_rank:2} M:{s.momentum_rank:2} "
                        f"Q:{s.quality_rank:2} L:{s.low_vol_rank:2} S:{s.sentiment_rank:2})")

        lines.append("")

        # Factor leaders (all 5 factors)
        for factor in ['value', 'momentum', 'quality', 'low_vol', 'sentiment']:
            lines.append(f"\nTop 5 by {factor.upper()}:")
            top_5 = self.get_top_stocks(scores, n=5, factor=factor)
            for s in top_5:
                score = getattr(s, f'{factor}_score')
                lines.append(f"  {s.symbol:12} {score:.3f}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf

    symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS']

    price_data = {}
    for sym in symbols:
        df = yf.download(sym, period='2y', progress=False)
        if len(df) > 0:
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            price_data[sym.replace('.NS', '')] = df

    engine = FactorEngine()
    scores = engine.compute_all_factors(price_data)

    print(engine.generate_report(scores))
