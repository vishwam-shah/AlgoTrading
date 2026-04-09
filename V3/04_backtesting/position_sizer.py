"""
Position sizing methods for portfolio construction.

Implements:
1. Fixed Fraction: allocate fixed % per signal (baseline)
2. Volatility-Adjusted: ATR-based sizing (standard in retail trading)
3. Kelly Criterion: mathematically optimal sizing (conservative: quarter Kelly)
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class PositionSizer:
    """Calculate optimal position sizes for stocks."""

    @staticmethod
    def fixed_fraction(
        portfolio_value: float,
        n_active_stocks: int,
        allocation_per_stock: float = 0.02,
    ) -> Dict[str, float]:
        """
        Simple fixed allocation per stock.

        Args:
            portfolio_value: Current portfolio value (₹)
            n_active_stocks: Number of stocks with active signals
            allocation_per_stock: Fraction per stock (default 2%)

        Returns:
            {symbol: allocation_weight (sums to 1)}
        """
        if n_active_stocks == 0:
            return {}

        weight_per_stock = allocation_per_stock / n_active_stocks
        return {f"STOCK_{i}": weight_per_stock for i in range(n_active_stocks)}

    @staticmethod
    def volatility_adjusted(
        prices: Dict[str, float],
        atr_values: Dict[str, float],
        confidences: Dict[str, float],
        portfolio_value: float,
        target_risk_pct: float = 0.01,
        stop_multiplier: float = 2.0,
    ) -> Dict[str, float]:
        """
        ATR-based position sizing (standard for retail traders).

        Risk per trade = target_risk_pct * portfolio_value
        Position size = risk_per_trade / (ATR * stop_multiplier)

        Args:
            prices: {symbol: current_price}
            atr_values: {symbol: atr_14d}
            confidences: {symbol: model_confidence [0, 1]}
            portfolio_value: Current portfolio value (₹)
            target_risk_pct: Risk per trade as % of portfolio (default 1%)
            stop_multiplier: ATR multiplier for stop-loss (default 2x)

        Returns:
            {symbol: allocation_weight}
        """
        risk_per_trade = target_risk_pct * portfolio_value
        position_sizes = {}

        for symbol, price in prices.items():
            if symbol not in atr_values or atr_values[symbol] == 0:
                position_sizes[symbol] = 0
                continue

            atr = atr_values[symbol]
            stop_distance = atr * stop_multiplier
            confidence = confidences.get(symbol, 0.5)

            # Shares to buy = risk / stop_distance
            shares = risk_per_trade / stop_distance if stop_distance > 0 else 0

            # Position value
            position_value = shares * price

            # Confidence scaling (high confidence = larger position)
            position_value *= confidence

            position_sizes[symbol] = position_value

        # Normalize to weights
        total_value = sum(position_sizes.values())
        if total_value == 0:
            return {s: 0 for s in position_sizes}

        weights = {s: v / total_value for s, v in position_sizes.items()}

        # Cap at 15% per stock (sector limits)
        weights = {s: min(w, 0.15) for s, w in weights.items()}

        # Renormalize after capping
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {s: w / total_weight for s, w in weights.items()}

        return weights

    @staticmethod
    def kelly_fraction(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25,
        confidence_scale: Dict[str, float] = None,
    ) -> Dict[str, float]:
        """
        Kelly Criterion for position sizing (mathematically optimal growth).

        f* = (p * b - q) / b
        where:
          p = win_rate (from backtest)
          b = avg_win / avg_loss
          q = 1 - p

        Using fractional Kelly (default 0.25) for safety. Full Kelly (1.0) is too aggressive.

        Args:
            win_rate: Win rate from OOS backtest (0.0-1.0)
            avg_win: Average winning trade size (log return)
            avg_loss: Average losing trade size (absolute, positive)
            kelly_fraction: Fraction of full Kelly (default 0.25 = quarter Kelly, safe)
            confidence_scale: {symbol: confidence} to scale per-stock sizing

        Returns:
            {symbol: allocation_weight}
        """
        if avg_loss == 0 or win_rate <= 0.5:
            return {}  # Not profitable enough for Kelly sizing

        # Kelly formula
        b = avg_win / avg_loss
        q = 1.0 - win_rate

        f_star = (win_rate * b - q) / b if b > 0 else 0

        # Use fractional Kelly (e.g., 0.25 * f_star)
        f = max(0, kelly_fraction * f_star)

        # If no per-symbol confidences, return uniform
        if confidence_scale is None:
            return {"default": f}

        # Scale by per-symbol confidence
        weights = {s: f * confidence_scale.get(s, 0.5) for s in confidence_scale}

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {s: w / total_weight for s, w in weights.items()}

        # Cap at 15% per stock
        weights = {s: min(w, 0.15) for s, w in weights.items()}

        # Renormalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {s: w / total_weight for s, w in weights.items()}

        return weights
