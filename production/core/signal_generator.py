"""
================================================================================
TRADING SIGNAL GENERATOR
================================================================================
Generates actionable trading signals from model predictions.

Features:
- Multi-factor signal scoring
- Risk-adjusted position sizing
- Support/Resistance-based targets
- Portfolio-level optimization
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from production.models import ModelPrediction


@dataclass
class TradingSignal:
    """Trading signal with full context"""
    symbol: str
    timestamp: datetime

    # Action
    action: str  # 'BUY', 'SELL', 'HOLD'
    strength: str  # 'STRONG', 'MODERATE', 'WEAK'

    # Prices
    current_price: float
    target_price: float
    stop_loss: float

    # Risk metrics
    expected_return_pct: float
    risk_pct: float
    risk_reward_ratio: float

    # Confidence
    confidence: float
    direction_probability: float

    # Position sizing
    suggested_position_pct: float  # % of portfolio
    suggested_quantity: int = 0

    # Additional context
    sector: str = ''
    market_regime: str = ''
    technical_summary: str = ''


@dataclass
class PortfolioSignals:
    """Portfolio-level signal aggregation"""
    timestamp: datetime
    signals: List[TradingSignal]
    buy_signals: List[TradingSignal] = field(default_factory=list)
    sell_signals: List[TradingSignal] = field(default_factory=list)
    total_suggested_allocation: float = 0.0
    market_exposure: str = ''  # 'BULLISH', 'BEARISH', 'NEUTRAL'


class SignalGenerator:
    """
    Generates trading signals from model predictions.

    Key features:
    1. Multi-factor scoring (model + technical + sentiment)
    2. Risk-based filtering (ATR-based stops)
    3. Position sizing based on confidence and volatility
    4. Portfolio-level constraints
    """

    def __init__(
        self,
        min_confidence: float = 0.60,  # Increased from 0.55 to reduce false signals
        min_risk_reward: float = 1.5,
        max_position_pct: float = 0.10,  # Max 10% per position
        max_portfolio_risk: float = 0.02,  # Max 2% portfolio risk per trade
        volume_confirmation: bool = True,  # Require volume confirmation
        volume_threshold: float = 1.2  # Volume must be 1.2x 20-day average
    ):
        self.min_confidence = min_confidence
        self.min_risk_reward = min_risk_reward
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.volume_confirmation = volume_confirmation
        self.volume_threshold = volume_threshold

    def generate_signal(
        self,
        symbol: str,
        prediction: ModelPrediction,
        current_data: pd.Series,
        portfolio_value: float = 100000
    ) -> TradingSignal:
        """
        Generate trading signal for a single stock.

        Args:
            symbol: Stock symbol
            prediction: Model prediction
            current_data: Current row of price data with features
            portfolio_value: Total portfolio value

        Returns:
            TradingSignal object
        """
        current_price = current_data['close']
        current_time = datetime.now()

        # Get ATR for stop loss calculation
        atr = current_data.get('atr_14', current_price * 0.02)
        atr_pct = atr / current_price

        # Volume confirmation check
        volume_confirmed = True
        if self.volume_confirmation:
            current_volume = current_data.get('volume', 0)
            avg_volume_20 = current_data.get('volume_sma_20', current_data.get('volume_ma_20', current_volume))
            if avg_volume_20 > 0:
                volume_ratio = current_volume / avg_volume_20
                volume_confirmed = volume_ratio >= self.volume_threshold
            else:
                volume_confirmed = True  # No volume data, allow trade

        # Determine action based on prediction
        if prediction.direction == 1 and prediction.confidence >= self.min_confidence and volume_confirmed:
            action = 'BUY'
            # Target based on expected return + buffer
            expected_return = max(prediction.expected_return, 0.005)
            target_price = current_price * (1 + expected_return * 1.5)
            # Stop loss based on ATR (widened to 2.5x)
            stop_loss = current_price - (atr * 2.5)

        elif prediction.direction == 0 and prediction.confidence >= self.min_confidence and volume_confirmed:
            action = 'SELL'
            expected_return = min(prediction.expected_return, -0.005)
            target_price = current_price * (1 + expected_return * 1.5)
            stop_loss = current_price + (atr * 2.5)

        else:
            action = 'HOLD'
            target_price = current_price
            stop_loss = current_price
            expected_return = 0

        # Calculate risk metrics
        expected_return_pct = (target_price - current_price) / current_price * 100
        risk_pct = abs(current_price - stop_loss) / current_price * 100

        if risk_pct > 0:
            risk_reward = abs(expected_return_pct) / risk_pct
        else:
            risk_reward = 0

        # Signal strength
        if prediction.confidence >= 0.7 and risk_reward >= 2.0:
            strength = 'STRONG'
        elif prediction.confidence >= 0.6 and risk_reward >= 1.5:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'

        # Position sizing (Kelly Criterion inspired)
        # Position size = (edge * win_rate - (1 - win_rate)) / edge
        # Simplified: size based on confidence and volatility
        base_position = self.max_position_pct

        # Reduce position for lower confidence
        confidence_factor = (prediction.confidence - 0.5) / 0.5  # 0 to 1
        confidence_factor = np.clip(confidence_factor, 0.2, 1.0)

        # Reduce position for high volatility
        vol_20 = current_data.get('volatility_20d', 0.20)
        vol_factor = np.clip(0.25 / (vol_20 + 0.01), 0.5, 1.0)

        # Reduce position for poor risk/reward
        rr_factor = np.clip(risk_reward / self.min_risk_reward, 0.5, 1.0) if risk_reward > 0 else 0

        suggested_position_pct = base_position * confidence_factor * vol_factor * rr_factor

        # Ensure max portfolio risk constraint
        if risk_pct > 0:
            max_position_by_risk = self.max_portfolio_risk / (risk_pct / 100)
            suggested_position_pct = min(suggested_position_pct, max_position_by_risk)

        # Calculate quantity
        position_value = portfolio_value * suggested_position_pct
        suggested_quantity = int(position_value / current_price)

        # Get sector
        sector = config.STOCK_SECTOR_MAP.get(symbol, 'Unknown')

        # Generate technical summary
        technical_summary = self._generate_technical_summary(current_data, prediction)

        # Determine market regime
        market_regime = self._determine_regime(current_data)

        # Regime-aware trading: adjust or filter based on regime
        regime_adjusted_action = action
        regime_note = ""
        
        if market_regime == 'HIGH_VOLATILITY':
            # In high volatility, reduce position size and be more selective
            suggested_position_pct *= 0.5
            suggested_quantity = int(portfolio_value * suggested_position_pct / current_price)
            regime_note = " [Reduced size: High Vol]"
            # Only take strong signals in high vol
            if strength == 'WEAK':
                regime_adjusted_action = 'HOLD'
                suggested_position_pct = 0
                suggested_quantity = 0
                
        elif market_regime == 'RANGING':
            # In ranging market, prefer mean reversion - contrarian signals
            # If RSI is in overbought/oversold, signal is more reliable
            rsi = current_data.get('rsi_14', 50)
            if action == 'BUY' and rsi > 60:
                regime_adjusted_action = 'HOLD'  # Don't buy in ranging market when not oversold
                suggested_position_pct = 0
                suggested_quantity = 0
            elif action == 'SELL' and rsi < 40:
                regime_adjusted_action = 'HOLD'  # Don't sell in ranging market when not overbought
                suggested_position_pct = 0
                suggested_quantity = 0
            regime_note = " [Ranging Market]"
                
        elif market_regime == 'TRENDING_UP' and action == 'BUY':
            # Trend following - increase position in trending up market
            suggested_position_pct *= 1.2
            suggested_quantity = int(portfolio_value * suggested_position_pct / current_price)
            regime_note = " [Trend Aligned]"
            
        elif market_regime == 'TRENDING_DOWN' and action == 'SELL':
            # Trend following - increase position in trending down market
            suggested_position_pct *= 1.2
            suggested_quantity = int(portfolio_value * suggested_position_pct / current_price)
            regime_note = " [Trend Aligned]"

        # Update action based on regime adjustment
        action = regime_adjusted_action
        technical_summary += regime_note

        return TradingSignal(
            symbol=symbol,
            timestamp=current_time,
            action=action,
            strength=strength,
            current_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_return_pct=expected_return_pct,
            risk_pct=risk_pct,
            risk_reward_ratio=risk_reward,
            confidence=prediction.confidence,
            direction_probability=prediction.direction_probability,
            suggested_position_pct=suggested_position_pct * 100,  # Convert to %
            suggested_quantity=suggested_quantity,
            sector=sector,
            market_regime=market_regime,
            technical_summary=technical_summary
        )

    def generate_portfolio_signals(
        self,
        symbols: List[str],
        predictions: Dict[str, ModelPrediction],
        current_data: Dict[str, pd.Series],
        portfolio_value: float = 100000
    ) -> PortfolioSignals:
        """
        Generate signals for entire portfolio with constraints.

        Args:
            symbols: List of stock symbols
            predictions: Dict of predictions per symbol
            current_data: Dict of current data per symbol
            portfolio_value: Total portfolio value

        Returns:
            PortfolioSignals with all signals and allocations
        """
        all_signals = []

        for symbol in symbols:
            if symbol in predictions and symbol in current_data:
                signal = self.generate_signal(
                    symbol,
                    predictions[symbol],
                    current_data[symbol],
                    portfolio_value
                )
                all_signals.append(signal)

        # Sort by confidence
        all_signals.sort(key=lambda x: x.confidence, reverse=True)

        # Filter actionable signals
        buy_signals = [s for s in all_signals if s.action == 'BUY' and s.strength != 'WEAK']
        sell_signals = [s for s in all_signals if s.action == 'SELL' and s.strength != 'WEAK']

        # Apply portfolio-level constraints
        buy_signals = self._apply_portfolio_constraints(buy_signals)

        # Calculate total allocation
        total_allocation = sum(s.suggested_position_pct for s in buy_signals)

        # Determine market exposure
        n_buy = len(buy_signals)
        n_sell = len(sell_signals)

        if n_buy > n_sell * 2:
            market_exposure = 'BULLISH'
        elif n_sell > n_buy * 2:
            market_exposure = 'BEARISH'
        else:
            market_exposure = 'NEUTRAL'

        return PortfolioSignals(
            timestamp=datetime.now(),
            signals=all_signals,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            total_suggested_allocation=total_allocation,
            market_exposure=market_exposure
        )

    def _apply_portfolio_constraints(
        self,
        signals: List[TradingSignal],
        max_positions: int = 10,
        max_sector_allocation: float = 0.30
    ) -> List[TradingSignal]:
        """Apply portfolio-level constraints to signals."""

        # Limit number of positions
        signals = signals[:max_positions]

        # Track sector allocation
        sector_allocation = {}
        constrained_signals = []

        for signal in signals:
            sector = signal.sector
            current_sector_alloc = sector_allocation.get(sector, 0)

            if current_sector_alloc + signal.suggested_position_pct <= max_sector_allocation * 100:
                constrained_signals.append(signal)
                sector_allocation[sector] = current_sector_alloc + signal.suggested_position_pct

        return constrained_signals

    def _generate_technical_summary(
        self,
        data: pd.Series,
        prediction: ModelPrediction
    ) -> str:
        """Generate human-readable technical summary."""
        parts = []

        # RSI
        rsi = data.get('rsi_14', 50)
        if rsi > 70:
            parts.append("RSI overbought")
        elif rsi < 30:
            parts.append("RSI oversold")

        # MACD
        macd_hist = data.get('macd_hist', 0)
        if macd_hist > 0:
            parts.append("MACD bullish")
        else:
            parts.append("MACD bearish")

        # Trend
        trend_strength = data.get('trend_strength', 0)
        if trend_strength >= 2:
            parts.append("Strong uptrend")
        elif trend_strength <= -2:
            parts.append("Strong downtrend")

        # Volume
        vol_ratio = data.get('volume_ratio', 1)
        if vol_ratio > 1.5:
            parts.append("High volume")

        # Confidence
        if prediction.confidence >= 0.7:
            parts.append("High confidence")
        elif prediction.confidence >= 0.6:
            parts.append("Moderate confidence")

        return " | ".join(parts) if parts else "Neutral technicals"

    def _determine_regime(self, data: pd.Series) -> str:
        """Determine current market regime."""
        ma_regime = data.get('ma_trend_regime', 0)
        vol_regime = data.get('vol_regime', 0)

        if ma_regime >= 1 and vol_regime <= 1:
            return 'TRENDING_UP'
        elif ma_regime <= -1 and vol_regime <= 1:
            return 'TRENDING_DOWN'
        elif vol_regime >= 2:
            return 'HIGH_VOLATILITY'
        else:
            return 'RANGING'

    def format_signals_report(self, portfolio_signals: PortfolioSignals) -> str:
        """Format signals as a readable report."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"  TRADING SIGNALS REPORT - {portfolio_signals.timestamp.strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 80)
        lines.append(f"\nMarket Exposure: {portfolio_signals.market_exposure}")
        lines.append(f"Total Suggested Allocation: {portfolio_signals.total_suggested_allocation:.1f}%")
        lines.append("")

        if portfolio_signals.buy_signals:
            lines.append("-" * 80)
            lines.append("BUY SIGNALS")
            lines.append("-" * 80)

            for signal in portfolio_signals.buy_signals:
                lines.append(f"\n[{signal.strength}] {signal.symbol} ({signal.sector})")
                lines.append(f"  Price: Rs {signal.current_price:,.2f}")
                lines.append(f"  Target: Rs {signal.target_price:,.2f} ({signal.expected_return_pct:+.2f}%)")
                lines.append(f"  Stop Loss: Rs {signal.stop_loss:,.2f} ({-signal.risk_pct:.2f}%)")
                lines.append(f"  Risk/Reward: 1:{signal.risk_reward_ratio:.1f}")
                lines.append(f"  Confidence: {signal.confidence:.1%}")
                lines.append(f"  Suggested: {signal.suggested_position_pct:.1f}% ({signal.suggested_quantity} shares)")
                lines.append(f"  {signal.technical_summary}")

        if portfolio_signals.sell_signals:
            lines.append("\n" + "-" * 80)
            lines.append("SELL/SHORT SIGNALS")
            lines.append("-" * 80)

            for signal in portfolio_signals.sell_signals:
                lines.append(f"\n[{signal.strength}] {signal.symbol} ({signal.sector})")
                lines.append(f"  Price: Rs {signal.current_price:,.2f}")
                lines.append(f"  Target: Rs {signal.target_price:,.2f} ({signal.expected_return_pct:+.2f}%)")
                lines.append(f"  Confidence: {signal.confidence:.1%}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)
