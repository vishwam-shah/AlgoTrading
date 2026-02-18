"""
================================================================================
POSITION SIZER - Kelly Criterion & Volatility-Adjusted Sizing
================================================================================
Calculates optimal position sizes based on:
1. Kelly Criterion (mathematical optimal)
2. Volatility adjustment (ATR-based)
3. Correlation limits (diversification)
4. Risk limits (max position size)

Target: Maximize returns while controlling risk
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calculate optimal position sizes using multiple methods.
    
    Methods:
    1. Kelly Criterion: (Win_Rate × Avg_Win - (1-Win_Rate) × Avg_Loss) / Avg_Win
    2. Volatility-Adjusted: Risk_Amount / (Stock_Volatility × Price)
    3. Fixed Fractional: Fixed % of capital per trade
    4. Correlation-Adjusted: Reduce size for correlated positions
    """
    
    def __init__(
        self,
        total_capital: float,
        max_position_pct: float = 0.10,  # Max 10% per position
        max_risk_per_trade: float = 0.01,  # Max 1% risk per trade
        kelly_fraction: float = 0.25,  # Use 25% of full Kelly
        use_volatility_adjustment: bool = True,
        use_correlation_adjustment: bool = True
    ):
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.max_risk_per_trade = max_risk_per_trade
        self.kelly_fraction = kelly_fraction
        self.use_volatility_adjustment = use_volatility_adjustment
        self.use_correlation_adjustment = use_correlation_adjustment
        
        # Track model performance for Kelly calculation
        self.win_rate = 0.60  # Initial estimate
        self.avg_win = 0.03  # 3%
        self.avg_loss = 0.015  # 1.5%
        self.total_trades = 0
        self.winning_trades = 0
        self.total_win_pnl = 0
        self.total_loss_pnl = 0
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        volatility: float,
        current_positions: Optional[Dict] = None,
        correlations: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate optimal position size.
        
        Args:
            symbol: Stock symbol
            entry_price: Planned entry price
            stop_loss: Stop loss price
            confidence: Model confidence (0-1)
            volatility: Stock volatility (ATR%)
            current_positions: Dict of current positions {symbol: size}
            correlations: Dict of correlations {symbol: correlation_value}
            
        Returns:
            Dict with position size, reasoning, and risk metrics
        """
        # 1. Calculate Kelly-based size
        kelly_size = self._calculate_kelly_size(confidence)
        
        # 2. Calculate risk-based size
        risk_per_share = abs(entry_price - stop_loss)
        risk_based_size = self._calculate_risk_based_size(
            entry_price, 
            risk_per_share
        )
        
        # 3. Calculate volatility-adjusted size
        if self.use_volatility_adjustment:
            vol_adjusted_size = self._calculate_volatility_adjusted_size(
                entry_price,
                volatility
            )
        else:
            vol_adjusted_size = risk_based_size
        
        # 4. Take minimum of all sizes (most conservative)
        base_size = min(kelly_size, risk_based_size, vol_adjusted_size)
        
        # 5. Apply correlation adjustment
        if self.use_correlation_adjustment and current_positions and correlations:
            correlation_factor = self._calculate_correlation_adjustment(
                symbol,
                current_positions,
                correlations
            )
            final_size = base_size * correlation_factor
        else:
            final_size = base_size
        
        # 6. Apply absolute limits
        max_size = self.total_capital * self.max_position_pct
        final_size = min(final_size, max_size)
        
        # 7. Calculate number of shares
        num_shares = int(final_size / entry_price)
        actual_size = num_shares * entry_price
        
        # 8. Calculate risk metrics
        position_risk = num_shares * risk_per_share
        risk_pct = position_risk / self.total_capital
        
        return {
            'symbol': symbol,
            'num_shares': num_shares,
            'position_value': actual_size,
            'position_pct': actual_size / self.total_capital,
            'risk_amount': position_risk,
            'risk_pct': risk_pct,
            'kelly_size': kelly_size,
            'risk_based_size': risk_based_size,
            'vol_adjusted_size': vol_adjusted_size,
            'final_size': actual_size,
            'reasoning': self._generate_reasoning(
                kelly_size, risk_based_size, vol_adjusted_size, 
                final_size, confidence
            )
        }
    
    def _calculate_kelly_size(self, confidence: float) -> float:
        """
        Calculate Kelly Criterion position size.
        
        Kelly% = (Win_Rate × Avg_Win - (1-Win_Rate) × Avg_Loss) / Avg_Win
        
        We use a fraction of Kelly to be conservative.
        """
        if self.avg_win == 0:
            return self.total_capital * self.max_position_pct
        
        kelly_pct = (
            self.win_rate * self.avg_win - 
            (1 - self.win_rate) * self.avg_loss
        ) / self.avg_win
        
        # Adjust by confidence
        kelly_pct = kelly_pct * confidence
        
        # Use fraction of Kelly (typically 25-50% for safety)
        kelly_pct = kelly_pct * self.kelly_fraction
        
        # Cap at max position%
        kelly_pct = min(kelly_pct, self.max_position_pct)
        kelly_pct = max(kelly_pct, 0)  # No negative
        
        return self.total_capital * kelly_pct
    
    def _calculate_risk_based_size(self, entry_price: float, risk_per_share: float) -> float:
        """
        Calculate position size based on fixed risk per trade.
        
        Position Size = Risk_Amount / Risk_Per_Share
        """
        risk_amount = self.total_capital * self.max_risk_per_trade
        
        if risk_per_share == 0:
            return self.total_capital * self.max_position_pct
        
        num_shares = risk_amount / risk_per_share
        return num_shares * entry_price
    
    def _calculate_volatility_adjusted_size(self, entry_price: float, volatility: float) -> float:
        """
        Calculate position size adjusted for volatility.
        
        Higher volatility = smaller position
        """
        if volatility == 0:
            volatility = 0.02  # Default 2%
        
        # Target volatility: 2% per position
        target_vol = 0.02
        
        # Adjust size inversely to volatility
        vol_factor = target_vol / volatility
        vol_factor = min(vol_factor, 1.5)  # Cap at 1.5x
        vol_factor = max(vol_factor, 0.3)  # Floor at 0.3x
        
        base_size = self.total_capital * self.max_position_pct
        return base_size * vol_factor
    
    def _calculate_correlation_adjustment(
        self,
        symbol: str,
        current_positions: Dict,
        correlations: Dict
    ) -> float:
        """
        Reduce position size if highly correlated with existing positions.
        """
        if not current_positions:
            return 1.0
        
        # Calculate average correlation with existing positions
        weighted_corr = 0
        total_weight = 0
        
        for existing_symbol, position_value in current_positions.items():
            if existing_symbol == symbol:
                continue
            
            corr = correlations.get(existing_symbol, 0)
            weight = position_value / self.total_capital
            
            weighted_corr += abs(corr) * weight
            total_weight += weight
        
        if total_weight == 0:
            return 1.0
        
        avg_corr = weighted_corr / total_weight
        
        # Reduce size based on correlation
        # High correlation (>0.7) = 50% reduction
        # Medium correlation (0.3-0.7) = 25% reduction
        # Low correlation (<0.3) = no reduction
        if avg_corr > 0.7:
            return 0.50
        elif avg_corr > 0.3:
            return 0.75
        else:
            return 1.0
    
    def _generate_reasoning(
        self,
        kelly_size: float,
        risk_based_size: float,
        vol_adjusted_size: float,
        final_size: float,
        confidence: float
    ) -> str:
        """Generate human-readable reasoning for position size."""
        reasons = []
        
        reasons.append(f"Kelly: ₹{kelly_size:,.0f}")
        reasons.append(f"Risk-based: ₹{risk_based_size:,.0f}")
        reasons.append(f"Vol-adjusted: ₹{vol_adjusted_size:,.0f}")
        
        # Identify limiting factor
        if final_size == kelly_size:
            reasons.append("Limited by Kelly")
        elif final_size == risk_based_size:
            reasons.append("Limited by risk (1% max)")
        elif final_size == vol_adjusted_size:
            reasons.append("Limited by volatility")
        else:
            reasons.append(f"Limited by max position ({self.max_position_pct:.0%})")
        
        reasons.append(f"Confidence: {confidence:.0%}")
        
        return " | ".join(reasons)
    
    def update_performance(self, pnl: float):
        """Update performance metrics for Kelly calculation."""
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
            self.total_win_pnl += pnl
        else:
            self.total_loss_pnl += abs(pnl)
        
        # Update win rate
        self.win_rate = self.winning_trades / self.total_trades
        
        # Update average win/loss
        if self.winning_trades > 0:
            self.avg_win = self.total_win_pnl / self.winning_trades
        
        losing_trades = self.total_trades - self.winning_trades
        if losing_trades > 0:
            self.avg_loss = self.total_loss_pnl / losing_trades
        
        logger.info(f"Updated performance: WR={self.win_rate:.1%}, "
                   f"Avg Win={self.avg_win:.2%}, Avg Loss={self.avg_loss:.2%}")


if __name__ == "__main__":
    # Example usage
    sizer = PositionSizer(
        total_capital=100000,
        max_position_pct=0.10,
        max_risk_per_trade=0.01,
        kelly_fraction=0.25
    )
    
    # Calculate position size
    result = sizer.calculate_position_size(
        symbol='SBIN',
        entry_price=500,
        stop_loss=485,  # 3% stop loss
        confidence=0.68,
        volatility=0.025,  # 2.5% ATR
        current_positions={'HDFCBANK': 10000},
        correlations={'HDFCBANK': 0.75}  # High correlation
    )
    
    print(f"\nPosition Size Calculation:")
    print(f"Symbol: {result['symbol']}")
    print(f"Shares: {result['num_shares']}")
    print(f"Position Value: ₹{result['position_value']:,.0f}")
    print(f"Position %: {result['position_pct']:.1%}")
    print(f"Risk Amount: ₹{result['risk_amount']:,.0f}")
    print(f"Risk %: {result['risk_pct']:.2%}")
    print(f"\nReasoning: {result['reasoning']}")
