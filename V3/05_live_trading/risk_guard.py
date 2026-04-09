"""
Risk Guard — Pre-trade risk checks and daily loss limits.

Prevents:
- Position sizes > 15% of portfolio
- Daily losses > 2% of portfolio
- Sector concentration > 30%
- Max holdings > 20 stocks
"""

import pandas as pd
from typing import Dict, List, Tuple


class RiskGuard:
    """Pre-trade risk validation."""

    @staticmethod
    def validate_order(
        symbol: str,
        quantity: int,
        price: float,
        portfolio_value: float,
        current_holdings: Dict[str, int],
        current_positions: Dict[str, float],
        sector_map: Dict[str, str],
        max_position_pct: float = 0.15,
        max_sector_pct: float = 0.30,
        max_holdings: int = 20,
    ) -> Tuple[bool, str]:
        """
        Validate an order before placement.

        Returns:
            (is_valid, reason)
        """
        position_value = quantity * price
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0

        # Check position size
        if position_pct > max_position_pct:
            return False, f"Position {position_pct:.1%} exceeds max {max_position_pct:.1%}"

        # Check sector concentration
        sector = sector_map.get(symbol, "OTHER")
        sector_value = sum(
            current_positions.get(s, 0)
            for s in current_positions
            if sector_map.get(s, "OTHER") == sector
        )
        sector_value += position_value  # Add new position
        sector_pct = sector_value / portfolio_value if portfolio_value > 0 else 0

        if sector_pct > max_sector_pct:
            return False, f"Sector {sector} would be {sector_pct:.1%} (max {max_sector_pct:.1%})"

        # Check holdings count
        if len(current_holdings) >= max_holdings and symbol not in current_holdings:
            return False, f"Already holding {max_holdings} stocks (max)"

        return True, "OK"

    @staticmethod
    def check_daily_loss_limit(
        portfolio_value: float,
        portfolio_value_yesterday: float,
        daily_loss_limit_pct: float = 0.02,
    ) -> Tuple[bool, str]:
        """
        Check if daily loss exceeds limit.

        Returns:
            (is_trading_allowed, reason)
        """
        if portfolio_value_yesterday == 0:
            return True, "OK"

        daily_loss = (portfolio_value_yesterday - portfolio_value) / portfolio_value_yesterday
        daily_loss_pct = max(0, daily_loss)

        if daily_loss_pct > daily_loss_limit_pct:
            return False, f"Daily loss {daily_loss_pct:.2%} exceeds limit {daily_loss_limit_pct:.2%}"

        return True, "OK"
