"""
NSE-specific transaction cost model.

Costs for delivery (CNC) trading:
- Brokerage: ~0.03% (Angel One/Zerodha style)
- STT: 0.1% on sell side only
- Exchange charges: 0.00345%
- GST: 18% on brokerage + exchange
- Stamp duty: 0.015% on buy side
- SEBI: ₹10 per crore

Total one-way: ~0.06-0.08%
Round-trip: ~0.12-0.16%

Slippage varies by liquidity:
- Nifty 50 stocks (very liquid): 0.02-0.05%
- Nifty Next 50 (less liquid): 0.05-0.10%
"""

import numpy as np
from typing import Dict, Tuple


class TransactionCosts:
    """Calculate NSE transaction costs."""

    # Flat brokerage per trade (Angel One pricing)
    FLAT_BROKERAGE = 20.0  # ₹20 per trade

    # Percentage-based charges
    STT_RATE = 0.001  # 0.1% on sell side only
    EXCHANGE_CHARGE_RATE = 0.00345  # 0.345%
    STAMP_DUTY_RATE = 0.00015  # 0.015% on buy side
    GST_RATE = 0.18  # 18% on brokerage + exchange charges
    SEBI_CHARGE_PER_CRORE = 10.0  # ₹10 per crore

    # Slippage model (fraction of bid-ask spread)
    SLIPPAGE_NIFTY50 = 0.0003  # 0.03% for liquid stocks
    SLIPPAGE_NIFTY_NEXT50 = 0.0007  # 0.07% for less liquid
    SLIPPAGE_DEFAULT = 0.0005  # 0.05% default

    @staticmethod
    def get_liquidity_category(symbol: str, is_nifty50: bool = True) -> str:
        """Classify stock liquidity (used for slippage estimation)."""
        if is_nifty50:
            return "nifty50"
        else:
            return "nifty_next50"

    @staticmethod
    def calculate_one_way_cost(
        symbol: str,
        price: float,
        quantity: int,
        is_buy: bool = True,
        is_nifty50: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate one-way transaction cost for a single trade.

        Args:
            symbol: Stock symbol
            price: Entry/exit price
            quantity: Number of shares
            is_buy: True for buy, False for sell
            is_nifty50: Is stock in Nifty 50 (more liquid)

        Returns:
            Dict with cost breakdown:
                - brokerage
                - stt
                - exchange_charge
                - stamp_duty
                - gst
                - slippage
                - total_cost_rupees
                - total_cost_pct
        """
        notional_value = price * quantity

        # Brokerage
        brokerage = TransactionCosts.FLAT_BROKERAGE

        # STT (only on sell side)
        stt = TransactionCosts.STT_RATE * notional_value if not is_buy else 0

        # Exchange charges
        exchange_charge = TransactionCosts.EXCHANGE_CHARGE_RATE * notional_value

        # Stamp duty (only on buy side)
        stamp_duty = (
            TransactionCosts.STAMP_DUTY_RATE * notional_value if is_buy else 0
        )

        # GST on brokerage + exchange charges
        gst = TransactionCosts.GST_RATE * (brokerage + exchange_charge)

        # SEBI charge (₹10 per crore notional)
        sebi = (notional_value / 10_000_000) * TransactionCosts.SEBI_CHARGE_PER_CRORE

        # Slippage (half the bid-ask spread)
        slippage_rate = (
            TransactionCosts.SLIPPAGE_NIFTY50
            if is_nifty50
            else TransactionCosts.SLIPPAGE_NIFTY_NEXT50
        )
        slippage = slippage_rate * notional_value

        # Total
        total_cost = brokerage + stt + exchange_charge + stamp_duty + gst + sebi + slippage
        total_cost_pct = total_cost / notional_value if notional_value > 0 else 0

        return {
            "brokerage": brokerage,
            "stt": stt,
            "exchange_charge": exchange_charge,
            "stamp_duty": stamp_duty,
            "gst": gst,
            "sebi": sebi,
            "slippage": slippage,
            "total_cost_rupees": total_cost,
            "total_cost_pct": total_cost_pct,
        }

    @staticmethod
    def calculate_round_trip_cost(
        symbol: str, price: float, quantity: int, is_nifty50: bool = True
    ) -> float:
        """
        Calculate round-trip cost (buy + sell) as percentage of notional value.

        Realistic estimate for NSE delivery trading: 0.12-0.16%
        """
        buy_cost = TransactionCosts.calculate_one_way_cost(
            symbol, price, quantity, is_buy=True, is_nifty50=is_nifty50
        )
        sell_cost = TransactionCosts.calculate_one_way_cost(
            symbol, price, quantity, is_buy=False, is_nifty50=is_nifty50
        )

        notional_value = price * quantity
        total_cost = buy_cost["total_cost_rupees"] + sell_cost["total_cost_rupees"]

        return total_cost / notional_value if notional_value > 0 else 0
