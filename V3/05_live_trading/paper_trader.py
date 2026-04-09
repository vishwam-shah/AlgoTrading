"""
Paper Trader — Dry-run trading mode using live market data.

Simulates execution WITHOUT placing real orders.

Validates:
- Order routing logic (without submitting)
- Fill price predictions (using WebSocket LTP)
- Transaction costs
- Portfolio tracking

Allows 2+ weeks of dry-run before live trading.
"""

from typing import Dict, List, Optional
from datetime import datetime


class PaperTrader:
    """
    Simulated trading without real money.

    Uses live market data (WebSocket LTP) but doesn't submit orders to Angel One.
    """

    def __init__(self, initial_cash: float = 100_000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings: Dict[str, int] = {}
        self.trades: List[Dict] = []

    def simulate_order(
        self,
        symbol: str,
        quantity: int,
        ltp: float,
        action: str = "BUY",
        slippage_pct: float = 0.0003,
    ) -> Dict[str, float]:
        """
        Simulate an order execution (without placing on exchange).

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            ltp: Last traded price from WebSocket
            action: BUY or SELL
            slippage_pct: Expected slippage (default 0.03%)

        Returns:
            {
                'executed_price': float (with slippage),
                'position_value': float,
                'cash_after': float,
                'status': str
            }
        """
        # Apply slippage
        if action == "BUY":
            executed_price = ltp * (1 + slippage_pct)
            position_value = quantity * executed_price
            if self.cash >= position_value:
                self.cash -= position_value
                self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
                status = "FILLED"
            else:
                status = "INSUFFICIENT_FUNDS"
                executed_price = 0
                position_value = 0

        else:  # SELL
            executed_price = ltp * (1 - slippage_pct)
            current_qty = self.holdings.get(symbol, 0)
            if current_qty >= quantity:
                position_value = quantity * executed_price
                self.cash += position_value
                self.holdings[symbol] = current_qty - quantity
                status = "FILLED"
            else:
                status = "INSUFFICIENT_POSITION"
                executed_price = 0
                position_value = 0

        # Record
        if status == "FILLED":
            self.trades.append(
                {
                    "date": datetime.now().isoformat(),
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "price": executed_price,
                    "position_value": position_value,
                }
            )

        return {
            "executed_price": executed_price,
            "position_value": position_value,
            "cash_after": self.cash,
            "status": status,
        }

    def get_portfolio_value(self, live_prices: Dict[str, float]) -> float:
        """Current portfolio value (cash + holdings at live prices)."""
        position_value = sum(
            self.holdings.get(s, 0) * live_prices.get(s, 0)
            for s in self.holdings
        )
        return self.cash + position_value
