"""
Order Manager — Track order fills and execution.

Monitors:
- Order status (PENDING, FILLED, PARTIAL, REJECTED)
- Partial fills (multiple orders for same symbol)
- Average fill price
- Execution costs
"""

from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime


@dataclass
class ExecutedOrder:
    """Record of an executed order."""
    date: str
    symbol: str
    action: str  # BUY or SELL
    ordered_qty: int
    filled_qty: int
    avg_price: float
    status: str  # FILLED, PARTIAL, REJECTED
    execution_cost: float


class OrderManager:
    """Track order execution and fills."""

    def __init__(self):
        self.orders: List[ExecutedOrder] = []
        self.pending_orders: Dict[str, Dict] = {}  # {order_id: order_data}

    def record_fill(
        self,
        symbol: str,
        ordered_qty: int,
        filled_qty: int,
        avg_price: float,
        action: str = "BUY",
        status: str = "FILLED",
        cost: float = 0.0,
    ) -> ExecutedOrder:
        """
        Record an order fill.

        Args:
            symbol: Stock symbol
            ordered_qty: Quantity ordered
            filled_qty: Quantity actually filled
            avg_price: Average execution price
            action: BUY or SELL
            status: FILLED, PARTIAL, REJECTED
            cost: Transaction costs (₹)

        Returns:
            ExecutedOrder record
        """
        order = ExecutedOrder(
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol=symbol,
            action=action,
            ordered_qty=ordered_qty,
            filled_qty=filled_qty,
            avg_price=avg_price,
            status=status,
            execution_cost=cost,
        )
        self.orders.append(order)
        return order

    def get_execution_summary(self, symbol: str) -> Dict[str, float]:
        """
        Get execution summary for a symbol.

        Returns:
            {
                'total_qty_bought': int,
                'total_qty_sold': int,
                'avg_buy_price': float,
                'avg_sell_price': float,
                'total_cost': float
            }
        """
        symbol_orders = [o for o in self.orders if o.symbol == symbol]

        buy_orders = [o for o in symbol_orders if o.action == "BUY"]
        sell_orders = [o for o in symbol_orders if o.action == "SELL"]

        total_buy_qty = sum(o.filled_qty for o in buy_orders)
        total_sell_qty = sum(o.filled_qty for o in sell_orders)

        avg_buy_price = (
            sum(o.filled_qty * o.avg_price for o in buy_orders) / total_buy_qty
            if total_buy_qty > 0
            else 0
        )
        avg_sell_price = (
            sum(o.filled_qty * o.avg_price for o in sell_orders) / total_sell_qty
            if total_sell_qty > 0
            else 0
        )

        total_cost = sum(o.execution_cost for o in symbol_orders)

        return {
            "total_qty_bought": total_buy_qty,
            "total_qty_sold": total_sell_qty,
            "avg_buy_price": avg_buy_price,
            "avg_sell_price": avg_sell_price,
            "total_cost": total_cost,
        }
