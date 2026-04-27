"""
order_manager.py — Execution engine connecting signal_publisher → Angel One
============================================================================
Takes approved orders from signal_publisher, places them on Angel One,
tracks fills, persists execution log, and handles partial/rejected orders.

Usage:
    from order_manager import OrderManager
    from angel_one_client import AngelOneClient

    client  = AngelOneClient()
    client.login()
    manager = OrderManager(client)
    fills   = manager.execute_orders(orders)           # from signal_publisher
    manager.wait_for_fills(timeout_min=15)
    manager.save_execution_log()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_LIVE_DIR  = Path(__file__).resolve().parent
_V3_ROOT   = _LIVE_DIR.parent
_EXEC_DIR  = _LIVE_DIR / "execution_logs"

# NSE round-trip cost model
_STT_SELL_PCT     = 0.001   # 0.1% STT on sell side
_BROKERAGE_FLAT   = 20.0    # ₹20 per order (Zerodha model; Angel One charges differently)
_EXCHANGE_PCT     = 0.00035 # NSE exchange + SEBI
_STAMP_PCT        = 0.00015 # stamp duty buy side
_GST_ON_BROKERAGE = 0.18


@dataclass
class FillRecord:
    symbol:         str
    order_id:       str
    side:           str     # BUY | SELL
    ordered_qty:    int
    filled_qty:     int
    avg_price:      float
    order_value:    float
    brokerage:      float
    stt:            float
    other_charges:  float
    net_cost:       float
    status:         str     # FILLED | PARTIAL | REJECTED | PENDING | TIMEOUT
    placed_at:      str     = ""
    filled_at:      str     = ""
    message:        str     = ""


def _compute_costs(qty: int, price: float, side: str) -> Dict[str, float]:
    value = qty * price
    brok  = min(_BROKERAGE_FLAT, value * 0.0003)   # ₹20 or 0.03%, whichever lower
    gst   = brok * _GST_ON_BROKERAGE
    stt   = value * _STT_SELL_PCT if side == "SELL" else 0.0
    exch  = value * _EXCHANGE_PCT
    stamp = value * _STAMP_PCT if side == "BUY" else 0.0
    total_charges = brok + gst + stt + exch + stamp
    return {
        "brokerage":     round(brok + gst, 4),
        "stt":           round(stt, 4),
        "other_charges": round(exch + stamp, 4),
        "net_cost":      round(value + total_charges if side == "BUY"
                               else value - total_charges, 2),
    }


class OrderManager:
    """
    Full execution manager with fill tracking and execution log.

    Works in both LIVE mode (real Angel One orders) and
    PAPER mode (simulated fills at last price).
    """

    def __init__(self,
                 client=None,
                 paper_mode: bool = False,
                 slippage_pct: float = 0.0003):
        """
        Args:
            client      : AngelOneClient instance (None for paper mode)
            paper_mode  : If True, simulate fills without real orders
            slippage_pct: Expected slippage for paper mode (0.03%)
        """
        self.client       = client
        self.paper_mode   = paper_mode or (client is None)
        self.slippage_pct = slippage_pct

        self.fills:   List[FillRecord] = []
        self.pending: Dict[str, FillRecord] = {}   # {order_id: fill}

    # ── Place single order ────────────────────────────────────────────────────

    def place_order(self,
                    symbol:     str,
                    qty:        int,
                    price:      float,
                    order_type: str = "LIMIT",
                    product:    str = "CNC",
                    side:       str = "BUY") -> FillRecord:
        """Place one order and return a FillRecord (status=PENDING until fill)."""
        costs = _compute_costs(qty, price, side)
        placed_at = datetime.now().isoformat()

        if self.paper_mode:
            sim_price = price * (1 + self.slippage_pct if side == "BUY"
                                 else 1 - self.slippage_pct)
            rec = FillRecord(
                symbol=symbol, order_id=f"PAPER_{symbol}_{int(time.time())}",
                side=side, ordered_qty=qty, filled_qty=qty,
                avg_price=round(sim_price, 2), order_value=round(qty * sim_price, 2),
                status="FILLED", placed_at=placed_at, filled_at=placed_at,
                **costs,
            )
            self.fills.append(rec)
            print(f"  [paper] {side} {qty}×{symbol} @ ₹{sim_price:.2f}  "
                  f"cost=₹{costs['net_cost']:.0f}")
            return rec

        # Live order
        resp = self.client.place_order(symbol=symbol, qty=qty, price=price,
                                       order_type=order_type, product=product, side=side)
        rec = FillRecord(
            symbol=symbol, order_id=resp.order_id,
            side=side, ordered_qty=qty, filled_qty=0,
            avg_price=price, order_value=round(qty * price, 2),
            status=resp.status, placed_at=placed_at, message=resp.message,
            **costs,
        )
        if resp.status != "REJECTED":
            self.pending[resp.order_id] = rec
        else:
            self.fills.append(rec)
            print(f"  [angel] REJECTED {symbol}: {resp.message}")
        return rec

    # ── Batch execution ────────────────────────────────────────────────────────

    def execute_orders(self, orders: List[Dict],
                       max_slippage_pct: float = 0.003) -> List[FillRecord]:
        """
        Execute a list of order dicts from signal_publisher.

        Checks live LTP vs expected price — skips if slippage > max_slippage_pct.

        Args:
            orders           : List of dicts (symbol, qty, price, direction, ...)
            max_slippage_pct : Skip order if LTP drifted > this % from expected (default 0.3%)

        Returns:
            List of FillRecord (one per order attempted)
        """
        results = []
        for o in orders:
            sym      = o["symbol"]
            qty      = o["qty"]
            exp_px   = o["price"]
            side     = str(o.get("direction", "BUY")).upper()

            # Check live price for slippage guard (live mode only)
            if not self.paper_mode and self.client:
                ltp = self.client.get_ltp(sym)
                if ltp:
                    slip = abs(ltp - exp_px) / exp_px
                    if slip > max_slippage_pct:
                        print(f"  [skip] {sym}: LTP={ltp:.2f} vs expected={exp_px:.2f} "
                              f"slippage={slip:.2%} > {max_slippage_pct:.2%}")
                        continue
                    exp_px = ltp  # use current LTP as limit price

            rec = self.place_order(sym, qty, exp_px, side=side)
            results.append(rec)
            time.sleep(0.06)   # ~16 orders/s, safely under rate limit

        return results

    # ── Fill tracking ─────────────────────────────────────────────────────────

    def wait_for_fills(self, timeout_min: int = 20,
                       poll_interval_sec: int = 30) -> None:
        """
        Poll order book until all pending orders are filled or timeout.

        Args:
            timeout_min      : Stop polling after this many minutes
            poll_interval_sec: How often to check (seconds)
        """
        if self.paper_mode or not self.pending:
            return

        deadline = time.time() + timeout_min * 60
        print(f"  [fills] Waiting for {len(self.pending)} orders "
              f"(timeout {timeout_min} min) …")

        while self.pending and time.time() < deadline:
            time.sleep(poll_interval_sec)
            order_book = {str(o["orderid"]): o
                          for o in (self.client.get_order_book() or [])}

            for oid in list(self.pending.keys()):
                ob = order_book.get(oid)
                if not ob:
                    continue
                status = ob.get("orderstatus", "").lower()
                if status in ("complete", "filled"):
                    rec = self.pending.pop(oid)
                    rec.filled_qty = int(ob.get("filledshares", rec.ordered_qty))
                    rec.avg_price  = float(ob.get("averageprice", rec.avg_price))
                    rec.status     = "FILLED"
                    rec.filled_at  = datetime.now().isoformat()
                    # Recalculate costs on actual fill
                    costs = _compute_costs(rec.filled_qty, rec.avg_price, rec.side)
                    rec.brokerage      = costs["brokerage"]
                    rec.stt            = costs["stt"]
                    rec.other_charges  = costs["other_charges"]
                    rec.net_cost       = costs["net_cost"]
                    self.fills.append(rec)
                    print(f"  [fill] {rec.symbol} {rec.filled_qty}×₹{rec.avg_price:.2f}  "
                          f"cost=₹{rec.net_cost:.0f}")
                elif status in ("rejected", "cancelled"):
                    rec = self.pending.pop(oid)
                    rec.status  = "REJECTED"
                    rec.message = ob.get("text", "")
                    self.fills.append(rec)
                    print(f"  [rejected] {rec.symbol}: {rec.message}")

        # Mark remaining pending as TIMEOUT
        for oid, rec in list(self.pending.items()):
            rec.status = "TIMEOUT"
            self.fills.append(rec)
            print(f"  [timeout] {rec.symbol} order {oid} not filled in {timeout_min} min")
        self.pending.clear()

    # ── Execution log ─────────────────────────────────────────────────────────

    def save_execution_log(self, run_id: Optional[str] = None) -> Path:
        """Persist fills to parquet + JSON for reconciliation."""
        _EXEC_DIR.mkdir(parents=True, exist_ok=True)
        today   = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag     = f"{run_id}_{today}" if run_id else today

        rows = [asdict(f) for f in self.fills]
        df   = pd.DataFrame(rows)

        json_path = _EXEC_DIR / f"execution_{tag}.json"
        parq_path = _EXEC_DIR / f"execution_log.parquet"

        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2, default=str)

        # Append to rolling parquet (atomic write to avoid corruption on kill)
        if parq_path.exists():
            try:
                existing = pd.read_parquet(parq_path)
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=["order_id"], keep="last")
            except Exception:
                print("  WARNING: execution log parquet corrupted — starting fresh")
        _tmp = parq_path.with_suffix(".parquet.tmp")
        df.to_parquet(_tmp, index=False, compression="snappy")
        _tmp.rename(parq_path)

        print(f"  [exec_log] {len(self.fills)} fills → {json_path.name}")
        return json_path

    def summary(self) -> Dict:
        """Return fill summary for logging/API."""
        total_fills     = [f for f in self.fills if f.status == "FILLED"]
        total_value     = sum(f.order_value for f in total_fills)
        total_charges   = sum(f.brokerage + f.stt + f.other_charges for f in total_fills)
        return {
            "total_orders":   len(self.fills),
            "filled":         len(total_fills),
            "rejected":       sum(1 for f in self.fills if f.status == "REJECTED"),
            "timeout":        sum(1 for f in self.fills if f.status == "TIMEOUT"),
            "total_value":    round(total_value, 2),
            "total_charges":  round(total_charges, 2),
            "symbols":        [f.symbol for f in total_fills],
        }
