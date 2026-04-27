"""
exit_runner.py — Time-based exit for v2 strategy
=================================================
Backtest holds each position exactly HOLD_DAYS_V2 (10) trading days. Live trading
must do the same — without this script, signal_publisher buys but no one ever
sells, so live ≠ backtest.

How it works
------------
1. Reads execution_log.parquet (all historical fills).
2. Computes net open position per symbol = sum(BUY filled_qty) − sum(SELL filled_qty).
3. For each symbol with positive open qty, finds the earliest unmatched BUY date
   (FIFO — oldest lot is the next to exit).
4. If `today >= entry_date + HOLD_DAYS_V2 trading days`, emits a SELL order.
5. Saves SELL orders to orders/exits_<today>.json.
6. If --execute is passed, hands the SELLs to OrderManager and saves an
   execution log alongside.

Trading-day arithmetic uses np.busday_count which skips Sat/Sun. NSE holidays
add ~12 days/year — for a 10-day hold this can advance the exit by 0–1 day vs
a strict NSE calendar; we accept that as a paper-test caveat.

Usage
-----
    python V3/05_live_trading/exit_runner.py             # dry-run, write orders/exits_*.json
    python V3/05_live_trading/exit_runner.py --execute   # also place via OrderManager
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_LIVE_DIR     = Path(__file__).resolve().parent
_EXEC_DIR     = _LIVE_DIR / "execution_logs"
_ORDERS_DIR   = _LIVE_DIR / "orders"
_PARQUET_PATH = _EXEC_DIR / "execution_log.parquet"

HOLD_DAYS_V2 = 10   # must match backtest.HOLD_DAYS_V2 and signal_publisher.TARGET_HOLD_DAYS


def _load_fills() -> pd.DataFrame:
    """Load all FILLED rows from execution_log.parquet (or scan JSONs as fallback)."""
    if _PARQUET_PATH.exists():
        try:
            df = pd.read_parquet(_PARQUET_PATH)
        except Exception as e:
            print(f"  [warn] parquet read failed ({e}) — falling back to JSON scan")
            df = _scan_json_logs()
    else:
        df = _scan_json_logs()

    if df.empty:
        return df

    df = df[df["status"] == "FILLED"].copy()
    df["filled_at"] = pd.to_datetime(df["filled_at"])
    df["entry_date"] = df["filled_at"].dt.date
    return df.sort_values("filled_at").reset_index(drop=True)


def _scan_json_logs() -> pd.DataFrame:
    rows = []
    for jp in sorted(_EXEC_DIR.glob("execution_*.json")):
        try:
            rows.extend(json.load(open(jp)))
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _open_positions_fifo(fills: pd.DataFrame) -> List[Dict]:
    """
    FIFO net-open positions. Returns one row per still-open BUY lot:
      {symbol, qty, entry_date, avg_price}
    """
    lots: Dict[str, List[Dict]] = defaultdict(list)   # symbol → list of open BUY lots
    for _, r in fills.iterrows():
        sym = r["symbol"]
        qty = int(r["filled_qty"])
        if r["side"] == "BUY":
            lots[sym].append({
                "symbol": sym, "qty": qty,
                "entry_date": r["entry_date"],
                "avg_price": float(r["avg_price"]),
            })
        elif r["side"] == "SELL":
            remaining = qty
            while remaining > 0 and lots[sym]:
                lot = lots[sym][0]
                if lot["qty"] <= remaining:
                    remaining -= lot["qty"]
                    lots[sym].pop(0)
                else:
                    lot["qty"] -= remaining
                    remaining = 0
    open_lots: List[Dict] = []
    for sym, lst in lots.items():
        for lot in lst:
            if lot["qty"] > 0:
                open_lots.append(lot)
    return open_lots


def _is_due_for_exit(entry_date, today: datetime.date, hold_days: int = HOLD_DAYS_V2) -> bool:
    """True if `hold_days` trading days have elapsed since entry."""
    elapsed = int(np.busday_count(entry_date, today))
    return elapsed >= hold_days


def _build_sell_order(lot: Dict, ltp_lookup: Optional[Dict[str, float]] = None) -> Dict:
    """Build a SELL order dict shaped like signal_publisher's BUY orders."""
    px = (ltp_lookup or {}).get(lot["symbol"], lot["avg_price"])
    return {
        "symbol":         lot["symbol"],
        "exchange":       "NSE",
        "direction":      "SELL",
        "qty":            int(lot["qty"]),
        "price":          round(float(px), 2),
        "order_value":    round(float(px) * int(lot["qty"]), 2),
        "order_type":     "LIMIT",
        "product":        "CNC",
        "validity":       "DAY",
        "entry_date":     str(lot["entry_date"]),
        "entry_price":    lot["avg_price"],
        "hold_days":      HOLD_DAYS_V2,
        "reason":         "time_exit_v2",
        "generated_at":   datetime.now().isoformat(),
    }


def find_due_exits() -> List[Dict]:
    """Scan execution log, return SELL orders for any open lot past hold horizon."""
    fills = _load_fills()
    if fills.empty:
        print("  [exit] no execution log yet — nothing to exit")
        return []
    open_lots = _open_positions_fifo(fills)
    today = datetime.now().date()
    due = [lot for lot in open_lots if _is_due_for_exit(lot["entry_date"], today)]
    if not open_lots:
        print("  [exit] no open positions")
        return []
    print(f"  [exit] open lots: {len(open_lots)}  due today: {len(due)}")
    return [_build_sell_order(lot) for lot in due]


def _save_orders(orders: List[Dict]) -> Optional[Path]:
    if not orders:
        return None
    _ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    path = _ORDERS_DIR / f"exits_{today}.json"
    with open(path, "w") as f:
        json.dump(orders, f, indent=2, default=str)
    print(f"  [exit] wrote {len(orders)} SELL orders → {path.name}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Place SELLs through OrderManager (paper or live per TRADING_MODE)")
    args = parser.parse_args()

    sells = find_due_exits()
    _save_orders(sells)

    if not args.execute or not sells:
        return

    # Honour TRADING_MODE for live vs paper
    paper = os.getenv("TRADING_MODE", "paper").strip().lower() != "live"

    sys.path.insert(0, str(_LIVE_DIR))
    from order_manager import OrderManager  # noqa
    if paper:
        mgr = OrderManager(client=None, paper_mode=True)
    else:
        from angel_one_client import AngelOneClient  # noqa
        client = AngelOneClient()
        if not client.login():
            print("  [exit] Angel login failed; aborting --execute"); sys.exit(1)
        # Use live LTP for limit price
        for o in sells:
            ltp = client.get_ltp(o["symbol"])
            if ltp:
                o["price"] = round(float(ltp), 2)
                o["order_value"] = round(o["price"] * o["qty"], 2)
        mgr = OrderManager(client=client, paper_mode=False)

    fills = mgr.execute_orders(sells)
    if not paper:
        mgr.wait_for_fills(timeout_min=15)
    mgr.save_execution_log()
    print(f"  [exit] executed {len(fills)} SELL orders ({'paper' if paper else 'live'})")


if __name__ == "__main__":
    main()
