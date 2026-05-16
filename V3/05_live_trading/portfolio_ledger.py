"""
portfolio_ledger.py — Canonical state of the live portfolio
============================================================
One source of truth for:
  • previous-day NAV
  • open lots (FIFO with entry date / price)
  • realized P&L (closed lots, gross + net of costs)
  • pending orders (not yet filled)
  • current holdings (= sum of open lots per symbol)

Consumed by:
  • signal_publisher (sizing capped by available capital)
  • exit_runner (which lots to close)
  • paper_pnl_reconciler (compare expected vs actual)
  • promotion_gate (rolling Sharpe, win rate, drawdown)
  • risk_guard (sector + drawdown circuit breakers)

Storage layout (under V3/05_live_trading/ledger/):
  state.json              — single canonical snapshot (re-written atomically)
  events.parquet          — append-only audit log of every BUY/SELL/MTM event
  nav_history.parquet     — daily NAV / equity / cash for charts and gates

The ledger is rebuilt from execution_log.parquet on demand (idempotent), so
losing state.json never loses history.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_LIVE_DIR  = Path(__file__).resolve().parent
_LEDGER    = _LIVE_DIR / "ledger"
_LEDGER.mkdir(parents=True, exist_ok=True)
_STATE     = _LEDGER / "state.json"
_EVENTS    = _LEDGER / "events.parquet"
_NAV       = _LEDGER / "nav_history.parquet"
_EXEC      = _LIVE_DIR / "execution_logs" / "execution_log.parquet"

sys.path.insert(0, str(_LIVE_DIR.parent / "00_config"))
from risk_config import HOT as _RC, get as _rcget  # type: ignore  # noqa: E402

DEFAULT_CAPITAL = float(_rcget("sizing", "capital_default_inr", default=500_000))


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Lot:
    symbol:      str
    qty:         int
    entry_price: float
    entry_date:  str       # ISO date string
    order_id:    str
    cost_at_entry: float = 0.0   # brokerage + STT + slippage charged on BUY


@dataclass
class ClosedTrade:
    symbol:        str
    qty:           int
    entry_price:   float
    exit_price:    float
    entry_date:    str
    exit_date:     str
    gross_pnl:     float
    cost_total:    float          # round-trip cost (BUY + SELL)
    net_pnl:       float
    hold_days:     int
    buy_order_id:  str
    sell_order_id: str


@dataclass
class PendingOrder:
    order_id:     str
    symbol:       str
    side:         str
    qty:          int
    price:        float
    placed_at:    str
    expected_fill: float = 0.0     # signal price; used for slippage drift


@dataclass
class LedgerState:
    # capital
    starting_capital:  float                  = DEFAULT_CAPITAL
    cash:              float                  = DEFAULT_CAPITAL
    realized_pnl:      float                  = 0.0
    nav_yesterday:     float                  = DEFAULT_CAPITAL
    # books
    open_lots:         List[Lot]              = field(default_factory=list)
    closed_trades:     List[ClosedTrade]      = field(default_factory=list)
    pending_orders:    List[PendingOrder]     = field(default_factory=list)
    # bookkeeping
    last_rebuilt_at:   str                    = ""
    last_event_id:     int                    = 0


# ── Persistence ──────────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(path)   # replace() overwrites on Windows; rename() does not


def _read_state() -> LedgerState:
    if not _STATE.exists():
        return LedgerState()
    try:
        with open(_STATE) as f:
            d = json.load(f)
        return LedgerState(
            starting_capital=float(d.get("starting_capital", DEFAULT_CAPITAL)),
            cash=float(d.get("cash", DEFAULT_CAPITAL)),
            realized_pnl=float(d.get("realized_pnl", 0.0)),
            nav_yesterday=float(d.get("nav_yesterday", DEFAULT_CAPITAL)),
            open_lots=[Lot(**lot) for lot in d.get("open_lots", [])],
            closed_trades=[ClosedTrade(**t) for t in d.get("closed_trades", [])],
            pending_orders=[PendingOrder(**p) for p in d.get("pending_orders", [])],
            last_rebuilt_at=d.get("last_rebuilt_at", ""),
            last_event_id=int(d.get("last_event_id", 0)),
        )
    except Exception as e:
        print(f"  [ledger] WARN — state corrupt ({e}); rebuilding from execution log")
        return rebuild_from_executions(starting_capital=DEFAULT_CAPITAL)


def _write_state(state: LedgerState) -> None:
    payload = {
        "starting_capital":  state.starting_capital,
        "cash":              state.cash,
        "realized_pnl":      state.realized_pnl,
        "nav_yesterday":     state.nav_yesterday,
        "open_lots":         [asdict(l) for l in state.open_lots],
        "closed_trades":     [asdict(t) for t in state.closed_trades],
        "pending_orders":    [asdict(p) for p in state.pending_orders],
        "last_rebuilt_at":   state.last_rebuilt_at,
        "last_event_id":     state.last_event_id,
    }
    _atomic_write_json(_STATE, payload)


def _append_event(event: dict) -> None:
    df = pd.DataFrame([{**event, "ts": datetime.now().isoformat()}])
    if _EVENTS.exists():
        try:
            existing = pd.read_parquet(_EVENTS)
            df = pd.concat([existing, df], ignore_index=True)
        except Exception:
            pass
    tmp = _EVENTS.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(_EVENTS)


# ── Rebuild from executions (truth source) ────────────────────────────────────

def rebuild_from_executions(starting_capital: float = DEFAULT_CAPITAL) -> LedgerState:
    """
    Idempotent rebuild — process every FILLED order in execution_log.parquet
    in chronological order, FIFO-match SELLs against the oldest open BUY, and
    recompute cash + realized P&L from scratch.
    """
    state = LedgerState(starting_capital=starting_capital,
                        cash=starting_capital,
                        nav_yesterday=starting_capital)
    if not _EXEC.exists():
        state.last_rebuilt_at = datetime.now().isoformat()
        _write_state(state)
        return state

    df = pd.read_parquet(_EXEC)
    df = df[df["status"] == "FILLED"].copy()
    if df.empty:
        state.last_rebuilt_at = datetime.now().isoformat()
        _write_state(state)
        return state

    df["filled_at_ts"] = pd.to_datetime(df["filled_at"])
    df = df.sort_values("filled_at_ts").reset_index(drop=True)

    open_lots: Dict[str, List[Lot]] = defaultdict(list)
    for _, r in df.iterrows():
        sym = str(r["symbol"]); qty = int(r["filled_qty"])
        px  = float(r["avg_price"]); when = pd.to_datetime(r["filled_at"]).date().isoformat()
        oid = str(r.get("order_id", ""))
        cost = float(r.get("brokerage", 0.0) + r.get("stt", 0.0) + r.get("other_charges", 0.0))
        if r["side"] == "BUY":
            state.cash -= float(r.get("net_cost", qty * px))
            open_lots[sym].append(Lot(
                symbol=sym, qty=qty, entry_price=px, entry_date=when,
                order_id=oid, cost_at_entry=cost,
            ))
        elif r["side"] == "SELL":
            remaining = qty
            sell_value = float(r.get("net_cost", qty * px))
            state.cash += sell_value
            sell_cost = cost
            while remaining > 0 and open_lots[sym]:
                lot = open_lots[sym][0]
                consume = min(lot.qty, remaining)
                gross = (px - lot.entry_price) * consume
                allocated_buy_cost = lot.cost_at_entry * (consume / max(lot.qty, 1))
                allocated_sell_cost = sell_cost * (consume / max(qty, 1))
                cost_total = allocated_buy_cost + allocated_sell_cost
                net = gross - cost_total
                state.realized_pnl += net
                state.closed_trades.append(ClosedTrade(
                    symbol=sym, qty=consume, entry_price=lot.entry_price, exit_price=px,
                    entry_date=lot.entry_date, exit_date=when,
                    gross_pnl=round(gross, 2), cost_total=round(cost_total, 2),
                    net_pnl=round(net, 2),
                    hold_days=_busdays(lot.entry_date, when),
                    buy_order_id=lot.order_id, sell_order_id=oid,
                ))
                lot.qty -= consume
                lot.cost_at_entry -= allocated_buy_cost
                if lot.qty <= 0:
                    open_lots[sym].pop(0)
                remaining -= consume

    # Flatten open lots into state
    for sym, lots in open_lots.items():
        for lot in lots:
            if lot.qty > 0:
                state.open_lots.append(lot)

    state.last_rebuilt_at = datetime.now().isoformat()
    state.last_event_id = len(df)
    _write_state(state)
    return state


def _busdays(d0: str, d1: str) -> int:
    try:
        return int(np.busday_count(date.fromisoformat(d0), date.fromisoformat(d1)))
    except Exception:
        return 0


# ── Public API ───────────────────────────────────────────────────────────────

def get_state() -> LedgerState:
    """Return current ledger state (lazy rebuild if missing)."""
    return _read_state()


def open_holdings(state: Optional[LedgerState] = None) -> Dict[str, int]:
    """Net-open quantity per symbol."""
    s = state or _read_state()
    out: Dict[str, int] = defaultdict(int)
    for lot in s.open_lots:
        out[lot.symbol] += lot.qty
    return dict(out)


def open_market_value(state: Optional[LedgerState] = None,
                      ltp: Optional[Dict[str, float]] = None) -> float:
    """Σ qty × LTP (or entry_price if LTP missing)."""
    s = state or _read_state()
    ltp = ltp or {}
    return float(sum(lot.qty * ltp.get(lot.symbol, lot.entry_price) for lot in s.open_lots))


def nav(state: Optional[LedgerState] = None,
        ltp: Optional[Dict[str, float]] = None) -> float:
    """Net asset value = cash + open market value (LTP-marked)."""
    s = state or _read_state()
    return float(s.cash + open_market_value(s, ltp))


def record_pending(orders: List[Dict]) -> None:
    """Persist freshly-placed orders before fills land."""
    s = _read_state()
    for o in orders:
        s.pending_orders.append(PendingOrder(
            order_id=str(o.get("order_id") or o.get("orderid", "")),
            symbol=o["symbol"], side=str(o.get("direction", o.get("side", "BUY"))).upper(),
            qty=int(o["qty"]), price=float(o["price"]),
            placed_at=str(o.get("placed_at", datetime.now().isoformat())),
            expected_fill=float(o.get("price", 0.0)),
        ))
    _write_state(s)


def mark_to_market(ltp: Dict[str, float]) -> Tuple[float, float]:
    """Roll forward NAV using today's LTPs. Returns (nav_today, daily_return)."""
    s = _read_state()
    nav_today = nav(s, ltp)
    nav_y = s.nav_yesterday or s.starting_capital
    daily_ret = (nav_today - nav_y) / nav_y if nav_y > 0 else 0.0
    # Append to nav_history
    row = pd.DataFrame([{
        "date": date.today().isoformat(),
        "nav": round(nav_today, 2),
        "cash": round(s.cash, 2),
        "open_value": round(open_market_value(s, ltp), 2),
        "realized_pnl": round(s.realized_pnl, 2),
        "daily_return": round(daily_ret, 6),
    }])
    if _NAV.exists():
        try:
            existing = pd.read_parquet(_NAV)
            row = pd.concat([existing, row], ignore_index=True)
            row = row.drop_duplicates(subset=["date"], keep="last")
        except Exception:
            pass
    tmp = _NAV.with_suffix(".parquet.tmp")
    row.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(_NAV)
    s.nav_yesterday = nav_today
    _write_state(s)
    return float(nav_today), float(daily_ret)


def summary(ltp: Optional[Dict[str, float]] = None) -> Dict:
    """One-call snapshot for dashboards / promotion gate."""
    s = _read_state()
    nv = nav(s, ltp)
    holdings = open_holdings(s)
    return {
        "nav":              round(nv, 2),
        "cash":             round(s.cash, 2),
        "realized_pnl":     round(s.realized_pnl, 2),
        "unrealized_pnl":   round(open_market_value(s, ltp) - sum(l.qty * l.entry_price for l in s.open_lots), 2),
        "open_lots":        len(s.open_lots),
        "open_symbols":     sorted(holdings.keys()),
        "holdings_qty":     holdings,
        "pending_orders":   len(s.pending_orders),
        "closed_trades":    len(s.closed_trades),
        "starting_capital": s.starting_capital,
        "ret_total_pct":    round((nv - s.starting_capital) / s.starting_capital * 100, 3) if s.starting_capital > 0 else 0.0,
        "last_rebuilt_at":  s.last_rebuilt_at,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true",
                   help="Rebuild state from execution_log.parquet")
    p.add_argument("--starting-capital", type=float, default=DEFAULT_CAPITAL)
    p.add_argument("--summary", action="store_true",
                   help="Print a snapshot summary (default action)")
    a = p.parse_args()

    if a.rebuild:
        s = rebuild_from_executions(starting_capital=a.starting_capital)
        print(f"  rebuilt: {len(s.open_lots)} open lots, "
              f"{len(s.closed_trades)} closed trades, "
              f"realized P&L ₹{s.realized_pnl:,.2f}, cash ₹{s.cash:,.2f}")
        return 0

    snap = summary()
    print(json.dumps(snap, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
