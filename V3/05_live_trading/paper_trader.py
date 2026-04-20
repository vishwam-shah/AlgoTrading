"""
paper_trader.py — Live paper trading session manager
=====================================================
Uses REAL market prices (Angel One WebSocket or yfinance fallback)
but does NOT submit orders to the exchange.

Purpose: run 2+ weeks of paper trading before going live.
Tracks simulated portfolio, fills, P&L vs benchmark.

Usage:
    # Start paper trading session from today's approved orders
    python V3/05_live_trading/paper_trader.py --capital 500000

    # Dry-run a specific orders JSON file
    python V3/05_live_trading/paper_trader.py --orders-file orders_20260416.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_LIVE_DIR    = Path(__file__).resolve().parent
_ORDERS_DIR  = _LIVE_DIR / "orders"
_PT_LOG_DIR  = _LIVE_DIR / "paper_trading_logs"

SLIPPAGE_PCT = 0.0003   # 0.03% simulated slippage per trade
STT_SELL_PCT = 0.001    # 0.1% STT on sell
BROKERAGE    = 20.0     # ₹20 flat per order


class PaperTrader:
    """
    Simulated trading at live market prices.

    State is persisted to paper_trading_logs/ so sessions survive restarts.
    """

    def __init__(self,
                 initial_cash: float = 500_000,
                 session_id: Optional[str] = None):
        self.initial_cash = initial_cash
        self.session_id   = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cash         = initial_cash
        self.holdings:    Dict[str, Dict] = {}   # {symbol: {qty, avg_price}}
        self.trades:      List[Dict]      = []
        self._log_path    = _PT_LOG_DIR / f"session_{self.session_id}.json"

    # ── Execution ──────────────────────────────────────────────────────────────

    def buy(self, symbol: str, qty: int, ltp: float) -> Dict:
        """Simulate a BUY fill at ltp + slippage."""
        fill_price = ltp * (1 + SLIPPAGE_PCT)
        cost       = qty * fill_price + BROKERAGE
        if cost > self.cash:
            return {"status": "INSUFFICIENT_FUNDS", "symbol": symbol}

        self.cash -= cost
        if symbol in self.holdings:
            h = self.holdings[symbol]
            total_qty  = h["qty"] + qty
            h["avg_price"] = (h["qty"] * h["avg_price"] + qty * fill_price) / total_qty
            h["qty"]   = total_qty
        else:
            self.holdings[symbol] = {"qty": qty, "avg_price": fill_price}

        rec = {
            "date":       datetime.now().isoformat(),
            "symbol":     symbol,
            "side":       "BUY",
            "qty":        qty,
            "fill_price": round(fill_price, 2),
            "cost":       round(cost, 2),
            "status":     "FILLED",
        }
        self.trades.append(rec)
        return rec

    def sell(self, symbol: str, qty: Optional[int] = None, ltp: float = 0) -> Dict:
        """Simulate a SELL fill at ltp - slippage."""
        if symbol not in self.holdings:
            return {"status": "NO_POSITION", "symbol": symbol}
        h = self.holdings[symbol]
        qty = qty or h["qty"]
        qty = min(qty, h["qty"])

        fill_price = ltp * (1 - SLIPPAGE_PCT) if ltp > 0 else h["avg_price"]
        proceeds   = qty * fill_price
        stt        = proceeds * STT_SELL_PCT
        net        = proceeds - stt - BROKERAGE
        pnl        = (fill_price - h["avg_price"]) * qty - stt - BROKERAGE

        self.cash += net
        h["qty"] -= qty
        if h["qty"] <= 0:
            del self.holdings[symbol]

        rec = {
            "date":       datetime.now().isoformat(),
            "symbol":     symbol,
            "side":       "SELL",
            "qty":        qty,
            "fill_price": round(fill_price, 2),
            "proceeds":   round(net, 2),
            "pnl":        round(pnl, 2),
            "status":     "FILLED",
        }
        self.trades.append(rec)
        return rec

    # ── Portfolio value ────────────────────────────────────────────────────────

    def portfolio_value(self, prices: Dict[str, float]) -> float:
        """Total value = cash + holdings at current prices."""
        holding_val = sum(
            h["qty"] * prices.get(sym, h["avg_price"])
            for sym, h in self.holdings.items()
        )
        return self.cash + holding_val

    def portfolio_pnl(self, prices: Dict[str, float]) -> Dict:
        """Return dict with total/realized/unrealized P&L and % return."""
        realized   = sum(t["pnl"] for t in self.trades if t["side"] == "SELL" and "pnl" in t)
        holding_val = sum(
            h["qty"] * prices.get(sym, h["avg_price"])
            for sym, h in self.holdings.items()
        )
        cost_basis = sum(h["qty"] * h["avg_price"] for h in self.holdings.values())
        unrealized = holding_val - cost_basis
        total_val  = self.cash + holding_val
        total_pnl  = total_val - self.initial_cash

        return {
            "total_value":    round(total_val, 2),
            "cash":           round(self.cash, 2),
            "holding_value":  round(holding_val, 2),
            "realized_pnl":   round(realized, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_pnl":      round(total_pnl, 2),
            "return_pct":     round(total_pnl / self.initial_cash * 100, 3),
        }

    # ── Run from order file ────────────────────────────────────────────────────

    def execute_order_file(self, orders_path: Path) -> List[Dict]:
        """
        Execute all BUY orders from a signal_publisher JSON file.
        Fetches live prices via yfinance for each symbol.
        """
        with open(orders_path) as f:
            orders = json.load(f)

        symbols = [o["symbol"] for o in orders]
        prices  = _fetch_prices_yf(symbols)
        fills   = []

        for o in orders:
            sym = o["symbol"]
            qty = o.get("qty", 0)
            ltp = prices.get(sym, o.get("price", 0))
            if qty > 0 and ltp > 0:
                rec = self.buy(sym, qty, ltp)
                print(f"  [paper] {rec.get('side','BUY')} {qty}×{sym} @ ₹{ltp:.2f}  "
                      f"status={rec.get('status')}")
                fills.append(rec)

        return fills

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self) -> None:
        _PT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "session_id":   self.session_id,
            "initial_cash": self.initial_cash,
            "cash":         self.cash,
            "holdings":     self.holdings,
            "trades":       self.trades,
            "saved_at":     datetime.now().isoformat(),
        }
        with open(self._log_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    @classmethod
    def load(cls, session_id: str) -> "PaperTrader":
        path = _PT_LOG_DIR / f"session_{session_id}.json"
        with open(path) as f:
            state = json.load(f)
        pt = cls(initial_cash=state["initial_cash"], session_id=session_id)
        pt.cash     = state["cash"]
        pt.holdings = state["holdings"]
        pt.trades   = state["trades"]
        return pt

    def summary(self, prices: Optional[Dict[str, float]] = None) -> None:
        prices = prices or {sym: h["avg_price"] for sym, h in self.holdings.items()}
        pnl    = self.portfolio_pnl(prices)
        print(f"\n{'═'*60}")
        print(f"  Paper Trading Session: {self.session_id}")
        print(f"  Portfolio Value : ₹{pnl['total_value']:>12,.2f}")
        print(f"  Cash            : ₹{pnl['cash']:>12,.2f}")
        print(f"  Holdings Value  : ₹{pnl['holding_value']:>12,.2f}")
        print(f"  Realized P&L    : ₹{pnl['realized_pnl']:>12,.2f}")
        print(f"  Unrealized P&L  : ₹{pnl['unrealized_pnl']:>12,.2f}")
        print(f"  Total Return    :  {pnl['return_pct']:>+.3f}%")
        print(f"  Trades executed : {len(self.trades)}")
        print(f"  Holdings        : {len(self.holdings)} stocks")
        print(f"{'═'*60}\n")


def _fetch_prices_yf(symbols: List[str]) -> Dict[str, float]:
    """Fetch latest close prices via yfinance."""
    try:
        import yfinance as yf
        tickers = [f"{s}.NS" for s in symbols]
        data    = yf.download(tickers, period="2d", auto_adjust=True,
                              progress=False, threads=True)
        prices  = {}
        closes  = data["Close"].iloc[-1] if "Close" in data.columns else data.iloc[-1]
        import numpy as np
        for sym, t in zip(symbols, tickers):
            val = closes.get(t, np.nan)
            if not (isinstance(val, float) and np.isnan(val)):
                prices[sym] = float(val)
        return prices
    except Exception as e:
        print(f"  [paper] price fetch failed: {e}")
        return {}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Paper trading session")
    parser.add_argument("--capital",     type=float, default=500_000)
    parser.add_argument("--orders-file", default=None,
                        help="Path to orders JSON (default: latest in orders/)")
    parser.add_argument("--session-id",  default=None,
                        help="Resume existing session ID")
    args = parser.parse_args()

    if args.session_id:
        pt = PaperTrader.load(args.session_id)
        print(f"  Resumed session {args.session_id} | {len(pt.trades)} trades")
    else:
        pt = PaperTrader(initial_cash=args.capital)
        print(f"  New paper trading session: {pt.session_id}")

    orders_path = (Path(args.orders_file) if args.orders_file
                   else sorted(_ORDERS_DIR.glob("orders_*.json"), reverse=True)[0]
                   if list(_ORDERS_DIR.glob("orders_*.json")) else None)

    if not orders_path:
        print("  No orders file found. Run signal_publisher.py first.")
        sys.exit(1)

    print(f"  Executing: {orders_path.name}")
    fills = pt.execute_order_file(orders_path)
    pt.save()

    # Show summary with live prices
    syms   = list(pt.holdings.keys())
    prices = _fetch_prices_yf(syms)
    pt.summary(prices)
