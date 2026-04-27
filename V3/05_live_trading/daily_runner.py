"""
daily_runner.py — Master daily automation for the AlgoTrading system
=====================================================================
Two modes:

  EVENING (T-1, ~6 PM IST):
    1. Fetch today's news sentiment (all 100 symbols)
    2. Run V3 pipeline (fast mode — generates predictions + next_day_predictions.csv)
    3. Build + validate orders → approved_orders.json
    Cron: 0 18 * * 1-5  (Mon–Fri 6 PM IST)

  MORNING (T, 9:00 AM IST):
    1. Angel One login
    2. Fetch live prices for approved symbols
    3. Check slippage guard
    4. Place LIMIT orders
    5. Wait for fills (up to 30 min)
    6. Save execution log
    Cron: 0 9 * * 1-5  (Mon–Fri 9 AM IST)

  EVENING RECONCILE (T, 3:45 PM IST):
    1. Fetch final holdings from Angel One
    2. Compare with execution log
    3. Append to trade_history.parquet
    Cron: 45 15 * * 1-5

Usage:
    python V3/05_live_trading/daily_runner.py --mode evening
    python V3/05_live_trading/daily_runner.py --mode morning --capital 500000
    python V3/05_live_trading/daily_runner.py --mode reconcile
    python V3/05_live_trading/daily_runner.py --mode morning --paper   # dry run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_LIVE_DIR = Path(__file__).resolve().parent
_V3_ROOT  = _LIVE_DIR.parent

# Load .env so TRADING_MODE / ANGEL_* are available when cron invokes this script.
_ENV_PATH = _V3_ROOT.parent / ".env"
if _ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH)
    except Exception:
        pass


def _resolve_paper_mode(cli_paper: bool) -> bool:
    """
    Precedence: CLI --paper flag (if set True) > TRADING_MODE env var > default paper.

    TRADING_MODE=live  → real orders on Angel One
    TRADING_MODE=paper → simulated fills only (DEFAULT for safety)
    """
    if cli_paper:
        return True
    mode = os.getenv("TRADING_MODE", "paper").strip().lower()
    return mode != "live"
_PIPELINE_DIR = _V3_ROOT / "07_pipeline"
_NEWS_DIR     = _V3_ROOT / "01_data" / "news"
_ORDERS_DIR   = _LIVE_DIR / "orders"
_RESULTS_DIR  = _V3_ROOT / "06_results" / "runs"
_HISTORY_PATH = _LIVE_DIR / "trade_history.parquet"

# Python executable (venv)
_PYTHON = str(Path(sys.executable))

# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str) -> int:
    print(f"\n  ┌─ {label}")
    result = subprocess.run(cmd, cwd=str(_V3_ROOT.parent))
    print(f"  └─ exit={result.returncode}")
    return result.returncode


def _latest_run_id() -> Optional[str]:
    runs = sorted(_RESULTS_DIR.glob("20*"), reverse=True)
    return runs[0].name if runs else None


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

# ══════════════════════════════════════════════════════════════════════════════
#  EVENING — Generate predictions & orders
# ══════════════════════════════════════════════════════════════════════════════

def run_evening(capital: float = 500_000) -> None:
    """
    T-1 Evening sequence (run after 4 PM IST, before market close next day).
    Takes ~30–60 min (fast mode, 100 stocks).
    """
    _log("=== EVENING RUN STARTED ===")

    # 1. Fetch news sentiment
    _log("Step 1/3 — Fetching news sentiment …")
    rc = _run([_PYTHON, str(_NEWS_DIR / "sentiment_history.py")], "sentiment_history.py")
    if rc != 0:
        _log("WARNING: Sentiment fetch failed — continuing with cached data")

    # 2. Run V3 pipeline (fast mode = LightGBM + XGBoost only)
    _log("Step 2/3 — Running V3 pipeline (fast mode) …")
    rc = _run([
        _PYTHON, str(_PIPELINE_DIR / "orchestrator.py"),
        "--fast", "--force-features",
    ], "orchestrator.py --fast")
    if rc != 0:
        _log("ERROR: Pipeline failed — aborting evening run")
        sys.exit(1)

    # 3. Generate orders from predictions
    _log("Step 3/3 — Building approved orders …")
    run_id = _latest_run_id()
    if not run_id:
        _log("ERROR: No run directory found after pipeline")
        sys.exit(1)

    rc = _run([
        _PYTHON, str(_LIVE_DIR / "signal_publisher.py"),
        "--run-id", run_id,
        "--capital", str(int(capital)),
        "--dry-run",
    ], f"signal_publisher.py  run_id={run_id}")
    if rc != 0:
        _log("WARNING: signal_publisher failed")

    _log(f"=== EVENING RUN COMPLETE | run_id={run_id} ===")


# ══════════════════════════════════════════════════════════════════════════════
#  MORNING — Place orders
# ══════════════════════════════════════════════════════════════════════════════

def run_morning(capital: float = 500_000, paper: bool = False) -> None:
    """
    T Morning sequence (run at 9:00 AM IST, 15 min before market open).

    Step 0 (NEW): close any positions whose 10-trading-day hold has elapsed.
    Step 1+    : place today's BUY orders.
    """
    _log(f"=== MORNING RUN STARTED | {'PAPER MODE' if paper else 'LIVE MODE'} ===")

    # ── Step 0: time-based exits ──────────────────────────────────────────────
    # Backtest holds positions exactly 10 trading days; live must do the same.
    # Run exit_runner first so the SELLs go in before we add new BUYs.
    _log("Step 0/3 — Running exit_runner …")
    rc = _run([
        _PYTHON, str(_LIVE_DIR / "exit_runner.py"), "--execute",
    ], "exit_runner.py --execute")
    if rc != 0:
        _log("WARNING: exit_runner returned non-zero — continuing")

    # Load today's approved orders
    order_files = sorted(_ORDERS_DIR.glob("orders_*.json"), reverse=True)
    if not order_files:
        _log("ERROR: No order files found. Did evening run complete?")
        sys.exit(1)

    latest = order_files[0]
    _log(f"Loading orders: {latest.name}")
    with open(latest) as f:
        orders = json.load(f)

    if not orders:
        _log("No orders to place today.")
        return

    _log(f"  {len(orders)} orders to place | ₹{sum(o['order_value'] for o in orders):,.0f} total")

    if paper:
        # Paper mode — simulate fills only
        from order_manager import OrderManager
        mgr = OrderManager(client=None, paper_mode=True)
        mgr.execute_orders(orders)
        mgr.save_execution_log()
        _log("=== MORNING RUN COMPLETE (PAPER) ===")
        return

    # Live mode
    try:
        from angel_one_client import AngelOneClient
        from order_manager import OrderManager
        from risk_guard import RiskGuard
    except ImportError as e:
        _log(f"ERROR: Import failed: {e}")
        sys.exit(1)

    client = AngelOneClient()
    if not client.login():
        _log("ERROR: Angel One login failed")
        sys.exit(1)

    # Get current portfolio state for risk checks
    holdings     = client.get_holdings()
    funds        = client.get_funds()
    cash         = funds.get("available", capital)
    current_vals = {sym: pos.qty * pos.ltp for sym, pos in holdings.items()}
    portval      = cash + sum(current_vals.values())

    _log(f"  Portfolio: ₹{portval:,.0f}  Cash: ₹{cash:,.0f}  "
         f"Holdings: {len(holdings)} stocks")

    # Daily loss circuit breaker
    can_trade, reason = RiskGuard.check_daily_loss(portval, capital)
    if not can_trade:
        _log(f"CIRCUIT BREAKER: {reason} — no orders placed")
        sys.exit(0)

    # Market hours guard
    can_trade, reason = RiskGuard.check_market_hours()
    # Allow running slightly before open (will place orders at 9:15 open)

    # Full risk validation
    approved, rejected = RiskGuard.validate_batch(
        orders, portval, {s: p.qty for s, p in holdings.items()}, current_vals
    )
    _log(f"  RiskGuard: {len(approved)} approved, {len(rejected)} blocked")

    # Subscribe WebSocket for live prices
    syms = [o["symbol"] for o in approved]
    client.subscribe_ticks(syms)
    import time; time.sleep(3)  # let WS warm up

    # Execute
    mgr = OrderManager(client=client, paper_mode=False)
    fills = mgr.execute_orders(approved, max_slippage_pct=0.003)
    _log(f"  Orders placed: {len(fills)}")

    # Wait for fills (30 min timeout)
    mgr.wait_for_fills(timeout_min=30, poll_interval_sec=60)

    summary = mgr.summary()
    _log(f"  Fills: {summary['filled']} filled, {summary['rejected']} rejected, "
         f"{summary['timeout']} timeout | charges ₹{summary['total_charges']:,.0f}")

    log_path = mgr.save_execution_log(run_id=_latest_run_id())
    client.stop_websocket()
    _log(f"=== MORNING RUN COMPLETE | log={log_path.name} ===")


# ══════════════════════════════════════════════════════════════════════════════
#  RECONCILE — Evening portfolio reconciliation
# ══════════════════════════════════════════════════════════════════════════════

def run_reconcile() -> None:
    """
    T Evening (3:45 PM IST) — reconcile Angel One holdings with execution log.
    Appends to trade_history.parquet.
    """
    _log("=== RECONCILE STARTED ===")

    import pandas as pd

    try:
        from angel_one_client import AngelOneClient
        client = AngelOneClient()
        client.login()
    except Exception as e:
        _log(f"ERROR: {e}")
        sys.exit(1)

    holdings  = client.get_holdings()
    order_bk  = client.get_order_book()
    funds     = client.get_funds()

    today = datetime.now().strftime("%Y-%m-%d")
    rows  = []

    for sym, pos in holdings.items():
        rows.append({
            "date":       today,
            "symbol":     sym,
            "qty":        pos.qty,
            "avg_price":  pos.avg_price,
            "ltp":        pos.ltp,
            "pnl":        pos.pnl,
            "source":     "holdings",
        })

    for o in order_bk:
        status = o.get("orderstatus", "").lower()
        if status in ("complete", "filled"):
            rows.append({
                "date":       today,
                "symbol":     o.get("tradingsymbol", ""),
                "qty":        int(o.get("filledshares", 0)),
                "avg_price":  float(o.get("averageprice", 0)),
                "ltp":        float(o.get("ltp", 0)),
                "pnl":        float(o.get("realizedprofitloss", 0)),
                "source":     "orderbook",
            })

    if rows:
        df = pd.DataFrame(rows)
        if _HISTORY_PATH.exists():
            try:
                existing = pd.read_parquet(_HISTORY_PATH)
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=["date", "symbol", "source"], keep="last")
            except Exception:
                _log("  WARNING: trade_history.parquet corrupted — starting fresh")
        _tmp = _HISTORY_PATH.with_suffix(".parquet.tmp")
        df.to_parquet(_tmp, index=False, compression="snappy")
        _tmp.rename(_HISTORY_PATH)
        _log(f"  {len(rows)} records → {_HISTORY_PATH.name}")

    _log(f"  Funds: available=₹{funds.get('available', 0):,.0f}  "
         f"net=₹{funds.get('net', 0):,.0f}")
    client.stop_websocket()

    # Paper-trading P&L reconciliation: live-paper round-trips vs backtest predictions.
    # Always-on (paper or live) — runs on the local execution_log.parquet.
    _log("Running paper P&L reconciler …")
    _run([_PYTHON, str(_LIVE_DIR / "paper_pnl_reconciler.py")], "paper_pnl_reconciler.py")

    _log("=== RECONCILE COMPLETE ===")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily algo-trading runner")
    parser.add_argument("--mode",    required=True,
                        choices=["evening", "morning", "reconcile"],
                        help="Which phase to run")
    parser.add_argument("--capital", type=float, default=500_000,
                        help="Total portfolio capital in INR (default 500000)")
    parser.add_argument("--paper",   action="store_true",
                        help="Paper mode — simulate without placing real orders")
    args = parser.parse_args()

    if args.mode == "evening":
        run_evening(capital=args.capital)
    elif args.mode == "morning":
        paper = _resolve_paper_mode(args.paper)
        _log(f"Trading mode resolved → {'PAPER' if paper else 'LIVE'} "
             f"(TRADING_MODE={os.getenv('TRADING_MODE', 'paper')})")
        run_morning(capital=args.capital, paper=paper)
    elif args.mode == "reconcile":
        run_reconcile()
