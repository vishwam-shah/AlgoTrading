"""
signal_publisher.py — Predictions → Position Sizes → Angel One Orders
======================================================================
Reads next_day_predictions.csv produced by orchestrator.py (or predict mode),
applies volatility-adjusted Kelly sizing, validates via risk_guard, and emits
structured order dicts ready for order_manager.place_order().

Usage (dry-run):
    python V3/05_live_trading/signal_publisher.py \
        --run-id 20260307_141956 --capital 500000 --dry-run

Usage (live):
    python V3/05_live_trading/signal_publisher.py \
        --run-id 20260307_141956 --capital 500000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_LIVE_DIR = Path(__file__).resolve().parent
_V3_ROOT  = _LIVE_DIR.parent
sys.path.insert(0, str(_V3_ROOT))
sys.path.insert(0, str(_V3_ROOT / "07_pipeline"))

# ── Constants ────────────────────────────────────────────────────────────────

MAX_POSITION_PCT   = 0.12   # 12% max per stock
MIN_CONFIDENCE     = 0.52   # only trade above this prob_up
MIN_PROB_REQUIRED  = True   # skip order if prob_up missing in csv
MAX_STOCKS         = 15     # max simultaneous positions
TARGET_HOLD_DAYS   = 7      # expected hold (for sizing context, not enforced here)
ROUND_LOT          = 1      # NSE CNC: 1-share lots allowed

# NSE round-trip cost estimate: STT 0.1% sell + ~0.09% other ≈ 0.19% one-way
ROUND_TRIP_COST_PCT = 0.0038  # ~0.38% full round-trip for 7-day hold sizing

RESULTS_DIR = _V3_ROOT / "06_results" / "runs"
ORDERS_DIR  = _V3_ROOT / "05_live_trading" / "orders"


# ── Kelly / volatility-adjusted sizing ──────────────────────────────────────

def kelly_fraction(prob_up: float, win_loss_ratio: float = 1.5) -> float:
    """
    Full Kelly: f* = (bp - q) / b  where b = win/loss ratio, p = prob up, q = 1 - p.
    Capped at 0.25 full Kelly then halved (half-Kelly) for safety.
    """
    q = 1.0 - prob_up
    b = win_loss_ratio
    f = (b * prob_up - q) / b
    return max(0.0, min(f * 0.5, MAX_POSITION_PCT))  # half-Kelly, hard cap


def vol_adjusted_size(base_frac: float, atr_pct: float,
                      target_vol: float = 0.015) -> float:
    """Scale down if stock is more volatile than target (1.5% daily ATR)."""
    if atr_pct <= 0:
        return base_frac
    return base_frac * (target_vol / atr_pct)


# ── Load predictions ─────────────────────────────────────────────────────────

def load_predictions(run_id: Optional[str] = None,
                     pred_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load next_day_predictions.csv from a run directory.
    Columns expected: symbol, direction, confidence, [prob_up], [atr_pct]
    """
    if pred_path is None:
        if run_id is None:
            # Auto-detect latest run
            runs = sorted(RESULTS_DIR.glob("*/next_day_predictions.csv"))
            if not runs:
                raise FileNotFoundError(f"No next_day_predictions.csv found under {RESULTS_DIR}")
            pred_path = runs[-1]
            run_id = pred_path.parent.name
        else:
            pred_path = RESULTS_DIR / run_id / "next_day_predictions.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    df = pd.read_csv(pred_path)

    # Normalise probability column — pipeline writes avg_prob, publisher expects prob_up
    if "avg_prob" in df.columns and "prob_up" not in df.columns:
        df["prob_up"] = df["avg_prob"]
    # Also use last_close as price hint if present
    if "last_close" in df.columns and "price" not in df.columns:
        df["price_hint"] = df["last_close"]

    try:
        rel = pred_path.relative_to(_V3_ROOT)
    except ValueError:
        rel = pred_path
    print(f"  Loaded {len(df)} predictions from {rel}")
    return df, run_id


# ── Build orders ─────────────────────────────────────────────────────────────

def build_orders(
    pred_df: pd.DataFrame,
    capital: float,
    price_map: Optional[dict] = None,
    run_id: Optional[str] = None,
) -> list[dict]:
    """
    Convert predictions DataFrame to a list of order dicts.

    Parameters
    ----------
    pred_df   : DataFrame with columns: symbol, direction, confidence, [prob_up], [atr_pct]
    capital   : Total portfolio capital in INR
    price_map : {symbol: last_price}. If None, fetched live via yfinance.
    run_id    : Used to load backtest_results.csv for sharpe-based filtering.

    Returns
    -------
    List of order dicts (sorted by position size descending):
        symbol, direction, confidence, prob_up, kelly_frac, target_pct,
        target_inr, qty, price, order_type, product, validity
    """
    df = pred_df.copy()

    # ── 1. Direction filter — UP signals only (NSE CNC: no short selling) ─
    df = df[df["direction"] == "UP"].copy()
    if df.empty:
        print("  No UP signals today.")
        return []

    # ── 2. Sharpe-profitable universe filter ──────────────────────────────
    # Only trade stocks that are both directionally valid (OOS>50%) AND
    # actually made money in backtest simulation (sharpe>0).
    # Prefer the tradeable flag in predictions (updated by orchestrator after backtest).
    # Fall back to loading backtest_results.csv directly.
    profitable_symbols: Optional[set] = None
    if "tradeable" in df.columns:
        profitable_symbols = set(df[df["tradeable"] == True]["symbol"].tolist())
    elif run_id:
        bt_path = RESULTS_DIR / run_id / "backtest_results.csv"
        if bt_path.exists():
            bt_df = pd.read_csv(bt_path)
            profitable_symbols = set(
                bt_df[(bt_df["sharpe"] > 0) & (bt_df["oos_accuracy"] >= 0.50)]["symbol"].tolist()
            )

    if profitable_symbols is not None:
        before = len(df)
        df = df[df["symbol"].isin(profitable_symbols)].copy()
        print(f"  {len(df)}/{before} signals in profitable universe (sharpe>0, OOS>50%)")
        if df.empty:
            print("  No signals in profitable universe today.")
            return []

    # ── 3. Confidence / prob_up filter ────────────────────────────────────
    if "prob_up" in df.columns and df["prob_up"].notna().any():
        df = df[df["prob_up"] >= MIN_CONFIDENCE].copy()
        print(f"  {len(df)} signals above prob_up ≥ {MIN_CONFIDENCE:.0%}")
    elif MIN_PROB_REQUIRED:
        if "confidence" in df.columns:
            df = df[df["confidence"] >= MIN_CONFIDENCE].copy()
            print(f"  {len(df)} signals above confidence ≥ {MIN_CONFIDENCE:.0%} (no prob_up)")
        else:
            print("  WARNING: no prob_up or confidence — using all UP signals")
    else:
        print(f"  {len(df)} UP signals (no prob_up filtering)")

    if df.empty:
        return []

    # ── 4. Sort by prob_up descending, take top N ─────────────────────────
    sort_col = "prob_up" if "prob_up" in df.columns else "confidence"
    df = df.sort_values(sort_col, ascending=False).head(MAX_STOCKS).reset_index(drop=True)

    # ── 4. Fetch prices if not provided ───────────────────────────────────
    if price_map is None:
        price_map = _fetch_prices(df["symbol"].tolist())

    # ── 5. Compute position sizes ─────────────────────────────────────────
    orders = []
    for _, row in df.iterrows():
        sym      = row["symbol"]
        prob_up  = float(row.get("prob_up", row.get("confidence", 0.55)))
        atr_pct  = float(row.get("atr_pct", 0.015))   # default 1.5% if missing
        price = price_map.get(sym)
        if (price is None or price <= 0) and "price_hint" in row and row["price_hint"] > 0:
            price = float(row["price_hint"])   # fallback to last_close from predictions
            print(f"  ℹ  {sym}: using last_close ₹{price:.2f} (live price unavailable)")
        if price is None or price <= 0:
            print(f"  ⚠  {sym}: no price available, skipping")
            continue

        kf         = kelly_fraction(prob_up)
        target_pct = vol_adjusted_size(kf, atr_pct)
        target_pct = min(target_pct, MAX_POSITION_PCT)
        target_inr = capital * target_pct
        qty        = max(1, int(target_inr / price))   # floor shares, min 1

        orders.append({
            "symbol":     sym,
            "exchange":   "NSE",
            "direction":  "BUY",
            "prob_up":    round(prob_up, 4),
            "kelly_frac": round(kf, 4),
            "target_pct": round(target_pct * 100, 2),   # as %
            "target_inr": round(target_inr, 0),
            "qty":        qty,
            "price":      round(price, 2),
            "order_value":round(qty * price, 0),
            "order_type": "LIMIT",
            "product":    "CNC",     # delivery, not intraday MIS
            "validity":   "DAY",
            "generated_at": datetime.now().isoformat(),
        })

    # Sort by target_inr desc
    orders.sort(key=lambda o: o["target_inr"], reverse=True)

    # ── 6. Portfolio-level capital check ──────────────────────────────────
    total_deployed = sum(o["order_value"] for o in orders)
    if total_deployed > capital:
        # Scale all down proportionally
        scale = capital / total_deployed
        for o in orders:
            o["qty"]         = max(1, int(o["qty"] * scale))
            o["order_value"] = round(o["qty"] * o["price"], 0)
            o["target_pct"]  = round(o["order_value"] / capital * 100, 2)
        total_deployed = sum(o["order_value"] for o in orders)

    return orders


# ── Price fetcher ─────────────────────────────────────────────────────────────

def _fetch_prices(symbols: list[str]) -> dict:
    """Fetch last closing prices via yfinance using verified ticker map."""
    try:
        import sys as _sys, io, contextlib
        _sys.path.insert(0, str(_V3_ROOT / "00_config"))
        try:
            from tickers import to_yf  # type: ignore
        except Exception:
            def to_yf(s: str) -> str: return f"{s}.NS"  # type: ignore

        import yfinance as yf
        tickers = [to_yf(s) for s in symbols]

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            data = yf.download(tickers, period="2d", auto_adjust=True,
                               progress=False, threads=True)
        if data.empty:
            return {}
        closes = data["Close"].iloc[-1] if "Close" in data else data.iloc[-1]
        prices = {}
        for sym, ticker in zip(symbols, tickers):
            val = closes.get(ticker, np.nan)
            if not (isinstance(val, float) and np.isnan(val)):
                prices[sym] = float(val)
        return prices
    except Exception as e:
        print(f"  ⚠  Price fetch failed: {e}")
        return {}


# ── Risk gate ─────────────────────────────────────────────────────────────────

def validate_orders(orders: list[dict]) -> list[dict]:
    """Run orders through risk_guard if available. Returns approved orders."""
    try:
        from risk_guard import RiskGuard
        rg       = RiskGuard()
        approved = [o for o in orders if rg.check_order(o)]
        refused  = len(orders) - len(approved)
        if refused:
            print(f"  RiskGuard: {refused} order(s) blocked")
        return approved
    except ImportError:
        # risk_guard not integrated yet — pass through
        return orders
    except Exception as e:
        print(f"  ⚠  RiskGuard error: {e} — passing all orders")
        return orders


# ── Save + print ──────────────────────────────────────────────────────────────

def save_orders(orders: list[dict], run_id: str) -> Path:
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    today     = datetime.now().strftime("%Y%m%d")
    out_path  = ORDERS_DIR / f"orders_{run_id}_{today}.json"
    with open(out_path, "w") as f:
        json.dump(orders, f, indent=2, default=str)
    return out_path


def print_orders(orders: list[dict], capital: float) -> None:
    total = sum(o["order_value"] for o in orders)
    print(f"\n{'═'*72}")
    print(f"  {'SYM':<14} {'DIR':>4}  {'P(UP)':>6}  {'%CAP':>5}  {'INR':>9}  {'QTY':>5}  {'PRICE':>8}")
    print(f"{'─'*72}")
    for o in orders:
        print(f"  {o['symbol']:<14} {o['direction']:>4}  {o['prob_up']:>6.3f}  "
              f"{o['target_pct']:>5.1f}%  {o['order_value']:>9,.0f}  "
              f"{o['qty']:>5d}  {o['price']:>8.2f}")
    print(f"{'─'*72}")
    print(f"  Total deployed: ₹{total:,.0f}  ({total/capital*100:.1f}% of ₹{capital:,.0f})")
    print(f"{'═'*72}\n")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def run(
    run_id: Optional[str]  = None,
    capital: float          = 500_000,
    dry_run: bool           = True,
    pred_path: Optional[Path] = None,
    price_map: Optional[dict] = None,
) -> list[dict]:
    """
    Full pipeline: load predictions → build orders → validate → (optionally) execute.

    Parameters
    ----------
    run_id    : Pipeline run ID (e.g. "20260307_141956"). Auto-detects latest if None.
    capital   : Total portfolio capital in INR.
    dry_run   : If True, print orders and save JSON but do NOT place on Angel One.
    pred_path : Override path to predictions CSV.
    price_map : Pre-loaded price dict. Fetched live if None.

    Returns
    -------
    Approved order list.
    """
    pred_df, run_id = load_predictions(run_id=run_id, pred_path=pred_path)

    print(f"\n  Capital  : ₹{capital:,.0f}")
    print(f"  Run ID   : {run_id}")
    print(f"  Signals  : {len(pred_df)} total, {(pred_df['direction']=='UP').sum()} UP")

    orders = build_orders(pred_df, capital, price_map=price_map, run_id=run_id)
    if not orders:
        print("  No actionable orders.")
        return []

    orders = validate_orders(orders)
    print_orders(orders, capital)

    out_path = save_orders(orders, run_id)
    print(f"  Orders saved → {out_path.relative_to(_V3_ROOT)}")

    if dry_run:
        print("  DRY RUN — no orders placed on Angel One.\n")
        return orders

    # Live execution
    try:
        from angel_one_client import AngelOneClient
        from order_manager   import OrderManager
        client  = AngelOneClient()
        client.login()
        manager = OrderManager(client)
        for o in orders:
            try:
                resp = manager.place_order(
                    symbol    = o["symbol"],
                    qty       = o["qty"],
                    price     = o["price"],
                    order_type= o["order_type"],
                    product   = o["product"],
                )
                print(f"  ✓ {o['symbol']} order placed: {resp.get('orderid', '?')}")
            except Exception as oe:
                print(f"  ✗ {o['symbol']} order failed: {oe}")
    except Exception as e:
        print(f"  Live execution failed: {e}")

    return orders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish trading signals to Angel One")
    parser.add_argument("--run-id",  default=None,    help="Pipeline run ID (auto-detects latest)")
    parser.add_argument("--capital", type=float, default=500_000, help="Portfolio capital in INR")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Print orders but do NOT place (default: True)")
    parser.add_argument("--live",    action="store_true", default=False,
                        help="Actually place orders on Angel One (overrides --dry-run)")
    args = parser.parse_args()

    run(
        run_id   = args.run_id,
        capital  = args.capital,
        dry_run  = not args.live,
    )
