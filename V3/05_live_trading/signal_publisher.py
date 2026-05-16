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

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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
sys.path.insert(0, str(_V3_ROOT / "00_config"))

from risk_config import HOT as _RC, get as _rcget  # type: ignore  # noqa: E402

# ── Constants (single source: V3/00_config/risk_config.yaml) ────────────────

MAX_POSITION_PCT   = _RC["MAX_POSITION_PCT"]
MIN_CONFIDENCE     = _RC["MIN_CONFIDENCE"]
META_THRESHOLD     = _RC["META_THRESHOLD"]
MIN_PROB_REQUIRED  = True
MAX_STOCKS         = _RC["MAX_HOLDINGS"]
MIN_TRADEABLE_UNIV = _RC["EXPAND_BELOW"]
TARGET_HOLD_DAYS   = _RC["HOLD_DAYS"]
ROUND_LOT          = int(_rcget("sizing", "min_lot", default=1))
ROUND_TRIP_COST_PCT = _RC["COST_RT"] + 2 * (_rcget("strategy", "slippage_one_way_bps", default=5) / 10000.0)

RESULTS_DIR = _V3_ROOT / "06_results" / "runs"
ORDERS_DIR  = _V3_ROOT / "05_live_trading" / "orders"


# ── Kelly / volatility-adjusted sizing ──────────────────────────────────────

_KELLY_CAP     = float(_rcget("sizing", "kelly_cap_full", default=0.25))
_KELLY_HAIRCUT = float(_rcget("sizing", "kelly_haircut", default=0.5))
_VOL_TARGET    = float(_rcget("sizing", "vol_target_daily", default=0.015))


def kelly_fraction(prob_up: float, win_loss_ratio: float = 1.5) -> float:
    """
    Full Kelly: f* = (bp - q) / b. Cap full-Kelly, then haircut by configured factor,
    then enforce per-stock exposure cap.
    """
    q = 1.0 - prob_up
    b = win_loss_ratio
    f = (b * prob_up - q) / b
    f = max(0.0, min(f, _KELLY_CAP))
    return min(f * _KELLY_HAIRCUT, MAX_POSITION_PCT)


def vol_adjusted_size(base_frac: float, atr_pct: float,
                      target_vol: float | None = None) -> float:
    """Scale down if stock is more volatile than the configured daily-ATR target."""
    if atr_pct <= 0:
        return base_frac
    return base_frac * ((target_vol if target_vol is not None else _VOL_TARGET) / atr_pct)


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
    # Primary gate: tradeable==True  (sharpe>0 AND OOS>=50%) — strict, often ~8 stocks.
    # Expansion:    cross_sectional_top15 (top-15 by Sharpe among OOS>=50%) — used
    # when the strict universe is small (< MIN_TRADEABLE_UNIV) to avoid
    # concentrating capital in too few names.
    strict_syms: Optional[set]   = None
    expanded_syms: Optional[set] = None

    if "tradeable" in df.columns:
        strict_syms = set(df[df["tradeable"] == True]["symbol"].tolist())
    if "cross_sectional_top15" in df.columns:
        expanded_syms = set(df[df["cross_sectional_top15"] == True]["symbol"].tolist())

    if (strict_syms is None or expanded_syms is None) and run_id:
        bt_path = RESULTS_DIR / run_id / "backtest_results.csv"
        if bt_path.exists():
            bt_df = pd.read_csv(bt_path)
            if strict_syms is None and "tradeable" in bt_df.columns:
                strict_syms = set(bt_df[bt_df["tradeable"] == True]["symbol"].tolist())
            elif strict_syms is None:
                strict_syms = set(
                    bt_df[(bt_df["sharpe"] > 0) & (bt_df["oos_accuracy"] >= 0.50)]["symbol"].tolist()
                )
            if expanded_syms is None and "cross_sectional_top15" in bt_df.columns:
                expanded_syms = set(bt_df[bt_df["cross_sectional_top15"] == True]["symbol"].tolist())

    profitable_symbols: Optional[set] = None
    if strict_syms is not None:
        profitable_symbols = strict_syms
        gate_label = f"strict tradeable (n={len(strict_syms)})"
        if len(strict_syms) < MIN_TRADEABLE_UNIV and expanded_syms:
            profitable_symbols = strict_syms | expanded_syms
            gate_label = (f"strict ({len(strict_syms)}) ∪ cross-sectional top15 "
                          f"({len(expanded_syms)}) → {len(profitable_symbols)} symbols")

    if profitable_symbols is not None:
        before = len(df)
        df = df[df["symbol"].isin(profitable_symbols)].copy()
        print(f"  {len(df)}/{before} signals in {gate_label}")
        if df.empty:
            print("  No signals in profitable universe today.")
            return []

    # ── 3. Primary (prob_up) gate ─────────────────────────────────────────
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

    # ── 3b. Meta (López de Prado) gate — only when secondary is present ───
    if "meta_prob" in df.columns and df["meta_prob"].notna().any() \
            and float(df["meta_prob"].std()) > 1e-6:
        before = len(df)
        df = df[df["meta_prob"] >= META_THRESHOLD].copy()
        print(f"  {len(df)}/{before} signals above meta_prob ≥ {META_THRESHOLD:.0%}")
        if df.empty:
            return []

    # ── 4. Score-rank (prob_up × meta_prob) desc, take top n_max ──────────
    if "meta_prob" in df.columns:
        df["_score"] = df.get("prob_up", 0.55) * df["meta_prob"]
    else:
        df["_score"] = df.get("prob_up", df.get("confidence", 0.55))
    df = df.sort_values("_score", ascending=False).head(MAX_STOCKS).reset_index(drop=True)

    # ── 4. Fetch prices if not provided ───────────────────────────────────
    if price_map is None:
        price_map = _fetch_prices(df["symbol"].tolist())

    # ── 5. Resolve prices and build raw candidate list ─────────────────────
    candidates: list[dict] = []
    for _, row in df.iterrows():
        sym      = row["symbol"]
        prob_up  = float(row.get("prob_up", row.get("confidence", 0.55)))
        atr_pct  = float(row.get("atr_pct", 0.015))
        price = price_map.get(sym)
        if (price is None or price <= 0) and "price_hint" in row and row["price_hint"] > 0:
            price = float(row["price_hint"])
            print(f"  ℹ  {sym}: using last_close ₹{price:.2f} (live price unavailable)")
        if price is None or price <= 0:
            print(f"  ⚠  {sym}: no price available, skipping")
            continue
        candidates.append({
            "symbol":     sym,
            "price":      round(price, 2),
            "prob_up":    round(prob_up, 4),
            "meta_prob":  round(float(row.get("meta_prob", 0.5)), 4),
            "score":      round(float(row.get("_score", prob_up)), 4),
            "atr_pct":    atr_pct,
        })

    # ── 6. Risk-aware allocation (correlation, MCR, sector cap, vol budget) ──
    try:
        from portfolio_optimizer import allocate as _alloc
        allocated = _alloc(candidates, capital=capital)
    except Exception as e:
        print(f"  ⚠  optimizer unavailable ({e}); falling back to half-Kelly equal weight")
        allocated = []
        for c in candidates[:MAX_STOCKS]:
            kf = kelly_fraction(c["prob_up"])
            tgt_pct = min(vol_adjusted_size(kf, c["atr_pct"]), MAX_POSITION_PCT)
            target_inr = capital * tgt_pct
            qty = max(1, int(target_inr / c["price"]))
            allocated.append({**c, "weight": tgt_pct,
                              "target_inr": round(target_inr, 0),
                              "qty": qty,
                              "target_pct": round(tgt_pct * 100, 2),
                              "order_value": round(qty * c["price"], 0)})

    from datetime import timedelta as _td
    planned_exit = (datetime.now() + _td(days=int(TARGET_HOLD_DAYS * 1.45))).date().isoformat()

    orders: list[dict] = []
    for a in allocated:
        kf = kelly_fraction(a["prob_up"])
        orders.append({
            "symbol":         a["symbol"],
            "exchange":       "NSE",
            "direction":      "BUY",
            "prob_up":        a["prob_up"],
            "meta_prob":      a["meta_prob"],
            "score":          a["score"],
            "kelly_frac":     round(kf, 4),
            "weight":         a.get("weight", a.get("target_pct", 0)),
            "weight_components": a.get("weight_components", {}),
            "sector":         a.get("sector", ""),
            "target_pct":     a["target_pct"],
            "target_inr":     a["target_inr"],
            "qty":            a["qty"],
            "price":          a["price"],
            "order_value":    a["order_value"],
            "order_type":     "LIMIT",
            "product":        "CNC",
            "validity":       "DAY",
            "hold_days":      TARGET_HOLD_DAYS,
            "planned_exit":   planned_exit,
            "generated_at":   datetime.now().isoformat(),
        })

    orders.sort(key=lambda o: o["target_inr"], reverse=True)

    # ── 7. Final capital check (defensive — optimizer should already respect this)
    total_deployed = sum(o["order_value"] for o in orders)
    if total_deployed > capital:
        scale = capital / total_deployed
        for o in orders:
            o["qty"]         = max(1, int(o["qty"] * scale))
            o["order_value"] = round(o["qty"] * o["price"], 0)
            o["target_pct"]  = round(o["order_value"] / capital * 100, 2)

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
