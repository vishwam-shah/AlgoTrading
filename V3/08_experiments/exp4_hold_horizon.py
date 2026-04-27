"""
Experiment 4: Hold-horizon portfolio backtest
==============================================
Key fix from Exp3: our primary target is the 5-day forward return sign, so
the signal is valid for ~5 days — NOT 1. Holding the position for the full
horizon amortises the 0.25% round-trip cost across 5 days of edge instead
of paying it daily.

Also tests stricter thresholds to further reduce trade frequency.

Portfolio mechanics:
  Entry  : On day D, if (p1>=t1) AND (p2>=t2) AND (not stress), go long next open.
  Hold   : Exactly HOLD_DAYS (default 5) trading days.
  Exit   : Close at day D+HOLD_DAYS close.
  Weight : Equal across concurrent positions, max N_MAX concurrent.
  Cost   : ROUND_TRIP_COST applied once per trade (round trip).

Compares:
  - 1-day hold (Exp3 baseline)
  - 3-day hold
  - 5-day hold  (matches target horizon)
  - 5-day hold with stricter (0.65, 0.65) thresholds
  - 10-day hold with loose thresholds

Writes per-trade-level CSV and aggregate metrics vs NIFTY buy-and-hold.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_EXP_DIR  = Path(__file__).resolve().parent
_V3_ROOT  = _EXP_DIR.parent
sys.path.insert(0, str(_V3_ROOT))
sys.path.insert(0, str(_V3_ROOT / "00_config"))

# Import models/utilities from exp3 (reuse)
sys.path.insert(0, str(_EXP_DIR))
from exp3_winning_config import (                             # type: ignore
    run_one, SUBSET, ROUND_TRIP_COST, ANN_FACTOR, sharpe, max_dd,
)
from config import RAW_DATA_DIR, SYMBOLS_100                  # type: ignore

OUT_DIR = _EXP_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HOLD-HORIZON BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def build_trade_list(stock_results: List[Dict], t1: float, t2: float,
                     respect_regime: bool = True) -> List[Dict]:
    """
    For each stock, emit a list of (entry_date, entry_signal_strength) trades
    where signal exceeds both thresholds. Returns list of dicts.
    """
    trades: List[Dict] = []
    for r in stock_results:
        sym = r["symbol"]; dates = r["dates"]
        p1  = r["p1"]; p2 = r["p2"]
        stress = r["stress"]
        mask = (p1 >= t1) & (p2 >= t2)
        if respect_regime:
            mask &= ~stress
        for i in np.where(mask)[0]:
            trades.append({
                "date":   pd.Timestamp(dates[i]),
                "symbol": sym,
                "p1":     float(p1[i]),
                "p2":     float(p2[i]),
                "score":  float(p1[i] * p2[i]),
                "entry_idx_in_stock": int(i),
            })
    return trades


def simulate_hold_horizon(
    stock_results: List[Dict],
    hold_days:    int = 5,
    t1:           float = 0.55,
    t2:           float = 0.60,
    n_max:        int = 5,
    rebalance_every: int = 1,      # 1 = consider new entries daily
    respect_regime: bool = True,
) -> Dict:
    """
    Event-driven portfolio simulation with fixed holding period.
    On each trading day we can open up to `n_max - currently_held` new positions
    from today's eligible trades (ranked by score).
    """
    # Flatten to a per-stock index of close prices across OOS window
    # Reconstruct price series from next_ret (nr) — each r["ret"][i] is ret from
    # close(t) to close(t+1), so close series can be reconstructed up to scale.
    # We'll use relative price index starting at 1.0 at the first OOS date.
    stock_price: Dict[str, Dict] = {}
    for r in stock_results:
        dates = r["dates"]; nr = r["ret"]
        # Build close-index series (starts at 1.0 at first OOS date).
        # close[i+1] / close[i] = 1 + nr[i]  (nr[i] = next_ret at i)
        ci = np.empty(len(nr) + 1); ci[0] = 1.0
        for i, r1 in enumerate(nr):
            ci[i+1] = ci[i] * (1 + r1)
        stock_price[r["symbol"]] = dict(
            dates=pd.to_datetime(dates).to_numpy(),
            price=ci[:-1],             # price at start of each OOS day (same len as dates)
            next_close=ci[1:],         # the close after 1 day (what nr gave us)
            nr=nr,
        )

    # All eligible trades
    all_trades = build_trade_list(stock_results, t1, t2, respect_regime)
    if not all_trades:
        return dict(n_trades=0, total_return=0.0, ann_return=0.0, sharpe=0.0, max_dd=0.0, daily=pd.DataFrame())

    # Sort by (date, -score)
    all_trades.sort(key=lambda t: (t["date"], -t["score"]))

    # Union of all dates across all stocks, sorted
    all_dates = sorted({d for r in stock_results for d in pd.to_datetime(r["dates"]).to_numpy()})
    all_dates = pd.to_datetime(all_dates).sort_values().unique()

    # Simulate
    open_positions: List[Dict] = []   # [{symbol, entry_date, entry_price, exit_date, shares}]
    daily_rows: List[Dict] = []
    executed_trades: List[Dict] = []

    # Index trades by date
    from collections import defaultdict
    trades_by_date = defaultdict(list)
    for t in all_trades:
        trades_by_date[t["date"]].append(t)

    portfolio_value = 1.0  # unit portfolio
    cash            = 1.0
    positions_book: Dict[str, Dict] = {}   # symbol → {entry_date, entry_price, weight, exit_date}

    for day_i, today in enumerate(all_dates):
        # 1. Close positions whose exit_date is today
        closed = []
        for sym, pos in list(positions_book.items()):
            if pos["exit_date"] <= today:
                sp = stock_price.get(sym)
                if sp is not None:
                    idx = np.searchsorted(sp["dates"], today)
                    idx = min(max(0, idx), len(sp["price"]) - 1)
                    exit_price = sp["price"][idx]
                    gross = exit_price / pos["entry_price"] - 1.0
                    net   = gross - ROUND_TRIP_COST
                    cash += pos["weight"] * (1 + net)
                    executed_trades.append({
                        "symbol": sym, "entry_date": pos["entry_date"], "exit_date": today,
                        "hold_days": (today - pos["entry_date"]).days,
                        "gross_return": gross, "net_return": net, "weight": pos["weight"],
                    })
                    closed.append(sym)
                else:
                    closed.append(sym)
        for sym in closed:
            del positions_book[sym]

        # 2. Open new positions from today's eligible trades
        candidates = trades_by_date.get(today, [])
        slots_free = n_max - len(positions_book)
        if slots_free > 0 and candidates:
            # Dedupe: don't re-enter a symbol we already hold
            candidates = [c for c in candidates if c["symbol"] not in positions_book]
            candidates.sort(key=lambda x: -x["score"])
            chosen = candidates[:slots_free]
            if chosen:
                weight_each = 1.0 / n_max  # always size as if we wanted n_max
                for t in chosen:
                    sp = stock_price.get(t["symbol"])
                    if sp is None: continue
                    idx = np.searchsorted(sp["dates"], today)
                    if idx >= len(sp["price"]): continue
                    entry_price = sp["price"][idx]
                    # Exit = hold_days later (by trading-date index within this stock)
                    exit_idx = min(idx + hold_days, len(sp["dates"]) - 1)
                    exit_date = sp["dates"][exit_idx]
                    positions_book[t["symbol"]] = dict(
                        entry_date=today,
                        entry_price=entry_price,
                        weight=weight_each,
                        exit_date=pd.Timestamp(exit_date),
                    )
                    cash -= weight_each  # commit cash to position

        # 3. Mark-to-market — compute portfolio value from open + cash
        mtm = cash
        for sym, pos in positions_book.items():
            sp = stock_price.get(sym)
            if sp is None:
                mtm += pos["weight"]; continue
            idx = np.searchsorted(sp["dates"], today)
            idx = min(max(0, idx), len(sp["price"]) - 1)
            cur_price = sp["price"][idx]
            mtm += pos["weight"] * (cur_price / pos["entry_price"])

        daily_rows.append({"date": today, "portfolio_value": mtm,
                           "n_open": len(positions_book), "cash": cash})

    # Close any still-open positions at final day
    if positions_book:
        last_day = all_dates[-1]
        for sym, pos in positions_book.items():
            sp = stock_price.get(sym)
            if sp is None: continue
            idx = min(np.searchsorted(sp["dates"], last_day), len(sp["price"]) - 1)
            exit_price = sp["price"][idx]
            gross = exit_price / pos["entry_price"] - 1.0
            net   = gross - ROUND_TRIP_COST
            executed_trades.append({
                "symbol": sym, "entry_date": pos["entry_date"], "exit_date": last_day,
                "hold_days": (last_day - pos["entry_date"]).days,
                "gross_return": gross, "net_return": net, "weight": pos["weight"],
            })

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return dict(n_trades=0, total_return=0.0, ann_return=0.0, sharpe=0.0, max_dd=0.0, daily=daily)

    daily["daily_return"] = daily["portfolio_value"].pct_change().fillna(0.0)
    n_days = len(daily)
    rets   = daily["daily_return"].values
    total  = float(daily["portfolio_value"].iloc[-1] - 1.0)
    # Safe annualisation — avoid complex numbers when portfolio_value < 0
    pv_end = max(daily["portfolio_value"].iloc[-1], 1e-6)
    ann    = pv_end ** (ANN_FACTOR / max(n_days, 1)) - 1

    et = pd.DataFrame(executed_trades)

    return dict(
        n_trades       = int(len(et)),
        total_return   = total,
        ann_return     = float(ann),
        sharpe         = sharpe(rets),
        max_dd         = max_dd(daily["portfolio_value"].values),
        win_rate_days  = float((rets > 0).mean()),
        avg_hold_days  = float(et["hold_days"].mean()) if not et.empty else 0,
        trade_win_rate = float((et["net_return"] > 0).mean()) if not et.empty else 0,
        daily          = daily,
        trades         = et,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  NIFTY COMPARE
# ══════════════════════════════════════════════════════════════════════════════

def nifty_return(d0: pd.Timestamp, d1: pd.Timestamp) -> Optional[float]:
    try:
        gc = pd.read_parquet(RAW_DATA_DIR / "global_cues.parquet")
        gc["date"] = pd.to_datetime(gc["date"])
        sub = gc[(gc["date"] >= d0) & (gc["date"] <= d1)]
        if "nifty50_close" in sub.columns and len(sub) >= 2:
            return float(sub["nifty50_close"].iloc[-1] / sub["nifty50_close"].iloc[0] - 1)
    except Exception:
        return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    symbols = SYMBOLS_100 if args.full else SUBSET
    print(f"\n═══ Exp4 — Hold-Horizon Portfolio  n_stocks={len(symbols)} ═══")

    stock_results: List[Dict] = []
    for i, sym in enumerate(symbols):
        r = run_one(sym, use_regime_gate=True)
        if r is not None:
            stock_results.append(r)
            print(f"  [{i+1:>3}/{len(symbols)}] ✓ {sym:<12} n_oos={r['n_oos']}")
        else:
            print(f"  [{i+1:>3}/{len(symbols)}] ✗ {sym:<12}")

    if not stock_results:
        print("No stock results."); return

    # Grid over (hold_days, t1, t2, n_max)
    CONFIGS = [
        ("hold_1d_t0.58_0.60_n5",    1, 0.58, 0.60, 5),
        ("hold_3d_t0.58_0.60_n5",    3, 0.58, 0.60, 5),
        ("hold_5d_t0.55_0.55_n5",    5, 0.55, 0.55, 5),
        ("hold_5d_t0.55_0.60_n5",    5, 0.55, 0.60, 5),
        ("hold_5d_t0.58_0.60_n5",    5, 0.58, 0.60, 5),
        ("hold_5d_t0.60_0.65_n5",    5, 0.60, 0.65, 5),
        ("hold_5d_t0.58_0.60_n3",    5, 0.58, 0.60, 3),
        ("hold_5d_t0.58_0.60_n8",    5, 0.58, 0.60, 8),
        ("hold_10d_t0.55_0.60_n5",  10, 0.55, 0.60, 5),
    ]

    results = []
    suffix = "full" if args.full else "subset"
    for name, h, t1, t2, nmax in CONFIGS:
        m = simulate_hold_horizon(stock_results, hold_days=h, t1=t1, t2=t2, n_max=nmax)
        results.append({
            "config": name, "hold_days": h, "t1": t1, "t2": t2, "n_max": nmax,
            "n_trades": m["n_trades"],
            "total_return": round(m["total_return"], 4),
            "ann_return":   round(m["ann_return"], 4),
            "sharpe":       round(m["sharpe"], 3),
            "max_dd":       round(m["max_dd"], 4),
            "trade_win_rate": round(m.get("trade_win_rate", 0), 4),
            "avg_hold_days":  round(m.get("avg_hold_days", 0), 1),
        })
        # Save daily equity for best-performing later
        m["daily"].to_csv(OUT_DIR / f"exp4_{suffix}_{name}_daily.csv", index=False)
        if not m["trades"].empty:
            m["trades"].to_csv(OUT_DIR / f"exp4_{suffix}_{name}_trades.csv", index=False)

    res_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    res_df.to_csv(OUT_DIR / f"exp4_summary_{suffix}.csv", index=False)

    print("\n" + "="*130)
    print(f" EXP4 — HOLD-HORIZON PORTFOLIO SUMMARY  (n_stocks={len(stock_results)})")
    print("="*130)
    print(res_df.to_string(index=False))

    # NIFTY comparison using best config's window
    best_name = res_df.iloc[0]["config"]
    best_daily = pd.read_csv(OUT_DIR / f"exp4_{suffix}_{best_name}_daily.csv")
    best_daily["date"] = pd.to_datetime(best_daily["date"])
    if len(best_daily) >= 2:
        d0, d1 = best_daily["date"].iloc[0], best_daily["date"].iloc[-1]
        nr = nifty_return(d0, d1)
        print(f"\n  Best config         : {best_name}")
        print(f"  Window              : {d0.date()} → {d1.date()}  ({len(best_daily)} days)")
        best_total = float(best_daily['portfolio_value'].iloc[-1] - 1)
        print(f"  Portfolio return    : {best_total:+.2%}")
        if nr is not None:
            print(f"  NIFTY buy-and-hold  : {nr:+.2%}")
            print(f"  Alpha               : {best_total - nr:+.2%}")


if __name__ == "__main__":
    main()
