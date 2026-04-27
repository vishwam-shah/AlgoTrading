"""
Experiment 4b: Clean hold-horizon portfolio backtest (simpler simulation)
==========================================================================
Rewrite of Exp4 with a much simpler mental model:

  For each eligible (date, stock) entry signal:
      - enter at close on entry_date
      - hold exactly `hold_days` trading days (on that stock's own calendar)
      - exit at close on entry_date + hold_days
      - trade net return = (exit_px/entry_px - 1) - ROUND_TRIP_COST

  Portfolio simulation (slot-based):
      - At most `n_max` concurrent slots
      - New signals rejected if all slots full OR same symbol already held
      - Slot weight = 1/n_max fixed  (equal-weight cap)
      - Daily portfolio return = Σ (weight × daily_stock_return) for open positions
                               + cash_weight × 0 (idle cash earns 0 to be conservative)
      - Entry cost split into daily drag: we apply the -COST hit on entry day only.

This avoids the cash-accounting bug from Exp4 by never subtracting weight from cash.
Instead, each open slot simply tracks its stock's daily return until closure.
"""

from __future__ import annotations

import sys, warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_EXP_DIR  = Path(__file__).resolve().parent
_V3_ROOT  = _EXP_DIR.parent
sys.path.insert(0, str(_V3_ROOT))
sys.path.insert(0, str(_V3_ROOT / "00_config"))
sys.path.insert(0, str(_EXP_DIR))

from exp3_winning_config import run_one, SUBSET, ROUND_TRIP_COST, ANN_FACTOR, sharpe, max_dd
from config import RAW_DATA_DIR, SYMBOLS_100   # type: ignore

OUT_DIR = _EXP_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLE SLOT-BASED SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def collect_eligible_trades(stock_results: List[Dict], t1: float, t2: float,
                            respect_regime: bool) -> pd.DataFrame:
    """Returns DataFrame: [date, symbol, score, entry_idx, stock_idx]."""
    rows = []
    for sidx, r in enumerate(stock_results):
        dates = pd.to_datetime(r["dates"])
        p1, p2 = r["p1"], r["p2"]; stress = r["stress"]
        mask = (p1 >= t1) & (p2 >= t2)
        if respect_regime: mask &= ~stress
        for i in np.where(mask)[0]:
            rows.append({"date": dates[i], "symbol": r["symbol"],
                         "score": float(p1[i] * p2[i]),
                         "entry_idx": int(i), "stock_idx": sidx})
    df = pd.DataFrame(rows)
    if df.empty: return df
    return df.sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)


def build_price_series(stock_results: List[Dict]) -> List[Dict]:
    """For each stock, build {dates, rel_close} where rel_close[i] is price at OOS day i."""
    out = []
    for r in stock_results:
        nr = np.asarray(r["ret"], dtype=float)
        n  = len(nr)
        px = np.empty(n + 1); px[0] = 1.0
        for i in range(n):
            px[i+1] = px[i] * (1 + nr[i])
        out.append({
            "symbol": r["symbol"],
            "dates":  pd.to_datetime(r["dates"]).to_numpy(),
            "px":     px[:n],            # price at OOS day i (entry point)
            "px_next": px[1:n+1],        # price at OOS day i+1 (next close)
        })
    return out


def simulate(stock_results: List[Dict], t1: float = 0.58, t2: float = 0.60,
             hold_days: int = 5, n_max: int = 5, respect_regime: bool = True) -> Dict:
    prices = build_price_series(stock_results)
    trades_df = collect_eligible_trades(stock_results, t1, t2, respect_regime)
    if trades_df.empty:
        return dict(n_trades=0, total_return=0.0, ann_return=0.0, sharpe=0.0,
                    max_dd=0.0, daily=pd.DataFrame(), trades=pd.DataFrame())

    all_dates = np.sort(np.unique(np.concatenate([p["dates"] for p in prices])))
    n_days = len(all_dates)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # slots[i] = dict(symbol, stock_idx, entry_day_idx, exit_day_idx, entry_px, cost_applied)
    slots: List[Optional[Dict]] = [None] * n_max

    # Daily portfolio return
    daily_ret = np.zeros(n_days)
    n_open    = np.zeros(n_days, dtype=int)
    open_syms: List[set] = [set() for _ in range(n_days)]

    executed: List[Dict] = []

    # Pre-index trades by date
    from collections import defaultdict
    by_date = defaultdict(list)
    for _, row in trades_df.iterrows():
        by_date[row["date"]].append(row.to_dict())

    for day_idx, d in enumerate(all_dates):
        # 1. MtM existing slots: add their stock's (day_idx → day_idx+1) return
        for s_i, slot in enumerate(slots):
            if slot is None:
                continue
            sp = prices[slot["stock_idx"]]
            # Find this day's index in that stock's own calendar
            sp_i = np.searchsorted(sp["dates"], d)
            if sp_i >= len(sp["dates"]):
                # past end for that stock; close position
                exit_px = sp["px"][-1]
                gross   = exit_px / slot["entry_px"] - 1
                net     = gross - ROUND_TRIP_COST
                executed.append({**slot, "exit_day_idx": day_idx - 1,
                                 "gross_return": gross, "net_return": net,
                                 "hold_days_real": day_idx - 1 - slot["entry_day_idx"]})
                slots[s_i] = None; continue
            # MtM: use stock's next-day return at this index
            if sp_i < len(sp["px"]) - 1:
                stock_daily_r = sp["px_next"][sp_i] / sp["px"][sp_i] - 1
            else:
                stock_daily_r = 0.0
            daily_ret[day_idx] += stock_daily_r / n_max  # equal-weight slot

            # Close if we've held enough
            if day_idx >= slot["exit_day_idx"]:
                exit_px = sp["px"][sp_i]
                gross   = exit_px / slot["entry_px"] - 1
                net     = gross - ROUND_TRIP_COST
                executed.append({**slot, "exit_day_idx": day_idx,
                                 "gross_return": gross, "net_return": net,
                                 "hold_days_real": day_idx - slot["entry_day_idx"]})
                slots[s_i] = None

        n_open[day_idx]     = sum(1 for s in slots if s is not None)
        open_syms[day_idx]  = {s["symbol"] for s in slots if s is not None}

        # 2. Consider new entries
        free = [i for i, s in enumerate(slots) if s is None]
        if free:
            candidates = by_date.get(d, [])
            candidates = [c for c in candidates if c["symbol"] not in open_syms[day_idx]]
            # already sorted by -score within date
            for c in candidates:
                if not free: break
                s_i = free.pop(0)
                sp  = prices[c["stock_idx"]]
                sp_i = np.searchsorted(sp["dates"], d)
                if sp_i >= len(sp["px"]): continue
                entry_px   = sp["px"][sp_i]
                exit_sp_i  = min(sp_i + hold_days, len(sp["px"]) - 1)
                exit_date  = sp["dates"][exit_sp_i]
                exit_day_i = date_to_idx.get(exit_date, day_idx + hold_days * 2)
                slots[s_i] = {
                    "symbol": c["symbol"], "stock_idx": c["stock_idx"],
                    "entry_day_idx": day_idx, "exit_day_idx": exit_day_i,
                    "entry_px": float(entry_px), "entry_date": d,
                    "entry_score": float(c["score"]),
                }
                # Apply entry cost as a one-day drag on portfolio
                daily_ret[day_idx] -= ROUND_TRIP_COST / n_max
                open_syms[day_idx].add(c["symbol"])

    # Close any still-open slots at the end
    for s_i, slot in enumerate(slots):
        if slot is None: continue
        sp = prices[slot["stock_idx"]]
        exit_px = sp["px"][-1]
        gross   = exit_px / slot["entry_px"] - 1
        net     = gross - ROUND_TRIP_COST
        executed.append({**slot, "exit_day_idx": n_days - 1,
                         "gross_return": gross, "net_return": net,
                         "hold_days_real": n_days - 1 - slot["entry_day_idx"]})

    pv = np.cumprod(1 + daily_ret)
    total = float(pv[-1] - 1)
    pv_end = max(pv[-1], 1e-6)
    ann = pv_end ** (ANN_FACTOR / max(n_days, 1)) - 1

    daily = pd.DataFrame({
        "date": all_dates, "daily_return": daily_ret,
        "portfolio_value": pv, "n_open": n_open,
    })
    trades = pd.DataFrame(executed)
    return dict(
        n_trades     = len(executed),
        total_return = total,
        ann_return   = float(ann),
        sharpe       = sharpe(daily_ret),
        max_dd       = max_dd(pv),
        win_rate_days= float((daily_ret > 0).mean()),
        avg_hold     = float(trades["hold_days_real"].mean()) if not trades.empty else 0.0,
        trade_win    = float((trades["net_return"] > 0).mean()) if not trades.empty else 0.0,
        avg_trade_net= float(trades["net_return"].mean()) if not trades.empty else 0.0,
        daily=daily, trades=trades,
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
    print(f"\n═══ Exp4b — Clean Hold-Horizon Portfolio  n_stocks={len(symbols)} ═══")
    stock_results = []
    for i, sym in enumerate(symbols):
        r = run_one(sym, use_regime_gate=True)
        if r is not None:
            stock_results.append(r)
            print(f"  [{i+1:>3}/{len(symbols)}] ✓ {sym:<12} n_oos={r['n_oos']}")
        else:
            print(f"  [{i+1:>3}/{len(symbols)}] ✗ {sym:<12}")

    if not stock_results:
        print("No results."); return

    CFGS = [
        ("hold_1d_t0.58_0.60_n5",   1, 0.58, 0.60, 5),
        ("hold_3d_t0.58_0.60_n5",   3, 0.58, 0.60, 5),
        ("hold_5d_t0.55_0.55_n5",   5, 0.55, 0.55, 5),
        ("hold_5d_t0.55_0.60_n5",   5, 0.55, 0.60, 5),
        ("hold_5d_t0.58_0.60_n5",   5, 0.58, 0.60, 5),
        ("hold_5d_t0.60_0.65_n5",   5, 0.60, 0.65, 5),
        ("hold_5d_t0.58_0.60_n3",   5, 0.58, 0.60, 3),
        ("hold_5d_t0.58_0.60_n8",   5, 0.58, 0.60, 8),
        ("hold_10d_t0.58_0.60_n5", 10, 0.58, 0.60, 5),
        ("hold_10d_t0.60_0.65_n5", 10, 0.60, 0.65, 5),
        ("hold_15d_t0.60_0.65_n5", 15, 0.60, 0.65, 5),
    ]

    rows = []
    suffix = "full" if args.full else "subset"
    for name, h, t1, t2, nmax in CFGS:
        m = simulate(stock_results, t1=t1, t2=t2, hold_days=h, n_max=nmax)
        rows.append({
            "config": name, "hold_days": h, "t1": t1, "t2": t2, "n_max": nmax,
            "n_trades":     m["n_trades"],
            "total_return": round(m["total_return"], 4),
            "ann_return":   round(m["ann_return"], 4),
            "sharpe":       round(m["sharpe"], 3),
            "max_dd":       round(m["max_dd"], 4),
            "win_rate_days": round(m["win_rate_days"], 4),
            "trade_win":    round(m["trade_win"], 4),
            "avg_hold":     round(m["avg_hold"], 1),
            "avg_trade_net":round(m["avg_trade_net"], 4),
        })
        m["daily"].to_csv(OUT_DIR / f"exp4b_{suffix}_{name}_daily.csv", index=False)
        if not m["trades"].empty:
            m["trades"].to_csv(OUT_DIR / f"exp4b_{suffix}_{name}_trades.csv", index=False)

    res = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    res.to_csv(OUT_DIR / f"exp4b_summary_{suffix}.csv", index=False)
    print("\n" + "="*135)
    print(f" EXP4b — CLEAN HOLD-HORIZON SUMMARY  (n_stocks={len(stock_results)})")
    print("="*135)
    print(res.to_string(index=False))

    # NIFTY comparison (over full simulation window)
    best = res.iloc[0]["config"]
    d = pd.read_csv(OUT_DIR / f"exp4b_{suffix}_{best}_daily.csv")
    d["date"] = pd.to_datetime(d["date"])
    d0, d1 = d["date"].iloc[0], d["date"].iloc[-1]
    nr = nifty_return(d0, d1)
    print(f"\n  Best config         : {best}")
    print(f"  Window              : {d0.date()} → {d1.date()}  ({len(d)} days)")
    tot = float(d['portfolio_value'].iloc[-1] - 1)
    print(f"  Portfolio return    : {tot:+.2%}")
    if nr is not None:
        print(f"  NIFTY buy-and-hold  : {nr:+.2%}")
        print(f"  Alpha               : {tot - nr:+.2%}")


if __name__ == "__main__":
    main()
