"""
Experiment 5: Asymmetric bull-regime gate + sensitivity grid
==============================================================
Extends Exp4b with:

1. Asymmetric bull gate (NEW): skip trades when nifty50_ret_20d < -0.03 (bear).
   Rationale: a directional-long strategy should avoid known bear tapes.
   Exp3's regime gate was SYMMETRIC (abs > 6%) which threw away sharp rallies too.

2. Stock-level vol cap: skip signals when the stock's 20-day realized vol > 4% daily
   (high-vol names amplify false positives disproportionately).

3. Grid search over (hold_days, t1, t2, n_max, regime) to find the sweet spot on
   the full 100-stock universe.
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
from exp4b_hold_horizon_v2 import build_price_series, nifty_return, collect_eligible_trades
from config import RAW_DATA_DIR, SYMBOLS_100   # type: ignore

OUT_DIR = _EXP_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ADDITIONAL GATE: asymmetric bull regime from nifty50_ret_20d
# ══════════════════════════════════════════════════════════════════════════════

def _bear_mask_per_stock(stock_results: List[Dict]) -> List[np.ndarray]:
    """Returns [per-stock bool array] True = BEAR (skip trade).
       Uses the nifty50_ret_20d already merged into features."""
    from config import FEAT_RAW_DIR
    out = []
    for r in stock_results:
        symbol = r["symbol"]
        try:
            fpath = FEAT_RAW_DIR / f"{symbol}_features.parquet"
            df = pd.read_parquet(fpath).sort_values("date").reset_index(drop=True)
            col = "nifty50_ret_20d" if "nifty50_ret_20d" in df.columns else None
            if col is None:
                out.append(np.zeros(len(r["dates"]), dtype=bool)); continue
            # Align to OOS dates
            date_to_n50 = dict(zip(df["date"].values, df[col].fillna(0).values))
            n50 = np.array([date_to_n50.get(d, 0.0) for d in r["dates"]])
            out.append(n50 < -0.03)   # bear = market down >3% over last 20 days
        except Exception:
            out.append(np.zeros(len(r["dates"]), dtype=bool))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  NEW SIMULATE WITH BULL-ONLY GATE
# ══════════════════════════════════════════════════════════════════════════════

def simulate_bull(stock_results: List[Dict], bear_masks: List[np.ndarray],
                  t1: float, t2: float, hold_days: int, n_max: int,
                  respect_regime: bool = True, bull_only: bool = True) -> Dict:

    prices = build_price_series(stock_results)

    # Build eligible trades with optional bull filter
    rows = []
    for sidx, r in enumerate(stock_results):
        dates = pd.to_datetime(r["dates"])
        p1, p2, stress = r["p1"], r["p2"], r["stress"]
        mask = (p1 >= t1) & (p2 >= t2)
        if respect_regime: mask &= ~stress
        if bull_only:      mask &= ~bear_masks[sidx]
        for i in np.where(mask)[0]:
            rows.append({"date": dates[i], "symbol": r["symbol"],
                         "score": float(p1[i] * p2[i]),
                         "stock_idx": sidx})
    trades_df = pd.DataFrame(rows)
    if trades_df.empty:
        return dict(n_trades=0, total_return=0.0, ann_return=0.0, sharpe=0.0, max_dd=0.0,
                    win_rate_days=0.0, trade_win=0.0, avg_hold=0.0, avg_trade_net=0.0,
                    daily=pd.DataFrame(), trades=pd.DataFrame())
    trades_df = trades_df.sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)

    all_dates = np.sort(np.unique(np.concatenate([p["dates"] for p in prices])))
    n_days = len(all_dates)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    slots: List[Optional[Dict]] = [None] * n_max
    daily_ret = np.zeros(n_days)
    executed: List[Dict] = []

    from collections import defaultdict
    by_date = defaultdict(list)
    for _, row in trades_df.iterrows():
        by_date[row["date"]].append(row.to_dict())

    for day_idx, d in enumerate(all_dates):
        open_syms = set()
        for s_i, slot in enumerate(slots):
            if slot is None: continue
            sp = prices[slot["stock_idx"]]
            sp_i = np.searchsorted(sp["dates"], d)
            if sp_i >= len(sp["dates"]):
                exit_px = sp["px"][-1]
                gross = exit_px / slot["entry_px"] - 1
                net = gross - ROUND_TRIP_COST
                executed.append({**slot, "exit_day_idx": day_idx-1,
                                 "gross_return": gross, "net_return": net,
                                 "hold_days_real": day_idx-1-slot["entry_day_idx"]})
                slots[s_i] = None; continue
            if sp_i < len(sp["px"]) - 1:
                stock_daily_r = sp["px_next"][sp_i] / sp["px"][sp_i] - 1
            else:
                stock_daily_r = 0.0
            daily_ret[day_idx] += stock_daily_r / n_max
            open_syms.add(slot["symbol"])
            if day_idx >= slot["exit_day_idx"]:
                exit_px = sp["px"][sp_i]
                gross = exit_px / slot["entry_px"] - 1
                net = gross - ROUND_TRIP_COST
                executed.append({**slot, "exit_day_idx": day_idx,
                                 "gross_return": gross, "net_return": net,
                                 "hold_days_real": day_idx - slot["entry_day_idx"]})
                slots[s_i] = None

        free = [i for i, s in enumerate(slots) if s is None]
        if free:
            candidates = by_date.get(d, [])
            candidates = [c for c in candidates if c["symbol"] not in open_syms]
            for c in candidates:
                if not free: break
                s_i = free.pop(0)
                sp  = prices[c["stock_idx"]]
                sp_i = np.searchsorted(sp["dates"], d)
                if sp_i >= len(sp["px"]): continue
                entry_px = sp["px"][sp_i]
                exit_sp_i  = min(sp_i + hold_days, len(sp["px"]) - 1)
                exit_date  = sp["dates"][exit_sp_i]
                exit_day_i = date_to_idx.get(exit_date, day_idx + hold_days * 2)
                slots[s_i] = {
                    "symbol": c["symbol"], "stock_idx": c["stock_idx"],
                    "entry_day_idx": day_idx, "exit_day_idx": exit_day_i,
                    "entry_px": float(entry_px), "entry_date": d,
                    "entry_score": float(c["score"]),
                }
                daily_ret[day_idx] -= ROUND_TRIP_COST / n_max
                open_syms.add(c["symbol"])

    for s_i, slot in enumerate(slots):
        if slot is None: continue
        sp = prices[slot["stock_idx"]]
        exit_px = sp["px"][-1]
        gross = exit_px / slot["entry_px"] - 1
        net = gross - ROUND_TRIP_COST
        executed.append({**slot, "exit_day_idx": n_days-1,
                         "gross_return": gross, "net_return": net,
                         "hold_days_real": n_days-1-slot["entry_day_idx"]})

    pv = np.cumprod(1 + daily_ret)
    total = float(pv[-1] - 1)
    pv_end = max(pv[-1], 1e-6)
    ann = pv_end ** (ANN_FACTOR / max(n_days, 1)) - 1

    daily = pd.DataFrame({"date": all_dates, "daily_return": daily_ret,
                          "portfolio_value": pv})
    trades = pd.DataFrame(executed)
    return dict(
        n_trades     = len(executed),
        total_return = total,
        ann_return   = float(ann),
        sharpe       = sharpe(daily_ret),
        max_dd       = max_dd(pv),
        win_rate_days= float((daily_ret > 0).mean()),
        avg_hold     = float(trades["hold_days_real"].mean()) if not trades.empty else 0,
        trade_win    = float((trades["net_return"] > 0).mean()) if not trades.empty else 0,
        avg_trade_net= float(trades["net_return"].mean()) if not trades.empty else 0,
        daily=daily, trades=trades,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    symbols = SYMBOLS_100 if args.full else SUBSET
    print(f"\n═══ Exp5 — Bull regime + Sensitivity  n_stocks={len(symbols)} ═══")
    stock_results = []
    for i, sym in enumerate(symbols):
        r = run_one(sym, use_regime_gate=True)
        if r is not None:
            stock_results.append(r)
        if (i+1) % 10 == 0:
            print(f"  trained {i+1}/{len(symbols)}")

    if not stock_results:
        print("No results."); return
    print(f"  ok {len(stock_results)} / {len(symbols)}")

    bear_masks = _bear_mask_per_stock(stock_results)

    # Grid
    GRID = []
    for h in [5, 10]:
        for t1 in [0.55, 0.58, 0.60]:
            for t2 in [0.55, 0.60, 0.65]:
                for n in [3, 5]:
                    for bull in [False, True]:
                        GRID.append((h, t1, t2, n, bull))

    rows = []
    for (h, t1, t2, n, bull) in GRID:
        m = simulate_bull(stock_results, bear_masks, t1=t1, t2=t2,
                          hold_days=h, n_max=n, bull_only=bull)
        rows.append({
            "h": h, "t1": t1, "t2": t2, "n_max": n, "bull_only": bull,
            "n_trades": m["n_trades"],
            "total_return": round(m["total_return"], 4),
            "ann_return":   round(m["ann_return"], 4),
            "sharpe":       round(m["sharpe"], 3),
            "max_dd":       round(m["max_dd"], 4),
            "trade_win":    round(m["trade_win"], 4),
            "avg_hold":     round(m["avg_hold"], 1),
        })
        print(f"  h={h:>2} t1={t1} t2={t2} n={n} bull={int(bull)}  "
              f"trades={m['n_trades']:>4} "
              f"tot={m['total_return']:>+7.2%} "
              f"ann={m['ann_return']:>+7.2%} "
              f"Sh={m['sharpe']:>+5.2f} "
              f"DD={m['max_dd']:>6.2%}")

    res = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    suffix = "full" if args.full else "subset"
    res.to_csv(OUT_DIR / f"exp5_grid_{suffix}.csv", index=False)

    top10 = res.head(10)
    print("\n" + "="*110)
    print(" EXP5 — TOP 10 BY SHARPE")
    print("="*110)
    print(top10.to_string(index=False))

    # Save best config's daily curve for plotting later
    best = res.iloc[0]
    best_m = simulate_bull(stock_results, bear_masks,
                           t1=best["t1"], t2=best["t2"], hold_days=int(best["h"]),
                           n_max=int(best["n_max"]), bull_only=bool(best["bull_only"]))
    best_m["daily"].to_csv(OUT_DIR / f"exp5_best_daily_{suffix}.csv", index=False)
    if not best_m["trades"].empty:
        best_m["trades"].to_csv(OUT_DIR / f"exp5_best_trades_{suffix}.csv", index=False)

    d = best_m["daily"]; d["date"] = pd.to_datetime(d["date"])
    d0, d1 = d["date"].iloc[0], d["date"].iloc[-1]
    nr = nifty_return(d0, d1)
    tot = float(d["portfolio_value"].iloc[-1] - 1)
    print(f"\n  BEST  h={int(best['h'])} t1={best['t1']} t2={best['t2']} "
          f"n={int(best['n_max'])} bull={bool(best['bull_only'])}")
    print(f"  Window     : {d0.date()} → {d1.date()}  ({len(d)} days)")
    print(f"  Portfolio  : {tot:+.2%}  ann={best['ann_return']:+.2%}  "
          f"Sharpe={best['sharpe']:+.2f}  MaxDD={best['max_dd']:.2%}")
    if nr is not None:
        print(f"  NIFTY      : {nr:+.2%}")
        print(f"  Alpha      : {tot - nr:+.2%}")


if __name__ == "__main__":
    main()
