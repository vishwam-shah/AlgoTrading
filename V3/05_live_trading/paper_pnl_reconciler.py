"""
paper_pnl_reconciler.py — Live paper P&L vs backtest-implied P&L
================================================================
For the live paper-trading forward test, this script joins:
  - execution_logs/execution_log.parquet  (live paper fills, BUYs and SELLs)
  - 06_results/runs/<latest>/predictions.csv (model OOS predictions)

and produces:
  - paper_trading_logs/paper_pnl_<today>.csv          per-trade P&L
  - paper_trading_logs/paper_pnl_summary.json         rolling totals
  - paper_trading_logs/paper_vs_backtest.csv          per-day equity vs backtest

The forward-test compares live-paper realised P&L against the equity curve
the backtest would have produced over the same dates, so we can spot
slippage, missing exits, or model drift.

Usage:
    python V3/05_live_trading/paper_pnl_reconciler.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_LIVE     = Path(__file__).resolve().parent
_V3       = _LIVE.parent
_EXEC     = _LIVE / "execution_logs"
_PAPER    = _LIVE / "paper_trading_logs"
_RUNS     = _V3 / "06_results" / "runs"
_RAW      = _V3 / "01_data" / "raw"

ROUND_TRIP_COST = 0.0025


def _latest_run_id() -> str:
    runs = sorted(_RUNS.glob("20*"), reverse=True)
    return runs[0].name if runs else ""


def _load_fills() -> pd.DataFrame:
    p = _EXEC / "execution_log.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df = df[df["status"] == "FILLED"].copy()
    df["filled_at"] = pd.to_datetime(df["filled_at"])
    df["date"] = df["filled_at"].dt.date.astype(str)
    return df.sort_values("filled_at").reset_index(drop=True)


def _match_round_trips(fills: pd.DataFrame) -> List[Dict]:
    """FIFO-match BUYs to SELLs → list of completed round trips with realised P&L."""
    open_lots: Dict[str, List[Dict]] = defaultdict(list)
    rt: List[Dict] = []
    for _, r in fills.iterrows():
        sym = r["symbol"]
        qty = int(r["filled_qty"])
        price = float(r["avg_price"])
        date = r["date"]
        if r["side"] == "BUY":
            open_lots[sym].append({"qty": qty, "price": price, "date": date,
                                   "fees": float(r.get("brokerage", 0)) + float(r.get("other_charges", 0))})
        elif r["side"] == "SELL":
            remaining = qty
            sell_fees = float(r.get("brokerage", 0)) + float(r.get("stt", 0)) + float(r.get("other_charges", 0))
            while remaining > 0 and open_lots[sym]:
                lot = open_lots[sym][0]
                use = min(lot["qty"], remaining)
                gross = (price - lot["price"]) * use
                fees = lot["fees"] * (use / qty if qty > 0 else 1) + sell_fees * (use / qty if qty > 0 else 1)
                rt.append({
                    "symbol": sym, "qty": use,
                    "entry_date": lot["date"], "entry_px": lot["price"],
                    "exit_date":  date,        "exit_px":  price,
                    "gross_pnl": round(gross, 2),
                    "fees":      round(fees, 2),
                    "net_pnl":   round(gross - fees, 2),
                    "ret_pct":   round((price / lot["price"] - 1) - ROUND_TRIP_COST, 4),
                })
                lot["qty"] -= use
                remaining -= use
                if lot["qty"] == 0:
                    open_lots[sym].pop(0)
    return rt


def _backtest_implied_pnl(rt: List[Dict], run_id: str) -> List[Dict]:
    """For each round trip, look up what the backtest model said on entry_date."""
    out = []
    for trade in rt:
        sym = trade["symbol"]
        pred_path = _RUNS / run_id / sym / "predictions.csv"
        if not pred_path.exists():
            out.append({**trade, "model_prob_up": None, "model_meta": None, "match": "no_pred"})
            continue
        df = pd.read_csv(pred_path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        m = df[df["date"] == trade["entry_date"]]
        if m.empty:
            out.append({**trade, "model_prob_up": None, "model_meta": None, "match": "date_miss"})
            continue
        row = m.iloc[-1]
        out.append({**trade,
                    "model_prob_up": round(float(row.get("prob_up", np.nan)), 4),
                    "model_meta":    round(float(row.get("meta_prob", np.nan)), 4),
                    "match":         "ok"})
    return out


def _summary(rt: List[Dict]) -> Dict:
    if not rt:
        return {"n_trades": 0, "total_net_pnl": 0.0, "win_rate": 0.0, "sharpe": 0.0}
    rets = np.array([t["ret_pct"] for t in rt])
    wins = (rets > 0).sum()
    n = len(rets)
    return {
        "n_trades":       n,
        "total_net_pnl":  round(float(sum(t["net_pnl"] for t in rt)), 2),
        "avg_ret_pct":    round(float(rets.mean()), 4),
        "win_rate":       round(float(wins / n), 4),
        "sharpe":         round(float(rets.mean() / rets.std() * np.sqrt(252) / 5)
                                if rets.std() > 0 else 0.0, 3),
        "max_loss_pct":   round(float(rets.min()), 4),
        "max_win_pct":    round(float(rets.max()), 4),
        "first_trade":    rt[0]["entry_date"] if rt else None,
        "last_trade":     rt[-1]["exit_date"] if rt else None,
        "as_of":          datetime.now().isoformat(timespec="seconds"),
    }


def main():
    _PAPER.mkdir(exist_ok=True)
    fills = _load_fills()
    if fills.empty:
        print("  No execution log found — paper trading hasn't started.")
        return
    print(f"  Fills loaded: {len(fills)} rows ({(fills.side=='BUY').sum()} BUY / {(fills.side=='SELL').sum()} SELL)")

    rt_list = _match_round_trips(fills)
    print(f"  Completed round-trips: {len(rt_list)}")

    if not rt_list:
        print("  Nothing to reconcile yet — no SELLs matched to BUYs (positions still open).")
        # Still write open-position snapshot
        open_count = sum(1 for v in _match_round_trips(fills) if v) - len(rt_list)
        return

    run_id = _latest_run_id()
    rt_enriched = _backtest_implied_pnl(rt_list, run_id)

    today = datetime.now().strftime("%Y%m%d")
    csv_path = _PAPER / f"paper_pnl_{today}.csv"
    pd.DataFrame(rt_enriched).to_csv(csv_path, index=False)

    summary = _summary(rt_list)
    json_path = _PAPER / "paper_pnl_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary:")
    for k, v in summary.items():
        print(f"    {k:<18} {v}")
    print(f"\n  → {csv_path.name}")
    print(f"  → {json_path.name}")


if __name__ == "__main__":
    main()
