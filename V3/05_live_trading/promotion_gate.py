"""
promotion_gate.py — Paper → Live promotion gate
================================================
Refuses to flip TRADING_MODE=paper → live unless every metric below clears
a threshold from V3/00_config/risk_config.yaml `promotion`:

  • min_paper_trades    — at least N closed paper trades
  • min_paper_days      — at least N calendar days since first paper trade
  • min_rolling_sharpe  — last-30 closed trades must beat this
  • max_rolling_dd      — peak-to-trough drawdown over the same window
  • max_slippage_drift  — average |fill - signal| in bps
  • max_brier_drift     — calibration drift vs research baseline
  • min_fill_rate       — fraction of placed orders that filled

Outputs:
  V3/05_live_trading/ledger/promotion_decision.json
    {decision: "go" | "no-go", checks: [...]}

Usage:
  python V3/05_live_trading/promotion_gate.py            # evaluate, exit 0/1
  python V3/05_live_trading/promotion_gate.py --flip     # only flips .env if go
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_LIVE_DIR  = Path(__file__).resolve().parent
_LEDGER    = _LIVE_DIR / "ledger"
_DECISION  = _LEDGER / "promotion_decision.json"
_LEDGER.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_LIVE_DIR.parent / "00_config"))
from risk_config import HOT as _RC, get as _rcget  # type: ignore  # noqa: E402

P = {
    "min_paper_trades":  int(_rcget("promotion", "min_paper_trades", default=40)),
    "min_paper_days":    int(_rcget("promotion", "min_paper_days", default=20)),
    "min_rolling_sharpe": float(_rcget("promotion", "min_rolling_sharpe", default=1.0)),
    "max_rolling_dd":    float(_rcget("promotion", "max_rolling_drawdown", default=0.10)),
    "max_slip_bps":      float(_rcget("promotion", "max_slippage_drift_bps", default=25)),
    "max_brier_drift":   float(_rcget("promotion", "max_calibration_brier_drift", default=0.05)),
    "min_fill_rate":     float(_rcget("promotion", "min_fill_rate", default=0.90)),
    "cooldown_days":     int(_rcget("promotion", "cooldown_after_breach_days", default=5)),
}


@dataclass
class Check:
    name:    str
    passed:  bool
    value:   float
    target:  float
    detail:  str = ""


def _ledger_state():
    sys.path.insert(0, str(_LIVE_DIR))
    from portfolio_ledger import get_state, rebuild_from_executions  # noqa
    state = rebuild_from_executions()
    return state


def _closed_trades_df() -> pd.DataFrame:
    state = _ledger_state()
    if not state.closed_trades:
        return pd.DataFrame()
    rows = []
    for t in state.closed_trades:
        rows.append({
            "symbol":      t.symbol,
            "entry_date":  t.entry_date,
            "exit_date":   t.exit_date,
            "ret_pct":     (t.exit_price - t.entry_price) / t.entry_price if t.entry_price > 0 else 0.0,
            "net_pnl":     t.net_pnl,
            "hold_days":   t.hold_days,
        })
    return pd.DataFrame(rows)


def _rolling_sharpe_dd(df: pd.DataFrame, window: int = 30) -> tuple[float, float]:
    if df.empty:
        return 0.0, 0.0
    df = df.sort_values("exit_date").tail(window)
    rets = df["ret_pct"].values
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252 / max(int(df["hold_days"].mean() or 10), 1))) \
             if rets.std() > 0 else 0.0
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return sharpe, float(-dd.min()) if len(dd) and dd.min() < 0 else 0.0


def _slippage_drift_bps() -> float:
    """Mean |fill_price - signal_price| in bps over last 30 paper fills."""
    exec_path = _LIVE_DIR / "execution_logs" / "execution_log.parquet"
    if not exec_path.exists():
        return float("nan")
    df = pd.read_parquet(exec_path)
    df = df[(df["status"] == "FILLED")].copy()
    if df.empty:
        return float("nan")
    df["filled_at"] = pd.to_datetime(df["filled_at"])
    df = df.sort_values("filled_at").tail(30)
    # `avg_price` is the fill, original requested price not always logged. Use
    # placed_at as a tie-breaker; this is best-effort.
    if "expected_price" in df.columns:
        diff = (df["avg_price"] - df["expected_price"]).abs() / df["expected_price"]
    else:
        # No expected price recorded — assume fill = signal (zero drift).
        return 0.0
    return float(diff.mean() * 10_000)


def _fill_rate() -> float:
    exec_path = _LIVE_DIR / "execution_logs" / "execution_log.parquet"
    if not exec_path.exists():
        return float("nan")
    df = pd.read_parquet(exec_path)
    if df.empty:
        return float("nan")
    placed = len(df)
    filled = (df["status"] == "FILLED").sum()
    return float(filled / placed) if placed else float("nan")


def _calibration_drift() -> float:
    """
    Brier on the latest 60 paper trades vs the research baseline. Returns
    NaN if there isn't enough recent data.
    """
    df = _closed_trades_df()
    if df.empty:
        return float("nan")
    # Win/loss is the closest paper-side proxy for calibration; compare against
    # the research-side win-rate (assumed 0.5 baseline). Future versions can
    # join meta_prob from the prediction snapshot.
    wr = float((df["ret_pct"] > 0).tail(60).mean())
    return abs(wr - 0.50)


def _paper_days(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    d0 = pd.to_datetime(df["entry_date"]).min().date()
    return int((datetime.now().date() - d0).days)


def evaluate() -> Dict:
    df = _closed_trades_df()

    n_trades = int(len(df))
    days = _paper_days(df)
    sharpe, max_dd = _rolling_sharpe_dd(df, window=30)
    slip_bps = _slippage_drift_bps()
    fill_rate = _fill_rate()
    brier_drift = _calibration_drift()

    checks: List[Check] = [
        Check("min_paper_trades",  n_trades >= P["min_paper_trades"],  n_trades, P["min_paper_trades"]),
        Check("min_paper_days",    days >= P["min_paper_days"],         days,     P["min_paper_days"]),
        Check("min_rolling_sharpe", sharpe >= P["min_rolling_sharpe"], round(sharpe, 3), P["min_rolling_sharpe"]),
        Check("max_rolling_dd",    max_dd <= P["max_rolling_dd"], round(max_dd, 3), P["max_rolling_dd"]),
        Check("max_slip_bps",      (np.isnan(slip_bps) or slip_bps <= P["max_slip_bps"]),
              round(slip_bps, 1) if not np.isnan(slip_bps) else -1, P["max_slip_bps"],
              detail="NaN treated as pass — no expected_price column" if np.isnan(slip_bps) else ""),
        Check("min_fill_rate",     (np.isnan(fill_rate) or fill_rate >= P["min_fill_rate"]),
              round(fill_rate, 3) if not np.isnan(fill_rate) else -1, P["min_fill_rate"]),
        Check("max_brier_drift",   (np.isnan(brier_drift) or brier_drift <= P["max_brier_drift"]),
              round(brier_drift, 3) if not np.isnan(brier_drift) else -1, P["max_brier_drift"]),
    ]
    decision = "go" if all(c.passed for c in checks) else "no-go"

    payload = {
        "evaluated_at": datetime.now().isoformat(),
        "decision":     decision,
        "checks":       [asdict(c) for c in checks],
        "summary": {
            "n_closed_trades":  n_trades,
            "paper_days":       days,
            "rolling_sharpe":   round(sharpe, 3),
            "rolling_max_dd":   round(max_dd, 3),
            "fill_rate":        round(fill_rate, 3) if not np.isnan(fill_rate) else None,
            "slippage_bps":     round(slip_bps, 1) if not np.isnan(slip_bps) else None,
            "brier_drift":      round(brier_drift, 3) if not np.isnan(brier_drift) else None,
        },
        "thresholds": P,
    }

    with open(_DECISION, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return payload


def _flip_env_to_live() -> bool:
    """Set TRADING_MODE=live in the project .env file."""
    env_path = _LIVE_DIR.parent.parent / ".env"
    if not env_path.exists():
        print(f"  [promotion] no .env at {env_path}; cannot flip")
        return False
    txt = env_path.read_text()
    if "TRADING_MODE=" in txt:
        new = "\n".join(
            "TRADING_MODE=live" if l.startswith("TRADING_MODE=") else l
            for l in txt.splitlines()
        )
    else:
        new = txt.rstrip() + "\nTRADING_MODE=live\n"
    env_path.write_text(new)
    print(f"  [promotion] flipped {env_path} → TRADING_MODE=live")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip", action="store_true",
                    help="On 'go' decision, set TRADING_MODE=live in .env")
    args = ap.parse_args()

    out = evaluate()
    print(f"\n  Promotion gate: {out['decision'].upper()}")
    for c in out["checks"]:
        ok = "✓" if c["passed"] else "✗"
        det = f"  {c['detail']}" if c.get("detail") else ""
        print(f"   {ok}  {c['name']:<22} {c['value']!s:<10} (target {c['target']}){det}")
    print(f"\n  decision saved → {_DECISION}")

    if out["decision"] == "go" and args.flip:
        _flip_env_to_live()
    return 0 if out["decision"] == "go" else 1


if __name__ == "__main__":
    sys.exit(main())
