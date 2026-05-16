"""
exit_policy.py — Multi-rule exit decision used by exit_runner + backtest
==========================================================================
Replaces the pure time-stop with four configurable layers:

  1. Vol-stop   : exit if drawdown > vol_stop_atr_mult × ATR(14)
  2. Trailing   : after position is +trailing_arm_pct in profit, exit if
                  it gives back trailing_stop_pct from peak
  3. Signal     : exit early if next-day prob_up < signal_decay_threshold
                  for `signal_decay_lookback` consecutive days
  4. Time stop  : exit on or after `time_stop_days` trading days
  5. Partial PT : sell `partial_take_profit_size` at +partial_take_profit_pct
                  (returned as a separate "partial" exit; the remaining qty
                  continues to use rules 1–4)

Inputs:
  lot         (dict)  : open BUY lot with `entry_price`, `entry_date`, `qty`
  bars        (DF)    : daily OHLCV from entry_date through today (inclusive)
  prob_series (Series): optional, indexed by date — daily prob_up forecasts

Returns:
  ExitDecision dict with: action ("hold" | "exit" | "partial"),
  reason (one of the rules), qty_to_sell, exit_price, ref_metrics dict.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_config"))
from risk_config import HOT as _RC, get as _rcget  # type: ignore  # noqa: E402

HOLD_DAYS               = int(_RC["HOLD_DAYS"])
VOL_STOP_ATR_MULT       = float(_RC["VOL_STOP_ATR_MULT"])
TRAIL_PCT               = float(_RC["TRAIL_PCT"])
TRAIL_ARM               = float(_RC["TRAIL_ARM"])
DECAY_THR               = float(_RC["DECAY_THR"])
DECAY_LOOKBACK          = int(_RC["DECAY_LOOKBACK"])
PARTIAL_PROFIT_PCT      = float(_RC["PARTIAL_PROFIT_PCT"])
PARTIAL_PROFIT_SIZE     = float(_RC["PARTIAL_PROFIT_SIZE"])


@dataclass
class ExitDecision:
    action:        str             # "hold" | "exit" | "partial"
    reason:        str             # which rule fired
    qty_to_sell:   int             # 0 if hold
    exit_price:    float           # last close (caller can override with LTP)
    pnl_pct:       float           # current paper-pnl on the lot
    ref:           Dict            # diagnostic metrics


def _atr(bars: pd.DataFrame, n: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    h = bars["high"].astype(float); l = bars["low"].astype(float); c = bars["close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    if len(tr.dropna()) < 1:
        return 0.0
    return float(tr.rolling(n, min_periods=1).mean().iloc[-1])


def evaluate(
    lot: Dict,
    bars: pd.DataFrame,
    prob_series: Optional[pd.Series] = None,
    today: Optional[date] = None,
    *,
    partial_taken: bool = False,
) -> ExitDecision:
    """
    Decide whether to hold, exit fully, or take partial profit on `lot`.

    `bars` must contain rows for every trading day from entry_date through
    today (inclusive), with at least open/high/low/close columns.

    `prob_series` (optional) is indexed by ISO date strings or pd.Timestamps
    and gives prob_up for each day; required only for the signal-decay rule.

    `partial_taken` should be True if the partial-profit rule has already
    fired for this lot — the rule will not fire twice.
    """
    if bars is None or len(bars) == 0:
        return ExitDecision("hold", "no_bars", 0, float(lot.get("entry_price", 0)), 0.0, {})

    bars = bars.sort_values("date").reset_index(drop=True)
    last = bars.iloc[-1]
    last_close = float(last["close"])
    entry_px = float(lot["entry_price"])
    qty = int(lot["qty"])
    pnl_pct = (last_close - entry_px) / entry_px if entry_px > 0 else 0.0

    # 1. Vol stop — drawdown vs entry exceeds ATR × multiplier
    atr = _atr(bars, n=14)
    atr_pct = atr / entry_px if entry_px > 0 else 0.0
    if atr_pct > 0 and pnl_pct < 0 and abs(pnl_pct) > VOL_STOP_ATR_MULT * atr_pct:
        return ExitDecision("exit", "vol_stop", qty, last_close, pnl_pct,
                            {"atr_pct": round(atr_pct, 4),
                             "threshold": round(VOL_STOP_ATR_MULT * atr_pct, 4)})

    # 2. Trailing stop — armed once trade is +TRAIL_ARM
    if pnl_pct >= TRAIL_ARM:
        peak = float(bars["close"].max())
        peak_pct = (peak - entry_px) / entry_px
        give_back = (peak - last_close) / peak if peak > 0 else 0.0
        if peak_pct >= TRAIL_ARM and give_back >= TRAIL_PCT:
            return ExitDecision("exit", "trailing_stop", qty, last_close, pnl_pct,
                                {"peak_pct": round(peak_pct, 4),
                                 "give_back": round(give_back, 4)})

    # 3. Signal-decay exit — 2 consecutive days of prob < threshold
    if prob_series is not None and len(prob_series) >= DECAY_LOOKBACK:
        recent = prob_series.dropna().tail(DECAY_LOOKBACK)
        if len(recent) >= DECAY_LOOKBACK and (recent < DECAY_THR).all():
            return ExitDecision("exit", "signal_decay", qty, last_close, pnl_pct,
                                {"decay_window": list(map(float, recent.values))})

    # 5. Partial take-profit (run before time stop so the remaining size
    #    can still benefit from a continued move; requires partial_taken=False)
    if not partial_taken and pnl_pct >= PARTIAL_PROFIT_PCT and qty > 1:
        partial_qty = max(1, int(round(qty * PARTIAL_PROFIT_SIZE)))
        return ExitDecision("partial", "partial_profit", partial_qty, last_close, pnl_pct,
                            {"target_pct": PARTIAL_PROFIT_PCT, "kept_qty": qty - partial_qty})

    # 4. Time stop — last resort
    today = today or date.today()
    try:
        entry_d = date.fromisoformat(str(lot["entry_date"]))
    except Exception:
        entry_d = today
    elapsed = int(np.busday_count(entry_d, today))
    if elapsed >= HOLD_DAYS:
        return ExitDecision("exit", "time_stop", qty, last_close, pnl_pct,
                            {"elapsed_busdays": elapsed})

    return ExitDecision("hold", "within_hold_window", 0, last_close, pnl_pct,
                        {"elapsed_busdays": elapsed,
                         "atr_pct": round(atr_pct, 4)})
