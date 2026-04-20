"""
earnings_calendar.py
====================
Provides `days_to_earnings` for NSE stocks.

Two modes:
  Live  (predict_today) : yfinance Ticker.calendar → exact announced date
  Historical (pipeline) : NSE quarterly reporting heuristic → estimated date

NSE quarterly reporting windows (large-cap companies):
  Q4 (Jan-Mar) → results Apr 15 – May 31
  Q1 (Apr-Jun) → results Jul 15 – Aug 31
  Q2 (Jul-Sep) → results Oct 15 – Nov 30
  Q3 (Oct-Dec) → results Jan 15 – Feb 28

Feature semantics:
  days_to_earnings : int, days until next earnings (0 = today, <0 = past)
  is_results_week  : binary, 1 if |days_to_earnings| <= 3
  pre_results_drift: binary, 1 if 1 <= days_to_earnings <= 5 (buy-the-rumour)
  post_results_day : binary, 1 if -2 <= days_to_earnings <= 0 (sell-the-news)
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

_CACHE_DIR  = Path(__file__).resolve().parent / "_earnings_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_YF_LOCK    = threading.Lock()

# Approximate mid-point of each quarter's results window
# (month, day) tuples — used for historical heuristic
_RESULT_WINDOWS = [
    (4, 25),   # Q4 FY: April 25
    (7, 25),   # Q1: July 25
    (10, 25),  # Q2: October 25
    (2, 10),   # Q3: February 10 (next year)
]


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE MODE  (yfinance)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_next_earnings_yf(symbol: str, cache_ttl_hours: int = 24) -> Optional[date]:
    """
    Fetch next earnings date from yfinance. Cached for 24h to avoid hammering.
    Returns None if not available.
    """
    cache_file = _CACHE_DIR / f"{symbol}.json"

    # Check cache
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            cached_at = pd.Timestamp(data["cached_at"])
            if (pd.Timestamp.now() - cached_at).total_seconds() < cache_ttl_hours * 3600:
                d = data.get("earnings_date")
                return date.fromisoformat(d) if d else None
        except Exception:
            pass

    # Fetch from yfinance
    try:
        import yfinance as yf
        with _YF_LOCK:
            t = yf.Ticker(f"{symbol}.NS")
            cal = t.calendar
        time.sleep(0.3)

        earnings_date = None
        if cal and "Earnings Date" in cal:
            ed = cal["Earnings Date"]
            if isinstance(ed, list) and ed:
                earnings_date = ed[0]
            elif isinstance(ed, date):
                earnings_date = ed

        result = earnings_date.isoformat() if earnings_date else None
        cache_file.write_text(json.dumps({
            "symbol": symbol,
            "earnings_date": result,
            "cached_at": pd.Timestamp.now().isoformat(),
        }))
        return earnings_date

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  HISTORICAL HEURISTIC  (pipeline use)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_next_earnings(current_date: date) -> date:
    """
    Estimate next NSE earnings announcement date from a given date.
    Uses standard NSE quarterly reporting window midpoints.
    """
    year = current_date.year

    candidates = []
    for month, day in _RESULT_WINDOWS:
        y = year if month >= 4 else year + 1  # Feb window is next calendar year
        try:
            candidates.append(date(y, month, day))
        except ValueError:
            pass
    # Add next year's Q4 window as fallback
    candidates.append(date(year + 1, 4, 25))

    future = sorted(d for d in candidates if d > current_date)
    return future[0] if future else date(year + 1, 4, 25)


def days_to_next_earnings(current_date: date, symbol: Optional[str] = None,
                          use_live: bool = True) -> int:
    """
    Return days until next earnings. Negative = earnings already passed.

    Args:
        current_date : Reference date (today for live, row date for historical)
        symbol       : NSE symbol (optional, enables yfinance live lookup)
        use_live     : If True and symbol given, try yfinance first

    Returns:
        int: days until earnings (positive = upcoming, negative = past)
    """
    next_date = None

    if use_live and symbol:
        next_date = fetch_next_earnings_yf(symbol)

    if next_date is None:
        next_date = estimate_next_earnings(current_date)

    return (next_date - current_date).days


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE VECTOR  (for both pipeline and predict_today)
# ══════════════════════════════════════════════════════════════════════════════

def compute_earnings_features(df: pd.DataFrame, symbol: Optional[str] = None,
                               use_live: bool = False) -> pd.DataFrame:
    """
    Add earnings-proximity features to a feature DataFrame.

    Columns added:
      days_to_earnings   : int, days until next results
      is_results_week    : float (0/1), within 3 days either side
      pre_results_drift  : float (0/1), 1-5 days before results (buy the rumour)
      post_results_day   : float (0/1), 0-2 days after results (sell the news)
      earnings_proximity : float [0,1], 1/(1+days) smoothed proximity score

    use_live=True only for predict_today (single latest row, yfinance fetch).
    use_live=False for historical pipeline (uses heuristic, no API calls).
    """
    d = df.copy()
    dates = pd.to_datetime(d["date"]).dt.date

    if use_live and symbol:
        # Single-fetch for inference — apply to last row only
        latest = dates.iloc[-1]
        n_days = days_to_next_earnings(latest, symbol=symbol, use_live=True)
        dte    = pd.Series([n_days] * len(d), index=d.index, dtype=float)
    else:
        # Vectorised heuristic for full historical column
        dte = dates.apply(lambda dt: days_to_next_earnings(dt, use_live=False))
        dte = dte.astype(float)

    d["days_to_earnings"]   = dte.values
    d["is_results_week"]    = (dte.abs() <= 3).astype(float).values
    d["pre_results_drift"]  = ((dte >= 1) & (dte <= 5)).astype(float).values
    d["post_results_day"]   = ((dte >= -2) & (dte <= 0)).astype(float).values
    # Smooth proximity: peaks at 1 on result day, decays as distance grows
    d["earnings_proximity"] = (1.0 / (1.0 + dte.abs().clip(lower=0))).values

    return d


if __name__ == "__main__":
    # Quick test
    from datetime import date as _date
    today = _date.today()

    print(f"Today: {today}")
    dte = days_to_next_earnings(today, symbol="HDFCBANK", use_live=True)
    print(f"HDFCBANK days_to_earnings: {dte}")

    est = estimate_next_earnings(today)
    print(f"Heuristic estimate: {est}")
