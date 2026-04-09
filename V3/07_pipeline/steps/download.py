"""
step 1 — Data Download
======================
Incremental OHLCV download for NSE stocks, USD/INR, and global market cues.
All functions write to V3/01_data/raw/ as .parquet files (date column).
"""

from __future__ import annotations

import calendar as _calendar
import concurrent.futures as _cf
import os
import sys
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_STEPS_DIR = Path(__file__).resolve().parent
_PIPE_DIR  = _STEPS_DIR.parent
_V3_ROOT   = _PIPE_DIR.parent
sys.path.insert(0, str(_V3_ROOT))

from config_v3 import (  # type: ignore  # noqa: E402
    RAW_DATA_DIR, DATA_START_DATE, YFINANCE_DELAY,
    GLOBAL_CUES_TICKERS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  I/O HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_parquet(path: Path) -> Optional[pd.DataFrame]:
    """Read a parquet; normalise date column (supports both 'date' and 'timestamp')."""
    try:
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "date"})
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return None


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def fetch_yfinance(ticker: str, start: str, end: str, retries: int = 3) -> Optional[pd.DataFrame]:
    """Download OHLCV from Yahoo Finance with retry logic. Returns DataFrame with 'date' column."""
    import yfinance as yf

    for attempt in range(retries):
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                if attempt < retries - 1:
                    _time.sleep(1.0 * (attempt + 1))
                    continue
                return None
            df = raw.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [str(c).lower() for c in df.columns]
            # Normalise date column name
            for dc in ["date", "datetime", "index"]:
                if dc in df.columns:
                    df = df.rename(columns={dc: "date"})
                    break
            df["date"] = pd.to_datetime(df["date"])
            keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[keep].sort_values("date").reset_index(drop=True)
            df = df[df["close"] > 0].reset_index(drop=True)
            return df if not df.empty else None
        except Exception as e:
            if attempt < retries - 1:
                _time.sleep(1.0 * (attempt + 1))
            else:
                print(f"  [yfinance] {ticker} failed after {retries} attempts: {e}")
    return None


def last_thursday_of_month(year: int, month: int) -> date:
    """Return the last Thursday of the given month (NSE monthly F&O expiry day)."""
    _, last_day = _calendar.monthrange(year, month)
    dt = date(year, month, last_day)
    offset = (dt.weekday() - 3) % 7
    return dt - timedelta(days=offset)


# ══════════════════════════════════════════════════════════════════════════════
#  SYMBOL DOWNLOAD  (incremental)
# ══════════════════════════════════════════════════════════════════════════════

def download_symbol(symbol: str) -> Optional[pd.DataFrame]:
    """Incremental download for one NSE symbol. Appends new rows only."""
    save_path = RAW_DATA_DIR / f"{symbol}.parquet"
    existing  = load_parquet(save_path)
    ticker    = f"{symbol}.NS"
    today     = datetime.now().strftime("%Y-%m-%d")

    if existing is not None and not existing.empty:
        last_date   = existing["date"].max()
        fetch_start = (last_date - timedelta(days=4)).strftime("%Y-%m-%d")
        check_date  = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        if check_date >= today:
            print(f"  {symbol:<12} ✓ up-to-date  [{last_date.date()}]")
            return existing
        print(f"  {symbol:<12} ↓ incremental  {last_date.date()} → {today} ...", end=" ", flush=True)
        new_df = fetch_yfinance(ticker, fetch_start, today)
        if new_df is not None and not new_df.empty:
            combined = (
                pd.concat([existing, new_df], ignore_index=True)
                .drop_duplicates("date").sort_values("date").reset_index(drop=True)
            )
            n_added = len(combined) - len(existing)
            if n_added > 0:
                save_parquet(combined, save_path)
                print(f"✓ +{n_added} rows  (total {len(combined)}  [{combined['date'].iloc[-1].date()}])")
                return combined
        print("✓ no new rows")
        return existing

    print(f"  {symbol:<12} ↓ full  {DATA_START_DATE} → {today} ...", end=" ", flush=True)
    df = fetch_yfinance(ticker, DATA_START_DATE, today)
    if df is None or df.empty:
        print(f"✗ failed ({ticker} not available)")
        return None
    save_parquet(df, save_path)
    print(f"✓ {len(df)} rows  [{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}]")
    return df


def download_usdinr() -> Optional[pd.DataFrame]:
    """Incremental download of USD/INR exchange rate."""
    save_path   = RAW_DATA_DIR / "usdinr.parquet"
    existing    = load_parquet(save_path)
    today       = datetime.now().strftime("%Y-%m-%d")
    fetch_start = DATA_START_DATE
    if existing is not None and not existing.empty:
        last_date   = existing["date"].max()
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        if fetch_start >= today:
            return existing
    df = fetch_yfinance("USDINR=X", fetch_start, today)
    if df is None:
        return existing
    df = df.rename(columns={"close": "usdinr_close"})[["date", "usdinr_close"]]
    df = df[df["usdinr_close"] > 0].reset_index(drop=True)
    if existing is not None and not existing.empty:
        df = (pd.concat([existing, df], ignore_index=True)
                .drop_duplicates("date").sort_values("date").reset_index(drop=True))
    save_parquet(df, save_path)
    return df


def download_global_cues() -> Optional[pd.DataFrame]:
    """
    Incremental download of global market cues (S&P500, Nasdaq, VIX, DXY, Crude, Nikkei).
    Each cue is stored as: {name}_close, {name}_ret (1-day log return).
    Returns a wide DataFrame with 'date' index.
    """
    save_path = RAW_DATA_DIR / "global_cues.parquet"
    existing  = load_parquet(save_path)
    today     = datetime.now().strftime("%Y-%m-%d")
    fetch_start = DATA_START_DATE

    if existing is not None and not existing.empty:
        last_date   = existing["date"].max()
        fetch_start = (last_date - timedelta(days=4)).strftime("%Y-%m-%d")
        check_date  = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        if check_date >= today:
            return existing

    all_series: Dict[str, pd.DataFrame] = {}
    for name, ticker in GLOBAL_CUES_TICKERS.items():
        df = fetch_yfinance(ticker, fetch_start, today)
        if df is None or df.empty:
            continue
        df = df[["date", "close"]].rename(columns={"close": f"{name}_close"})
        df[f"{name}_ret"] = np.log(df[f"{name}_close"] / df[f"{name}_close"].shift(1))
        all_series[name] = df.set_index("date")

    if not all_series:
        return existing

    wide = pd.concat(all_series.values(), axis=1).reset_index()
    wide = wide.rename(columns={"index": "date"}) if "index" in wide.columns else wide
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.sort_values("date").reset_index(drop=True)

    if existing is not None and not existing.empty:
        wide = (pd.concat([existing, wide], ignore_index=True)
                .drop_duplicates("date").sort_values("date").reset_index(drop=True))

    save_parquet(wide, save_path)
    return wide


def download_all_symbols(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Download all symbols in parallel threads. Returns {symbol: DataFrame}."""
    raw_data: Dict[str, pd.DataFrame] = {}
    with _cf.ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
        futures = {pool.submit(download_symbol, s): s for s in symbols}
        for fut in _cf.as_completed(futures):
            sym = futures[fut]
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    raw_data[sym] = df
                    _time.sleep(YFINANCE_DELAY)
            except Exception as exc:
                print(f"  {sym}: download error — {exc}")
    return raw_data
