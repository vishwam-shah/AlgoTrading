"""
sentiment_history.py — Build and maintain daily sentiment parquet.
===================================================================
Appends today's FinBERT sentiment for all 100 NSE symbols to
V3/01_data/news/sentiment_history.parquet

Run daily (after 4 PM IST) to accumulate sentiment history.
Over time this becomes a training feature for the walk-forward pipeline.

Usage:
    python V3/01_data/news/sentiment_history.py           # all 100 symbols
    python V3/01_data/news/sentiment_history.py SBIN HDFCBANK  # specific
    python V3/01_data/news/sentiment_history.py --date 2026-04-15  # backfill
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_NEWS_DIR = Path(__file__).resolve().parent
_V3_ROOT  = _NEWS_DIR.parent.parent
sys.path.insert(0, str(_V3_ROOT))
sys.path.insert(0, str(_NEWS_DIR))

import pandas as pd

HIST_PATH = _NEWS_DIR / "sentiment_history.parquet"

# All 100 NSE symbols from config
_DEFAULT_SYMBOLS = [
    "ABB","ADANIENT","ADANIPORTS","ALKEM","AMBUJACEM","ASIANPAINT","AUBANK",
    "AUROPHARMA","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BANDHANBNK",
    "BERGEPAINT","BHARTIARTL","BHEL","BPCL","BRITANNIA","CHOLAFIN","CIPLA",
    "COALINDIA","COFORGE","COLPAL","CUMMINSIND","DIVISLAB","DLF","DMART",
    "DRREDDY","EICHERMOT","FEDERALBNK","GAIL","GODREJCP","GODREJPROP","GRASIM",
    "HAVELLS","HCLTECH","HDFCBANK","HDFCLIFE","HEROMOTOCO","HINDALCO",
    "HINDUNILVR","ICICIBANK","ICICIGI","IDFCFIRSTB","INDUSINDBK","INDUSTOWER",
    "INFY","ITC","JSWSTEEL","KOTAKBANK","LT","LTIM","LUPIN","M&M","MARICO",
    "MARUTI","MOTHERSON","MPHASIS","MUTHOOTFIN","NESTLEIND","NTPC","ONGC",
    "PAGEIND","PERSISTENT","PIDILITIND","POLYCAB","POWERGRID","RELIANCE",
    "SAIL","SBILIFE","SBIN","SHREECEM","SHRIRAMFIN","SIEMENS","SUNPHARMA",
    "TATACONSUM","TATAELXSI","TATAPOWER","TATASTEEL","TCS","TECHM","TITAN",
    "TORNTPHARM","TVSMOTOR","ULTRACEMCO","VEDL","VOLTAS","WIPRO",
]


def load_history() -> pd.DataFrame:
    """Load existing sentiment history, return empty DataFrame if none."""
    if HIST_PATH.exists():
        df = pd.read_parquet(HIST_PATH)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df
    return pd.DataFrame(columns=[
        "date", "symbol", "raw_score", "positive_ratio",
        "negative_ratio", "neutral_ratio", "n_articles", "model_used",
    ])


def already_fetched(hist: pd.DataFrame, symbol: str, date: str) -> bool:
    """Return True if we already have a record for this symbol+date."""
    if hist.empty:
        return False
    d = pd.Timestamp(date).normalize()
    return not hist[(hist["symbol"] == symbol) & (hist["date"] == d)].empty


def fetch_one(symbol: str, date: str) -> dict:
    """Fetch and score news for one symbol. Returns result dict."""
    from news_fetcher import NewsFeaturizer
    nf     = NewsFeaturizer()
    result = nf.fetch_and_score(symbol, date=date)
    return {
        "date":           pd.Timestamp(date).normalize(),
        "symbol":         symbol,
        "raw_score":      result.get("raw_score",       0.0),
        "positive_ratio": result.get("positive_ratio",  0.0),
        "negative_ratio": result.get("negative_ratio",  0.0),
        "neutral_ratio":  result.get("neutral_ratio",   1.0),
        "n_articles":     result.get("n_articles",      0),
        "model_used":     result.get("model_used",      "unknown"),
    }


def append_today(symbols: list[str], date: str, verbose: bool = True) -> int:
    """
    Fetch today's sentiment for all symbols not yet in history.
    Returns number of new records appended.
    """
    hist    = load_history()
    new_rows = []

    for sym in symbols:
        if already_fetched(hist, sym, date):
            if verbose:
                print(f"  {sym:<14} ✓ already in history")
            continue
        try:
            row = fetch_one(sym, date)
            new_rows.append(row)
            if verbose:
                score = row["raw_score"]
                n     = row["n_articles"]
                sign  = "▲" if score > 0.1 else ("▼" if score < -0.1 else "─")
                print(f"  {sym:<14} {sign} score={score:+.3f}  n={n:2d}  [{row['model_used']}]")
        except Exception as e:
            print(f"  {sym:<14} ✗ error: {e}")

    if not new_rows:
        if verbose:
            print("  Nothing new to append.")
        return 0

    new_df = pd.DataFrame(new_rows)
    updated = pd.concat([hist, new_df], ignore_index=True)
    updated = updated.drop_duplicates(subset=["date", "symbol"], keep="last")
    updated = updated.sort_values(["date", "symbol"]).reset_index(drop=True)
    updated.to_parquet(HIST_PATH, index=False, compression="snappy")
    if verbose:
        print(f"\n  Saved {len(new_rows)} new rows → {HIST_PATH.name}"
              f"  (total: {len(updated)} rows, "
              f"{updated['symbol'].nunique()} symbols, "
              f"{updated['date'].nunique()} dates)")
    return len(new_rows)


# ── Called by features.py to load the full history for feature computation ────

def load_for_features() -> pd.DataFrame:
    """
    Load sentiment history in the format expected by features._add_sentiment_features().
    Returns wide DataFrame with columns: date, symbol, raw_score, ...
    Returns empty DataFrame if no history file exists yet.
    """
    if not HIST_PATH.exists():
        return pd.DataFrame()
    df = load_history()
    df["date"] = df["date"].astype("datetime64[us]")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbols", nargs="*", help="Symbols (default: all 100)")
    parser.add_argument("--date",  default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date to fetch (default: today)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    syms = args.symbols if args.symbols else _DEFAULT_SYMBOLS
    print(f"\nFetching sentiment for {len(syms)} symbols on {args.date} ...")
    n = append_today(syms, args.date, verbose=not args.quiet)
    print(f"\nDone. {n} new records added.")
