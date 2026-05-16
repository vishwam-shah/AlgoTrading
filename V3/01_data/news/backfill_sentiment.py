"""
backfill_sentiment.py — Historical sentiment via yfinance news archive.
======================================================================
Google News RSS returns only current headlines. yfinance's Ticker.news
exposes ~10-30 historical articles per ticker with pubDate, which lets
us reconstruct daily sentiment for the last 1-4 weeks.

Groups headlines by pubDate (IST date), scores each daily batch with
FinBERT India, and appends to sentiment_history.parquet — the same
schema produced by sentiment_history.py so downstream features work.

Usage:
    python V3/01_data/news/backfill_sentiment.py                # all 100
    python V3/01_data/news/backfill_sentiment.py SBIN HDFCBANK  # specific
    python V3/01_data/news/backfill_sentiment.py --min-articles 2
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_NEWS_DIR = Path(__file__).resolve().parent
_V3_ROOT  = _NEWS_DIR.parent.parent
sys.path.insert(0, str(_V3_ROOT))
sys.path.insert(0, str(_NEWS_DIR))

import pandas as pd

from sentiment_history import _DEFAULT_SYMBOLS, HIST_PATH, load_history


def _fetch_yfinance_news(symbol: str) -> list[dict]:
    """Fetch raw news items from yfinance for one symbol. Returns [] on failure."""
    try:
        import yfinance as yf
        # Use canonical ticker map if available
        try:
            sys.path.insert(0, str(_V3_ROOT / "00_config"))
            from tickers import to_yf  # type: ignore
            ticker = to_yf(symbol)
        except Exception:
            ticker = f"{symbol}.NS"
        items = yf.Ticker(ticker).news or []
        return items
    except Exception as e:
        print(f"  {symbol:<14} ✗ yfinance error: {e}")
        return []


def _extract_title_date(item: dict) -> tuple[str, pd.Timestamp] | None:
    """Normalize a yfinance news item to (title, date_normalized)."""
    content = item.get("content") or item
    title = content.get("title") or ""
    pub = content.get("pubDate") or content.get("displayTime") or content.get("providerPublishTime")
    if not title or not pub:
        return None
    try:
        if isinstance(pub, (int, float)):
            ts = pd.Timestamp(int(pub), unit="s", tz="UTC")
        else:
            ts = pd.Timestamp(pub)
        # Convert to IST market day
        ts = ts.tz_convert("Asia/Kolkata") if ts.tzinfo else ts.tz_localize("UTC").tz_convert("Asia/Kolkata")
        return title.strip(), ts.normalize().tz_localize(None)
    except Exception:
        return None


def _score_and_build_rows(symbol: str, by_date: dict[pd.Timestamp, list[str]]) -> list[dict]:
    """Score each date's headlines with FinBERT, return rows matching parquet schema."""
    from news_fetcher import NewsFeaturizer
    nf = NewsFeaturizer()
    rows = []
    for date, headlines in sorted(by_date.items()):
        if not headlines:
            continue
        scores = nf.score_headlines(headlines)
        rows.append({
            "date":           date,
            "symbol":         symbol,
            "raw_score":      scores["raw_score"],
            "positive_ratio": scores["positive_ratio"],
            "negative_ratio": scores["negative_ratio"],
            "neutral_ratio":  scores["neutral_ratio"],
            "n_articles":     len(headlines),
            "model_used":     scores["model_used"],
        })
    return rows


def backfill_one(symbol: str, hist: pd.DataFrame, min_articles: int = 1,
                 verbose: bool = True) -> list[dict]:
    """Backfill historical sentiment for one symbol. Returns new rows."""
    items = _fetch_yfinance_news(symbol)
    if not items:
        if verbose:
            print(f"  {symbol:<14} – no news items returned")
        return []

    by_date: dict[pd.Timestamp, list[str]] = defaultdict(list)
    for it in items:
        parsed = _extract_title_date(it)
        if parsed is None:
            continue
        title, date = parsed
        by_date[date].append(title)

    # Skip dates already in history
    existing = set()
    if not hist.empty:
        existing = set(
            hist[hist["symbol"] == symbol]["date"].dt.normalize().tolist()
        )

    new_by_date = {
        d: h for d, h in by_date.items()
        if d not in existing and len(h) >= min_articles
    }

    if not new_by_date:
        if verbose:
            print(f"  {symbol:<14} ✓ up-to-date ({len(by_date)} dates cached)")
        return []

    rows = _score_and_build_rows(symbol, new_by_date)
    if verbose and rows:
        span = f"{min(r['date'] for r in rows).strftime('%m-%d')}..{max(r['date'] for r in rows).strftime('%m-%d')}"
        print(f"  {symbol:<14} ✓ +{len(rows)} dates  [{span}]")
    return rows


def main(symbols: list[str], min_articles: int = 1) -> None:
    print(f"\nBackfilling FinBERT sentiment from yfinance news for {len(symbols)} symbols ...\n")
    hist = load_history()
    initial = len(hist)

    all_new = []
    for i, sym in enumerate(symbols, 1):
        try:
            rows = backfill_one(sym, hist, min_articles=min_articles, verbose=True)
            all_new.extend(rows)
            # Small delay to avoid rate-limiting yfinance
            if i % 10 == 0:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n  ⚠ interrupted — saving progress so far")
            break
        except Exception as e:
            print(f"  {sym:<14} ✗ error: {e}")

    if not all_new:
        print("\nNothing new to append.")
        return

    new_df  = pd.DataFrame(all_new)
    updated = pd.concat([hist, new_df], ignore_index=True)
    updated = updated.drop_duplicates(subset=["date", "symbol"], keep="last")
    updated = updated.sort_values(["date", "symbol"]).reset_index(drop=True)
    updated.to_parquet(HIST_PATH, index=False, compression="snappy")

    print(f"\n  Appended {len(all_new)} new rows  (was {initial}, now {len(updated)})")
    print(f"  Coverage: {updated['symbol'].nunique()} symbols × "
          f"{updated['date'].nunique()} dates "
          f"({updated['date'].min().date()} → {updated['date'].max().date()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbols", nargs="*", help="Symbols (default: all 100)")
    parser.add_argument("--min-articles", type=int, default=1,
                        help="Minimum articles per day to record (default: 1)")
    args = parser.parse_args()

    syms = args.symbols if args.symbols else _DEFAULT_SYMBOLS
    main(syms, min_articles=args.min_articles)
