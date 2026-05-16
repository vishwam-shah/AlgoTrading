"""
instrument_master.py — Daily refresh of Angel One symbol→token map
====================================================================
Replaces the hard-coded NSE_TOKEN_MAP in angel_one_client.py with the
live OpenAPIScripMaster.json from Angel — fetched once per session and
cached locally so we never hit the upstream more than necessary.

Source:
  https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json

Cache:
  V3/01_data/raw/angel_instrument_master.parquet  (full master, ~5MB)
  V3/01_data/raw/angel_token_map_NSE.json         (compact symbol→token)

Refresh policy:
  - If cache is older than today (IST) OR missing, re-fetch.
  - On network failure, fall back to the cached file (warn loudly).
  - On total miss (no cache, no network), raise — caller must decide.

Public surface:
  get_token(symbol, exchange="NSE", segment="EQ") -> str | None
  load_token_map(exchange="NSE", segment="EQ") -> dict[str, str]
  refresh(force=False) -> Path  # path to refreshed parquet
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

_LIVE_DIR = Path(__file__).resolve().parent
_RAW_DIR = _LIVE_DIR.parent / "01_data" / "raw"
_RAW_DIR.mkdir(parents=True, exist_ok=True)

_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/"
    "OpenAPIScripMaster.json"
)
_MASTER_PARQUET = _RAW_DIR / "angel_instrument_master.parquet"
_TOKEN_MAP_JSON = _RAW_DIR / "angel_token_map_NSE.json"
_FETCH_TIMEOUT_SEC = 30


def _is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
    return mtime != datetime.now().date()


def _download_master() -> pd.DataFrame:
    """Pull the JSON master and return as DataFrame. Raises on network error.

    Prefers `requests` (uses certifi's trust store on macOS where stdlib
    urllib often fails with CERTIFICATE_VERIFY_FAILED). Falls back to urllib.
    """
    payload = None
    try:
        import requests  # type: ignore
        r = requests.get(_MASTER_URL, timeout=_FETCH_TIMEOUT_SEC,
                         headers={"User-Agent": "AlgoTrading/1.0 (research)"})
        r.raise_for_status()
        payload = r.json()
    except Exception:
        import urllib.request, ssl
        ctx = ssl.create_default_context()
        try:
            import certifi  # type: ignore
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
        req = urllib.request.Request(
            _MASTER_URL, headers={"User-Agent": "AlgoTrading/1.0 (research)"}
        )
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_SEC, context=ctx) as r:
            payload = json.load(r)
    df = pd.DataFrame(payload)
    if df.empty:
        raise RuntimeError("Angel instrument master returned empty payload")
    return df


def refresh(force: bool = False) -> Path:
    """
    Refresh the local instrument master if stale, return parquet path.

    On network failure, falls back to cached parquet and prints a warning.
    Raises FileNotFoundError if there is no cache and the network is down.
    """
    if not force and not _is_stale(_MASTER_PARQUET):
        return _MASTER_PARQUET

    try:
        df = _download_master()
        # Persist full master + a compact NSE-EQ map for fast lookups.
        df.to_parquet(_MASTER_PARQUET, index=False, compression="snappy")
        nse = df[(df["exch_seg"] == "NSE") & (df.get("instrumenttype", "").fillna("") == "")]
        token_map = dict(zip(nse["symbol"].astype(str), nse["token"].astype(str)))
        # Symbols arrive as e.g. "SBIN-EQ" — strip the "-EQ" suffix for our use.
        clean = {s.replace("-EQ", "").upper(): t for s, t in token_map.items()}
        with open(_TOKEN_MAP_JSON, "w") as f:
            json.dump(clean, f, separators=(",", ":"))
        print(f"  [instrument_master] refreshed: {len(clean):,} NSE symbols → {_TOKEN_MAP_JSON.name}")
        return _MASTER_PARQUET
    except Exception as e:
        if _MASTER_PARQUET.exists():
            print(f"  [instrument_master] WARN — fetch failed ({e}); using cached "
                  f"{_MASTER_PARQUET.name} from {datetime.fromtimestamp(_MASTER_PARQUET.stat().st_mtime).date()}")
            return _MASTER_PARQUET
        raise FileNotFoundError(
            f"Angel instrument master fetch failed and no cache at {_MASTER_PARQUET}: {e}"
        )


def load_token_map(exchange: str = "NSE", segment: str = "EQ") -> Dict[str, str]:
    """Return {symbol: token} for the requested exchange/segment."""
    refresh()
    if exchange == "NSE" and segment == "EQ" and _TOKEN_MAP_JSON.exists():
        with open(_TOKEN_MAP_JSON) as f:
            return json.load(f)
    # Generic path — recompute from parquet
    df = pd.read_parquet(_MASTER_PARQUET)
    seg = f"{exchange}"
    sub = df[df["exch_seg"] == seg]
    if segment == "EQ":
        sub = sub[sub.get("instrumenttype", "").fillna("") == ""]
    return {str(s).replace(f"-{segment}", "").upper(): str(t)
            for s, t in zip(sub["symbol"], sub["token"])}


def get_token(symbol: str, exchange: str = "NSE", segment: str = "EQ") -> Optional[str]:
    """Look up a single symbol — refreshes on-demand."""
    m = load_token_map(exchange=exchange, segment=segment)
    return m.get(symbol.upper())


# ── CLI ──────────────────────────────────────────────────────────────────────
def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-fetch even if cache is fresh")
    p.add_argument("--lookup", help="Print the token for one symbol and exit")
    args = p.parse_args()
    if args.lookup:
        t = get_token(args.lookup)
        print(f"{args.lookup}: {t or '(not found)'}")
        return 0 if t else 1
    refresh(force=args.force)
    m = load_token_map()
    print(f"  total NSE symbols cached: {len(m):,}")
    print("  sample:", {k: m[k] for k in list(m)[:5]})
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
