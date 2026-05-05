"""
tickers.py — Single source of truth for NSE symbol → yfinance ticker mapping.
Updated April 2026. All tickers verified against Yahoo Finance.

Key changes from default SYMBOL.NS:
  - LTIM  → LTM.NS   (LTIMindtree renamed ticker Feb 27, 2026)
  - M&M   → M&M.NS   (ampersand kept as-is, NOT URL-encoded)
"""

# NSE symbol → yfinance ticker
YFINANCE_TICKERS: dict[str, str] = {
    # Banking
    "SBIN":       "SBIN.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "ICICIBANK":  "ICICIBANK.NS",
    "AXISBANK":   "AXISBANK.NS",
    "KOTAKBANK":  "KOTAKBANK.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "BANDHANBNK": "BANDHANBNK.NS",
    "IDFCFIRSTB": "IDFCFIRSTB.NS",
    "FEDERALBNK": "FEDERALBNK.NS",
    "AUBANK":     "AUBANK.NS",
    "RBLBANK":    "RBLBANK.NS",
    # Finance / NBFC
    "BAJFINANCE": "BAJFINANCE.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "HDFCLIFE":   "HDFCLIFE.NS",
    "SBILIFE":    "SBILIFE.NS",
    "ICICIGI":    "ICICIGI.NS",
    "MUTHOOTFIN": "MUTHOOTFIN.NS",
    "CHOLAFIN":   "CHOLAFIN.NS",
    "SHRIRAMFIN": "SHRIRAMFIN.NS",
    "MANAPPURAM": "MANAPPURAM.NS",
    "LICHSGFIN":  "LICHSGFIN.NS",   # swapped 2026-05-05 from BAJAJHFL (IPO 2024-09; insufficient history)
    # IT
    "TCS":        "TCS.NS",
    "INFY":       "INFY.NS",
    "HCLTECH":    "HCLTECH.NS",
    "WIPRO":      "WIPRO.NS",
    "TECHM":      "TECHM.NS",
    "LTIM":       "LTM.NS",       # ← renamed Feb 27 2026
    "MPHASIS":    "MPHASIS.NS",
    "PERSISTENT": "PERSISTENT.NS",
    "COFORGE":    "COFORGE.NS",
    "TATAELXSI":  "TATAELXSI.NS",
    "OFSS":       "OFSS.NS",
    "NAUKRI":     "NAUKRI.NS",
    # Auto
    "MARUTI":     "MARUTI.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "EICHERMOT":  "EICHERMOT.NS",
    "TVSMOTOR":   "TVSMOTOR.NS",
    "M&M":        "M&M.NS",
    "MOTHERSON":  "MOTHERSON.NS",
    "BOSCHLTD":   "BOSCHLTD.NS",
    "EXIDEIND":   "EXIDEIND.NS",
    # Pharma
    "SUNPHARMA":  "SUNPHARMA.NS",
    "DRREDDY":    "DRREDDY.NS",
    "CIPLA":      "CIPLA.NS",
    "DIVISLAB":   "DIVISLAB.NS",
    "LUPIN":      "LUPIN.NS",
    "TORNTPHARM": "TORNTPHARM.NS",
    "AUROPHARMA": "AUROPHARMA.NS",
    "ALKEM":      "ALKEM.NS",
    # FMCG
    "HINDUNILVR": "HINDUNILVR.NS",
    "ITC":        "ITC.NS",
    "NESTLEIND":  "NESTLEIND.NS",
    "BRITANNIA":  "BRITANNIA.NS",
    "TATACONSUM": "TATACONSUM.NS",
    "MARICO":     "MARICO.NS",
    "COLPAL":     "COLPAL.NS",
    "GODREJCP":   "GODREJCP.NS",
    # Energy
    "RELIANCE":   "RELIANCE.NS",
    "ONGC":       "ONGC.NS",
    "BPCL":       "BPCL.NS",
    "NTPC":       "NTPC.NS",
    "POWERGRID":  "POWERGRID.NS",
    "COALINDIA":  "COALINDIA.NS",
    "GAIL":       "GAIL.NS",
    "TATAPOWER":  "TATAPOWER.NS",
    # Metals
    "TATASTEEL":  "TATASTEEL.NS",
    "HINDALCO":   "HINDALCO.NS",
    "JSWSTEEL":   "JSWSTEEL.NS",
    "VEDL":       "VEDL.NS",
    "SAIL":       "SAIL.NS",
    "NMDC":       "NMDC.NS",
    # Infra / Capital goods
    "LT":         "LT.NS",
    "BHEL":       "BHEL.NS",
    "SIEMENS":    "SIEMENS.NS",
    "ABB":        "ABB.NS",
    "HAVELLS":    "HAVELLS.NS",
    "POLYCAB":    "POLYCAB.NS",
    "CUMMINSIND": "CUMMINSIND.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "INDUSTOWER": "INDUSTOWER.NS",
    # Cement
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "GRASIM":     "GRASIM.NS",
    "AMBUJACEM":  "AMBUJACEM.NS",
    "SHREECEM":   "SHREECEM.NS",
    # Consumer / Discretionary
    "TITAN":      "TITAN.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "PIDILITIND": "PIDILITIND.NS",
    "BERGEPAINT": "BERGEPAINT.NS",
    "VOLTAS":     "VOLTAS.NS",
    "PAGEIND":    "PAGEIND.NS",
    # Real estate / Retail
    "DLF":        "DLF.NS",
    "DMART":      "DMART.NS",
    "GODREJPROP": "GODREJPROP.NS",
    # Conglomerate
    "ADANIENT":   "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    # Defence / PSU
    "BEL":        "BEL.NS",
    "HAL":        "HAL.NS",
    "IRFC":       "IRFC.NS",
    # Others
    "ETERNAL":    "ETERNAL.NS",
    "LTIM":       "LTM.NS",       # duplicate key intentional — alias
}

# Reverse map: yfinance ticker → NSE symbol
TICKER_TO_NSE: dict[str, str] = {v: k for k, v in YFINANCE_TICKERS.items()}


def to_yf(nse_symbol: str) -> str:
    """Convert NSE symbol to yfinance ticker. Falls back to SYMBOL.NS."""
    return YFINANCE_TICKERS.get(nse_symbol, f"{nse_symbol}.NS")


def to_nse(yf_ticker: str) -> str:
    """Convert yfinance ticker back to NSE symbol."""
    return TICKER_TO_NSE.get(yf_ticker, yf_ticker.replace(".NS", ""))


def yf_tickers(nse_symbols: list[str]) -> list[str]:
    """Batch convert NSE symbols to yfinance tickers."""
    return [to_yf(s) for s in nse_symbols]
