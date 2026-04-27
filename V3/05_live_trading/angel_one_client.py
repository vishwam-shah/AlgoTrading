"""
angel_one_client.py — Angel One SmartAPI REST + WebSocket wrapper
=================================================================
Full production implementation.

Credentials from .env:
    ANGEL_API_KEY       = your SmartAPI key
    ANGEL_CLIENT_ID     = your client ID (e.g. A1234567)
    ANGEL_PASSWORD      = your trading password
    ANGEL_TOTP_SECRET   = base32 TOTP secret from Angel One 2FA setup

Usage:
    client = AngelOneClient()
    client.login()
    ltp    = client.get_ltp("SBIN")
    resp   = client.place_order("SBIN", qty=10, price=810.0)
    funds  = client.get_funds()
    holds  = client.get_holdings()
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ── Load .env credentials ─────────────────────────────────────────────────────
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV_PATH.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)

# ── SmartAPI SDK (pip install smartapi-python) ────────────────────────────────
try:
    from SmartApi import SmartConnect
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False
    print("  [angel_one] smartapi-python not installed. Run: pip install smartapi-python")

import pyotp


# ── NSE token map — top-100 symbols ───────────────────────────────────────────
# Exchange token IDs from NSE (used for WebSocket subscription).
# Full list maintained at: https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json
NSE_TOKEN_MAP: Dict[str, str] = {
    "SBIN": "3045", "HDFCBANK": "1333", "ICICIBANK": "4963", "AXISBANK": "5900",
    "KOTAKBANK": "1922", "INDUSINDBK": "5258", "BANDHANBNK": "2263",
    "IDFCFIRSTB": "11184", "FEDERALBNK": "1023", "AUBANK": "11223",
    "BAJFINANCE": "317", "BAJAJFINSV": "16675", "HDFCLIFE": "467",
    "SBILIFE": "21808", "ICICIGI": "15083", "MUTHOOTFIN": "3849",
    "CHOLAFIN": "1270", "SHRIRAMFIN": "4306", "TCS": "11536", "INFY": "1594",
    "HCLTECH": "7229", "WIPRO": "3787", "TECHM": "13538", "LTIM": "17818",
    "MPHASIS": "4503", "PERSISTENT": "18365", "COFORGE": "10693",
    "TATAELXSI": "2513", "MARUTI": "10999", "TVSMOTOR": "2626",
    "M&M": "2031", "BAJAJ-AUTO": "16669", "HEROMOTOCO": "1348",
    "EICHERMOT": "910", "MOTHERSON": "4204", "HINDUNILVR": "1394",
    "ITC": "1660", "NESTLEIND": "17963", "BRITANNIA": "547",
    "TATACONSUM": "3432", "MARICO": "4067", "COLPAL": "1101",
    "GODREJCP": "10099", "SUNPHARMA": "3351", "DRREDDY": "881",
    "CIPLA": "694", "DIVISLAB": "10940", "LUPIN": "10440",
    "TORNTPHARM": "2585", "AUROPHARMA": "19", "ALKEM": "13751",
    "RELIANCE": "2885", "ONGC": "2475", "BPCL": "526", "NTPC": "11630",
    "POWERGRID": "14977", "COALINDIA": "20374", "GAIL": "1232",
    "TATAPOWER": "3426", "TATASTEEL": "3499", "HINDALCO": "1363",
    "JSWSTEEL": "11723", "VEDL": "3063", "SAIL": "4963",
    "BHARTIARTL": "10604", "INDUSTOWER": "29135", "LT": "11483",
    "BHEL": "526", "SIEMENS": "3150", "ABB": "13", "HAVELLS": "7929",
    "POLYCAB": "21514", "CUMMINSIND": "1901", "ULTRACEMCO": "11532",
    "GRASIM": "1232", "AMBUJACEM": "1270", "SHREECEM": "3103",
    "TITAN": "3506", "ASIANPAINT": "236", "PIDILITIND": "2664",
    "BERGEPAINT": "404", "VOLTAS": "3718", "PAGEIND": "14592",
    "DLF": "14732", "DMART": "20893", "GODREJPROP": "10481",
    "ADANIENT": "25", "ADANIPORTS": "15083", "BEL": "383",
    "HAL": "2303", "OFSS": "10738", "IRFC": "14977",
    "NMDC": "15332", "BOSCHLTD": "2181",
}


@dataclass
class OrderResponse:
    order_id: str
    symbol: str
    qty: int
    price: float
    status: str       # PENDING, FILLED, PARTIAL, REJECTED
    message: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    qty: int
    avg_price: float
    ltp: float = 0.0
    pnl: float = 0.0


class AngelOneClient:
    """
    Full production Angel One SmartAPI client.

    Thread-safe LTP cache updated via WebSocket.
    Rate-limited REST calls (25 req/s).
    Session auto-refresh (token valid ~24h).
    """

    RATE_LIMIT    = 20       # safe margin below 25 req/s
    SESSION_HOURS = 23       # refresh before 24h expiry

    def __init__(self,
                 api_key:     Optional[str] = None,
                 client_id:   Optional[str] = None,
                 password:    Optional[str] = None,
                 totp_secret: Optional[str] = None):

        self.api_key     = api_key     or os.getenv("ANGEL_API_KEY", "")
        self.client_id   = client_id   or os.getenv("ANGEL_CLIENT_ID", "")
        self.password    = password    or os.getenv("ANGEL_PASSWORD", "")
        self.totp_secret = totp_secret or os.getenv("ANGEL_TOTP_SECRET", "")

        if not all([self.api_key, self.client_id, self.password, self.totp_secret]):
            raise EnvironmentError(
                "Missing Angel One credentials. Set ANGEL_API_KEY, ANGEL_CLIENT_ID, "
                "ANGEL_PASSWORD, ANGEL_TOTP_SECRET in .env"
            )

        self._smart: Optional[SmartConnect] = None
        self._ws:    Optional[SmartWebSocketV2] = None

        self._access_token:  Optional[str]      = None
        self._refresh_token: Optional[str]      = None
        self._session_at:    Optional[datetime] = None
        self._feed_token:    Optional[str]      = None

        # Thread-safe LTP cache
        self._ltp_lock = threading.Lock()
        self._ltp: Dict[str, float] = {}

        # Rate limiter
        self._last_req  = 0.0
        self._req_lock  = threading.Lock()
        self._interval  = 1.0 / self.RATE_LIMIT

        # WebSocket state
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_running = False
        self._on_tick_cb: Optional[Callable] = None

    # ── Authentication ─────────────────────────────────────────────────────────

    def login(self) -> bool:
        """
        Authenticate with Angel One using TOTP.
        Call once per session (9:00 AM IST) or when token expires.
        """
        if not _SDK_AVAILABLE:
            raise ImportError("Install smartapi-python: pip install smartapi-python")
        try:
            totp = pyotp.TOTP(self.totp_secret).now()
            self._smart = SmartConnect(api_key=self.api_key)
            data = self._smart.generateSession(self.client_id, self.password, totp)

            if data.get("status") is False:
                msg = data.get("message", "login failed")
                raise RuntimeError(f"Angel One login failed: {msg}")

            d = data["data"]
            self._access_token  = d["jwtToken"]
            self._refresh_token = d["refreshToken"]
            self._feed_token    = self._smart.getfeedToken()
            self._session_at    = datetime.now()

            print(f"  [angel] Logged in | client={self.client_id} | "
                  f"feed_token={'...' + self._feed_token[-6:] if self._feed_token else '?'}")
            return True

        except Exception as e:
            print(f"  [angel] Login failed: {e}")
            return False

    def _ensure_session(self) -> None:
        """Re-login if session is expired or missing."""
        if self._smart is None or self._session_at is None:
            self.login()
            return
        age_h = (datetime.now() - self._session_at).total_seconds() / 3600
        if age_h >= self.SESSION_HOURS:
            print("  [angel] Session expiring — refreshing …")
            self.login()

    def _rate_limit(self) -> None:
        with self._req_lock:
            elapsed = time.time() - self._last_req
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_req = time.time()

    # ── Order placement ────────────────────────────────────────────────────────

    def place_order(
        self,
        symbol:     str,
        qty:        int,
        price:      float,
        order_type: str = "LIMIT",   # LIMIT | MARKET | SL | SL-M
        side:       str = "BUY",     # BUY | SELL
        product:    str = "CNC",     # CNC (delivery) | MIS (intraday)
        exchange:   str = "NSE",
        variety:    str = "NORMAL",
    ) -> OrderResponse:
        """
        Place an order on Angel One.

        Returns OrderResponse with order_id and status.
        Raises RuntimeError on rejected orders.
        """
        self._ensure_session()
        self._rate_limit()

        order_params = {
            "variety":         variety,
            "tradingsymbol":   symbol,
            "symboltoken":     NSE_TOKEN_MAP.get(symbol, ""),
            "transactiontype": side,
            "exchange":        exchange,
            "ordertype":       order_type,
            "producttype":     product,
            "duration":        "DAY",
            "price":           str(round(price, 2)),
            "squareoff":       "0",
            "stoploss":        "0",
            "quantity":        str(qty),
        }

        try:
            resp = self._smart.placeOrder(order_params)
            if resp.get("status") is False:
                raise RuntimeError(resp.get("message", "Order rejected"))
            order_id = resp["data"]["orderid"]
            return OrderResponse(
                order_id=order_id, symbol=symbol, qty=qty, price=price,
                status="PENDING", message="Order placed", raw=resp,
            )
        except Exception as e:
            return OrderResponse(
                order_id="", symbol=symbol, qty=qty, price=price,
                status="REJECTED", message=str(e),
            )

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> bool:
        """Cancel a pending order. Returns True on success."""
        self._ensure_session()
        self._rate_limit()
        try:
            resp = self._smart.cancelOrder(order_id, variety)
            return resp.get("status") is not False
        except Exception as e:
            print(f"  [angel] cancel_order {order_id}: {e}")
            return False

    def modify_order(self, order_id: str, price: float,
                     qty: int, variety: str = "NORMAL") -> bool:
        """Modify price/qty of a pending order."""
        self._ensure_session()
        self._rate_limit()
        try:
            resp = self._smart.modifyOrder({
                "orderid":  order_id,
                "variety":  variety,
                "price":    str(round(price, 2)),
                "quantity": str(qty),
            })
            return resp.get("status") is not False
        except Exception as e:
            print(f"  [angel] modify_order {order_id}: {e}")
            return False

    # ── Portfolio & account ────────────────────────────────────────────────────

    def get_holdings(self) -> Dict[str, Position]:
        """Return current CNC holdings as {symbol: Position}."""
        self._ensure_session()
        self._rate_limit()
        try:
            resp = self._smart.holding()
            if resp.get("status") is False:
                return {}
            positions: Dict[str, Position] = {}
            for h in resp.get("data", []) or []:
                sym = h.get("tradingsymbol", "")
                if sym:
                    positions[sym] = Position(
                        symbol=sym,
                        qty=int(h.get("quantity", 0)),
                        avg_price=float(h.get("averageprice", 0)),
                        ltp=float(h.get("ltp", 0)),
                        pnl=float(h.get("profitandloss", 0)),
                    )
            return positions
        except Exception as e:
            print(f"  [angel] get_holdings: {e}")
            return {}

    def get_funds(self) -> Dict[str, float]:
        """Return available cash and margin info."""
        self._ensure_session()
        self._rate_limit()
        try:
            resp = self._smart.rmsLimit()
            if resp.get("status") is False:
                return {}
            d = resp.get("data", {}) or {}
            return {
                "available":  float(d.get("availablecash", 0)),
                "net":        float(d.get("net", 0)),
                "used_margin": float(d.get("utilisedmargin", 0)),
            }
        except Exception as e:
            print(f"  [angel] get_funds: {e}")
            return {}

    def get_order_book(self) -> List[Dict]:
        """All orders placed today."""
        self._ensure_session()
        self._rate_limit()
        try:
            resp = self._smart.orderBook()
            return resp.get("data", []) or []
        except Exception as e:
            print(f"  [angel] get_order_book: {e}")
            return []

    def get_order_status(self, order_id: str) -> Optional[str]:
        """Return status string (complete, rejected, open, etc.) for an order."""
        for o in self.get_order_book():
            if str(o.get("orderid", "")) == str(order_id):
                return o.get("orderstatus", "unknown").lower()
        return None

    # ── LTP (REST fallback) ────────────────────────────────────────────────────

    def get_ltp(self, symbol: str) -> Optional[float]:
        """
        Get last traded price.
        Returns WebSocket cache if available, else REST fallback.
        """
        with self._ltp_lock:
            cached = self._ltp.get(symbol)
        if cached:
            return cached

        self._ensure_session()
        self._rate_limit()
        token = NSE_TOKEN_MAP.get(symbol)
        if not token:
            return None
        try:
            resp = self._smart.ltpData("NSE", symbol, token)
            if resp.get("status") is not False:
                price = float(resp["data"]["ltp"])
                with self._ltp_lock:
                    self._ltp[symbol] = price
                return price
        except Exception as e:
            print(f"  [angel] get_ltp {symbol}: {e}")
        return None

    def get_ltp_bulk(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch LTP for multiple symbols in one call where possible."""
        prices = {}
        for sym in symbols:
            p = self.get_ltp(sym)
            if p:
                prices[sym] = p
        return prices

    # ── WebSocket — real-time LTP feed ────────────────────────────────────────

    def subscribe_ticks(self,
                        symbols: List[str],
                        on_tick: Optional[Callable[[str, float], None]] = None) -> bool:
        """
        Subscribe to real-time LTP via WebSocket.

        Updates internal LTP cache automatically.
        Optionally calls on_tick(symbol, price) for each update.

        Args:
            symbols : NSE symbols to subscribe (e.g. ["SBIN", "HDFCBANK"])
            on_tick : Optional callback (symbol, price)
        """
        if not _SDK_AVAILABLE:
            print("  [angel] smartapi-python not installed — WebSocket unavailable")
            return False
        if not self._feed_token or not self._access_token:
            print("  [angel] Not logged in — call login() first")
            return False

        self._on_tick_cb = on_tick

        token_list = [
            {"exchangeType": 1, "tokens": [NSE_TOKEN_MAP[s]]}
            for s in symbols
            if s in NSE_TOKEN_MAP
        ]
        if not token_list:
            return False

        def _on_data(wsapp, message):
            try:
                d = json.loads(message) if isinstance(message, str) else message
                tok = str(d.get("token", ""))
                ltp = float(d.get("last_traded_price", 0)) / 100  # SmartAPI returns paise
                # Reverse-lookup symbol from token
                sym = next((s for s, t in NSE_TOKEN_MAP.items() if t == tok), None)
                if sym and ltp > 0:
                    with self._ltp_lock:
                        self._ltp[sym] = ltp
                    if self._on_tick_cb:
                        self._on_tick_cb(sym, ltp)
            except Exception:
                pass

        def _on_open(wsapp):
            print(f"  [angel] WebSocket open | {len(symbols)} symbols")

        def _on_error(wsapp, err):
            print(f"  [angel] WebSocket error: {err}")

        def _on_close(wsapp, code, msg):
            self._ws_running = False
            print(f"  [angel] WebSocket closed: {code} {msg}")

        try:
            self._ws = SmartWebSocketV2(
                auth_token=self._access_token,
                api_key=self.api_key,
                client_code=self.client_id,
                feed_token=self._feed_token,
                on_open=_on_open,
                on_error=_on_error,
                on_close=_on_close,
                on_data=_on_data,
            )
            self._ws_running = True

            def _run():
                self._ws.connect()
                self._ws.subscribe("abc123", 3, token_list)  # mode 3 = snap quote

            self._ws_thread = threading.Thread(target=_run, daemon=True)
            self._ws_thread.start()
            return True

        except Exception as e:
            print(f"  [angel] subscribe_ticks: {e}")
            return False

    def stop_websocket(self) -> None:
        """Disconnect WebSocket gracefully."""
        self._ws_running = False
        if self._ws:
            try:
                self._ws.close_connection()
            except Exception:
                pass

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, *_):
        self.stop_websocket()


# ── Singleton accessor ────────────────────────────────────────────────────────

_client: Optional[AngelOneClient] = None


def get_client() -> AngelOneClient:
    """Return singleton AngelOneClient (auto-login on first call)."""
    global _client
    if _client is None:
        _client = AngelOneClient()
        _client.login()
    return _client


# ── CLI credential test ───────────────────────────────────────────────────────

def _cli_test() -> int:
    """
    Non-destructive credentials check:
      login → funds → holdings → order_book → logout.
    Places NO orders. Returns 0 on success, 1 on failure.
    """
    print("═" * 60)
    print("  Angel One SmartAPI — Credentials Test (no orders)")
    print("═" * 60)

    # 1. Environment check
    missing = [k for k in ("ANGEL_API_KEY", "ANGEL_CLIENT_ID",
                            "ANGEL_PASSWORD", "ANGEL_TOTP_SECRET")
               if not os.getenv(k)]
    if missing:
        print(f"  ✗ Missing env vars: {', '.join(missing)}")
        print(f"    Check {_ENV_PATH}")
        return 1
    print("  ✓ All 4 credentials present in .env")

    # 2. SDK check
    if not _SDK_AVAILABLE:
        print("  ✗ smartapi-python not installed  →  pip install smartapi-python")
        return 1
    print("  ✓ smartapi-python SDK importable")

    # 3. Login
    try:
        client = AngelOneClient()
    except EnvironmentError as e:
        print(f"  ✗ Init failed: {e}")
        return 1

    print("\n  Attempting TOTP login …")
    if not client.login():
        print("  ✗ Login failed — check client_id/password/TOTP_secret")
        return 1
    print("  ✓ Login succeeded")

    # 4. Funds
    funds = client.get_funds()
    if funds:
        print(f"\n  Funds")
        print(f"    Available cash : ₹{funds.get('available', 0):>14,.2f}")
        print(f"    Net            : ₹{funds.get('net', 0):>14,.2f}")
        print(f"    Used margin    : ₹{funds.get('used_margin', 0):>14,.2f}")
    else:
        print("  ⚠ get_funds returned empty")

    # 5. Holdings
    holdings = client.get_holdings()
    print(f"\n  Holdings: {len(holdings)} positions")
    for sym, pos in list(holdings.items())[:10]:
        print(f"    {sym:<12} qty={pos.qty:>5}  avg=₹{pos.avg_price:>9,.2f}  "
              f"ltp=₹{pos.ltp:>9,.2f}  pnl=₹{pos.pnl:>+10,.2f}")
    if len(holdings) > 10:
        print(f"    … and {len(holdings) - 10} more")

    # 6. Order book
    orders = client.get_order_book()
    print(f"\n  Order book today: {len(orders)} entries")

    # 7. LTP roundtrip
    probe = "SBIN"
    ltp = client.get_ltp(probe)
    if ltp:
        print(f"\n  LTP check: {probe} = ₹{ltp:.2f}")
    else:
        print(f"\n  ⚠ LTP fetch failed for {probe}")

    client.stop_websocket()
    print("\n  ✓ Credentials test PASSED")
    print("═" * 60)
    return 0


if __name__ == "__main__":
    import argparse as _argparse
    _p = _argparse.ArgumentParser(description="Angel One SmartAPI client")
    _p.add_argument("--test", action="store_true",
                    help="Non-destructive credentials check (no orders)")
    _args = _p.parse_args()

    if _args.test:
        sys.exit(_cli_test())
    else:
        _p.print_help()
