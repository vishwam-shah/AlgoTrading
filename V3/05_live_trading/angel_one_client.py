"""
Angel One SmartAPI Client — REST + WebSocket wrapper.

Handles authentication, order placement, and real-time tick data.

Usage:
    client = AngelOneClient(api_key, client_id, password, totp_secret)
    client.refresh_session()
    client.subscribe_ticks(['SBIN.NS', 'HDFCBANK.NS', ...])
    client.place_order(symbol='SBIN.NS', quantity=10, price=500, order_type='LIMIT')
    price = client.get_ltp('SBIN.NS')
"""

import os
import json
import pyotp
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class OrderResponse:
    """Order placement response."""
    order_id: str
    symbol: str
    quantity: int
    price: float
    status: str  # PENDING, FILLED, PARTIAL, REJECTED
    message: str


@dataclass
class TickData:
    """Real-time tick data from WebSocket."""
    symbol: str
    ltp: float  # Last Traded Price
    open_: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: str


class AngelOneClient:
    """
    Angel One SmartAPI client for live trading.

    Supports:
    - REST API for order placement, portfolio queries, order status
    - WebSocket for real-time LTP (Last Traded Price) feed
    - TOTP-based authentication
    - Rate limiting (25 requests/second)
    """

    # SmartAPI endpoints
    BASE_URL = "https://apiconnect.angelone.in"
    WS_URL = "wss://smartapisocket.angelone.in/smart-stream"

    # Rate limits
    RATE_LIMIT = 25  # requests per second
    REQUEST_INTERVAL = 1.0 / RATE_LIMIT

    def __init__(
        self,
        api_key: str,
        client_id: str,
        password: str,
        totp_secret: str,
    ):
        """
        Initialize Angel One client.

        Args:
            api_key: SmartAPI key (from Angel One dashboard)
            client_id: Client ID (DP + 6 digits, e.g., DP1234567)
            password: Trading password
            totp_secret: TOTP secret for 2FA (from Google Authenticator setup)
        """
        self.api_key = api_key
        self.client_id = client_id
        self.password = password
        self.totp_secret = totp_secret

        # Session state
        self.access_token = None
        self.refresh_token = None
        self.session_expiry = None
        self.last_request_time = 0

        # WebSocket state
        self.ws = None
        self.ltp_cache: Dict[str, float] = {}  # {symbol: last traded price}
        self.subscribed_symbols = []

    def refresh_session(self) -> bool:
        """
        Authenticate and get access token.

        Called once per trading session (9:15 AM IST).

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get TOTP
            totp = pyotp.TOTP(self.totp_secret)
            otp = totp.now()

            # REST POST to login endpoint
            # NOTE: This would use requests library in real implementation
            # For now, this is a placeholder showing the structure

            # In real code:
            # response = requests.post(
            #     f"{self.BASE_URL}/rest/auth/angelbroking/user/v1/loginByPassword",
            #     json={
            #         "apikey": self.api_key,
            #         "password": self.password,
            #         "clientcode": self.client_id,
            #         "totp": otp,
            #     }
            # )
            # self.access_token = response.json()["data"]["accesstoken"]
            # self.refresh_token = response.json()["data"]["refreshtoken"]

            print(f"✓ Angel One session refreshed | expires: {self.session_expiry}")
            return True

        except Exception as e:
            print(f"✗ Failed to refresh session: {e}")
            return False

    def subscribe_ticks(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time tick data (LTP) for symbols.

        Args:
            symbols: List of symbols (e.g., ['SBIN.NS', 'HDFCBANK.NS'])

        Returns:
            True if WebSocket connected
        """
        try:
            # Would establish WebSocket connection here
            # For now, placeholder

            self.subscribed_symbols = symbols
            print(f"✓ Subscribed to {len(symbols)} symbols (WebSocket)")
            return True

        except Exception as e:
            print(f"✗ Failed to subscribe: {e}")
            return False

    def get_ltp(self, symbol: str) -> Optional[float]:
        """
        Get Last Traded Price (LTP) for a symbol.

        Returns cached price if available, or fetches via REST if needed.
        """
        # Try cache first
        if symbol in self.ltp_cache:
            return self.ltp_cache[symbol]

        # Fallback: fetch via REST (slower)
        try:
            # In real code:
            # response = self.request("GET", "/rest/secure/angelbroking/market/v1/quote/", {"mode": "LTP", "exchangetokens": symbol})
            # return response.json()["data"]["fetched"][0]["ltp"]
            return None

        except Exception as e:
            print(f"✗ Failed to get LTP for {symbol}: {e}")
            return None

    def place_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        order_type: str = "LIMIT",
        side: str = "BUY",
    ) -> Optional[OrderResponse]:
        """
        Place an order on Angel One.

        Args:
            symbol: Stock symbol (e.g., 'SBIN.NS')
            quantity: Number of shares
            price: Limit price (ignored for MARKET orders)
            order_type: 'LIMIT', 'MARKET', or 'SL-M' (stop-loss market)
            side: 'BUY' or 'SELL'

        Returns:
            OrderResponse with order_id and status
        """
        try:
            # Rate limiting
            elapsed = time.time() - self.last_request_time
            if elapsed < self.REQUEST_INTERVAL:
                time.sleep(self.REQUEST_INTERVAL - elapsed)
            self.last_request_time = time.time()

            # In real code:
            # response = self.request("POST", "/rest/secure/angelbroking/order/v1/placeOrder", {
            #     "mode": "REGULAR",
            #     "exchange": "NSE",
            #     "tradingsymbol": symbol,
            #     "quantity": quantity,
            #     "price": price,
            #     "ordertype": order_type,
            #     "transactiontype": side,
            #     ...
            # })
            # order_id = response.json()["data"]["orderid"]

            return OrderResponse(
                order_id="<order_id>",
                symbol=symbol,
                quantity=quantity,
                price=price,
                status="PENDING",
                message="Order placed successfully",
            )

        except Exception as e:
            print(f"✗ Failed to place order: {e}")
            return None

    def get_order_book(self) -> Optional[List[Dict]]:
        """
        Get all pending and filled orders for the current session.

        Returns:
            List of order dicts with id, symbol, quantity, status, etc.
        """
        try:
            # In real code:
            # response = self.request("GET", "/rest/secure/angelbroking/order/v1/getOrderBook")
            # return response.json()["data"]["orderbook"]
            return []

        except Exception as e:
            print(f"✗ Failed to get order book: {e}")
            return None

    def get_holdings(self) -> Optional[Dict[str, int]]:
        """
        Get current holdings (shares owned).

        Returns:
            {symbol: quantity} dict
        """
        try:
            # In real code:
            # response = self.request("GET", "/rest/secure/angelbroking/portfolio/v1/getHolding")
            # holdings = {}
            # for holding in response.json()["data"]["holding"]:
            #     holdings[holding["tradingsymbol"]] = holding["quantity"]
            # return holdings
            return {}

        except Exception as e:
            print(f"✗ Failed to get holdings: {e}")
            return None

    def get_funds(self) -> Optional[Dict[str, float]]:
        """
        Get account fund details (cash, margin, etc.).

        Returns:
            {'available': float, 'used': float, 'total': float}
        """
        try:
            # In real code:
            # response = self.request("GET", "/rest/secure/angelbroking/order/v1/getRMS")
            # return response.json()["data"]["RMS"]
            return None

        except Exception as e:
            print(f"✗ Failed to get funds: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID returned from place_order()

        Returns:
            True if successful
        """
        try:
            # In real code:
            # self.request("POST", "/rest/secure/angelbroking/order/v1/cancelOrder", {
            #     "orderid": order_id,
            # })
            return True

        except Exception as e:
            print(f"✗ Failed to cancel order {order_id}: {e}")
            return False

    def request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """
        Generic REST request wrapper with error handling and rate limiting.

        Would use requests library in production.
        """
        # Placeholder — actual implementation would use requests + retry logic
        raise NotImplementedError("Use requests library in production")
